import os
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video as hf_load_video
from accelerate import Accelerator


def calculate_centroid(mask):
    """Calculate the centroid of a mask."""
    mask_array = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    mask_bool = mask_array > 0.5

    if not mask_bool.any():
        return None

    y_coords, x_coords = np.where(mask_bool)
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))

    return (centroid_x, centroid_y)


def save_video_with_masks(video_frames, video_segments, output_path, fps=30):
    """Save video with colored mask overlays."""
    if len(video_frames) == 0:
        return

    # Get dimensions from first frame
    height, width = video_frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Color map for objects
    cmap = plt.cm.get_cmap("tab10")

    for frame_idx, frame in enumerate(video_frames):
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        frame_display = frame.copy()

        # Overlay masks if available for this frame
        if frame_idx in video_segments:
            frame_data = video_segments[frame_idx]
            masks = frame_data["masks"]
            obj_ids = frame_data["obj_ids"]

            # masks shape: [num_objects, H, W]
            for i, obj_id in enumerate(obj_ids):
                mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]

                # Get color for this object (use obj_id for consistent coloring)
                color = cmap(obj_id % 10)[:3]
                color = tuple(int(c * 255) for c in color)

                # Create colored overlay
                mask_bool = mask > 0.5
                frame_display[mask_bool] = (
                    frame_display[mask_bool] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)

        # Write frame (convert RGB to BGR for cv2)
        out.write(cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✓ Segmented video saved to {output_path}")


def save_video_with_tracking(video_frames, video_segments, output_path, fps=30):
    """Save video with colored mask overlays, object IDs, and movement paths."""
    if len(video_frames) == 0:
        return

    # Get dimensions from first frame
    height, width = video_frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Color map for objects
    cmap = plt.cm.get_cmap("tab10")

    # Track object paths: {obj_id: [(x, y), ...]} - keyed by actual object ID
    object_paths = {}

    for frame_idx, frame in enumerate(video_frames):
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        frame_display = frame.copy()

        # Overlay masks if available for this frame
        if frame_idx in video_segments:
            frame_data = video_segments[frame_idx]
            masks = frame_data["masks"]
            obj_ids = frame_data["obj_ids"]

            # First pass: draw all masks
            for i, obj_id in enumerate(obj_ids):
                mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]

                # Get color for this object (use obj_id for consistent coloring)
                color = cmap(obj_id % 10)[:3]
                color_rgb = tuple(int(c * 255) for c in color)

                # Create colored overlay
                mask_bool = mask > 0.5
                frame_display[mask_bool] = (
                    frame_display[mask_bool] * 0.5 + np.array(color_rgb) * 0.5
                ).astype(np.uint8)

                # Calculate centroid and store in path history
                centroid = calculate_centroid(mask)
                if centroid is not None:
                    if obj_id not in object_paths:
                        object_paths[obj_id] = []
                    object_paths[obj_id].append(centroid)

            # Convert to BGR for drawing
            frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            # Second pass: draw IDs and paths for all objects
            for i, obj_id in enumerate(obj_ids):
                mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]

                # Get color for this object
                color = cmap(obj_id % 10)[:3]
                color_bgr = tuple(int(c * 255) for c in reversed(color))

                centroid = calculate_centroid(mask)

                if centroid is not None:
                    # Draw object ID
                    text = f"ID: {obj_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, font, font_scale, thickness
                    )

                    # Draw background rectangle
                    text_x, text_y = centroid
                    cv2.rectangle(
                        frame_bgr,
                        (text_x - 5, text_y - text_height - 10),
                        (text_x + text_width + 5, text_y + 5),
                        (0, 0, 0),
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        frame_bgr,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        color_bgr,
                        thickness,
                        cv2.LINE_AA,
                    )

                    # Draw movement path
                    if obj_id in object_paths and len(object_paths[obj_id]) > 1:
                        path_points = object_paths[obj_id]
                        for j in range(len(path_points) - 1):
                            cv2.line(
                                frame_bgr,
                                path_points[j],
                                path_points[j + 1],
                                color_bgr,
                                2,
                                cv2.LINE_AA,
                            )

                        # Draw small circles at each point in the path
                        for point in path_points[:-1]:  # Skip the current point
                            cv2.circle(frame_bgr, point, 3, color_bgr, -1)

            # Write frame
            out.write(frame_bgr)
        else:
            # No masks for this frame, write as-is
            out.write(cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
            continue

    out.release()
    print(f"✓ Tracked video saved to {output_path}")


def visualize_frame(frame, frame_data, output_path, frame_idx):
    """Visualize a single frame with masks."""
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)

    masks = frame_data["masks"] if frame_data is not None else None
    obj_ids = frame_data["obj_ids"] if frame_data is not None else []

    if masks is not None and masks.shape[0] > 0:
        cmap = plt.cm.get_cmap("tab10")

        # Iterate over the actual number of masks
        num_masks = masks.shape[0]
        for i in range(num_masks):
            obj_id = obj_ids[i] if i < len(obj_ids) else i
            mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
            color = cmap(obj_id % 10)[:3]

            # Show mask overlay
            mask_bool = mask > 0.5
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask_bool] = (*color, 0.5)
            plt.imshow(colored_mask)

    num_objects = masks.shape[0] if masks is not None else 0
    plt.title(f"Frame {frame_idx} - Detected {num_objects} objects")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


def segment_video(video_path, prompt_text_str="object", output_dir="output"):
    print("=" * 60)
    print("SAM3 Video Segmentation with Transformers")
    print("=" * 60)

    # Setup device
    device = Accelerator().device
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading SAM3 model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
        device, dtype=torch.bfloat16
    )
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("✓ Model loaded successfully")

    # Load video frames
    print(f"Loading video frames from {video_path}...")
    video_frames, _ = hf_load_video(video_path)
    print(f"✓ Loaded {len(video_frames)} frames")

    # Initialize video inference session
    print("Initializing inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    print("✓ Session initialized")

    # Add text prompt
    print(f"Adding text prompt: '{prompt_text_str}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt_text_str,
    )
    print("✓ Prompt added")

    # Process all frames
    print("Processing video frames...")
    video_segments = {}

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=len(video_frames)
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        frame_idx = model_outputs.frame_idx

        # Store both masks and object IDs for consistent tracking
        # object_ids come from the model and remain consistent across frames
        obj_ids = processed_outputs.get("object_ids")
        num_masks = (
            processed_outputs["masks"].shape[0]
            if processed_outputs["masks"] is not None
            else 0
        )

        # Ensure obj_ids matches the number of masks
        if len(obj_ids) > num_masks:
            obj_ids = obj_ids[:num_masks]
        elif len(obj_ids) < num_masks:
            # Extend with sequential IDs if needed
            max_id = max(obj_ids) if obj_ids else -1
            obj_ids.extend(range(max_id + 1, max_id + 1 + (num_masks - len(obj_ids))))

        video_segments[frame_idx] = {
            "masks": processed_outputs["masks"],
            "obj_ids": obj_ids,
        }

        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1}/{len(video_frames)} frames...")

    print(f"✓ Processed all {len(video_segments)} frames")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    # Get first frame results
    frame_0_data = video_segments[0]
    num_objects = (
        frame_0_data["masks"].shape[0] if frame_0_data["masks"] is not None else 0
    )
    print(f"✓ Detected {num_objects} objects in first frame")
    print(f"  Object IDs: {frame_0_data['obj_ids']}")

    # Visualize first frame
    print("Creating visualizations...")
    visualize_frame(
        video_frames[0],
        frame_0_data,
        os.path.join(output_dir, "frames", "frame_0000.png"),
        0,
    )

    # Visualize every 60th frame
    for frame_idx in range(0, len(video_frames), 60):
        if frame_idx in video_segments:
            visualize_frame(
                video_frames[frame_idx],
                video_segments[frame_idx],
                os.path.join(output_dir, "frames", f"frame_{frame_idx:04d}.png"),
                frame_idx,
            )

    # Save segmented video
    print("Saving segmented video...")
    save_video_with_masks(
        video_frames,
        video_segments,
        os.path.join(output_dir, "segmented_video.mp4"),
        fps=30,
    )

    # Save tracked video with IDs and paths
    print("Saving tracked video with IDs and paths...")
    save_video_with_tracking(
        video_frames,
        video_segments,
        os.path.join(output_dir, "tracked_video.mp4"),
        fps=30,
    )

    print("=" * 60)
    print("✓ Segmentation complete!")
    print(f"  - Detected objects: {num_objects}")
    print(f"  - Processed frames: {len(video_segments)}")
    print(f"  - Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # video_path = os.path.join(script_dir, "videos", "real", "basketball.mp4")
    video_path = os.path.join(script_dir, "videos", "ai", "ai (23).mp4")
    output_dir = os.path.join(script_dir, "results")

    segment_video(video_path, prompt_text_str="body parts", output_dir=output_dir)
