import os
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video as hf_load_video
from accelerate import Accelerator


# Color map for consistent object coloring
CMAP = plt.cm.get_cmap("tab10")


def get_object_color(obj_id, as_bgr=False):
    """Get consistent color for an object ID."""
    color = CMAP(obj_id % 10)[:3]
    color_tuple = tuple(int(c * 255) for c in color)
    if as_bgr:
        return tuple(reversed(color_tuple))
    return color_tuple


def calculate_centroid(mask):
    """Calculate the centroid of a mask."""
    mask_array = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    mask_bool = mask_array > 0.5
    if not mask_bool.any():
        return None
    y_coords, x_coords = np.where(mask_bool)
    return (int(np.mean(x_coords)), int(np.mean(y_coords)))


def draw_mask_overlay(frame, mask, color, alpha=0.5):
    """Draw a colored mask overlay on frame."""
    mask_array = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    mask_bool = mask_array > 0.5
    frame[mask_bool] = (
        frame[mask_bool] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)


def draw_object_label(frame_bgr, centroid, obj_id, color_bgr):
    """Draw object ID label at centroid."""
    text = f"ID: {obj_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = centroid
    cv2.rectangle(
        frame_bgr, (x - 5, y - text_h - 10), (x + text_w + 5, y + 5), (0, 0, 0), -1
    )
    cv2.putText(
        frame_bgr, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA
    )


def draw_movement_path(frame_bgr, path_points, color_bgr):
    """Draw movement path with lines and circles."""
    if len(path_points) < 2:
        return
    for i in range(len(path_points) - 1):
        cv2.line(
            frame_bgr, path_points[i], path_points[i + 1], color_bgr, 2, cv2.LINE_AA
        )
    for point in path_points[:-1]:
        cv2.circle(frame_bgr, point, 3, color_bgr, -1)


def prepare_frame(frame):
    """Convert frame to RGB if needed."""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    return frame.copy()


def save_video_with_masks(video_frames, video_segments, output_path, fps=30):
    """Save video with colored mask overlays."""
    if len(video_frames) == 0:
        return

    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx, frame in enumerate(video_frames):
        frame_display = prepare_frame(frame)

        if frame_idx in video_segments:
            masks = video_segments[frame_idx]["masks"]
            obj_ids = video_segments[frame_idx]["obj_ids"]
            for i, obj_id in enumerate(obj_ids):
                draw_mask_overlay(frame_display, masks[i], get_object_color(obj_id))

        out.write(cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✓ Segmented video saved to {output_path}")


def save_video_with_tracking(video_frames, video_segments, output_path, fps=30):
    """Save video with colored mask overlays, object IDs, and movement paths."""
    if len(video_frames) == 0:
        return

    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    object_paths = {}

    for frame_idx, frame in enumerate(video_frames):
        frame_display = prepare_frame(frame)

        if frame_idx not in video_segments:
            out.write(cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
            continue

        masks = video_segments[frame_idx]["masks"]
        obj_ids = video_segments[frame_idx]["obj_ids"]
        current_centroids = {}

        # Draw masks and collect centroids
        for i, obj_id in enumerate(obj_ids):
            draw_mask_overlay(frame_display, masks[i], get_object_color(obj_id))
            centroid = calculate_centroid(masks[i])
            if centroid:
                current_centroids[obj_id] = centroid
                object_paths.setdefault(obj_id, []).append(centroid)

        # Draw labels and paths
        frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
        for obj_id, centroid in current_centroids.items():
            color_bgr = get_object_color(obj_id, as_bgr=True)
            draw_object_label(frame_bgr, centroid, obj_id, color_bgr)
            draw_movement_path(frame_bgr, object_paths.get(obj_id, []), color_bgr)

        out.write(frame_bgr)

    out.release()
    print(f"✓ Tracked video saved to {output_path}")


def load_model(device):
    """Load SAM3 model and processor."""
    print("Loading SAM3 model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
        device, dtype=torch.bfloat16
    )
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("✓ Model loaded successfully")
    return model, processor


def process_video_frames(model, processor, inference_session, num_frames):
    """Process all video frames and extract masks with object IDs."""
    print("Processing video frames...")
    video_segments = {}

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=num_frames
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        frame_idx = model_outputs.frame_idx
        masks = processed["masks"]
        num_masks = masks.shape[0] if masks is not None else 0

        # Get object IDs from model for consistent tracking
        obj_ids = model_outputs.object_ids
        if obj_ids is None:
            obj_ids = list(range(num_masks))
        elif torch.is_tensor(obj_ids):
            obj_ids = obj_ids.cpu().tolist()
        else:
            obj_ids = list(obj_ids)

        # Sync obj_ids with mask count
        if len(obj_ids) > num_masks:
            obj_ids = obj_ids[:num_masks]
        elif len(obj_ids) < num_masks:
            max_id = max(obj_ids) if obj_ids else -1
            obj_ids.extend(range(max_id + 1, max_id + 1 + num_masks - len(obj_ids)))

        video_segments[frame_idx] = {"masks": masks, "obj_ids": obj_ids}

        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")

    print(f"✓ Processed all {len(video_segments)} frames")
    return video_segments


def segment_video(video_path, prompt_text_str="object", output_dir="output"):
    """Main function to segment video and save outputs."""
    print("=" * 60)
    print("SAM3 Video Segmentation with Transformers")
    print("=" * 60)

    device = Accelerator().device
    print(f"Using device: {device}")

    model, processor = load_model(device)

    # Load video
    print(f"Loading video frames from {video_path}...")
    video_frames, _ = hf_load_video(video_path)
    print(f"✓ Loaded {len(video_frames)} frames")

    # Initialize session
    print("Initializing inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    print("✓ Session initialized")

    # Add prompt
    print(f"Adding text prompt: '{prompt_text_str}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session, text=prompt_text_str
    )
    print("✓ Prompt added")

    # Process frames
    video_segments = process_video_frames(
        model, processor, inference_session, len(video_frames)
    )

    # Get stats
    num_objects = (
        video_segments[0]["masks"].shape[0]
        if video_segments[0]["masks"] is not None
        else 0
    )
    print(f"✓ Detected {num_objects} objects (IDs: {video_segments[0]['obj_ids']})")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    print("Saving segmented video...")
    save_video_with_masks(
        video_frames, video_segments, os.path.join(output_dir, "segmented_video.mp4")
    )

    print("Saving tracked video with IDs and paths...")
    save_video_with_tracking(
        video_frames, video_segments, os.path.join(output_dir, "tracked_video.mp4")
    )

    print("=" * 60)
    print("✓ Segmentation complete!")
    print(f"  - Detected objects: {num_objects}")
    print(f"  - Processed frames: {len(video_segments)}")
    print(f"  - Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "videos", "real", "basketball.mp4")
    # video_path = os.path.join(script_dir, "videos", "ai", "ai (9).mp4")
    output_dir = os.path.join(script_dir, "results")

    segment_video(video_path, prompt_text_str="people", output_dir=output_dir)
