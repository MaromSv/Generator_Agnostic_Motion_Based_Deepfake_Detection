import os
import glob
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for cluster/headless environments
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# Font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


def download_video(url, output_path="basketball.mp4"):
    """Download video from URL if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"Downloading video from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Video saved to {output_path}")
    return output_path


def load_video_frames(video_path):
    """Load video frames for visualization."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            video_frames.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            print(
                "Frame names not in '<frame_index>.jpg' format, using lexicographic sort."
            )
            video_frames.sort()
    return video_frames


def propagate_in_video(predictor, session_id):
    """Propagate segmentation from frame 0 to end of video."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def segment_video(video_path, prompt_text_str="object", output_dir="output"):
    """
    Segment objects in a video using SAM3.

    Args:
        video_path: Path to the input video file
        prompt_text_str: Text prompt for segmentation (default: "object" to segment everything)
        output_dir: Directory to save output files (default: "output")
    """
    # Use all available GPUs
    gpus_to_use = range(torch.cuda.device_count())

    # Build the SAM3 video predictor
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Load video frames for visualization
    video_frames_for_vis = load_video_frames(video_path)

    # Start inference session on this video
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    # Add text prompt to segment objects
    frame_idx = 0

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    out = response["outputs"]

    # Visualize initial frame
    plt.close("all")
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "frame_0.png"))

    # Propagate outputs through the entire video
    outputs_per_frame = propagate_in_video(predictor, session_id)

    # Reformat outputs for visualization
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # Save masks and visualizations for each frame
    vis_frame_stride = 60
    for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        plt.close("all")
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )
        plt.savefig(os.path.join(output_dir, "masks", f"frame_{frame_idx:05d}.png"))

    # Save segmented video
    video_filename = os.path.basename(video_path).replace(".mp4", "_segmented.mp4")
    save_segmented_video(
        video_frames_for_vis,
        outputs_per_frame,
        os.path.join(output_dir, video_filename),
        fps=60,
    )

    # Close session to free resources
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    # Shutdown predictor
    predictor.shutdown()

    print(f"Segmentation complete! Output saved to '{output_dir}/' directory.")


def save_segmented_video(video_frames, outputs_per_frame, output_path, fps=30):
    """Save the segmented video with mask overlays."""
    if not video_frames:
        return

    # Get frame dimensions
    if isinstance(video_frames[0], str):
        sample_frame = cv2.imread(video_frames[0])
        height, width = sample_frame.shape[:2]
    else:
        height, width = video_frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(len(video_frames)):
        # Load frame
        if isinstance(video_frames[frame_idx], str):
            frame = cv2.imread(video_frames[frame_idx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = video_frames[frame_idx].copy()

        # Overlay masks if available
        if frame_idx in outputs_per_frame:
            frame_output = outputs_per_frame[frame_idx]
            if "masks" in frame_output:
                for obj_id, mask in frame_output["masks"].items():
                    # Create colored overlay for each object
                    color = plt.cm.tab10(obj_id % 10)[:3]
                    color = tuple(int(c * 255) for c in color)
                    mask_bool = (
                        mask > 0.5
                        if isinstance(mask, np.ndarray)
                        else mask.numpy() > 0.5
                    )
                    frame[mask_bool] = (
                        frame[mask_bool] * 0.5 + np.array(color) * 0.5
                    ).astype(np.uint8)

        # Write frame
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Segmented video saved to {output_path}")


if __name__ == "__main__":
    # Example usage - download and segment a sample video

    # Get script directory to make paths work regardless of working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(script_dir, "videos", "ai", "ai (7).mp4")
    output_dir = os.path.join(script_dir, "results")

    segment_video(video_path, prompt_text_str="object", output_dir=output_dir)
