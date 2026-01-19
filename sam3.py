import sam3  # ensure Meta backend is importable first

import importlib
import samgeo.samgeo3 as s3

importlib.reload(s3)  # recompute SAM3_META_AVAILABLE inside this process

from samgeo.samgeo3 import SamGeo3Video
from samgeo import download_file
import os


def segment_video():
    sam = SamGeo3Video()

    url = (
        "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/basketball.mp4"
    )
    video_path = download_file(url)

    sam.set_video(video_path)
    sam.generate_masks("player")

    player_names = {i: f"Player {i}" for i in range(15)}
    sam.show_frame(0, axis="on", show_ids=player_names)

    os.makedirs("output", exist_ok=True)
    sam.save_masks("output/masks")
    sam.save_video("output/players_segmented.mp4", fps=60, show_ids=player_names)


if __name__ == "__main__":
    segment_video()
