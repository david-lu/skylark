import os

from scenedetect import AdaptiveDetector

from helpers import (
    split_video_into_scenes, 
)
def get_scene_folder(input_video_file: str):
    return os.path.join(
        os.path.dirname(input_video_file),
        os.path.splitext(os.path.basename(input_video_file))[0]
    )

if __name__ == '__main__':
    input_video_file: str = "data/little_nemo.mkv"
    output_dir = get_scene_folder(input_video_file)
    print(output_dir)
    split_video_into_scenes(input_video_file, output_dir, [AdaptiveDetector(window_width=3)])
