from helpers import split_video_into_scenes
import os

SAM2_MODEL = "YxZhang/evf-sam2"


def process_scene_video(scene_path: str):
    
    None

if __name__ == '__main__':
    input_video_file: str = "data/Basil.mp4"
    split_video_into_scenes(input_video_file, 'data/scenes')
