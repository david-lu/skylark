from helpers import split_video_into_scenes
import os

SAM2_MODEL = "YxZhang/evf-sam2"

if __name__ == '__main__':
    input_video_file: str = "data/Basil.mp4"
    
    if not os.path.isfile(input_video_file):
        print(f"Input video file not found: {input_video_file}")
    else:
        split_video_into_scenes(input_video_file, 'data/scenes')