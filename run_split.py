from helpers import (
    split_video_into_scenes, 
)


if __name__ == '__main__':
    input_video_file: str = "data/Basil.mp4"
    split_video_into_scenes(input_video_file, 'data/scenes')
