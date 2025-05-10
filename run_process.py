from helpers import (
    prepare_temp_dir, 
    save_video_to_image_sequence,
)
from remove_bg import init_config, init_models, process_frames
import os

SAM2_MODEL = "YxZhang/evf-sam2"
TEMP_DIR = "tmp"

init_config()
model, tokenizer = init_models(SAM2_MODEL)

def process_scene_video(scene_path: str):
    prepare_temp_dir(TEMP_DIR)
    save_video_to_image_sequence(scene_path, TEMP_DIR)
    frames = process_frames(
        tokenizer=tokenizer,
        model=model,
        input_frames_path=os.path.join(TEMP_DIR, "input"),
        output_frames_path=os.path.join(TEMP_DIR, "output"),
        prompt="animated cartoon characters",
        image_size=224
    )
    return frames

if __name__ == '__main__':
    process_scene_video("data/Basil-Scene-0004.mp4")
