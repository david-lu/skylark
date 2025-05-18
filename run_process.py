from helpers import (
    prepare_temp_dir, 
    save_video_to_image_sequence,
)
from remove_bg import init_config, init_model, process_frames, init_tokenizer
import os

SAM2_MODEL = "YxZhang/evf-sam2"
TEMP_DIR = "tmp"

init_config()
tokenizer = init_tokenizer(SAM2_MODEL)
model = init_model(SAM2_MODEL)

def process_scene_video(scene_path: str):
    input_dir = os.path.join(TEMP_DIR, 'input')
    output_dir = os.path.join(TEMP_DIR, 'output')
    prepare_temp_dir(input_dir)
    save_video_to_image_sequence(scene_path, input_dir)
    frames = process_frames(
        tokenizer=tokenizer,
        model=model,
        input_frames_path=input_dir,
        output_frames_path=output_dir,
        prompt="animated cartoon characters",
        image_size=224
    )
    return frames

if __name__ == '__main__':
    process_scene_video("data/test.mp4")
