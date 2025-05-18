from constants import SAM2_MODEL, TMP_INPUT_FRAME_DIR, TMP_OUTPUT_FRAME_DIR
import shutil
import os
from helpers import (
    prepare_temp_dir, 
    save_video_to_image_sequence
)
from remove_bg import init_config, init_model, process_frames, init_tokenizer

init_config()
tokenizer = init_tokenizer(SAM2_MODEL)
model = init_model(SAM2_MODEL)

def process_scene(scene_path: str):
    # Setup temp environment
    prepare_temp_dir(TMP_INPUT_FRAME_DIR)

    # Break video into frames
    save_video_to_image_sequence(scene_path, TMP_INPUT_FRAME_DIR)

    # Process
    frames = process_frames(
        tokenizer=tokenizer,
        model=model,
        input_frames_path=TMP_INPUT_FRAME_DIR,
        output_frames_path=TMP_OUTPUT_FRAME_DIR,
        prompt="animated cartoon characters",
        image_size=224,
    )
    return frames

def process_scenes(scene_folder_path: str):
    for scene_file in os.listdir(scene_folder_path):
        scene_path = os.path.join(scene_folder_path, scene_file)
        if os.path.isfile(scene_path):
            process_scene(scene_path)
            scene_output_dir = os.path.join(scene_folder_path, os.path.splitext(scene_file)[0])
            os.makedirs(scene_output_dir, exist_ok=True)
            for frame_file in os.listdir(TMP_OUTPUT_FRAME_DIR):
                frame_path = os.path.join(TMP_OUTPUT_FRAME_DIR, frame_file)
                shutil.move(frame_path, os.path.join(scene_output_dir, frame_file))

if __name__ == '__main__':
    process_scene("data/test.mp4")
