from constants import SAM2_MODEL, INPUT_FRAME_DIR, OUTPUT_FRAME_DIR
from helpers import (
    prepare_temp_dir, 
    save_video_to_image_sequence,
    split_video_into_scenes
)
from remove_bg import init_config, init_model, process_frames, init_tokenizer

init_config()
tokenizer = init_tokenizer(SAM2_MODEL)
model = init_model(SAM2_MODEL)

def process_scene(scene_path: str):
    # Setup temp environment
    prepare_temp_dir(INPUT_FRAME_DIR)

    # Break video into frames
    save_video_to_image_sequence(scene_path, INPUT_FRAME_DIR)

    # Process
    frames = process_frames(
        tokenizer=tokenizer,
        model=model,
        input_frames_path=INPUT_FRAME_DIR,
        output_frames_path=OUTPUT_FRAME_DIR,
        prompt="animated cartoon characters",
        image_size=224,
    )
    return frames

def process_scenes(scene_folder_path: str):
    None

if __name__ == '__main__':
    process_scene("data/test.mp4")
