# Ensure you have PySceneDetect and FFmpeg installed.
# You can install PySceneDetect using pip:
# pip install scenedetect[opencv]
#
# FFmpeg needs to be installed on your system and accessible in your PATH
# for the scene splitting functionality.
# You can download it from https://ffmpeg.org/download.html
# OpenCV (cv2) is used for image writing.

import os
import cv2 # OpenCV for image writing
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg # For splitting video
from scenedetect.video_manager import VideoManager # For type hinting video object
from typing import List, Union, Optional, Tuple # For type hinting
import numpy # For type hinting frame_data

def split_video_into_scenes(video_path: str, output_path: str) -> List[str]:
    """
    Detects scenes in a video and saves them as individual clips.

    Args:
        video_path (str): The full path to the input video file.
        output_path (str): The directory where scene clips will be saved.
    
    Returns:
        List[str]: A list of filenames of the generated scene video files (relative to output_path).
                   Returns an empty list if no scenes are detected or an error occurs.
    """
    video: Optional[VideoManager] = None
    generated_scene_filenames: List[str] = [] # Changed variable name for clarity
    try:
        if not os.path.isfile(video_path):
            print(f"Error: Video file not found at {video_path}")
            return generated_scene_filenames

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory for scene clips: {output_path}")

        video = open_video(video_path)
        scene_manager: SceneManager = SceneManager()
        scene_manager.add_detector(ContentDetector()) 

        base_timecode: FrameTimecode = video.frame_rate
        print(f"Detecting scenes in '{os.path.basename(video_path)}' for video splitting...")
        scene_manager.detect_scenes(video=video, show_progress=True)
        scene_list: List[Tuple[FrameTimecode, FrameTimecode]] = scene_manager.get_scene_list(base_timecode)

        if not scene_list:
            print("No scenes detected in the video for splitting.")
            return generated_scene_filenames

        print(f"Detected {len(scene_list)} scenes for video splitting.")
        video_filename: str = os.path.basename(video_path)
        video_name_without_ext, video_ext = os.path.splitext(video_filename)

        output_template = f'{video_name_without_ext}_$SCENE_NUMBER{video_ext}'
        print(f"Splitting video into scene clips and saving to '{output_path}' with template '{output_template}'...")
        
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=output_path,
            output_file_template=output_template,
            show_progress=True,
        )

        for i in range(len(scene_list)):
            scene_number_str = f"{i+1:03d}" # $SCENE_NUMBER is 1-indexed and typically 3 digits
            expected_filename = output_template.replace("$SCENE_NUMBER", scene_number_str)
            full_path = os.path.join(output_path, expected_filename) # Still need full_path to check existence
            if os.path.exists(full_path): 
                generated_scene_filenames.append(expected_filename) # Append only the filename
            else:
                print(f"Warning: Expected scene file not found: {full_path}")
        
        if generated_scene_filenames:
            print(f"Scene video splitting complete. {len(generated_scene_filenames)} scene clips created.")
        else:
            print("Scene video splitting finished, but no scene files were confirmed.")
            
    except Exception as e:
        print(f"An error occurred during scene video splitting: {e}")
    finally:
        if video and video.is_open():
            video.release()
    return generated_scene_filenames

def save_video_to_image_sequence(video_path: str, output_images_path: str, image_prefix: str = "frame_", image_format: str = "png") -> List[str]:
    """
    Converts an entire video (or video clip) into a sequence of its frames.
    Filename format: image_prefix<frame_number>.<image_format>

    Args:
        video_path (str): The full path to the input video file (can be a scene clip).
        output_images_path (str): The directory where image sequences will be saved.
        image_prefix (str): Prefix for the output image filenames.
        image_format (str): Format for the output images (e.g., 'png', 'jpg').

    Returns:
        List[str]: A list of full paths to the generated image files.
                   Returns an empty list if an error occurs or no frames are saved.
    """
    video: Optional[VideoManager] = None
    generated_image_files: List[str] = []
    try:
        if not os.path.isfile(video_path):
            print(f"Error: Video file for frame extraction not found at {video_path}")
            return generated_image_files

        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
            print(f"Created output directory for frame images: {output_images_path}")

        video = open_video(video_path)
        print(f"Extracting frames from '{os.path.basename(video_path)}' to '{output_images_path}' with prefix '{image_prefix}'...")
        frame_num: int = 0
        while True:
            frame_data: Union[numpy.ndarray, bool] = video.read()
            if frame_data is False: 
                break
            
            current_frame: numpy.ndarray = frame_data
            image_filename: str = os.path.join(output_images_path, f"{image_prefix}{frame_num:04d}.{image_format}")
            if cv2.imwrite(image_filename, current_frame):
                generated_image_files.append(image_filename) # Add to list if save is successful
                if frame_num % 100 == 0: 
                    print(f"  Saved {image_filename}")
            else:
                print(f"  Warning: Could not write frame {frame_num} from {os.path.basename(video_path)} to {image_filename}")
            frame_num += 1
        print(f"  Finished extracting {frame_num} frames from '{os.path.basename(video_path)}'.")
    except Exception as e:
        print(f"An error occurred during image sequence extraction for {video_path}: {e}")
    finally:
        if video and video.is_open():
            video.release()
    return generated_image_files
