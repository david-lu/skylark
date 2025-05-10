# Ensure you have PySceneDetect and FFmpeg installed.
# You can install PySceneDetect using pip:
# pip install scenedetect[opencv]
#
# FFmpeg needs to be installed on your system and accessible in your PATH
# for the scene splitting functionality.
# You can download it from https://ffmpeg.org/download.html
# OpenCV (cv2) is used for image writing.

import os
import shutil
import logging
import cv2 # OpenCV for image writing
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.video_splitter import split_video_ffmpeg # For splitting video
from scenedetect.video_manager import VideoManager # For type hinting video object
from typing import List, Union, Optional, Tuple # For type hinting
import numpy # For type hinting frame_data

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_video_into_scenes(
    video_path: str,
    output_dir: str,
    detectors: List[ContentDetector] = None,
) -> List[str]:
    """
    Splits `video_path` into scene clips under `output_dir`.
    Returns list of generated clip filenames (not full paths).
    """
    detectors = detectors or [ContentDetector(threshold=12.0, min_scene_len=12)]
    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        return []
    os.makedirs(output_dir, exist_ok=True)

    # 1) Detect scenes
    vm = VideoManager([video_path])
    sm = SceneManager()
    for det in detectors:
        sm.add_detector(det)
    vm.start()
    sm.detect_scenes(vm, show_progress=True)
    scene_list = sm.get_scene_list(vm.get_base_timecode())
    vm.release()

    if not scene_list:
        logger.warning("No scenes detected.")
        return []

    # 2) Split via FFmpeg
    basename, ext = os.path.splitext(os.path.basename(video_path))
    template = f"{basename}_$SCENE_NUMBER{ext}"
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=output_dir,
        output_file_template=template,
        show_progress=True,
    )

    # 3) Collect filenames
    clips = []
    for idx in range(len(scene_list)):
        num = f"{idx+1:03d}"
        fname = template.replace("$SCENE_NUMBER", num)
        if os.path.exists(os.path.join(output_dir, fname)):
            clips.append(fname)
        else:
            logger.warning(f"Missing expected clip: {fname}")
    return clips

def save_video_to_image_sequence(
    video_path: str,
    output_dir: str,
    prefix: str = "frame_",
) -> List[str]:
    """
    Reads `video_path` with OpenCV and writes every frame as JPEG under `output_dir`.
    Returns list of full paths to the saved images.
    """
    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        return []
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    saved_files: List[str] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(output_dir, f"{prefix}{idx:04d}.jpg")
        if cv2.imwrite(fname, frame):
            saved_files.append(fname)
        else:
            logger.warning(f"Could not write frame {idx}")
        idx += 1

    cap.release()
    logger.info(f"Extracted {idx} frames to {output_dir}")
    return saved_files


def prepare_temp_dir(path: str):
    if os.path.exists(path):
        # Remove only if not empty
        if os.listdir(path):  # non-empty
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)