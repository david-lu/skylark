import cv2
import numpy as np
import os
import shutil
import argparse
import sys # Import sys for exit

# --- Constants for Smoothness Calculation ---
# These values are used as defaults in calculate_image_smoothness
DEFAULT_OPENING_KERNEL_SIZE = 3
DEFAULT_STD_DEV_WINDOW_SIZE = 7
DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD = 5.0

def _calculate_std_dev_map(channel_img: np.ndarray,
                          opening_kernel_size: int,
                          std_dev_window_size: int) -> np.ndarray:
    """
    Calculates the local standard deviation map for a single image channel.
    Helper function for calculate_image_smoothness.

    Args:
        channel_img (np.ndarray): The single-channel image (grayscale or one color channel).
        opening_kernel_size (int): Size of the kernel for morphological opening.
        std_dev_window_size (int): Size of the sliding window for local std dev calculation.

    Returns:
        np.ndarray: A map of the local standard deviation values.
    """
    # Apply Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_channel = cv2.morphologyEx(channel_img, cv2.MORPH_OPEN, kernel)

    # Calculate Local Standard Deviation
    # Ensure calculations are done in float to avoid overflow/clipping
    img_float = opened_channel.astype(np.float32)
    mean_img = cv2.blur(img_float, (std_dev_window_size, std_dev_window_size))
    mean_sq_img = cv2.blur(img_float**2, (std_dev_window_size, std_dev_window_size))
    variance_img = mean_sq_img - mean_img**2
    variance_img = np.maximum(variance_img, 0) # Ensure non-negative variance
    std_dev_map = np.sqrt(variance_img)
    return std_dev_map

def calculate_image_smoothness(image_path: str,
                               use_rgb: bool = False, # Flag to process in RGB
                               opening_kernel_size: int = DEFAULT_OPENING_KERNEL_SIZE,
                               std_dev_window_size: int = DEFAULT_STD_DEV_WINDOW_SIZE,
                               smoothness_std_dev_threshold: float = DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD) -> float:
    """
    Evaluates the smoothness of an image, prioritizing large flat areas.

    Can operate in grayscale (default, use_rgb=False) or RGB mode (use_rgb=True).
    In RGB mode, a pixel neighborhood is considered smooth only if the standard
    deviation is below the threshold in *all* color channels (B, G, R).

    Args:
        image_path (str): Path to the image file.
        use_rgb (bool): If True, process in RGB; otherwise, process in grayscale.
        opening_kernel_size (int): Size of the kernel for morphological opening.
        std_dev_window_size (int): Size of the sliding window for local std dev.
        smoothness_std_dev_threshold (float): Std dev value below which a pixel's
                                              neighborhood is considered smooth (per channel if RGB).

    Returns:
        float: A smoothness score between 0.0 and 1.0.
               Returns -1.0 if the image cannot be loaded or other errors occur.
    """
    # --- Input Validation ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
        return -1.0
    # Validate kernel/window sizes and threshold (using defaults if invalid)
    if opening_kernel_size <= 0 or opening_kernel_size % 2 == 0:
        print(f"Warning: opening_kernel_size ({opening_kernel_size}) must be a positive odd integer. Using default {DEFAULT_OPENING_KERNEL_SIZE}.", file=sys.stderr)
        opening_kernel_size = DEFAULT_OPENING_KERNEL_SIZE
    if std_dev_window_size <= 0 or std_dev_window_size % 2 == 0:
        print(f"Warning: std_dev_window_size ({std_dev_window_size}) must be a positive odd integer. Using default {DEFAULT_STD_DEV_WINDOW_SIZE}.", file=sys.stderr)
        std_dev_window_size = DEFAULT_STD_DEV_WINDOW_SIZE
    if smoothness_std_dev_threshold < 0:
        print(f"Warning: smoothness_std_dev_threshold ({smoothness_std_dev_threshold}) cannot be negative. Using default {DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD}.", file=sys.stderr)
        smoothness_std_dev_threshold = DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD

    try:
        # 1. Load Image
        read_flag = cv2.IMREAD_COLOR if use_rgb else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(image_path, read_flag)
        if img is None:
            print(f"Error: Could not load image (check format/permissions): '{image_path}'", file=sys.stderr)
            return -1.0

        # Ensure image has dimensions
        if img.size == 0:
             print(f"Error: Image has zero size: '{image_path}'", file=sys.stderr)
             return -1.0
        total_pixels = img.shape[0] * img.shape[1]
        if total_pixels == 0:
             print(f"Error: Image has zero pixels: '{image_path}'", file=sys.stderr)
             return -1.0 # Avoid division by zero

        # --- Calculate Smoothness ---
        if not use_rgb: # Process grayscale
            std_dev_map = _calculate_std_dev_map(img, opening_kernel_size, std_dev_window_size)
            # Count pixels where local standard deviation is below the threshold
            smooth_pixels = np.sum(std_dev_map < smoothness_std_dev_threshold)
        else: # Process RGB
            # Split channels (OpenCV uses BGR order)
            channels = cv2.split(img)
            if len(channels) != 3:
                 print(f"Warning: Expected 3 channels for RGB mode, got {len(channels)}. Processing first channel as grayscale for '{image_path}'.", file=sys.stderr)
                 # Fallback to grayscale-like processing on the first channel
                 std_dev_map = _calculate_std_dev_map(channels[0], opening_kernel_size, std_dev_window_size)
                 smooth_pixels = np.sum(std_dev_map < smoothness_std_dev_threshold)
            else:
                # Calculate std dev map for each channel using the helper function
                std_dev_maps = [_calculate_std_dev_map(ch, opening_kernel_size, std_dev_window_size) for ch in channels]

                # Create boolean maps indicating smoothness for each channel
                smooth_map_b = std_dev_maps[0] < smoothness_std_dev_threshold
                smooth_map_g = std_dev_maps[1] < smoothness_std_dev_threshold
                smooth_map_r = std_dev_maps[2] < smoothness_std_dev_threshold

                # Combine: a pixel is smooth only if it's smooth in ALL channels
                overall_smooth_map = np.logical_and(smooth_map_b, np.logical_and(smooth_map_g, smooth_map_r))
                smooth_pixels = np.sum(overall_smooth_map)

        # Calculate the proportion of smooth pixels (score between 0.0 and 1.0)
        smoothness_proportion = float(smooth_pixels) / total_pixels
        return smoothness_proportion

    except cv2.error as e:
        print(f"OpenCV Error processing '{image_path}': {e}", file=sys.stderr)
        return -1.0
    except Exception as e:
        print(f"Unexpected error processing '{image_path}': {e}", file=sys.stderr)
        return -1.0


def process_images(input_folder: str, output_folder: str, smoothness_score_threshold: float = 0.8, use_rgb: bool = False):
    """
    Processes images from the input folder. If an image's smoothness score
    (calculated using specified color mode and default internal parameters)
    is >= smoothness_score_threshold, it's copied to the output folder.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder where smooth images will be copied.
        smoothness_score_threshold (float): Minimum smoothness score (0.0 to 1.0) required
                                            to copy the image.
        use_rgb (bool): If True, process images in RGB; otherwise, use grayscale.
    """
    # --- Validate Folders ---
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1) # Exit if input folder is invalid

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: '{output_folder}'")
        except OSError as e:
            print(f"Error: Could not create output folder '{output_folder}': {e}", file=sys.stderr)
            sys.exit(1) # Exit if output folder cannot be created
    elif not os.path.isdir(output_folder):
         print(f"Error: Output path '{output_folder}' exists but is not a directory.", file=sys.stderr)
         sys.exit(1)

    # --- Validate Smoothness Threshold ---
    if not 0.0 <= smoothness_score_threshold <= 1.0:
         print(f"Error: Smoothness score threshold ({smoothness_score_threshold}) must be between 0.0 and 1.0.", file=sys.stderr)
         sys.exit(1)

    print(f"Processing images from: '{input_folder}'")
    print(f"Saving smooth images to: '{output_folder}'")
    print(f"Smoothness Score Threshold: {smoothness_score_threshold:.2f}")
    print(f"Color Mode: {'RGB' if use_rgb else 'Grayscale'}") # Indicate selected mode based on boolean
    # Print the constants being used for calculation
    print(f"Using Calculation Defaults: Opening Kernel Size={DEFAULT_OPENING_KERNEL_SIZE}, Std Dev Window Size={DEFAULT_STD_DEV_WINDOW_SIZE}, Std Dev Threshold={DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD:.2f}")
    print("-" * 30)

    copied_count = 0
    skipped_count = 0
    error_count = 0
    processed_count = 0

    # --- Iterate and Process ---
    for filename in os.listdir(input_folder):
        # Construct full path
        input_path = os.path.join(input_folder, filename)

        # Check if it's a file and has a common image extension
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            processed_count += 1
            print(f"Processing '{filename}'...", end=' ')

            # Calculate smoothness score using specified color mode
            smoothness = calculate_image_smoothness(
                input_path,
                use_rgb=use_rgb # Pass the boolean flag
                # Uses default kernel size, window size, std dev threshold internally
            )

            if smoothness == -1.0:
                # Error occurred during calculation (already printed)
                error_count += 1
                print(" -> Error")
                continue # Skip to next file

            # Check against the score threshold
            if smoothness >= smoothness_score_threshold:
                output_path = os.path.join(output_folder, filename)
                try:
                    shutil.copy2(input_path, output_path)  # copy2 preserves metadata
                    print(f" -> Copied (Score: {smoothness:.4f})")
                    copied_count += 1
                except Exception as e:
                    print(f" -> Error copying file: {e}", file=sys.stderr)
                    error_count += 1
            else:
                print(f" -> Skipped (Score: {smoothness:.4f})")
                skipped_count += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total files processed: {processed_count}")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Files with errors: {error_count}")


if __name__ == "__main__":
    # --- Run Processing ---
    process_images(
        './data/input',
        './data/output',
        0.8,
        True
    )
