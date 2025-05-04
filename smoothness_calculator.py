import cv2
import numpy as np
import os
import shutil
import argparse
import sys # Import sys for exit
import math # For sqrt

# --- Constants for Smoothness Calculation ---
DEFAULT_OPENING_KERNEL_SIZE = 3 # For initial noise/thin line removal in std dev expert
DEFAULT_STD_DEV_WINDOW_SIZE = 7 # For local std dev calculation
DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD = 2.5 # Threshold for std dev in non-edge areas (Expert 1)

# --- Constants for Gradient Threshold Expert (Expert 2) ---
DEFAULT_SOBEL_KERNEL_SIZE = 3 # Kernel size for Sobel gradient calculation
DEFAULT_GRADIENT_ZERO_THRESHOLD = 0.5 # Magnitude below this is considered 'flat' for Expert 2
# DEFAULT_GRADIENT_PRE_BLUR_KSIZE = (3, 3) # Removed pre-blur

# --- Constants for Edge Masking (Used by both experts) ---
DEFAULT_CANNY_THRESHOLD1 = 50   # Lower threshold for Canny edge detection
DEFAULT_CANNY_THRESHOLD2 = 150  # Upper threshold for Canny edge detection
DEFAULT_EDGE_DILATION_KERNEL_SIZE = 3 # Kernel size for thickening edges into a mask (must be odd)
DEFAULT_EDGE_DILATION_ITERATIONS = 1 # How many times to dilate the edges

# --- Constants for Mixture of Experts ---
WEIGHT_STD_DEV_EXPERT = 0.4 # Weight for the standard deviation based expert (Expert 1)
WEIGHT_GRADIENT_THRESH_EXPERT = 0.6 # Weight for the gradient threshold based expert (Expert 2) (Weights should sum to 1.0)

# --- Constants for Optional Resizing (Aspect Ratio Preserved) ---
RESIZE_ENABLED_BY_DEFAULT = False # Set to True if resizing should happen unless --no-resize is passed
# *** UPDATED: Now represents the maximum dimension for resizing ***
RESIZE_MAX_DIMENSION = 768 # Target maximum dimension (width or height) if resizing is enabled


# --- Helper: Edge Mask Creation ---
def _create_relevant_pixel_mask(img_gray: np.ndarray,
                                canny_threshold1: float,
                                canny_threshold2: float,
                                edge_dilation_kernel_size: int,
                                edge_dilation_iterations: int) -> np.ndarray:
    """
    Creates a boolean mask where True indicates pixels *not* part of detected edges.

    Args:
        img_gray (np.ndarray): Grayscale image for edge detection.
        canny_threshold1 (float): Lower Canny threshold.
        canny_threshold2 (float): Upper Canny threshold.
        edge_dilation_kernel_size (int): Kernel size for dilating edges.
        edge_dilation_iterations (int): Number of dilation iterations.

    Returns:
        np.ndarray: Boolean mask (True for non-edge pixels). Returns None if input is invalid.
    """
    if img_gray is None or img_gray.ndim != 2:
        print("Error (_create_relevant_pixel_mask): Invalid grayscale image input.", file=sys.stderr)
        return None

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, canny_threshold1, canny_threshold2)

    # Dilate the edges to create a mask covering lines/boundaries
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_dilation_kernel_size, edge_dilation_kernel_size))
    edge_mask = cv2.dilate(edges, dilation_kernel, iterations=edge_dilation_iterations)

    # Invert the mask: We want to calculate smoothness where there are NO edges
    inverted_edge_mask = cv2.bitwise_not(edge_mask) # Make non-edge areas white (255), edges black (0)

    # Convert mask to boolean for easier indexing (True for non-edge areas)
    relevant_pixel_mask = inverted_edge_mask == 255
    return relevant_pixel_mask


# --- Expert 1: Standard Deviation in Non-Edge Areas ---
def _calculate_std_dev_map(channel_img: np.ndarray,
                          opening_kernel_size: int,
                          std_dev_window_size: int) -> np.ndarray:
    """
    Calculates the local standard deviation map for a single image channel
    after applying morphological opening. (Helper for Expert 1)

    Args:
        channel_img (np.ndarray): The single-channel image.
        opening_kernel_size (int): Size of the kernel for morphological opening.
        std_dev_window_size (int): Size of the sliding window for local std dev calculation.

    Returns:
        np.ndarray: A map of the local standard deviation values. Returns None if input is invalid.
    """
    if channel_img is None or channel_img.ndim != 2:
        print("Error (_calculate_std_dev_map): Invalid channel image input.", file=sys.stderr)
        return None

    # Apply Morphological Opening first to reduce noise/thin lines before std dev
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_channel = cv2.morphologyEx(channel_img, cv2.MORPH_OPEN, opening_kernel)

    # Calculate Local Standard Deviation on the opened image
    img_float = opened_channel.astype(np.float32)
    mean_img = cv2.blur(img_float, (std_dev_window_size, std_dev_window_size))
    mean_sq_img = cv2.blur(img_float**2, (std_dev_window_size, std_dev_window_size))
    variance_img = mean_sq_img - mean_img**2
    variance_img = np.maximum(variance_img, 0) # Ensure non-negative variance
    std_dev_map = np.sqrt(variance_img)
    return std_dev_map

def calculate_smoothness_expert_stddev(img_target: np.ndarray,
                                       relevant_pixel_mask: np.ndarray,
                                       use_rgb: bool,
                                       opening_kernel_size: int = DEFAULT_OPENING_KERNEL_SIZE,
                                       std_dev_window_size: int = DEFAULT_STD_DEV_WINDOW_SIZE,
                                       smoothness_std_dev_threshold: float = DEFAULT_SMOOTHNESS_STD_DEV_THRESHOLD
                                       ) -> float:
    """
    Expert 1: Calculates smoothness based on the proportion of non-edge pixels
              with low local standard deviation.

    Args:
        img_target (np.ndarray): Target image (grayscale or BGR), potentially resized.
        relevant_pixel_mask (np.ndarray): Boolean mask (True for non-edge pixels), sized as img_target.
        use_rgb (bool): Indicates if img_target is RGB.
        opening_kernel_size (int): Kernel size for initial morphological opening.
        std_dev_window_size (int): Window size for local standard deviation calculation.
        smoothness_std_dev_threshold (float): Std dev threshold for smoothness.

    Returns:
        float: Smoothness score (0.0 to 1.0) based on std dev. Returns 0.0 on error.
    """
    if img_target is None or relevant_pixel_mask is None: return 0.0

    total_relevant_pixels_initial = np.sum(relevant_pixel_mask)
    if total_relevant_pixels_initial == 0:
        return 0.0 # No relevant pixels to evaluate

    score = 0.0
    total_relevant_pixels = 0

    if not use_rgb: # Process grayscale
        if img_target.shape != relevant_pixel_mask.shape:
             print(f"Warning (StdDev Expert): Mismatched shapes between grayscale map {img_target.shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
             return 0.0
        std_dev_map_gray = _calculate_std_dev_map(img_target, opening_kernel_size, std_dev_window_size)
        if std_dev_map_gray is None: return 0.0
        relevant_std_devs = std_dev_map_gray[relevant_pixel_mask]
        smooth_relevant_pixels = np.sum(relevant_std_devs < smoothness_std_dev_threshold)
        total_relevant_pixels = len(relevant_std_devs)
    else: # Process RGB
        channels = cv2.split(img_target)
        if len(channels) != 3:
            print(f"Warning (StdDev Expert): Expected 3 channels for RGB, got {len(channels)}. Using first channel.", file=sys.stderr)
            if channels[0].shape != relevant_pixel_mask.shape:
                print(f"Warning (StdDev Expert): Mismatched shapes between channel 0 {channels[0].shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
                return 0.0
            std_dev_map_ch0 = _calculate_std_dev_map(channels[0], opening_kernel_size, std_dev_window_size)
            if std_dev_map_ch0 is None: return 0.0
            relevant_std_devs = std_dev_map_ch0[relevant_pixel_mask]
            smooth_relevant_pixels = np.sum(relevant_std_devs < smoothness_std_dev_threshold)
            total_relevant_pixels = len(relevant_std_devs)
        else:
            # Check shape consistency before processing channels
            if channels[0].shape != relevant_pixel_mask.shape:
                 print(f"Warning (StdDev Expert): Mismatched shapes between channels {channels[0].shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
                 return 0.0

            std_dev_maps = [_calculate_std_dev_map(ch, opening_kernel_size, std_dev_window_size) for ch in channels]
            if any(m is None for m in std_dev_maps): return 0.0 # Check if any map failed

            smooth_map_b = (std_dev_maps[0] < smoothness_std_dev_threshold)[relevant_pixel_mask]
            smooth_map_g = (std_dev_maps[1] < smoothness_std_dev_threshold)[relevant_pixel_mask]
            smooth_map_r = (std_dev_maps[2] < smoothness_std_dev_threshold)[relevant_pixel_mask]
            overall_smooth_map_relevant = np.logical_and(smooth_map_b, np.logical_and(smooth_map_g, smooth_map_r))
            smooth_relevant_pixels = np.sum(overall_smooth_map_relevant)
            total_relevant_pixels = len(smooth_map_b) # Length is same for all

    if total_relevant_pixels > 0:
        score = float(smooth_relevant_pixels) / total_relevant_pixels

    return score


# --- Expert 2: Gradient Thresholding in Non-Edge Areas ---
def calculate_smoothness_expert_gradient_thresh(img_target: np.ndarray,
                                                relevant_pixel_mask: np.ndarray,
                                                use_rgb: bool,
                                                sobel_kernel_size: int = DEFAULT_SOBEL_KERNEL_SIZE,
                                                gradient_zero_threshold: float = DEFAULT_GRADIENT_ZERO_THRESHOLD
                                                ) -> float:
    """
    Expert 2: Calculates smoothness based on the proportion of non-edge pixels
              with gradient magnitude below a strict threshold (near-zero).
              * No pre-blurring is applied *

    Args:
        img_target (np.ndarray): Target image (grayscale or BGR), potentially resized.
        relevant_pixel_mask (np.ndarray): Boolean mask (True for non-edge pixels), sized as img_target.
        use_rgb (bool): Indicates if img_target is RGB.
        sobel_kernel_size (int): Kernel size for Sobel operator.
        gradient_zero_threshold (float): Magnitude below this is considered 'flat'.

    Returns:
        float: Smoothness score (0.0 to 1.0) based on gradient thresholding. Returns 0.0 on error.
    """
    if img_target is None or relevant_pixel_mask is None: return 0.0

    total_relevant_pixels = np.sum(relevant_pixel_mask)
    if total_relevant_pixels == 0:
        return 0.0 # No relevant pixels to evaluate

    smooth_relevant_pixels = 0
    score = 0.0

    if not use_rgb: # Process grayscale
        if img_target.shape != relevant_pixel_mask.shape:
             print(f"Warning (Gradient Thresh Expert): Mismatched shapes between grayscale map {img_target.shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
             return 0.0
        # Calculate gradients using Sobel on original image data
        grad_x = cv2.Sobel(img_target, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        grad_y = cv2.Sobel(img_target, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        # Calculate magnitude
        magnitude = cv2.magnitude(grad_x, grad_y)
        # Count relevant pixels below threshold
        relevant_magnitudes = magnitude[relevant_pixel_mask]
        smooth_relevant_pixels = np.sum(relevant_magnitudes < gradient_zero_threshold)
        total_relevant_pixels = len(relevant_magnitudes) # Use length of actual data considered

    else: # Process RGB
        channels = cv2.split(img_target)
        if len(channels) != 3:
            print(f"Warning (Gradient Thresh Expert): Expected 3 channels for RGB, got {len(channels)}. Using first channel.", file=sys.stderr)
            if channels[0].shape != relevant_pixel_mask.shape:
                 print(f"Warning (Gradient Thresh Expert): Mismatched shapes between channel 0 {channels[0].shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
                 return 0.0
            grad_x = cv2.Sobel(channels[0], cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
            grad_y = cv2.Sobel(channels[0], cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
            magnitude = cv2.magnitude(grad_x, grad_y)
            relevant_magnitudes = magnitude[relevant_pixel_mask]
            smooth_relevant_pixels = np.sum(relevant_magnitudes < gradient_zero_threshold)
            total_relevant_pixels = len(relevant_magnitudes)
        else:
            # Check shape consistency before processing channels
            if channels[0].shape != relevant_pixel_mask.shape:
                 print(f"Warning (Gradient Thresh Expert): Mismatched shapes between channels {channels[0].shape} and mask {relevant_pixel_mask.shape}. Skipping.", file=sys.stderr)
                 return 0.0

            # Check threshold per channel, smooth only if ALL channels are below threshold
            smooth_map_b = None
            smooth_map_g = None
            smooth_map_r = None

            all_channel_maps_ok = True
            channel_smooth_maps = []
            for i, ch in enumerate(channels):
                grad_x = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
                grad_y = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
                magnitude = cv2.magnitude(grad_x, grad_y)
                # Get boolean map for this channel where magnitude is low in relevant area
                channel_smooth_map = (magnitude < gradient_zero_threshold)[relevant_pixel_mask]
                channel_smooth_maps.append(channel_smooth_map)

            if len(channel_smooth_maps) == 3:
                 smooth_map_b, smooth_map_g, smooth_map_r = channel_smooth_maps
                 # Combine: a relevant pixel is smooth only if smooth in ALL channels
                 overall_smooth_map_relevant = np.logical_and(smooth_map_b, np.logical_and(smooth_map_g, smooth_map_r))
                 smooth_relevant_pixels = np.sum(overall_smooth_map_relevant)
                 total_relevant_pixels = len(smooth_map_b) # Length is same for all
            else:
                 # Should not happen if len(channels) == 3 check passed, but defensive
                 print("Error (Gradient Thresh Expert): Failed to process all RGB channels.", file=sys.stderr)
                 return 0.0

    # Calculate score
    if total_relevant_pixels > 0:
        score = float(smooth_relevant_pixels) / total_relevant_pixels

    return score


# --- Combined Calculation (Mixture of Experts) ---
def calculate_combined_smoothness(image_path: str,
                                  use_rgb: bool = False,
                                  perform_resize: bool = False) -> float: # Added perform_resize argument
    """
    Calculates a combined smoothness score using a mixture of experts.
    Optionally resizes the image before processing, preserving aspect ratio.

    Args:
        image_path (str): Path to the image file.
        use_rgb (bool): If True, process in RGB; otherwise, process in grayscale.
        perform_resize (bool): If True, resize image to fit within RESIZE_MAX_DIMENSION
                               while preserving aspect ratio, before processing.

    Returns:
        float: Combined smoothness score (0.0 to 1.0).
               Returns -1.0 if the image cannot be loaded or critical errors occur.
    """
    # --- Shared Setup: Load Images ---
    try:
        # Load grayscale version for edge detection
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Error: Could not load grayscale image: '{image_path}'", file=sys.stderr)
            return -1.0

        # Load target image (gray or color)
        read_flag = cv2.IMREAD_COLOR if use_rgb else cv2.IMREAD_GRAYSCALE
        img_target = cv2.imread(image_path, read_flag)
        if img_target is None:
            print(f"Error: Could not load target image: '{image_path}'", file=sys.stderr)
            return -1.0

        # --- Optional Resizing (Aspect Ratio Preserved) ---
        if perform_resize:
            max_dim = RESIZE_MAX_DIMENSION
            if max_dim > 0:
                h_orig, w_orig = img_gray.shape[:2]
                # Only resize if a dimension exceeds the max
                if h_orig > max_dim or w_orig > max_dim:
                    print(f"Resizing '{os.path.basename(image_path)}' to fit max dimension {max_dim} (preserving aspect ratio)...")
                    # Determine the scaling factor
                    scale = max_dim / float(max(h_orig, w_orig))
                    target_w = int(w_orig * scale)
                    target_h = int(h_orig * scale)

                    # Ensure dimensions are at least 1 pixel
                    target_w = max(1, target_w)
                    target_h = max(1, target_h)

                    # Choose interpolation (always shrinking or staying same size in this logic)
                    interpolation = cv2.INTER_AREA

                    try:
                        img_gray = cv2.resize(img_gray, (target_w, target_h), interpolation=interpolation)
                        img_target = cv2.resize(img_target, (target_w, target_h), interpolation=interpolation)
                        print(f"   New dimensions: {target_w}x{target_h}")
                    except cv2.error as e:
                         print(f"Error during resize for '{image_path}': {e}", file=sys.stderr)
                         return -1.0 # Treat resize error as critical
                else:
                    print(f"Skipping resize for '{os.path.basename(image_path)}': Dimensions ({w_orig}x{h_orig}) already within max ({max_dim}).")
            else:
                print(f"Warning: Invalid resize constant defined (RESIZE_MAX_DIMENSION={max_dim}). Skipping resize.", file=sys.stderr)


        # Basic dimension checks (after potential resize)
        if img_target.size == 0 or img_gray.size == 0:
             print(f"Error: Image has zero size (after potential resize): '{image_path}'", file=sys.stderr)
             return -1.0
        rows, cols = img_gray.shape[:2]
        if rows * cols == 0:
             print(f"Error: Image has zero pixels (after potential resize): '{image_path}'", file=sys.stderr)
             return -1.0

        # Create the mask for relevant (non-edge) pixels *after* potential resize
        relevant_pixel_mask = _create_relevant_pixel_mask(
            img_gray,
            DEFAULT_CANNY_THRESHOLD1,
            DEFAULT_CANNY_THRESHOLD2,
            DEFAULT_EDGE_DILATION_KERNEL_SIZE,
            DEFAULT_EDGE_DILATION_ITERATIONS
        )
        if relevant_pixel_mask is None: # Check if mask creation failed
             print(f"Error: Failed to create edge mask for '{image_path}'.", file=sys.stderr)
             return -1.0

        total_relevant_pixels_check = np.sum(relevant_pixel_mask)
        if total_relevant_pixels_check == 0:
             print(f"Warning: No relevant pixels found (mask covered entire image?) for '{image_path}'. Combined score set to 0.0", file=sys.stderr)
             return 0.0

        # --- Calculate Scores from Experts ---
        score_std_dev = calculate_smoothness_expert_stddev(
            img_target, relevant_pixel_mask, use_rgb
        )
        score_gradient_thresh = calculate_smoothness_expert_gradient_thresh(
            img_target, relevant_pixel_mask, use_rgb
        )

        # --- Combine Scores ---
        combined_score = (WEIGHT_STD_DEV_EXPERT * score_std_dev +
                          WEIGHT_GRADIENT_THRESH_EXPERT * score_gradient_thresh)

        # Ensure score is within bounds
        combined_score = max(0.0, min(1.0, combined_score))

        return combined_score

    except cv2.error as e:
        print(f"OpenCV Error processing '{image_path}': {e}", file=sys.stderr)
        return -1.0
    except Exception as e:
        print(f"Unexpected error during combined calculation for '{image_path}': {e}", file=sys.stderr)
        return -1.0


def process_images(input_folder: str, output_folder: str, smoothness_score_threshold: float = 0.8, use_rgb: bool = False, perform_resize: bool = True): # Added perform_resize
    """
    Processes images using the combined smoothness score. Copies images exceeding
    the threshold to the output folder. Optionally resizes images first.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder where smooth images will be copied.
        smoothness_score_threshold (float): Minimum combined smoothness score required.
        use_rgb (bool): If True, process images in RGB; otherwise, use grayscale.
        perform_resize (bool): If True, resize images before processing.
    """
    # --- Validate Folders ---
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: '{output_folder}'")
        except OSError as e:
            print(f"Error: Could not create output folder '{output_folder}': {e}", file=sys.stderr)
            sys.exit(1)
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
    print(f"Color Mode: {'RGB' if use_rgb else 'Grayscale'}")
    if perform_resize:
        # *** UPDATED: Print statement for aspect ratio preserving resize ***
        print(f"Resizing enabled: Target max dimension {RESIZE_MAX_DIMENSION}px (preserving aspect ratio)")
    else:
        print("Resizing disabled.")
    print(f"Using Mixture of Experts: StdDev Weight={WEIGHT_STD_DEV_EXPERT}, Gradient Thresh Weight={WEIGHT_GRADIENT_THRESH_EXPERT}")
    print(f"Gradient Threshold for 'Flatness': {DEFAULT_GRADIENT_ZERO_THRESHOLD}")
    print("-" * 30)

    copied_count = 0
    skipped_count = 0
    error_count = 0
    processed_count = 0

    # --- Iterate and Process ---
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            processed_count += 1
            print(f"Processing '{filename}'...", end=' ')

            # Calculate the combined smoothness score, passing resize flag
            combined_smoothness = calculate_combined_smoothness(
                input_path,
                use_rgb=use_rgb,
                perform_resize=perform_resize # Pass the flag
            )

            if combined_smoothness == -1.0:
                error_count += 1
                print(" -> Error")
                continue

            # Check against the score threshold
            if combined_smoothness >= smoothness_score_threshold:
                output_path = os.path.join(output_folder, filename)
                try:
                    shutil.copy2(input_path, output_path)
                    print(f" -> Copied (Score: {combined_smoothness:.4f})")
                    copied_count += 1
                except Exception as e:
                    print(f" -> Error copying file: {e}", file=sys.stderr)
                    error_count += 1
            else:
                print(f" -> Skipped (Score: {combined_smoothness:.4f})")
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
        0.9,
        True
    )
