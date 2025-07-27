import cv2
import numpy as np
from tqdm import tqdm
import time
from collections import deque

def smooth_highlights_optical_flow_improved(input_path, output_path, alpha=0.7, 
                                          highlight_threshold=100, temporal_window=5,
                                          flow_quality='high', adaptive_alpha=True):
    """
    Improved optical flow temporal smoothing with multiple enhancements:
    - Better optical flow parameters
    - Adaptive alpha blending
    - Multi-frame temporal smoothing
    - Improved highlight detection
    - Better motion compensation
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Set optical flow parameters based on quality
    if flow_quality == 'high':
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = 0.5, 5, 21, 10, 7, 1.5
    elif flow_quality == 'medium':
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = 0.5, 4, 15, 5, 5, 1.2
    else:  # 'fast'
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = 0.5, 3, 15, 3, 5, 1.2

    # Initialize frame buffer for temporal smoothing
    frame_buffer = deque(maxlen=temporal_window)
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Initialize
    frame_buffer.append(first_frame.astype(np.float32))
    smoothed_frame = first_frame.copy().astype(np.float32)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(first_frame)
    
    print("Applying improved motion-compensated smoothing...")
    for frame_idx in tqdm(range(total_frames - 1)):
        ret, current_frame = cap.read()
        if not ret:
            break

        current_frame_f32 = current_frame.astype(np.float32)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate high-quality optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0
        )
        
        # 2. Create motion compensation grid
        h, w = current_gray.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to get sampling coordinates
        sample_x = x_coords + flow[:, :, 0]
        sample_y = y_coords + flow[:, :, 1]
        
        # 3. Warp the previous smoothed frame
        warped_smoothed = cv2.remap(smoothed_frame, sample_x, sample_y, 
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 4. Improved highlight detection with adaptive thresholding
        # Convert to LAB color space for better highlight detection
        current_lab = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)
        l_channel = current_lab[:, :, 0]
        
        # Create highlight mask using both absolute and relative thresholds
        highlight_mask1 = l_channel > highlight_threshold
        
        # Adaptive threshold based on local statistics
        blur_l = cv2.GaussianBlur(l_channel, (15, 15), 0)
        highlight_mask2 = (l_channel - blur_l) > 30  # Relative highlights
        
        # Combine masks
        highlight_mask = highlight_mask1 | highlight_mask2
        highlight_mask = highlight_mask.astype(np.uint8) * 255
        
        # Morphological operations for cleaner mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)
        highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_CLOSE, kernel)
        
        # Smooth mask edges
        highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
        highlight_mask_3ch = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # 5. Calculate motion magnitude for adaptive alpha
        if adaptive_alpha:
            motion_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            motion_weight = np.clip(motion_magnitude / 5.0, 0, 1)  # Normalize motion
            # Reduce smoothing in high-motion areas
            adaptive_alpha_map = alpha * (1 - motion_weight * 0.5)
            adaptive_alpha_map = np.stack([adaptive_alpha_map] * 3, axis=2)
        else:
            adaptive_alpha_map = alpha
        
        # 6. Multi-frame temporal smoothing
        frame_buffer.append(current_frame_f32)
        
        if len(frame_buffer) >= 3:
            # Weighted average of recent frames
            weights = np.array([0.2, 0.3, 0.5])  # More weight on recent frames
            temporal_average = np.zeros_like(current_frame_f32)
            for i, weight in enumerate(weights):
                temporal_average += frame_buffer[-(i+1)] * weight
        else:
            temporal_average = current_frame_f32
        
        # 7. Blend current frame with warped smoothed frame
        if adaptive_alpha:
            blended_frame = (current_frame_f32 * (1.0 - adaptive_alpha_map) + 
                           warped_smoothed * adaptive_alpha_map)
        else:
            blended_frame = current_frame_f32 * (1.0 - alpha) + warped_smoothed * alpha
        
        # 8. Apply temporal smoothing to the blended result
        temporal_smoothed = (blended_frame * 0.7 + temporal_average * 0.3)
        
        # 9. Final composition with highlight mask
        # Apply stronger smoothing in highlight areas

        # Define how much to smooth the highlights
        highlight_smoothing_amount = 0.8

        # Create a mask that represents the blend weight
        effective_mask = highlight_mask_3ch * highlight_smoothing_amount

        # Blend between the current and smoothed frames using the mask
        final_frame = (current_frame_f32 * (1.0 - effective_mask) + 
                    temporal_smoothed * effective_mask)
        
        # final_frame = (current_frame_f32 * (1.0 - highlight_mask_3ch * 0.8) + 
        #               temporal_smoothed * highlight_mask_3ch * 0.8 +
        #               temporal_smoothed * (1.0 - highlight_mask_3ch) * 0.2)
        
        # 10. Ensure valid pixel range
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
        
        out.write(final_frame)
        
        # Update for next iteration
        prev_gray = current_gray
        smoothed_frame = final_frame.astype(np.float32)

    print(f"Improved motion-compensated smoothing complete. Video saved to {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def smooth_highlights_bilateral_filter(input_path, output_path, alpha=0.6, 
                                     highlight_threshold=100, bilateral_d=15):
    """
    Alternative approach using bilateral filtering for edge-preserving smoothing
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    smoothed_frame = prev_frame.copy().astype(np.float32)
    out.write(prev_frame)
    
    print("Applying bilateral filter smoothing...")
    for _ in tqdm(range(total_frames - 1)):
        ret, current_frame = cap.read()
        if not ret:
            break

        current_frame_f32 = current_frame.astype(np.float32)
        
        # Apply bilateral filter to preserve edges while smoothing
        bilateral_filtered = cv2.bilateralFilter(current_frame, bilateral_d, 80, 80)
        bilateral_filtered_f32 = bilateral_filtered.astype(np.float32)
        
        # Create highlight mask
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(current_gray, highlight_threshold, 255, cv2.THRESH_BINARY)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
        highlight_mask_3ch = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # Temporal smoothing
        smoothed_frame = smoothed_frame * alpha + current_frame_f32 * (1 - alpha)
        
        # Apply stronger smoothing in highlight areas
        final_frame = (current_frame_f32 * (1.0 - highlight_mask_3ch) + 
                      (smoothed_frame * 0.7 + bilateral_filtered_f32 * 0.3) * highlight_mask_3ch)
        
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
        
        out.write(final_frame)
        smoothed_frame = final_frame.astype(np.float32)

    print(f"Bilateral filter smoothing complete. Video saved to {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def smooth_highlights_combined(input_path, output_path, 
                             # Optical flow parameters
                             optical_alpha=0.6, optical_highlight_threshold=80, 
                             temporal_window=5, flow_quality='high', adaptive_alpha=True,
                             # Bilateral filter parameters
                             bilateral_alpha=0.7, bilateral_highlight_threshold=80, 
                             bilateral_d=15,
                             # General parameters
                             keep_intermediate=False, temp_dir=None):
    """
    Combined smoothing approach that applies optical flow smoothing first,
    then bilateral filtering for enhanced flicker reduction.
    
    Args:
        input_path: Path to input video
        output_path: Path to final output video
        optical_alpha: Alpha for optical flow smoothing (0.0-1.0)
        optical_highlight_threshold: Threshold for optical flow highlight detection
        temporal_window: Number of frames for temporal smoothing
        flow_quality: 'high', 'medium', or 'fast' for optical flow quality
        adaptive_alpha: Whether to use adaptive alpha blending
        bilateral_alpha: Alpha for bilateral filter smoothing (0.0-1.0)
        bilateral_highlight_threshold: Threshold for bilateral highlight detection
        bilateral_d: Bilateral filter diameter
        keep_intermediate: Whether to keep the intermediate video file
        temp_dir: Directory for temporary files (if None, uses same dir as output)
    """
    import os
    import tempfile
    
    # Create temporary file path for intermediate result
    if temp_dir is None:
        temp_dir = os.path.dirname(output_path)
    
    # Generate unique temporary filename
    temp_filename = f"temp_optical_flow_{int(time.time())}.mp4"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        print("=== STAGE 1: Applying Optical Flow Smoothing ===")
        smooth_highlights_optical_flow_improved(
            input_path=input_path,
            output_path=temp_path,
            alpha=optical_alpha,
            highlight_threshold=optical_highlight_threshold,
            temporal_window=temporal_window,
            flow_quality=flow_quality,
            adaptive_alpha=adaptive_alpha
        )
        
        print("\n=== STAGE 2: Applying Bilateral Filter Smoothing ===")
        smooth_highlights_bilateral_filter(
            input_path=temp_path,
            output_path=output_path,
            alpha=bilateral_alpha,
            highlight_threshold=bilateral_highlight_threshold,
            bilateral_d=bilateral_d
        )
        
        print(f"\n=== COMBINED SMOOTHING COMPLETE ===")
        print(f"Final video saved to: {output_path}")
        
        # Clean up temporary file unless user wants to keep it
        if not keep_intermediate:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
        else:
            print(f"Intermediate file kept: {temp_path}")
            
    except Exception as e:
        print(f"Error during combined smoothing: {e}")
        # Clean up temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise