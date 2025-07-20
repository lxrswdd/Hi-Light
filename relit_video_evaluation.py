import cv2
import numpy as np
import matplotlib.pyplot as plt # Moved import to top for standard practice
from skimage.metrics import structural_similarity as ssim
import argparse


def compare_videos_ssim(video_path1, video_path2, frame_interval=1):
    """
    Calculates the average Structural Similarity Index (SSIM) between two videos.

    Args:
        video_path1 (str): Path to the first video file.
        video_path2 (str): Path to the second video file.
        frame_interval (int): Interval at which to sample and compare frames.
                             Default is 1 (every frame). Use a higher number for faster processing.

    Returns:
        float: The average SSIM score between the two videos, or None if an error occurs.
    """
    # Open video capture objects
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return None

    ssim_scores = []
    frame_count = 0

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video ends, break the loop
        if not ret1 or not ret2:
            break

        # Process frames only at the specified interval
        if frame_count % frame_interval == 0:
            # Convert frames to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Ensure frames are the same size
            if gray1.shape != gray2.shape:
                print(f"Frame dimensions do not match at frame {frame_count}. Resizing frame 2 to match frame 1.")
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

            # Calculate SSIM for the pair of frames
            # The data_range is the dynamic range of the image, which is 255 for 8-bit grayscale images.
            score, _ = ssim(gray1, gray2, full=True, data_range=255)
            ssim_scores.append(score)

        frame_count += 1

    # Release video capture objects
    cap1.release()
    cap2.release()
    
    if not ssim_scores:
        print("Warning: No frames were compared. Check video lengths or frame interval.")
        return 0.0

    # Calculate the average SSIM score
    average_ssim = np.mean(ssim_scores)
    return average_ssim



def calculate_smoothness_score(signal_values, amplification_factor=20.0, signal_name="Signal"):
    """
    Calculates a smoothness score. A spiky plot (large frame-to-frame changes
    relative to its overall range) gets a low score.
    Score = exp(-amplification_factor * normalized_mean_absolute_change)
    where normalized_mean_absolute_change = mean_abs_change / signal_range.

    Args:
        signal_values (np.array): The input 1D signal.
        amplification_factor (float): Factor to scale the normalized unsmoothness metric.
                                      Typical values might now be between 5 and 50.
        signal_name (str): Name of the signal for debug printing.
    """
    debug = False
    cleaned_signal = signal_values[~np.isnan(signal_values)]

    if cleaned_signal.size < 2:
        mean_abs_change = 0.0
        normalized_unsmoothness = 0.0
    else:
        first_derivative = np.diff(cleaned_signal)
        if first_derivative.size == 0: 
            mean_abs_change = 0.0
        else:
            mean_abs_change = np.mean(np.abs(first_derivative))

        if mean_abs_change < 1e-9: 
            normalized_unsmoothness = 0.0
        else:
            signal_range = np.ptp(cleaned_signal) 
            if signal_range < 1e-9:
                normalized_unsmoothness = 1.0 
            else:
                normalized_unsmoothness = mean_abs_change / signal_range
    
        # --- DEBUG PRINT (Essential for tuning amplification_factor) ---
    if debug:
        print(f"\n--- Debugging Smoothness for '{signal_name}' ---")
        print(f"  Signal points (cleaned): {cleaned_signal.size}")
        if cleaned_signal.size >= 2:
            print(f"  Mean abs change (MAC)  : {mean_abs_change:.4f}")
            # Corrected condition for printing signal_range
            if mean_abs_change > 1e-9 and 'signal_range' in locals(): 
                print(f"  Signal range (PTP)     : {signal_range:.4f}")
            elif mean_abs_change > 1e-9:
                print(f"  Signal range (PTP)     : N/A (Range was ~0 but MAC > 0 or signal_range not defined due to MAC being 0 initially)")
        print(f"  Normalized Unsmoothness: {normalized_unsmoothness:.4f} (MAC/PTP; 0=perfectly smooth, higher=spikier)")
    
    exponent_argument = -amplification_factor * normalized_unsmoothness
    score = np.exp(exponent_argument)

    if debug:
        print(f"  Amplification factor   : {amplification_factor:.4f}")
        print(f"  Exponent argument      : {exponent_argument:.4f}")
        print(f"  Calculated Score       : {score:.4f} (Higher is smoother)")
        print(f"--------------------------------------------------")

    return score

# --- Data Extraction Function ---
def get_dynamic_bright_pixel_timeseries(video_path, brightness_threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None, None

    avg_intensities = []
    pixel_counts = []
    frame_times = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps is None:
        print("Warning: Could not get FPS from video. Using default of 30. Analysis might be affected.")
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_pixels_mask = gray_frame > brightness_threshold
        bright_pixels = gray_frame[bright_pixels_mask]

        if bright_pixels.size > 0:
            avg_intensity_of_bright = np.mean(bright_pixels)
            count_of_bright_pixels = bright_pixels.size
        else:
            avg_intensity_of_bright = np.nan
            count_of_bright_pixels = 0

        avg_intensities.append(avg_intensity_of_bright)
        pixel_counts.append(count_of_bright_pixels)
        frame_times.append(frame_count / fps)
        frame_count += 1
    cap.release()

    if frame_count == 0:
        print("Error: No frames processed from the video.")
        return None, None, None, None
    print(f"Processed {frame_count} frames at {fps:.2f} FPS.")
    return np.array(avg_intensities), np.array(pixel_counts), np.array(frame_times), fps

# --- Plotting and Analysis Function ---
def plot_and_analyze_with_smoothness(video_path, brightness_threshold_value,plot=True,
                                     amp_factor_avg_intensity=20.0,
                                     amp_factor_pixel_count=20.0,
                                     amp_factor_intensity_changes=40.0,verbose = False):
    """
    Generates plots (each in a separate figure) and calculates instability 
    and smoothness metrics using the normalized mean absolute change method.
    """
    # BRIGHTNESS_THRESHOLD is used in f-strings for plot labels
    # It's better to pass it or define it where it's used if not truly global.
    # For now, this approach works if BRIGHTNESS_THRESHOLD is set before plotting calls.
    # However, it's directly available as brightness_threshold_value in this scope.
    # Let's use brightness_threshold_value directly in labels for clarity.

    avg_bright_intensity_signal, bright_pixel_count_signal, time_axis, video_fps = \
        get_dynamic_bright_pixel_timeseries(video_path, brightness_threshold_value)

    if avg_bright_intensity_signal is not None:
        # --- Plotting ---

        # Figure 1: Average Intensity of Bright Pixels Over Time
        if plot:
            plt.figure(figsize=(15, 5)) # Adjusted figsize for a single plot
            plt.plot(time_axis, avg_bright_intensity_signal, label=f'Avg. Intensity of Pixels > {brightness_threshold_value}')
            plt.title('Average Intensity of Bright Pixels Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Avg. Brightness (0-255)')
            plt.legend()
            plt.grid(True)
            plt.show() 

            # Figure 2: Number of Bright Pixels Over Time
            plt.figure(figsize=(15, 5)) # Adjusted figsize for a single plot
            plt.plot(time_axis, bright_pixel_count_signal, label=f'Count of Pixels > {brightness_threshold_value}', color='green')
            plt.title('Number of Bright Pixels Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Pixel Count')
            plt.legend()
            plt.grid(True)
            plt.show()

        intensity_changes = np.array([])
        time_axis_changes = np.array([])
        
        valid_intensity_indices = ~np.isnan(avg_bright_intensity_signal)
        if np.sum(valid_intensity_indices) > 1:
            intensity_changes = np.diff(avg_bright_intensity_signal)
            time_axis_changes = time_axis[1:] if len(time_axis) > 1 else np.array([])

            if time_axis_changes.size > 0 and intensity_changes.size > 0 and plot:
                # Figure 3: Frame-to-Frame Change in Avg. Bright Pixel Intensity
                plt.figure(figsize=(15, 5))
                plt.plot(time_axis_changes, intensity_changes, label='Change in Avg. Bright Pixel Intensity', color='orange')
                plt.title('Frame-to-Frame Change in Avg. Bright Pixel Intensity')
                plt.xlabel('Time (s)')
                plt.ylabel('Change in Brightness')
                plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                plt.legend()
                plt.grid(True)
                plt.show()

        # --- Calculate Smoothness Scores ---
        if verbose:
            print("\nCalculating smoothness scores (tune amplification factors based on DEBUG prints):")
        
        smoothness_avg_intensity = calculate_smoothness_score(
            avg_bright_intensity_signal,
            amplification_factor=amp_factor_avg_intensity,
            signal_name="Avg. Intensity of Bright Pixels"
        )
        
        smoothness_pixel_count = calculate_smoothness_score(
            bright_pixel_count_signal,
            amplification_factor=amp_factor_pixel_count,
            signal_name="Number of Bright Pixels"
        )

        smoothness_intensity_changes = 0.0
        if intensity_changes.size > 1 : 
             smoothness_intensity_changes = calculate_smoothness_score(
                intensity_changes, 
                amplification_factor=amp_factor_intensity_changes,
                signal_name="Changes in Avg. Intensity"
            )
        elif intensity_changes.size <=1: 
            smoothness_intensity_changes = 1.0

        # --- Print Results in Table Format ---
        # Calculate average score
        average_score = (smoothness_avg_intensity + smoothness_pixel_count + smoothness_intensity_changes) / 3
        
        print("\n" + "="*80)
        print("SMOOTHNESS SCORES AND AMPLIFICATION FACTORS")
        print("="*80)
        print(f"{'Signal Name':<35} {'Amplification Factor':<20} {'Smoothness Score':<15}")
        print("-"*80)
        print(f"{'Avg. Intensity of Bright Pixels':<35} {amp_factor_avg_intensity:<20.2f} {smoothness_avg_intensity:<15.3f}")
        print(f"{'Number of Bright Pixels':<35} {amp_factor_pixel_count:<20.2f} {smoothness_pixel_count:<15.3f}")
        print(f"{'Changes in Avg. Intensity':<35} {amp_factor_intensity_changes:<20.2f} {smoothness_intensity_changes:<15.3f}")
        print("-"*80)
        print(f"{'AVERAGE SCORE':<35} {'N/A':<20} {average_score:<15.3f}")
        print("-"*80)
        print(f"Note: Higher smoothness scores indicate smoother signals (less spiky)")
        print("="*80)

        # --- Print Statistical Metrics (only if verbose is True) ---
        if verbose:
            print("\n--- Analysis Metrics ---")
            print(f"Threshold for 'bright' pixels: {brightness_threshold_value}")

            if np.sum(valid_intensity_indices) > 0:
                mean_intensity = np.nanmean(avg_bright_intensity_signal)
                std_dev_intensity = np.nanstd(avg_bright_intensity_signal)
                print("\nMetrics for 'Average Intensity of Bright Pixels':")
                print(f"  Mean Avg. Intensity: {mean_intensity:.2f}")
                print(f"  Std. Dev. of Avg. Intensity: {std_dev_intensity:.2f}")
                print(f"  Smoothness Score (norm. MAC based, amp by {amp_factor_avg_intensity:.2f}): {smoothness_avg_intensity:.4f}")

            mean_pixel_count = np.mean(bright_pixel_count_signal)
            std_dev_pixel_count = np.std(bright_pixel_count_signal)
            print("\nMetrics for 'Number of Bright Pixels':")
            print(f"  Mean Count: {mean_pixel_count:.2f}")
            print(f"  Std. Dev. of Count: {std_dev_pixel_count:.2f}")
            print(f"  Smoothness Score (norm. MAC based, amp by {amp_factor_pixel_count:.2f}): {smoothness_pixel_count:.4f}")

            if intensity_changes.size > 0: 
                mean_abs_change_val = np.nanmean(np.abs(intensity_changes)) 
                std_dev_change_val = np.nanstd(intensity_changes) 
                print("\nMetrics for 'Frame-to-Frame Change in Avg. Bright Pixel Intensity':")
                print(f"  Mean Absolute Change (of avg. intensity): {mean_abs_change_val:.2f}")
                print(f"  Std. Dev. of Changes (of avg. intensity): {std_dev_change_val:.2f}")
                print(f"  Smoothness Score (of these changes, norm. MAC based, amp by {amp_factor_intensity_changes:.2f}): {smoothness_intensity_changes:.3f}")
            print("-----------------------------------------------")
    else:
        print("Could not generate bright pixel timeseries for analysis.")
        print("running amplication facors")

    return smoothness_avg_intensity, smoothness_pixel_count, smoothness_intensity_changes, average_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate relighted video using SSIM and smoothness analysis.")
    parser.add_argument('--raw_video_path', type=str, required=True, help='Path to the raw video file.')
    parser.add_argument('--relit_video_path', type=str, required=True, help='Path to the relighted video file.')
    parser.add_argument('--brightness_threshold', type=int, default=125, help='Brightness threshold for pixel analysis.')
    args = parser.parse_args()

    raw_video_path = args.raw_video_path
    relighted_video_path = args.relit_video_path
    brightness_threshold = args.brightness_threshold


    print()
    print("Starting evaluation...")
    print()
    print("Raw video path:", raw_video_path)
    print("Relighted video path:", relighted_video_path)
    print()

    print("\n******************Comparing videos using SSIM ***********************************")
    SSIM_avg_score=compare_videos_ssim(video_path1 = raw_video_path, video_path2=relighted_video_path, frame_interval=5)
    if SSIM_avg_score is not None:
        print(f"The average SSIM score between the two videos is: {SSIM_avg_score:.4f}")
        print("(A score closer to 1.0 indicates higher structural similarity)")
    print('\n\n')
    print("\n****************** Analyzing lighting smoothness for the relit video*************")
    smoothness_avg_intensity, smoothness_pixel_count, smoothness_intensity_changes, average_score = plot_and_analyze_with_smoothness(video_path=relighted_video_path,brightness_threshold_value=brightness_threshold,plot=False,amp_factor_intensity_changes=5,verbose = False)