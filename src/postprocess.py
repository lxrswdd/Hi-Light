
import cv2
import os
import numpy as np
import time

import cv2

def upsample_video(input_path, output_path, width, height):


    cap = cv2.VideoCapture(input_path)


    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        upscaled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        out.write(upscaled)

    cap.release()
    out.release()
    print(f"Upscaled video saved to: {output_path}")





def _transfer_detail_spatial(original_frame, relighted_frame, strength):
    """
    Helper function containing the core logic to transfer detail between two single frames.
    This is extracted from the previous script to keep the main loop clean.
    """
    # Resize the original (detail source) to match the relighted (target).
    if original_frame.shape != relighted_frame.shape:
        original_frame = cv2.resize(
            original_frame,
            (relighted_frame.shape[1], relighted_frame.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )

    # Convert to LAB and extract Luminance
    original_lab = cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB)
    relighted_lab = cv2.cvtColor(relighted_frame, cv2.COLOR_BGR2LAB)
    original_L, _, _ = cv2.split(original_lab)
    relighted_L, relighted_a, relighted_b = cv2.split(relighted_lab)

    # Isolate the Detail Layer from the Original
    original_L_float = original_L.astype(np.float32)
    relighted_L_float = relighted_L.astype(np.float32)

    blurred_L = cv2.GaussianBlur(original_L_float, (21, 21), 0)
    blurred_relighted_L = cv2.GaussianBlur(relighted_L_float, (21, 21), 0)

    detail_layer = original_L_float - blurred_L

    # Add the detail to the relighted luminance channel
    # enhanced_L_float = relighted_L_float + (detail_layer * strength)

    enhanced_L_float = original_L_float + (blurred_relighted_L * strength)
    enhanced_L = np.clip(enhanced_L_float, 0, 255).astype(np.uint8)

    # Recombine channels and convert back to BGR
    enhanced_lab = cv2.merge([enhanced_L, relighted_a, relighted_b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image

def LAB_merge(
    video1_path, video2_path, output_path,
    strength
):
    """
    Processes two videos frame-by-frame, transferring detail from video 1 to video 2,
    and saves the result as a new video file.
    """
    # --- Step 1: Open Video Captures ---
    cap1 = cv2.VideoCapture(video1_path) # Source of detail
    cap2 = cv2.VideoCapture(video2_path) # Target to enhance

    if not cap1.isOpened():
        print(f"Error: Could not open detail source video: {video1_path}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open target video: {video2_path}")
        return

    # Dimensions
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert (w1, h1) == (w2, h2), f"Dimension mismatch: {(w1,h1)} vs {(w2,h2)}"

    # Frame counts
    n1, n2 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n1 == n2, f"Frame count mismatch: {n1} vs {n2}"
    # assert the videos have the same dimensions and number of frames


    # --- Step 2: Get Properties and Create Video Writer ---
    # Use properties from the target video for the output
    frame_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap2.get(cv2.CAP_PROP_FPS)
    total_frames_v2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_v1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames_v2 == 0:
        print("Error: Target video has zero frames.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 files
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to '{output_path}' with {fps:.2f} FPS.")

    # --- Step 3: The Main Processing Loop ---
    current_frame_idx = 0
    start_time = time.time()
    
    while True:
        # Read a frame from the target video
        ret2, relighted_frame = cap2.read()

        # If the target video ends, break the loop
        if not ret2:
            break

        # Calculate the corresponding frame position in the source video
        # This handles different lengths and framerates by mapping progress percentage
        progress = current_frame_idx / total_frames_v2
        pos_v1 = int(progress * total_frames_v1)
        
        cap1.set(cv2.CAP_PROP_POS_FRAMES, pos_v1)
        ret1, original_frame = cap1.read()

        if not ret1:
            # If we can't read the source frame, something is wrong, so we stop.
            print(f"Warning: Could not read corresponding frame {pos_v1} from source video. Stopping.")
            break
            
        # --- Step 4: Perform Detail Transfer ---
        enhanced_frame = _transfer_detail_spatial(original_frame, relighted_frame, strength)
        
        # --- Step 5: Write Frame to Output Video ---
        out_writer.write(enhanced_frame)

        # Print progress
        if (current_frame_idx + 1) % 10 == 0: # Print every 10 frames
            print(f"Processed frame {current_frame_idx + 1}/{total_frames_v2}")
            
        current_frame_idx += 1

    # --- Step 6: Cleanup and Finalize ---
    end_time = time.time()
    print("\nProcessing finished.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print(f"Output video saved successfully: {output_path}")

    cap1.release()
    cap2.release()
    out_writer.release()

# --- Example Usage ---
if __name__ == '__main__':
    # You would replace these with your actual video files
    # For demonstration, we can use the dummy video creator
    # _create_dummy_videos() 


    raw_video = '/scratch/xiangrui/project/video_edit/Light-A-Video/input_wan/portrait/split/model_1.mp4'
    raw_name = os.path.splitext(os.path.basename(raw_video))[0]
    relighted_root = '/scratch/xiangrui/project/video_edit/Light-A-Video/output/portrait/relight_size_restored/jun28'


    out_dir = '/scratch/xiangrui/project/video_edit/Light-A-Video/output/portrait/LAB_weighted_BlurrLumiance_original_size/jun29'
    os.makedirs(out_dir, exist_ok=True)

    for folder_name in os.listdir(relighted_root):
        folder_path = os.path.join(relighted_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Look for a relighted video file that includes the raw video name
        for file in os.listdir(folder_path):
            if file.endswith('.mp4') and raw_name in file:
                relighted_video = os.path.join(folder_path, file)
                output_name = f'merged_{raw_name}.mp4'
                

                output_root = f'{out_dir}/strength1'
                output_path = os.path.join(output_root, folder_name, output_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                LAB_merge(
                    video1_path=raw_video,
                    video2_path=relighted_video,
                    output_path=output_path,
                    strength=0.1
                )
                
                output_root = f'{out_dir}/strength3'
                output_path = os.path.join(output_root, folder_name, output_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                LAB_merge(
                    video1_path=raw_video,
                    video2_path=relighted_video,
                    output_path=output_path,
                    strength=0.3
                )

                output_root = f'{out_dir}/strength5'
                output_path = os.path.join(output_root, folder_name, output_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                LAB_merge(
                    video1_path=raw_video,
                    video2_path=relighted_video,
                    output_path=output_path,
                    strength=0.5
                )

                output_root = f'{out_dir}/strength7'
                output_path = os.path.join(output_root, folder_name, output_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                LAB_merge(
                    video1_path=raw_video,
                    video2_path=relighted_video,
                    output_path=output_path,
                    strength=0.7
                )