
import yaml
import random
import cv2
import numpy as np
import subprocess
import os
import tempfile
import shutil
from tqdm import tqdm
from datetime import datetime

import src.preprocess
import src.postprocess


def update_yaml_video_path(yaml_file_path, new_video_path,out_dir,new_dim,fps):
    """
    Update the video_path attribute in the YAML configuration file
    """
    try:
        # Read the YAML file
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update the video_path attribute
        config['video_path'] = new_video_path
        config['save_path'] = out_dir
        config['fps'] = fps  # Update the fps if needed
        config['width'] = new_dim[0]
        config['height']= new_dim[1]  # Update the video size if needed
        print(f'Updated video_path in YAML to: {new_video_path}')
        
        # Write back to the YAML file
        with open(yaml_file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f'Successfully updated {yaml_file_path}')
        
    except Exception as e:
        print(f'Error updating YAML file: {e}')
        raise

def run_relighting(out_dir,yaml_config_file,fps):

    # Define the path to your original script
    your_main_script = './lav_wan_relight.py' # Assuming your main code is in main.py
    
    base_config_template_path = yaml_config_file

    # Base output directory where all relighted results will be saved
    base_results_dir = os.path.join(out_dir)
    os.makedirs(base_results_dir, exist_ok=True) # Ensure this base directory exists

    # ramdon light direction
    print('running config:', base_config_template_path)
    # --- Define your experiment parameters ---

    run_count = 0
    try:
        # Load the base configuration once from your default config file
        with open(base_config_template_path, 'r') as f:
            base_config = yaml.safe_load(f)
       
        command = [
            os.sys.executable, # Ensures the script runs with the same Python interpreter
            your_main_script,
            "--config", base_config_template_path,
            "--sd_model", base_config.get("sd_model", "stablediffusionapi/realistic-vision-v51"), 
            "--vdm_model", base_config.get("vdm_model", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            "--ic_light_model", base_config.get("ic_light_model", "./models/iclight_sd15_fc.safetensors"),
        ]
        
        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate() # Wait for the process to complete

        if process.returncode == 0:
            print(f"Script run {run_count} completed successfully.")
            # print("Script Output:\n", stdout) # Uncomment if you want to see full stdout
        else:
            print(f"Script run {run_count} FAILED with exit code {process.returncode}.")
            print("Script Output:\n", stdout)
            print("Script Error:\n", stderr)

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Make sure '{e.filename}' exists.")
        print(f"Please check if `{your_main_script}` and `{base_config_template_path}` exist and are in the correct paths.")
    except Exception as e:
        print(f"An unexpected error occurred during the loop: {e}")
    finally:
        print("\n--- Relight process finished ---")
         
if __name__ == "__main__":

    
    long_video_path = None ### only if you wish to split a long video into segments, otherwise set to None.
    raw_input_video = '/scratch/xiangrui/project/video_edit/Light-A-Video/input_wan/portrait/81frames_24fps_split/man_taking_photo_1.mp4' ### Change this to your input video segment path. Must be 49 frames.
    video_name = os.path.splitext(os.path.basename(raw_input_video))[0]
    yaml_config_file = '/scratch/xiangrui/project/video_edit/Light-A-Video/configs/wan_relight/man_taking_photo.yaml' ### Change this to your YAML config file path.
    out_dir = f'/scratch/xiangrui/project/video_edit/Light-A-Video/test_out/{video_name}' ### Change this to your output directory.



    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(raw_input_video)
    raw_video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()


    split_long_video = False
    # step 0. Split the input video into segments of maximum 81 frames for WAN2.1
    if split_long_video:
        print('Splitting the video into segments...')
        video_seg_out_dir = ''
        src.preprocess.split_video_into_segments(long_video_path, output_dir=video_seg_out_dir, frames_per_segment=49, target_fps=14)




    # Step 1: Downsize the original video
    print('='*90)
    print('Step 1: Downsizing the video')

    if raw_video_size[0] > raw_video_size[1]:# width > height
        downsizing_width = 480
        downsizing_height = 854
    elif raw_video_size[0] < raw_video_size[1]:# width < height
        downsizing_width = 854
        downsizing_height = 480
    else: # width == height
        downsizing_width = 512
        downsizing_height = 512

    downsized_dir = os.path.join(out_dir, 'downsized_video.mp4')
    downsized_video_dir = src.preprocess.downsize_video(
        input_path=raw_input_video, 
        output_path=downsized_dir, 
        width=downsizing_width, 
        height=downsizing_height
    )

    print(f'Step 1 complete. Output: {downsized_video_dir}')
    print()
    


    # Step 2: Run the relighting loop
    print('Step 2: Running the relighting process')
    # Update YAML file with the downsized video path
    update_yaml_video_path(yaml_config_file, downsized_video_dir,out_dir,new_dim = (downsizing_width,downsizing_height),fps=fps)
    run_relighting(out_dir=out_dir, yaml_config_file=yaml_config_file,fps=fps)
    # The relighting process should output to: out_dir/relight_Left_downsized_video.mp4
    relighted_video_path = os.path.join(out_dir, 'relight_downsized_video.mp4')
    print(f'Step 2 complete. Output: {relighted_video_path}')
    print()



    # Step 3: Upsample the size of the relighted video to match the original video
    print('Step 3: Upsampling the relighted video to match the original video size')

    
    restored_video_path = os.path.join(out_dir, 'relighted_size_restored_video.mp4')
    src.postprocess.upsample_video(
        input_path=relighted_video_path, 
        output_path=restored_video_path, 
        width=raw_video_size[0], 
        height=raw_video_size[1]
    )
    print(f'Step 3 complete. Output: {restored_video_path}')
    print()

    # Step 4: LAB merge the relighted video with the original video
    print('Step 4: LAB merging the relighted video with the original video')
    final_merged_video = os.path.join(out_dir, 'final_merged_video.mp4')
    src.postprocess.LAB_merge(
        video1_path=raw_input_video, 
        video2_path=restored_video_path, 
        output_path=final_merged_video,
        strength=0.1
    )
    print(f'Step 4 complete. Final output: {final_merged_video}')
    
    print('Pipeline completed successfully!')