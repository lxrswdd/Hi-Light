import yaml
import cv2
import numpy as np
import subprocess
import os
from tqdm import tqdm
from datetime import datetime
import time
import src.preprocess
import src.postprocess
import src.flickerling_smooth

def print_header(title, width=60):
    """Print a formatted header with consistent styling"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_subheader(title, width=60):
    """Print a formatted subheader"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)

def print_info(label, value, indent=2):
    """Print formatted information with consistent indentation"""
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")

def print_success(message, indent=2):
    """Print success message with checkmark"""
    spaces = " " * indent
    print(f"{spaces}✓ {message}")

def print_progress(step, total_steps, description):
    """Print progress indicator"""
    progress = f"[{step}/{total_steps}]"
    print(f"\n{progress} {description}")

def update_yaml_video_path(yaml_file_path, new_video_path,out_dir,new_dim,fps,num_frames):
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
        config['num_frames'] = num_frames
        print(f'Updated video_path in YAML to: {new_video_path}')
        
        # Write back to the YAML file
        with open(yaml_file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f'Successfully updated {yaml_file_path}')
        
    except Exception as e:
        print(f'Error updating YAML file: {e}')
        raise


def run_relighting(out_dir,yaml_config_file,fps,vdm):

    # Define the path to your original script
    if vdm == 'wan21':
        your_main_script = './lav_wan_relight.py'
    elif vdm == 'animatediff':
        your_main_script = './lav_relight.py'
    elif vdm == 'cog':
        your_main_script = './lav_cog_relight.py'
    elif vdm =='wan22':
        your_main_script = './lav_wan_relight_wan22.py'
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

    
    raw_input_video = '' ### Change this to your input video segment path. Must be 49 frames.

    video_name = os.path.splitext(os.path.basename(raw_input_video))[0]
    yaml_config_file = './woman_cry.yaml' ### Change this to your YAML config file path.
    vdm = 'wan21'# VDM backbone 'wan21' or 'animatediff' or "cog"


    # Configuration for post-processing
    merge_strength = 0.3
    smooth = True
    scale_brightness = True
    LAB_transfer_mode = 'light'
    LAB_fuse_mode = 'weighted_sum'

    saturation_factor = 2
    brightness_factor = 1.2


    cap = cv2.VideoCapture(raw_input_video)
    raw_video_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    cap.release()

    out_dir = f'./test_out/{vdm}/{video_name}_{frame_count}frames' ### Change this to your output directory.
    os.makedirs(out_dir, exist_ok=True)


    def print_config_block(title: str, **items):
        width = max(len(k) for k in items)
        line = "=" * 120
        print("\n" + line)
        print(f"{title}".center(120))
        print("-" * 120)
        for k, v in items.items():
            print(f"{k:<{width}} : {v}")
        print(line + "\n")

    print_config_block(
        title=f"VIDEO RELIGHTING PIPELINE  |  VDM: {vdm.upper()}",
        VDM=vdm,
        Input_Video=raw_input_video,
        Output_Directory=out_dir,
        Original_Video_Size=f"{raw_video_size[0]}x{raw_video_size[1]}",
        FPS=fps,
        Merge_Strength=merge_strength,
        Smooth=smooth,
        Scale_Brightness=scale_brightness,
        LAB_Transfer_Mode=LAB_transfer_mode,
        LAB_Fuse_Mode=LAB_fuse_mode,
        Saturation_Factor=saturation_factor,
        Brightness_Factor=brightness_factor,
    )

    print_progress(1, 5, "DOWNSIZING VIDEO")
    print_subheader("Downsampling the video dimensions")

w, h = raw_video_size
orientation = None

if vdm == "animatediff":
    downsizing_width = downsizing_height = 512

elif vdm in {"cog", "wan21", "wan22"}:
    # width > height
    if w > h:
        if vdm == "cog":
            downsizing_width, downsizing_height = 720, 480
        elif vdm == "wan21":
            downsizing_width, downsizing_height = 848, 480
        else:  # wan22
            downsizing_width, downsizing_height = 832, 480
        orientation = "Landscape"

    # width < height
    elif w < h:
        if vdm == "cog":
            downsizing_width, downsizing_height = 480, 720
        elif vdm == "wan21":
            downsizing_width, downsizing_height = 480, 848
        else:  # wan22
            downsizing_width, downsizing_height = 480, 832
        orientation = "Portrait"

    # width == height
    else:
        downsizing_width = downsizing_height = 512
    

    print_info("Processing", "Downsizing video...")
    downsized_dir = os.path.join(out_dir, 'Intermediate_downsized_video.mp4')
    downsized_video_dir = src.preprocess.downsize_video(
        input_path=raw_input_video, 
        output_path=downsized_dir, 
        width=downsizing_width, 
        height=downsizing_height
    )

    print_success(f"Step 1 Complete - Downsized video saved")
    print_info("Output Path", downsized_video_dir, indent=4)
    
    # Step 2: Run the relighting loop
    print_progress(2, 5, "RUNNING RELIGHTING PROCESS")
    print_subheader("Applying relighting")
    
    start_time = time.time()
    relighted_video_path = os.path.join(out_dir, 'relight_Intermediate_downsized_video.mp4')

    print_info("Status", "Updating configuration to yaml file...")
    # Update YAML file with the downsized video path
    update_yaml_video_path(yaml_config_file, downsized_video_dir, out_dir, 
                          new_dim=(downsizing_width, downsizing_height), fps=fps, num_frames=frame_count)

    print_info("Status", "Running relighting model...")
    run_relighting(out_dir=out_dir, yaml_config_file=yaml_config_file, fps=fps, vdm=vdm)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print_success(f"Step 2 Complete - Relighting applied")
    print_info("Output Path", relighted_video_path, indent=4)
    print_info("Processing Time", f"{processing_time:.2f} seconds", indent=4)


    # Step 3: Upsample the size of the relighted video to match the original video
    print_progress(3, 5, "UPSAMPLING TO ORIGINAL RESOLUTION")
    print_subheader("Restoring original video dimensions")
    
    print_info("Target Size", f"{raw_video_size[0]}x{raw_video_size[1]}")
    print_info("Status", "Upsampling video...")
    
    restored_video_path = os.path.join(out_dir, 'Intermediate_relighted_size_restored_video.mp4')
    src.postprocess.upsample_video(
        input_path=relighted_video_path, 
        output_path=restored_video_path, 
        width=raw_video_size[0], 
        height=raw_video_size[1]
    )
    
    print_success(f"Step 3 Complete - Video upsampled to original size")
    print_info("Output Path", restored_video_path, indent=4)
    
    if smooth:
        print_info("Status", "Applying temporal smoothing...")
        post_upsample_stable_out_path = os.path.join(out_dir, 'Intermediate_post_dual_smoothed_video.mp4')
        src.flickerling_smooth.smooth_highlights_combined(
            input_path=restored_video_path, 
            output_path=post_upsample_stable_out_path,
            flow_quality='medium'
        )
        print_success("Temporal smoothing applied")
        print_info("Smoothed Output", post_upsample_stable_out_path, indent=4)

    # Step 4: LAB merge the relighted video with the original video
    print_progress(4, 5, "LAB COLOR SPACE MERGING")
    print_subheader("LAB Detial Preserving Fusion")

    
    print_info("Merge Strength", f"{merge_strength} (30%)")
    print_info("Blend Mode", "Absolute Scaling")
    print_info("Status", "Performing LAB color space merge...")

    final_merged_video = os.path.join(out_dir, 'FINAL_original_merged_video.mp4')

    if not smooth:
        src.postprocess.LAB_merge(
            video1_path=raw_input_video, 
            video2_path=restored_video_path, 
            output_path=final_merged_video,
            strength=merge_strength,
            blend_mode=LAB_fuse_mode,
            transfer_mode=LAB_transfer_mode
        )
        print_info("Output Path", final_merged_video, indent=4)


    if smooth:
        print_info("Status", "Applying LAB-DF to smoothed version...")
        post_upsample_merged_path = os.path.join(out_dir, f'FINAL_smooth_MS{merge_strength}.mp4')
        src.postprocess.LAB_merge(
            video1_path=raw_input_video, 
            video2_path=post_upsample_stable_out_path, 
            output_path=post_upsample_merged_path,
            strength=merge_strength,
            blend_mode=LAB_fuse_mode,
            transfer_mode=LAB_transfer_mode
        )
        print_success("LAB merge applied to smoothed version")
        print_info("Smoothed Video Output", post_upsample_merged_path, indent=4)

    print_success("Step 4 Complete - LAB-DF applied")

    # Step 5: Brightness and saturation scaling (optional)
    if scale_brightness:
        print_progress(5, 5, "BRIGHTNESS & SATURATION SCALING")
        print_subheader("Enhancing final video appearance")   
        print_info("Brightness Factor", f"{brightness_factor}x")
        print_info("Saturation Factor", f"{saturation_factor}x")
        print_info("Status", "Scaling brightness and saturation...")
        
        if not smooth:
            brightness_scaled_unsmoothed_video_out = os.path.join(out_dir, 'FINAL_scaled_unsmoothed_video.mp4')

            src.postprocess.brightness_scaling(
                input_path=final_merged_video,
                output_path=brightness_scaled_unsmoothed_video_out,
                brightness_factor=brightness_factor,
                saturation_factor=saturation_factor
            )

            print_success("Brightness scaling applied to unsmoothed version")
            print_info("Output Path", brightness_scaled_unsmoothed_video_out, indent=4)

        if smooth:
            print_info("Status", "Scaling smoothed version...")
            brightness_scaled_video_out = os.path.join(out_dir, f'FINAL_smoothed_brt{brightness_factor}_sat{saturation_factor}_video.mp4')

            src.postprocess.brightness_scaling(
                input_path=post_upsample_merged_path,
                output_path=brightness_scaled_video_out,
                brightness_factor=brightness_factor,
                saturation_factor=saturation_factor
            )
            
            print_success("Brightness scaling applied to smoothed version")
            print_info("Final Output", brightness_scaled_video_out, indent=4)

    # Pipeline completion summary
    print_header("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print_info("Total Processing Steps", "5/5 completed")
    
    final_outputs = []

    if scale_brightness:
        if smooth:
            final_outputs.append(("Final Enhanced Video (Recommended)", 
                                os.path.join(out_dir, f'FINAL_smoothed_brt{brightness_factor}_sat{saturation_factor}_video.mp4')))

    else:
        if smooth:
            final_outputs.append(("Final Merged Video (Smoothed)", post_upsample_merged_path))
        final_outputs.append(("Final Merged Video", final_merged_video))
    
    print_subheader("Final Output Files")
    for i, (description, path) in enumerate(final_outputs, 1):
        print_info(f"Output {i}", f"{description}")
        print_info("", f"→ {path}", indent=6)
    
    print("\n" + "=" * 60 + "\n")
