import argparse
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
import shutil
import torch

# ------------------------------- Printing helpers --------------------------------
def print_header(title, width=60):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_subheader(title, width=60):
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)

def print_info(label, value, indent=2):
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")

def print_success(message, indent=2):
    spaces = " " * indent
    print(f"{spaces}✓ {message}")

def print_progress(step, total_steps, description):
    progress = f"[{step}/{total_steps}]"
    print(f"\n{progress} {description}")

def print_config_block(title: str, **items):
    width = max(len(k) for k in items) if items else 0
    line = "=" * 120
    print("\n" + line)
    print(f"{title}".center(120))
    print("-" * 120)
    for k, v in items.items():
        print(f"{k:<{width}} : {v}")
    print(line + "\n")

# ------------------------------- CLI ---------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hi-Light video relighting pipeline."
    )
    parser.add_argument(
        "--raw_input_video", type=str, required=True,default=".demo/woman_holding_a_phone.mp4",
        help="Path to the raw input video."
    )
    parser.add_argument(
        "--config", dest="yaml_config_file", type=str, required=True,default="./configs/wan_relight/woman_holding_a_phone.yaml",
        help="Path to YAML config file to update and use."
    )
    parser.add_argument(
        "--outdir", type=str, default="./relit_output",
        help="Output directory. If omitted, a timestamped folder is created."
    )
    parser.add_argument(
        "--vdm", type=str, default="wan21",
        choices=["cog", "wan21",  "animatediff"],
        help="Video diffusion model Backbone."
    )
    parser.add_argument(
        "--merge_strength", type=float, default=0.3,
        help="LAB-DF merge strength (0~1)."
    )
    parser.add_argument(
        "--smooth", action="store_true",default=True,
        help="Enable HMA-LSF temporal smoothing."
    )
    parser.add_argument(
        "--scale_brightness", action="store_true",default=True,
        help="Enable final brightness/saturation scaling."
    )
    parser.add_argument(
        "--saturation_factor", type=float, default=2.0,
        help="Saturation scale factor if brightness scaling enabled."
    )
    parser.add_argument(
        "--brightness_factor", type=float, default=1.2,
        help="Brightness scale factor if brightness scaling enabled."
    )
    parser.add_argument(
        "--lab_transfer_mode", type=str, default="light",
        choices=["light", "detail"],
        help="LAB transfer mode."
    )
    parser.add_argument(
        "--lab_fuse_mode", type=str, default="weighted_sum",
        choices=["weighted_sum", "absolute_scaling", "ratio_scaling"],
        help="LAB-DF fuse (blend) mode."
    )
    return parser.parse_args()

# ------------------------------- YAML update -------------------------------------
def update_yaml_video_path(yaml_file_path, new_video_path, out_dir, new_dim, fps, num_frames):
    """
    Update the video_path and related fields in the YAML configuration file.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        config['video_path'] = new_video_path
        config['save_path'] = out_dir
        config['fps'] = fps
        config['width'] = new_dim[0]
        config['height'] = new_dim[1]
        config['num_frames'] = num_frames
        print(f'Updated video_path in YAML to: {new_video_path}')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f'Successfully updated {yaml_file_path}')
    except Exception as e:
        print(f'Error updating YAML file: {e}')
        raise

# ------------------------------- Relighting runner -------------------------------
def run_relighting(out_dir, yaml_config_file, fps, vdm):
    if vdm == 'wan21':
        your_main_script = './lav_wan_relight.py'
    elif vdm == 'animatediff':
        your_main_script = './lav_relight.py'
    elif vdm == 'cog':
        your_main_script = './lav_cog_relight.py'
    elif vdm == 'wan22':
        your_main_script = './lav_wan_relight_wan22.py' # not available yet
    else:
        raise ValueError(f"Unsupported vdm: {vdm}")

    os.makedirs(out_dir, exist_ok=True)
    print('running config:', yaml_config_file)

    try:
        # ensure config can be read
        with open(yaml_config_file, 'r') as f:
            _ = yaml.safe_load(f)

        command = [
            os.sys.executable,
            your_main_script,
            "--config", yaml_config_file,
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("Script run completed successfully.")
        else:
            print(f"Script FAILED with exit code {process.returncode}.")
            print("Script Output:\n", stdout)
            print("Script Error:\n", stderr)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. '{e.filename}'")
        print(f"Check `{your_main_script}` and `{yaml_config_file}` paths.")
    except Exception as e:
        print(f"Unexpected error during relight: {e}")
    finally:
        print("\n--- Relight process finished ---")

# ------------------------------- Main --------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    raw_input_video = args.raw_input_video
    yaml_config_file = args.yaml_config_file
    vdm = args.vdm

    # Gather input video info
    cap = cv2.VideoCapture(raw_input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {raw_input_video}")
    raw_video_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_name = os.path.splitext(os.path.basename(raw_input_video))[0]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Output directory
    if args.outdir:
        out_dir = args.outdir
    else:
        out_dir = os.path.join(
            os.getcwd(),
            "outputs",
            vdm,
            f"{video_name}_{frame_count}FS_{ts}"
        )

    os.makedirs(out_dir, exist_ok=True)

    # Pull remaining args
    merge_strength = args.merge_strength
    smooth = args.smooth                     # default False unless flag is provided
    scale_brightness = args.scale_brightness # default False unless flag is provided
    saturation_factor = args.saturation_factor
    brightness_factor = args.brightness_factor
    LAB_transfer_mode = args.lab_transfer_mode
    LAB_fuse_mode = args.lab_fuse_mode

    start_time = time.time()

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

    # Step 1: Downsize
    print_progress(1, 5, "DOWNSIZING VIDEO")
    print_subheader("Downsampling the video dimensions")

    w, h = raw_video_size
    if vdm == "animatediff":
        downsizing_width = downsizing_height = 512
    elif vdm in {"cog", "wan21", "wan22"}:
        if w > h:
            if vdm == "cog":
                downsizing_width, downsizing_height = 720, 480
            elif vdm == "wan21":
                downsizing_width, downsizing_height = 848, 480
            else:  # wan22
                downsizing_width, downsizing_height = 832, 480
        elif w < h:
            if vdm == "cog":
                downsizing_width, downsizing_height = 480, 720
            elif vdm == "wan21":
                downsizing_width, downsizing_height = 480, 848
            else:  # wan22
                downsizing_width, downsizing_height = 480, 832
        else:
            downsizing_width = downsizing_height = 512
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
    print_success("Step 1 Complete - Downsized video saved")
    print_info("Output Path", downsized_video_dir, indent=4)

    # Step 2: Relight
    print_progress(2, 5, "RUNNING RELIGHTING PROCESS")
    print_subheader("Applying relighting")

    relighted_video_path = os.path.join(out_dir, 'relight_Intermediate_downsized_video.mp4')

    print_info("Status", "Updating configuration to yaml file...")
    update_yaml_video_path(
        yaml_config_file,
        downsized_video_dir,
        out_dir,
        new_dim=(downsizing_width, downsizing_height),
        fps=fps,
        num_frames=frame_count
    )

    print_info("Status", "Running relighting model...")
    run_relighting(out_dir=out_dir, yaml_config_file=yaml_config_file, fps=fps, vdm=vdm)

    if torch.cuda.is_available():
        try:
            peak = torch.cuda.max_memory_allocated()
            print(f"Peak VRAM: {peak / 1024**2:.2f} MB")
        except Exception:
            pass

    end_time = time.time()
    processing_time = end_time - start_time
    print_success("Step 2 Complete - Relighting applied")
    print_info("Output Path", relighted_video_path, indent=4)
    print_info("Processing Time", f"{processing_time:.2f} seconds", indent=4)

    # Step 3: Upsample back to original size
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
    print_success("Step 3 Complete - Video upsampled to original size")
    print_info("Output Path", restored_video_path, indent=4)

    # Optional smoothing
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

    end_time = time.time()
    processing_time = end_time - start_time
    print_info("Total Processing Time", f"{processing_time:.2f} seconds")

    # Step 4: LAB merge
    print_progress(4, 5, "LAB COLOR SPACE MERGING")
    print_subheader("LAB Detail Preserving Fusion")

    print_info("Merge Strength", f"{merge_strength}")
    print_info("Blend Mode", f"{LAB_fuse_mode}")
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
    else:
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

    end_time = time.time()
    processing_time = end_time - start_time
    print_info("Total Processing Time", f"{processing_time:.2f} seconds")
    print_success("Step 4 Complete - LAB-DF applied")

    # Step 5: Optional brightness/saturation scaling
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
        else:
            print_info("Status", "Scaling smoothed version...")
            brightness_scaled_video_out = os.path.join(
                out_dir, f'FINAL_smoothed_brt{brightness_factor}_sat{saturation_factor}_video.mp4'
            )
            src.postprocess.brightness_scaling(
                input_path=post_upsample_merged_path,
                output_path=brightness_scaled_video_out,
                brightness_factor=brightness_factor,
                saturation_factor=saturation_factor
            )
            print_success("Brightness scaling applied to smoothed version")
            print_info("Final Output", brightness_scaled_video_out, indent=4)

    # Summary
    print_header("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print_info("Total Processing Steps", "5/5 completed")

    final_outputs = []
    if scale_brightness:
        if smooth:
            final_outputs.append(("Final Enhanced Video (Recommended)",
                                  os.path.join(out_dir, f'FINAL_smoothed_brt{brightness_factor}_sat{saturation_factor}_video.mp4')))
        else:
            final_outputs.append(("Final Enhanced Video (Unsmoothed)",
                                  os.path.join(out_dir, 'FINAL_scaled_unsmoothed_video.mp4')))
    else:
        if smooth:
            final_outputs.append(("Final Merged Video (Smoothed)", post_upsample_merged_path))
        final_outputs.append(("Final Merged Video", final_merged_video))

    print_subheader("Final Output Files")
    for i, (description, path) in enumerate(final_outputs, 1):
        print_info(f"Output {i}", f"{description}")
        print_info("", f"→ {path}", indent=6)

    # Copy YAML used into output folder
    yaml_copy_path = os.path.join(out_dir, os.path.basename(yaml_config_file))
    shutil.copy2(yaml_config_file, yaml_copy_path)
    print_success("YAML configuration copied to output folder")
    print_info("YAML Path", yaml_copy_path, indent=4)

    print("\n" + "=" * 60 + "\n")
    end_time = time.time()
    processing_time = end_time - start_time
    print_info("Total Processing Time", f"{processing_time:.2f} seconds")
    print("\n" + "=" * 60 + "\n")