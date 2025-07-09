import cv2
import os
import glob
import argparse
from pathlib import Path

def split_video_into_segments(input_path, output_dir, frames_per_segment=49, target_fps=14):
    """
    Split a video into segments of specified frame count and fps.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save output segments
        frames_per_segment: Number of frames per segment (default: 49)
        target_fps: Target fps for output videos (default: 14)
    """
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing: {input_path}")
    print(f"Total frames: {total_frames}, Original FPS: {original_fps}")
    print(f"Resolution: {width}x{height}")
    
    # Calculate how many complete segments we can create
    complete_segments = total_frames // frames_per_segment
    print(f"Will create {complete_segments} segments of {frames_per_segment} frames each")
    
    # Get video name without extension for output naming
    video_name = Path(input_path).stem
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    segment_count = 0
    
    for segment_idx in range(complete_segments):
        # Create output filename
        output_filename = f"{video_name}_{segment_idx + 1}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Initialize video writer for this segment
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            continue
        
        # Read and write frames for this segment
        frames_written = 0
        for frame_idx in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} for segment {segment_idx + 1}")
                break
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        # Verify we wrote the correct number of frames
        if frames_written == frames_per_segment:
            segment_count += 1
            print(f"Created: {output_filename} ({frames_written} frames)")
        else:
            # Remove incomplete segment
            os.remove(output_path)
            print(f"Removed incomplete segment: {output_filename} (only {frames_written} frames)")
    
    cap.release()
    print(f"Completed processing {input_path}: {segment_count} segments created\n")

def process_video_folder(input_folder, output_folder, frames_per_segment=49, target_fps=14):
    """
    Process all videos in a folder.
    
    Args:
        input_folder: Path to folder containing input videos
        output_folder: Path to folder for output segments
        frames_per_segment: Number of frames per segment (default: 49)
        target_fps: Target fps for output videos (default: 14)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
        video_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Output directory: {output_folder}")
    print(f"Settings: {frames_per_segment} frames per segment, {target_fps} fps\n")
    
    # Process each video
    for video_file in video_files:
        split_video_into_segments(
            video_file, 
            output_folder, 
            frames_per_segment, 
            target_fps
        )
    
    print("All videos processed!")

# Example usage
if __name__ == "__main__":
    # Set your input and output folders here
    argparser = argparse.ArgumentParser(description="Split videos into segments.")
    argparser.add_argument('--input_video_path', type=str, required=True, help='Path tothe long video')
    argparser.add_argument('--output_folder', type=str, required=True, help='Path to output folder for video segments')
    argparser.add_argument('--frames_per_segment', type=int, default=49, help='Number of frames per segment (default: 49)')     
    argparser.add_argument('--target_fps', type=int, default=14, help='Target FPS for output videos (default: 14)')
    args = argparser.parse_args()
    
    # Input and output folders
    input_video_path = args.input_video_path  # Change this to your input folder that contains videos
    output_folder = args.output_folder  # Change this to your output folder
    os.makedirs(output_folder, exist_ok=True)
    # Parameters
    frames_per_segment = args.frames_per_segment
    target_fps = args.target_fps
    
    # Process all videos in the folder
    # process_video_folder(
    #     intput_folder, 
    #     output_folder, 
    #     frames_per_segment, 
    #     target_fps
    # )

    # Process a single video
    
    split_video_into_segments(input_path=input_video_path, output_dir=output_folder, frames_per_segment=frames_per_segment, target_fps=target_fps)