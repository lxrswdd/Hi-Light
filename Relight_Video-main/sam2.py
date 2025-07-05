import os
import torch
import argparse
import numpy as np
from PIL import Image
from ultralytics.models.sam import SAM2VideoPredictor


def main(args):
    
    # Create SAM2VideoPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
    predictor = SAM2VideoPredictor(overrides=overrides)

    video_name = args.video_name
    results = predictor(source=f"input_animatediff/{video_name}.mp4",points=[args.x, args.y],labels=[1])

    for i in range(len(results)):
        mask = (results[i].masks.data).squeeze().to(torch.float16)
        mask = (mask * 255).cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask)
        mask_dir = f'masks_animatediff/{video_name}'
        if not os.path.exists(mask_dir):  
            os.makedirs(mask_dir)        
        mask_image.save(mask_dir + f'/{str(i).zfill(3)}.png')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process a video and generate masks using SAM2VideoPredictor.")
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video file (without extension).")
    parser.add_argument("--x", type=int, default=255, help="X coordinate of the point.")
    parser.add_argument("--y", type=int, default=255, help="Y coordinate of the point.")
    
    args = parser.parse_args()
    main(args)