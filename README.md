## Installation
```
git clone 
cd 

conda create -n Slight python=3.10
conda activate Slight

pip install -r requirements.txt
```

##  Quick Start
## Input Video specification

The framework supports the Wan2.1 backbone supporting frame numbers of 49 and 81.

### To split a long video into video segments of the desired number of frames and fps

```
python ./tool/split_long_video.py --input_video_path './demo/demo.mp4' --output_folder './input/81frames_24fps/' --frames_per_segment 81 --target_fps 24
```


### To perform the video relighting pipeline
### Change the following path to yours and edit the prompts in the yaml configuration file.
```bash
# path
raw_input_video = 'Change to your video path' ### Change this to your input video segment path. Set to 49 or 81 frames.
yaml_config_file = 'change to your yaml' ### Change this to your YAML config file path.
out_dir = 'Change this to your output directory.' 

# run script
python run_relight.py
```


### To get the newest/dev diffusers version 
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install .
```

## 📎 Citation 

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝 
```bibtex

```

## 📣 Disclaimer

This is the official code of Slight.
All the copyrights of some of the demo videos are from community users. 
Feel free to contact us if you would like to remove them.

## 💞 Acknowledgements
The code is built upon the following repositories. We thank all the contributors for open-sourcing.
* [Light-A-Video](https://github.com/bcmi/Light-A-Video)
* [IC-Light](https://github.com/lllyasviel/IC-Light)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)
* [CogVideoX](https://github.com/THUDM/CogVideo)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1)


