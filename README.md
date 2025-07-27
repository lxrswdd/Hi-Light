## 💡 Demo
<p align="center">
  <a href="https://github.com/user-attachments/assets/becdc6bb-5238-4817-84e3-5b90f7a7313c">
    <img src="https://github.com/user-attachments/assets/becdc6bb-5238-4817-84e3-5b90f7a7313c" alt="Demo Video" />
  </a>
</p>

<table class="center">
    <tr>
      <td><p style="text-align: center">Input video</p></td>
      <td><p style="text-align: center">Sunset lighting over sea</p></td>
    </tr>
    <tr>
      <td><img src="__assets__/sea_gull_1.gif"></td>
      <td><img src="__assets__/sea_gull_1_sunset.gif"></td>
    </tr>
    <tr>
      <td><p style="text-align: center">Input video </p></td>
      <td><p style="text-align: center">Green aurora lighting</p></td>
    </tr>
    <tr>
      <td><img src="__assets__/1440p_woman.gif"></td>
      <td><img src="__assets__/1440p_woman_aurora.gif"></td>
    </tr>
</table>

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

The framework adopts the Wan2.1 backbone, supporting frame numbers of 49 and 81.

Ensure the video has a smallest resolution of 480p.

### 1. To split a long video into video segments of the desired number of frames and fps

```
python ./utils/split_long_video.py --input_video_path './demo/man_taking_notes.mp4' --output_folder './input/81frames_24fps/' --frames_per_segment 81 --target_fps 24
```


### 2. To run the video relighting pipeline: Change the following path to yours and edit the prompts in the yaml configuration file.
```bash
# path
raw_input_video = 'Change to your video path' ### Change this to your input video segment path. Set to 49 or 81 frames.
yaml_config_file = 'change to your yaml' ### Change this to your YAML config file path.
out_dir = 'Change this to your output directory.' 

# run script
python run_relight.py
```
### 3. To evaluate the relit video
```
python relit_video_evaluation.py --raw_video_path "Path to your raw input video" --relit_video_path "Path to the relit video"
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
The the copyrights of some of the demo videos are from community users. 
Feel free to contact us if you would like to remove them.

## 💞 Acknowledgements
The code is built upon the following repositories. We thank all the contributors for open-sourcing.
* [Light-A-Video](https://github.com/bcmi/Light-A-Video)
* [IC-Light](https://github.com/lllyasviel/IC-Light)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1)


