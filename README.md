```
git clone 
cd 

conda create -n lav python=3.10
conda activate lav

pip install -r requirements.txt
```

## 🎈 Quick Start
## Input Video specification

The framework supports the Wan2.1 backbone supporting frame numbers of 49 and 81.

### To split a long video into video segments of the desired number of frames and fps

```
python ./tool/split_long_video.py 
```


### Perform video relighting pipeline with customized illumination control
```bash
# relight
python lav_relight.py --config "configs/relight/car.yaml"
```




### Update Diffusers from source
```bash
conda activate lav

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install.
```

### Perform video relighting with Wan2.1
Wan2.1 with Flow-Matching scheduler.
The VDM checkpoint is [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) and it will be downloaded automatically.
```bash
python lav_wan_relight.py --config "configs/wan_relight/bear.yaml"
```

## 📎 Citation 

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝 
```bibtex

```

## 📣 Disclaimer

This is the official code of .
All the copyrights of some of the demo videos are from community users. 
Feel free to contact us if you would like to remove them.

## 💞 Acknowledgements
The code is built upon the following repositories. We thank all the contributors for open-sourcing.
* [Light-A-_](https://github.com/lllyasviel/IC-Light)
* [IC-Light](https://github.com/lllyasviel/IC-Light)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)
* [CogVideoX](https://github.com/THUDM/CogVideo)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1)


