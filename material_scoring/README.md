
To robustly assess the quality of 3D assets, we adopt a video-based evaluation paradigm that captures spatio-temporal information from rendered videos. You can simply evaluate the generated asset from multiple dimensions with following steps.

## Installation
Create a conda environment and install dependencies:
```
conda create -n hi3d_video python=3.10 -y
conda activate hi3d_video
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Quick Start
- change your path to ckpts and videos in `infer/config.py`, where your data should be organized as json files in the `data` folder
- run the script `infer/run.sh` for evaluation
- output scores will be saved in `infer` folder, you can change it in the config file

## Training
- list your data according to the json examples in `data` folder
- modify the ckpt path and training stage in `train/config.py`
- run the script `train/run.sh` for training

## Acknowledgements
This codebase is built on top of the open-source implementation of [InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2) repo.

