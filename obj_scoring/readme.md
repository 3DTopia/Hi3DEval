# Usage Guide

## 1. Prepare Result Set JSON
You need to organize the result set JSON according to the prompts defined by Hi3DEval.  
For concrete examples, please refer to `data/worgb.json` and `data/wrgb.json`:  

- `worgb.json` contains examples without RGB input format.  
- `wrgb.json` contains examples with RGB input format.  

Different evaluation metrics require different input formats. Please check `infer/config.py` for detailed descriptions.

## 2. Modify Configuration
Depending on the metric you want to evaluate, update the `anno_path` in `infer/config.py` to point to the corresponding result JSON file.  
`infer/config.py` also explains which models are used for inference under different metrics, so please adjust the configuration accordingly.

## 3. Download Checkpoints

- CLIP-ViT
  - Download `CLIP-ViT-H-14` from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K, and place at `object_scoring/ckpts/CLIP-ViT-H-14` 
  - Download `mp_rank_00_model_states.pt` from https://huggingface.co/datasets/3DTopia/Hi3DEval/tree/main/object_scoring/ckpts, and place at `object_scoring/ckpts/mp_rank_00_model_states.pt`

- Pretrained Checkpoints

  Download `trained` folder from  https://huggingface.co/datasets/3DTopia/Hi3DEval/tree/main/object_scoring/trained, and place at `object_scoring/trained`

## 4. Run Inference Script
Change directory into the `infer` folder and run the model inference with:

```
bash run.sh
```