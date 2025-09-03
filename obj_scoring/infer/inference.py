import sys
sys.path.append('../')
import datetime
import logging
import time
from os.path import join
import copy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
import json
import os
from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from utils.retrieval_utils import evaluation_wrapper
from utils.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    
    test_datasets = create_dataset(f"{mode}_eval", config)
    media_types = ['video']#get_media_types(test_datasets)

    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test[d] for d in media_types],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    return test_loaders


def main(config):
    # import ipdb;ipdb.set_trace()
    with open(config.available_corpus['infer']['anno_path'], 'r') as f:
        anno_data = json.load(f)
    metrics = ['Test']#,['Geo_plausibility','Geo_detail','Texture_quality','Texture_coherence','Prompt_alignment']
    
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    
    test_loaders = setup_dataloaders(config, mode=config.mode)
    media_types = ['video']#get_media_types(test_loaders)
    test_loader = MetaLoader(name2loader=dict(list(zip(media_types, test_loaders))))

    metric_logger = MetricLogger(delimiter="  ")
    for metric in metrics:
        metric_logger.add_meter(metric, SmoothedValue(window=1, fmt="{value:1.1f}"))
    header = f"Test sample: "
    iterator = metric_logger.log_every(test_loader, config.log_freq, header)
    
    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    logger.info("Creating model")
    config = copy.deepcopy(config)
    model = model_cls(config=config, is_pretrain=False)
    model = model.to(torch.device(config.device))
    if config.get('use_bf16', True):
        logger.info("Change to bfloat16 for model")
        model = model.to(torch.bfloat16)
        data_type = torch.bfloat16
    else:
        logger.info("Change to float16 for model")
        model = model.half()
        data_type = torch.float16

    tokenizer = model.tokenizer
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    start_time = time.time()
    losses = {}
    # for key in metrics:
    #     losses[key] = []
    prediction_and_score = {}
    cnt = 0
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            for a, (b, (image, text, idx, prompt, is_text_prompt, score)) in enumerate(iterator):
                # print(image.shape)
                image = image.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                text_input = tokenizer(text).to(device)
                # import ipdb; ipdb.set_trace()
                predicts = model.generate(image, text_input, idx=idx, prompt=prompt, is_text_prompt=is_text_prompt)
                # import ipdb;ipdb.set_trace()
                for i in range(len(idx)):
                    losses[cnt] = (predicts[i]-score[i][:1].to(device)).cpu().tolist()
                    prediction_and_score[cnt] = [predicts[i].cpu().tolist(), score[i][:1].cpu().tolist()]
                    cnt += 1
                # import ipdb; ipdb.set_trace()
                print ('comparison: ',predicts[0],score[0][0])
                for i, metric in enumerate(metrics):
                    metric_logger.update(**{f"{metric}":predicts[0]-score[0][i].to(device)})
    
    
    align_dict = {}
    # import ipdb;ipdb.set_trace()
    for i in range(len(anno_data)):
        video_path = anno_data[i]["video"][0]
        pred_key = str(i)
        align_dict[video_path] = prediction_and_score[i]
    with open(os.path.join(config.output_dir, 'prediction_and_score_align.json'), 'w') as f:
        json.dump(align_dict, f, indent=2, ensure_ascii=False)
    with open(os.path.join(config.output_dir, 'prediction_and_score.json'), 'w') as f:
        json.dump(prediction_and_score, f, indent=4)
    with open(os.path.join(config.output_dir, 'losses.json'), 'w') as f:
        json.dump(losses, f, indent=4)
    
    metric_loss = {}
    for i,metric in enumerate(metrics):
        metric_loss[metric] = []
        for key in losses.keys():
            metric_loss[metric] += [losses[key][i]]
        metric_loss[metric] = sum(metric_loss[metric])/len(metric_loss[metric])           

    with open(os.path.join(config.output_dir, 'metric_loss.json'), 'w') as f:
        json.dump(metric_loss, f, indent=4)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Inference time {total_time_str}")
    logger.info(f"Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
