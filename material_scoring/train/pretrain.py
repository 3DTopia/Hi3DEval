import datetime
import logging
import time
from os.path import join
import json
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader_rs, MetaLoader, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from utils.retrieval_utils import evaluation_wrapper
from utils.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
import os
logger = logging.getLogger(__name__)


def train(
    model,
    train_loaders,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type,
    skip_num=0
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    loss_names = [k for k, v in config.criterion.loss_weight.items() if v != 0]

    media_types = get_media_types(train_loaders)

    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(
                f"{m}-{name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )
    # for _ in range(config.batch_size):
    #     metric_logger.add_meter(
    #         f"idx{_}", SmoothedValue(window=100, fmt="{value:.1f}")
    #     )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)
    
    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for _, (media_type, (image, text, idx, prompt, is_text_prompt, score)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text).to(device)
        score = score.unsqueeze(-1) if len(score.shape) == 1 else score
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            loss_dict = model(image, text_input, idx=idx, prompt=prompt, is_text_prompt=is_text_prompt, score=score)
            loss = sum([loss_dict[k]*v for k, v in config.criterion.loss_weight.items() if loss_dict[k] is not None])

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(loss)
            model.step()
        else: 
            if not config.use_half_precision or config.get('use_bf16', True):
                optimizer.zero_grad()
                loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()
                scheduler.step()
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if config.optimizer.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item() if value is not None else 0
            metric_logger.update(**{f"{media_type}-{name}": value})
        # for _ in range(config.batch_size):
        #     metric_logger.update(**{f"idx{_}": score[_]})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix=config.wandb.prefix)

        global_step += 1

        if config.debug and global_step % 20 == 0:
            logger.info("debug mode, break training loop")
            break

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

        if config.get('save_iter', 0) and global_step % config.save_iter == 0:
            if hasattr(config, "deepspeed") and config.deepspeed.enable:
                tag = f"ckpt_iter{global_step:02d}.pth"
                model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)
            elif is_main_process():
                state_dict = model_without_ddp.state_dict()
                param_grad_dict = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                for k in list(state_dict.keys()):
                    if k in param_grad_dict.keys() and not param_grad_dict[k]:
                        # delete parameters that do not require gradient
                        logger.info(f"Not saving {k}")
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(save_obj, join(config.output_dir, f"ckpt_iter{global_step:02d}.pth"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def infer(model, test_loaders, tokenizer, device, config, data_type, epoch):
    model.eval()
    metrics = ['Detail']#,'Color','Light','Material']
    eval_res = {k:[] for k in metrics}
    
    media_types = get_media_types(test_loaders)
    test_loader = MetaLoader(name2loader=dict(list(zip(media_types, test_loaders))))

    metric_logger = MetricLogger(delimiter="  ")
    for metric in metrics:
        metric_logger.add_meter(metric, SmoothedValue(window=1, fmt="{value:1.1f}"))
    header = f"Test MAE: "
    iterator = metric_logger.log_every(test_loader, config.log_freq, header)
    
    start_time = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            for _, (_, (image, text, idx, prompt, is_text_prompt, score)) in enumerate(iterator):
                image = image.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                text_input = tokenizer(text).to(device)

                predicts = model.generate(image, text_input, idx=idx, prompt=prompt, is_text_prompt=is_text_prompt)
                score = score.unsqueeze(-1) if len(score.shape) == 1 else score
                for i, metric in enumerate(metrics):
                    metric_logger.update(**{f"{metric}": torch.abs(predicts[:,i]-score[:,i].to(device)).mean()})
                    for index in range(len(idx)):
                        eval_res[metric].append({
                            "index": int(idx[index].detach().cpu().numpy()), 
                            "Predict": float(predicts[index][i].detach().cpu().float().numpy()), 
                            "GT": float(score[index][i].detach().cpu().float().numpy()),
                        })
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Inference time {total_time_str}")
    logger.info(f"Logs saved at {config.output_dir}")

    plt.figure(figsize=(8, 8*len(metrics)))
    for i, metric in enumerate(metrics):
        a2 = [item['GT'] for item in eval_res[metric]]
        b2 = [item['Predict'] for item in eval_res[metric]] 
        plt.subplot(len(metrics),1,i+1)
        plt.scatter(a2,b2,color='blue', marker='o')
        plt.plot(range(5),color='red',linestyle='--')
        plt.xlabel('GT')
        plt.ylabel('Predict')
        plt.title(metric)
    os.makedirs(f"{config.output_dir}/figs", exist_ok=True)
    plt.savefig(f"{config.output_dir}/figs/epoch_{epoch}_test.png")
    return eval_res

def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if config.distributed:
        batch_size = [config.inputs.batch_size[k] for k in media_types] # batch_size for each GPU
        samplers = create_stateful_sampler(train_datasets, batch_size)
    else:
        raise NotImplementedError
    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[None] * len(media_types),
    )
    # test datasets, a mapping from dataset name to data loader
    test_datasets = create_dataset(f"{mode}_eval", config)
    media_types = get_media_types(test_datasets)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test[d] for d in media_types],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    return train_loaders, test_loaders, media_types


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    # setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_loaders, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=is_pretrain,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    logger.info(f"Epoch: {start_epoch}")
    start_time = time.time()
    start_step = start_epoch * num_steps_per_epoch
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type,
                skip_num = global_step - start_step
            )

        # save checkpoint befor evaluation
        # only save those with gradient
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if config.get("save_latest", False):
                tag = "ckpt_latest.pth"
            else:
                tag = f"ckpt_{epoch:02d}.pth"
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)
            
        elif is_main_process():
            state_dict = model_without_ddp.state_dict()
            param_grad_dict = {
                k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            }
            for k in list(state_dict.keys()):
                if k in param_grad_dict.keys() and not param_grad_dict[k]:
                    # delete parameters that do not require gradient
                    logger.info(f"Not saving {k}")
                    del state_dict[k]

            save_obj = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        # evaluation
        eval_res = {}
        if config.test: 
            eval_res = infer(model_without_ddp, test_loaders, tokenizer, device, config, data_type=data_type, epoch=epoch)

        # save the best checkpoint
        if is_main_process():
            # log to wandb
            if config.wandb.enable:
                log_dict_to_wandb({k:str(v) for k,v in eval_res.items()}, step=global_step, prefix='test/')

            with open(join(config.output_dir, "eval_res_latest.json"), 'w') as f:
                json.dump(eval_res, f, indent=4)

        if config.evaluate:
            break
        
        start_step = global_step

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
