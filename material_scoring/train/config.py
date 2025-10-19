# ========================= data ==========================
available_corpus = dict(
    stage1 = dict(
        anno_path="./data/train_stage1_example.json",       # training stage 1 data json file
        data_root="./data",                                 # root path for videos in json file
        media_type="video",
    ),
    stage2 = dict(
        anno_path="./data/train_stage2_example.json",       # training stage 2 data json file
        data_root="./data",
        media_type="video",
    ),
    test = dict(
        anno_path="./data/test_example.json",               # test data json file
        data_root="./data",
        media_type="video",
    )
)
train_stage=2
train_file = available_corpus[f"stage{train_stage}"]
test_file = available_corpus[f"test"]
num_workers = 0

stop_key = None

# ========================= input ==========================
num_frames = 16
num_frames_test = 16
batch_size = 8
batch_size_test = 8
max_txt_l = 32

inputs = dict(
    image_res=448,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    model_cls="InternVideo2_CLIP",
    vision_encoder=dict(
        name="InternViT",
        config="configs/encoder",
        ckpt="/path/to/stage1/vision_encoder.pt",
    ),
    clip_encoder=dict(
        name='ViT-H-14',
        ckpt="/path/to/open_clip_model.safetensors"
    ),
    qformer=dict(
        config="configs/head",
        ckpt="/path/to/stage2/head.pt"
    ),
    tokenizer='ViT-H-14',
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_clip=True,
)

criterion = dict(
    loss_weight=dict(
        loss_format=0.2,
        loss_prompt=0.8,
        loss_score=1.0
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    lr=4e-4,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=2, min_lr_multi=0.01, warmup_epochs=0.6)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = "."  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
auto_resume = False
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=True,
    stage=1,
)