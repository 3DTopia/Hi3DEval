"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""
import os
import random
import io
import av
import cv2
import decord
import imageio
import torch
import numpy as np
import math
decord.bridge.set_bridge("torch")

import logging
logger = logging.getLogger(__name__)

import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
        local_num_frames = 4        
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)


def get_frame_indices_by_fps():
    pass


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None, max_num_frames=-1):
    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
         max_num_frames=-1, client=None, trimmed30=False,
    ):
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, None


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, trimmed30=False
    ):
    num_threads = 1 if video_path.endswith('.webm') else 0 # make ssv2 happy
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        # print(f"\033[1;31;40m {video_path} ok: {video_bytes is None} \033[0m")
        if video_bytes is None:
            logger.warning(f"Failed to load {video_path}")
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=num_threads) 
    else:
        video_reader = VideoReader(video_path, num_threads=num_threads)
    vlen = len(video_reader)
 
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # only use top 30 seconds
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )

    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, trimmed30=False
    ):
    img_list=[]
    if "s3://" in video_path:
        for path in client.list(video_path):
            if path.startswith('img'):
                img_list.append(path)
    else:
        for path in os.listdir(video_path):
            if path.startswith('img'):
                img_list.append(path)

    vlen = len(img_list)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )

    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    frames = torch.tensor(np.array(imgs), dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, None


VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'img': read_frames_img,
}
