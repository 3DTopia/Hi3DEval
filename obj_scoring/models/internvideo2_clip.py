import logging
import open_clip
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel
from .criterions import VTC_VTM_Loss, Ranking_Loss, get_sim
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class InternVideo2_CLIP(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        # create modules.
        if tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer(config.model.tokenizer)
        self.vision_encoder = self.build_vision_encoder()
        self.clip_encoder = self.build_clip_encoder()
        self.qformer = self.build_qformer()
        self.load_checkpoint()

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min


        # freeze model
        if self.config.train_stage == 1 and not self.config.train_full:
            for name, p in self.qformer.named_parameters():
                logger.info(f"Freeze {name}")
                p.requires_grad = False        
        if self.config.train_stage == 2:
            for name, p in self.vision_encoder.named_parameters():
                if name.startswith('mlp') and self.config.train_full:
                    continue
                logger.info(f"Freeze {name}")
                p.requires_grad = False

        if self.config.model.freeze_clip:
            for name, p in self.clip_encoder.named_parameters():
                logger.info(f"Freeze {name}")
                p.requires_grad = False

        print('Total Params:', sum([p.numel() for _, p in self.qformer.named_parameters()])/1024/1024, 'MB')
        # criterions
        self.format_loss = VTC_VTM_Loss(False)
        self.prompt_loss = VTC_VTM_Loss(False)
        self.score_loss = nn.MSELoss() if config.criterion.loss_type=='mse' \
                    else nn.CrossEntropyLoss() if config.criterion.loss_type=='ce' \
                    else nn.SmoothL1Loss()
        self.rank_loss = Ranking_Loss()
        # self.plcc_loss = 

    def no_weight_decay(self):
        ret = {"temp"}
        # ret.update(
        #     {"vision_encoder." + k for k, _ in self.vision_encoder.named_parameters()}
        # )
        # no weight decay for LLM if training
        # ret.update(
        #     {"text_encoder." + k for k, _ in self.clip_encoder.named_parameters()}
        # )

        return ret
    
    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def forward(self, image, text, idx, prompt, is_text_prompt, score, visualize=False,is_prompt_alignment=False):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        image.requires_grad = True

        B, device = image.shape[0], image.device

        if self.config.train_stage == 2:
            image = image.flatten(0, 1)
            
        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        
        prompt_embeds = torch.zeros([B,vision_embeds.shape[-1]], dtype=vision_embeds.dtype).to(device)
        text_prompts = [p for p, v in zip(prompt, is_text_prompt) if v]
        image_prompts = [p for p, v in zip(prompt, is_text_prompt) if not v]
        is_image_prompt = torch.where(~is_text_prompt)
        is_text_prompt = torch.where(is_text_prompt)
        if text_prompts:
            prompt_embeds[is_text_prompt] = self.encode_text(self.tokenizer(text_prompts).to(device))
        if image_prompts:
            prompt_embeds[is_image_prompt] = self.encode_image(image_prompts).to(device)

        if self.config.train_stage == 1:
            text_embeds = self.encode_text(text)
            loss_format = self.format_loss.vtc_loss(
                vision_embeds, text_embeds, idx, self.temp, all_gather=True
            )
            loss_prompt = self.prompt_loss.vtc_loss(
                vision_embeds, prompt_embeds, idx, self.temp, all_gather=True
            )
            return dict(
                loss_format=loss_format,
                loss_prompt=loss_prompt,
                loss_score=None,
                loss_rank=None,
            )
        
        elif self.config.train_stage == 2:
            
            if hasattr(self.config, 'is_prompt_alignment_cosine_mlp'):
                is_prompt_alignment_cosine_mlp = self.config.is_prompt_alignment_cosine_mlp
            else:
                is_prompt_alignment_cosine_mlp = False

            output = self.qformer(
                    vision_embeds.reshape(B, -1, vision_embeds.shape[-1]),
                    prompt_embeds.unsqueeze(dim=1),
                    is_prompt_alignment=is_prompt_alignment,
                    is_prompt_alignment_cosine_mlp=is_prompt_alignment_cosine_mlp
                )
            

            if visualize and torch.distributed.get_rank() == 0:
                print('*****************************Comparing score: predicted:',output[-1],'gt:',score[-1],'*****************************')
                print('Prompt alignment:',is_prompt_alignment)
                print('is_prompt_alignment_cosine_mlp:',is_prompt_alignment_cosine_mlp)
            if self.config.criterion.loss_type=='ce':
                score = (5*score.flatten()).to(torch.long)
            
            if self.config.criterion.loss_type=='pair':
                '''
                output difference between two models denoted as p
                score difference between two models denoted as s
                L = ( p * s > 0 ) + abs( p - s )
                '''
                indices = output[1]
                output = output[0]
                scores = torch.zeros(output.shape, dtype=output.dtype).to(output.device)
                for i in range(len(indices)//2):
                    m, n = indices[2*i], indices[2*i+1]
                    scores[i] = score[m] - score[n]
                loss_score = self.score_loss(output, scores)
            elif self.config.criterion.loss_type=='mae':
                weight = ((score - 2)**2).to(output.device)
                loss_score = self.score_loss(weight*output, weight*score.to(output.device))
            else:    
                loss_score = self.score_loss(output, score.to(device))
            loss_rank = self.rank_loss.rk_loss(output, score.to(device))
            return dict(
                loss_format=None,
                loss_prompt=None,
                loss_score=loss_score,
                loss_rank=loss_rank,
            )
        
    def generate(self, image, text, idx, prompt, is_text_prompt,is_prompt_alignment=False):

        B, device = image.shape[0], image.device
        image = image.flatten(0, 1)

        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        
        prompt_embeds = torch.zeros([B,vision_embeds.shape[-1]], dtype=vision_embeds.dtype).to(device)
        text_prompts = [p for p, v in zip(prompt, is_text_prompt) if v]
        image_prompts = [p for p, v in zip(prompt, is_text_prompt) if not v]
        is_image_prompt = torch.where(~is_text_prompt)
        is_text_prompt = torch.where(is_text_prompt)
        if text_prompts:
            prompt_embeds[is_text_prompt] = self.encode_text(self.tokenizer(text_prompts).to(device))
        if image_prompts:
            prompt_embeds[is_image_prompt] = self.encode_image(image_prompts).to(device)
        

        if hasattr(self.config, 'is_prompt_alignment'):
            is_prompt_alignment = self.config.is_prompt_alignment
        else:
            is_prompt_alignment = False
            
        if hasattr(self.config, 'is_prompt_alignment_cosine_mlp'):
            is_prompt_alignment_cosine_mlp = self.config.is_prompt_alignment_cosine_mlp
        else:
            is_prompt_alignment_cosine_mlp = False

        output = self.qformer(
                vision_embeds.reshape(B, -1, vision_embeds.shape[-1]),
                prompt_embeds.unsqueeze(dim=1),
                is_prompt_alignment=is_prompt_alignment,
                is_prompt_alignment_cosine_mlp=is_prompt_alignment_cosine_mlp
            )
        return output

    def sim(self, image, text_input, idx, prompt, is_text_prompt):
        image = image.flatten(0, 1)

        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image).transpose(0,1)
        vision_embeds = F.normalize(vision_embeds, dim=-1)
        
        return (vision_embeds @ vision_embeds.transpose(-2,-1)).mean(dim=0)

    def encode_vision(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,C].

        """
        vision_embeds = self.vision_encoder(image)
        
        return vision_embeds
    
    def encode_image(self, image):
        image = [self.processor(Image.open(i)) for i in image]
        image = torch.stack(image).to(torch.bfloat16).to(self.config.device)
        image_embeds = self.clip_encoder.encode_image(image)
        return image_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,C].

        """
        text_embeds = self.clip_encoder.encode_text(text)
        return text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder_config = AutoConfig.from_pretrained(self.config.model.vision_encoder.config, trust_remote_code=True)
        vision_encoder = AutoModel.from_config(vision_encoder_config, trust_remote_code=True).to(torch.bfloat16)

        return vision_encoder

    def build_clip_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        clip_encoder, _, clip_processor = open_clip.create_model_and_transforms(self.config.model.clip_encoder.name, pretrained=False)
        clip_encoder.eval()
        self.processor = clip_processor
        return clip_encoder

    def build_qformer(self):
        
        
        qformer_config = AutoConfig.from_pretrained(self.config.model.qformer.config, trust_remote_code=True)
        qformer = AutoModel.from_config(qformer_config, trust_remote_code=True).to(torch.bfloat16)

        return qformer

    def load_checkpoint(self):
        new_ckpt = {}

        ### load vision_encoder
        if hasattr(self.config.model.vision_encoder, 'ckpt'):
            vision_ckpt_path = self.config.model.vision_encoder.ckpt
            logger.info(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
            if vision_ckpt_path.endswith('.safetensors'):
                ckpt = load_file(vision_ckpt_path)
                for k, v in ckpt.items():
                    new_ckpt['vision_encoder.'+k] = v
            elif vision_ckpt_path.endswith('.pt'):
                ckpt = torch.load(vision_ckpt_path)
                ckpt = ckpt['module'] if 'module' in ckpt.keys() else ckpt
                for k,v in ckpt.items():
                    if k.startswith('vision_encoder.'):
                        new_ckpt[k] = v
        else:
            logger.info(f"Load vision_encoder checkpoint from scratch")
            for k,v in self.vision_encoder.named_parameters():
                new_ckpt['vision_encoder.'+k] = v

        ### load qformer
        # import ipdb; ipdb.set_trace()
        if self.config.auto_resume2:
            import os
            import glob
            ckpt_list = [ck for ck in glob.glob(os.path.join(self.config.output_dir, '*.pth')) if 'iter' not in ck]
            ckpt_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(ckpt_list) > 0:
                ckpt_path = ckpt_list[-1]+'/mp_rank_00_model_states.pt'
                self.config.model.qformer.ckpt = ckpt_path
            else:
                ckpt_list = [ck for ck in glob.glob(os.path.join(self.config.output_dir, '*.pth')) if 'iter' in ck]
                ckpt_list.sort(key=lambda x: int(x.split('_iter')[-1].split('.')[0]))
                if len(ckpt_list) > 0:
                    ckpt_path = ckpt_list[-1]+'/mp_rank_00_model_states.pt'
                    self.config.model.qformer.ckpt = ckpt_path
                else:
                    logger.info(f"Load qformer checkpoint from scratch")

        if hasattr(self.config.model.qformer, 'ckpt'):
            qformer_ckpt_path = self.config.model.qformer.ckpt
            logger.info(f"Load qformer checkpoint from {qformer_ckpt_path}")
            if qformer_ckpt_path.endswith('.safetensors'):
                ckpt = load_file(qformer_ckpt_path)
                for k, v in ckpt.items():
                    new_ckpt['qformer.'+k] = v
            elif qformer_ckpt_path.endswith('.pt'):
                ckpt = torch.load(qformer_ckpt_path)
                ckpt = ckpt['module'] if 'module' in ckpt.keys() else ckpt
                for k,v in ckpt.items():
                    if k.startswith('qformer.'):
                        new_ckpt[k] = v
                    if k.startswith('vision_encoder.mlp'):
                        new_ckpt[k] = v
        else:
            logger.info(f"Load qformer checkpoint from scratch")
            for k,v in self.qformer.named_parameters():
                new_ckpt['qformer.'+k] = v
        if hasattr(self.qformer, "Fuser"):
            new_ckpt['qformer.Fuser.embeddings.position_ids'] = self.qformer.Fuser.embeddings.position_ids
        if hasattr(self.qformer, "Qformer"):
            new_ckpt['qformer.Qformer.embeddings.position_ids'] = self.qformer.Qformer.embeddings.position_ids

        ### load clip
        assert hasattr(self.config.model.clip_encoder, 'ckpt')
        clip_ckpt_path = self.config.model.clip_encoder.ckpt
        logger.info(f"Load clip checkpoint from {clip_ckpt_path}")
        if clip_ckpt_path.endswith('.safetensors'):
            ckpt = load_file(clip_ckpt_path)
            for k, v in ckpt.items():
                new_ckpt['clip_encoder.'+k] = v

        msg = self.load_state_dict(new_ckpt, strict=True)
        logger.info(msg)
