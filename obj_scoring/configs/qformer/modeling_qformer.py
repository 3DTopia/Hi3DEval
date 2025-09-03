"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from .configuration_qformer import BertConfig
from lavis.models.blip2_models.Qformer import BertModel
from transformers.modeling_utils import PreTrainedModel

class Blip2Qformer(PreTrainedModel):

    def __init__(
        self,
        config: BertConfig,
    ):
        super().__init__(config)
        self.config = config
        self.conv = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(in_channels=config.hidden_size, out_channels=128, kernel_size=(1,1,1)),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Conv3d(in_channels=128, out_channels=1, kernel_size=(1, 1, 1)),
        )

        self.vision_proj = nn.Linear(256, 1) 
        
        
    def forward(self, image_embeds, prompt_embeds,is_prompt_alignment=False,is_prompt_alignment_cosine_mlp=False):
        B, T, D = image_embeds.shape
        S = int( (T // 16) ** 0.5 )
        
        if is_prompt_alignment_cosine_mlp and is_prompt_alignment:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) #[b,256,1024]
            prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True) #[b,1,1024]
            cosine_sim = torch.bmm(prompt_embeds, image_embeds.transpose(1, 2)) #[b,1,256]
            cosine_sim_flat = cosine_sim.squeeze(1) #[b,256]
            
            output = self.vision_proj(cosine_sim_flat) #[b,1]
            # output = torch.bmm(cosine_sim, self.weight) + self.bias #[b,1]
            # output = cosine_sim_flat @ self.weight.T + self.bias  # [B, 1]
            return output
        elif is_prompt_alignment:
            image_embeds = image_embeds - prompt_embeds
            

        output = self.conv(image_embeds.reshape(B, D, -1, S, S))
        output = self.vision_proj(output.reshape(B, -1))

        return output

