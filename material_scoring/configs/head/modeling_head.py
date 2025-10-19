import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel

class ScoreHead(PreTrainedModel):

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        self.config = config
        self.conv = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Conv3d(in_channels=config.hidden_size, out_channels=config.middle_size, kernel_size=(1,1,1)),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Conv3d(in_channels=config.middle_size, out_channels=1, kernel_size=(1, 1, 1)),
        )

        self.vision_proj = nn.Linear(config.proj_size, 1)

    def forward(self, image_embeds, prompt_embeds=None):
        
        B, T, D = image_embeds.shape
        
        assert T == self.config.proj_size, f'Incorrect Frame Token Length {T}!'

        if prompt_embeds is not None:
            image_embeds -= prompt_embeds.expand(-1,T,-1)
        
        S = int( (T // 16) ** 0.5 )
        output = self.conv(image_embeds.reshape(B, D, -1, S, S))
        output = self.vision_proj(output.reshape(B, -1))

        return output

