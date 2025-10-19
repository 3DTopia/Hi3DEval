from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ScoreHeadConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=1024,
        middle_size=128,
        proj_size=256,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.middle_size = middle_size
        self.proj_size = proj_size
        self.dropout = dropout

