from typing import Union

import torch.nn as nn

from peftnet.peft_module.lora_linear import LoraLinear
from peftnet._peftnet import PEFTNet


class LoraNet(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            ignore_list: list = None,
            factorize_list: list = None,
            *args, **kwargs
    ):
        """ LoRa PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace

        """
        super().__init__(
            model,
            ignore_list,
            factorize_list,
            replacement_module=LoraLinear,
            replacement_kwargs=dict(rank=rank)
        )
