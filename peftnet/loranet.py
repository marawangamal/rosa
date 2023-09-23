from typing import Union

import torch.nn as nn

from peftnet.peft_module.loralinear import LoraLinear
from peftnet._peftnet import PEFTNet


class LoraNet(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            init_method: str = "zero",
            *args, **kwargs
    ):
        """ LoRa PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """
        super().__init__(
            model,
            ignore_list,
            factorize_list,
            replacement_module=LoraLinear,
            replacement_kwargs=dict(rank=rank, use_scale=use_scale, init_method=init_method),
        )
