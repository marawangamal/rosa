from typing import Union

import torch.nn as nn

from peftnet.peft_module.rosalinear import RosaLinear
from peftnet._peftnet import PEFTNet


class RosaNet(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            factorize_mode: str = 'random',
            *args, **kwargs
    ):
        """ ROSA PEFT model for efficient adaptation of linear layers

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
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            factorize_mode=factorize_mode,
            replacement_module=RosaLinear,
            replacement_kwargs=dict(rank=rank, use_scale=use_scale),
        )

        # ROSA Model initializes low rank matrices with values obtained from SVD
        self.factorize()
