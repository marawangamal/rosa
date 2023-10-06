from typing import Union
import logging

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
            replacement_module: nn.Module = None,
            fan_in_fan_out_map: dict = None,
            adapt_method: str = 'ab',  # 'a', 'b', 'ab'
            init_method: str = "zero",
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ LoRa PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            replacement_module: replacement module
            fan_in_fan_out_map: map from module type to fan_in_fan_out flag
            adapt_method: adaptation method [`a`, `b`, `ab`]
            debug: whether to use debug mode

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """

        # Defaults to factorize only linear layers
        factorize_list = ['Linear, Conv1D'] if factorize_list is None else factorize_list
        replacement_module = LoraLinear if replacement_module is None else replacement_module
        fan_in_fan_out_map = {
            "Conv1D": True, "Linear": False
        } if fan_in_fan_out_map is None else fan_in_fan_out_map

        # Validation for Linear layers
        if any([k in factorize_list for k in ["Linear", "Conv1D"]]):
            assert (factorize_list is None and fan_in_fan_out_map is None) or \
                   (factorize_list is not None and fan_in_fan_out_map is not None), \
                "`factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously"

        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=replacement_module,

            # Parameters passed to `replacement_module` from_module method
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                adapt_method=adapt_method,
                init_method=init_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode,
                fan_in_fan_out_map=fan_in_fan_out_map
            ),
        )
        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=replacement_module,

            # Parameters passed to `replacement_module` from_module method
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                adapt_method=adapt_method,
                init_method=init_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode,
                fan_in_fan_out_map=fan_in_fan_out_map
            ),
        )
        logging.info(f'Initialized LoRA model with params:')
        logging.info(
            f'rank: {rank}, '
            f'bias_requires_grad: {bias_requires_grad} '
            f'debug: {debug}, '
            f'fast_mode: {fast_mode}'
        )
