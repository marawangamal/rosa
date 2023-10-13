import logging

import torch.nn as nn

from peftnet._peftnet import PEFTNet as _PEFTNet
from peftnet.peft_module.rosalinear import RosaLinear
from peftnet.peft_module.loralinear import LoraLinear
from peftnet.peft_module.ia3linear import IA3Linear
from peftnet.peft_module.loraconv2d import LoraConv2d

# fan_in_fan_out_map = {
#     "Conv1D": True,
#     "Linear": False,
# }

default_peft_map = {
    "rosa": {
        "Linear": RosaLinear,
    },
    "lora": {
        "Linear": LoraLinear,
    },
    "ia3": {
        "Linear": IA3Linear,
    },
    "loraconv2d": {
        "Conv2d": LoraConv2d
    },
    "lorafull": {
        "Linear": LoraLinear,
        "Conv2d": LoraConv2d
    }
}

default_peft_map_kwargs = {
    "rosa": {
        "Linear": {
            "fan_in_fan_out": True
        }
    },
    "lora": {
        "Linear": {
            "fan_in_fan_out": False
        }
    },
    "ia3": {
        "Linear": {
            "fan_in_fan_out": False
        }
    },
    "loraconv2d": {
        "Conv2d": {}
    },
    "lorafull": {
        "Linear": {
            "fan_in_fan_out": False
        },
        "Conv2d": {}
    },
}


class PEFTNet(_PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            method: str = "lora",
            peft_map: dict = None,
            peft_kwargs: dict = None,
            ignore_regex: str = None,
            ignore_list: list = None,
            *args, **kwargs
    ):
        """ PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            method: {'rosa', 'lora', 'ia3', 'loraconv2d'}
            target_modules: names of modules types to peft {'Linear', 'Conv2d'}. Default: ['Linear']
            ignore_regex: regex to match layers to ignore (e.g. ['bert.embeddings'])
            ignore_list: list of layers to ignore (e.g. [bert.embeddings])

        Notes:
            - only modules types in `factorize_list` will be factorized
            - kwargs are passed to replacement module `from_module` method

        Warning:
            If `model` has both Linear and Conv1D layers, this will cause an error

        """

        peft_map = default_peft_map[method] if peft_map is None else peft_map
        peft_kwargs = default_peft_map_kwargs[method] if peft_kwargs is None else peft_kwargs

        # Add **kwargs to peft_kwargs to all keys in peft_kwargs
        for k, v in peft_kwargs.items():
            peft_kwargs[k] = {**v, **kwargs}

        if ignore_regex == "" or not ignore_regex:
            ignore_regex = None

        super().__init__(
            model,
            peft_map=peft_map,
            peft_kwargs=peft_kwargs,
            ignore_regex=ignore_regex,
            ignore_list=ignore_list if ignore_list else [],
        )
