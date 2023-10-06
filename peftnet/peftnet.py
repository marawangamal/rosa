import logging

import torch.nn as nn

from peftnet._peftnet import PEFTNet as _PEFTNet
from peftnet.peft_module.rosalinear import RosaLinear
from peftnet.peft_module.loralinear import LoraLinear
from peftnet.peft_module.ia3linear import IA3Linear
from peftnet.peft_module.loraconv2d import LoraConv2d


class PEFTNet(_PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            peft_method: str = "lora",
            factorize_list: list = None,
            ignore_list: list = None,
            *args, **kwargs
    ):
        """ PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            peft_method: {'rosa', 'lora', 'ia3', 'loraconv2d'}
            factorize_list: names of modules types to replace {'Linear', 'Conv2d'}. Default: ['Linear']
            ignore_list: names of layers to ignore (e.g. ['bert.embeddings'])

        Notes:
            - only modules types in `factorize_list` will be factorized
            - kwargs are passed to replacement module `from_module` method

        """

        replacement_module = {
            "rosa": RosaLinear,
            "lora": LoraLinear,
            "ia3": IA3Linear,
            "loraconv2d": LoraConv2d
        }[peft_method]

        if factorize_list is None:
            factorize_list = ['Linear', 'Conv1D']

        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=replacement_module,
            replacement_kwargs=kwargs
        )

        logging.info(f"PEFTNet: {self.factorize_list} -> {replacement_module}")
