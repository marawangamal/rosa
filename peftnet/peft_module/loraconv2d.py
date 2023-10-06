from typing import Union
from ._peftconv2d import PeftConv2d


class LoraConv2d(PeftConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False,
            *args, **kwargs
    ):
        """ LORA linear layer with trainable and fixed parameters in parallel.

        Args:
            in_channels: number of input features
            out_channels: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            rank=rank,
            bias=bias,
            *args, **kwargs
        )
