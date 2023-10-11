from typing import Union, Tuple

import torch
import torch.nn as nn

# import tensorly as tl
from tensorly.decomposition import partial_tucker

# from vbmf import EVBMF


class TuckerConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            rank: Union[int, Tuple[int, int]],
            bias: bool = True,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()

        if isinstance(rank, int):
            self.rank = (rank, rank)
        elif isinstance(rank, (tuple, list)) and len(rank) == 2:
            self.rank = rank
        else:
            raise ValueError("rank should be int or tuple/list of length 2")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # A pointwise convolution that reduces the channels from S to R3
        self.first_layer = torch.nn.Conv2d(in_channels=self.in_channels,
                                           out_channels=self.rank[0], kernel_size=1,
                                           stride=1, padding=0, dilation=self.dilation, bias=False)

        # A regular 2D convolution layer with R3 input channels
        # and R3 output channels
        self.core_layer = torch.nn.Conv2d(in_channels=self.rank[0],
                                          out_channels=self.rank[1], kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, dilation=self.dilation,
                                          bias=False)

        # A pointwise convolution that increases the channels from R4 to T
        self.last_layer = torch.nn.Conv2d(in_channels=self.rank[1],
                                          out_channels=self.out_channels, kernel_size=1, stride=1,
                                          padding=0, dilation=self.dilation, bias=bias)

    def init_weights(self, first: torch.Tensor, core: torch.Tensor, last: torch.Tensor):
        """Initialize weights of the Tucker convolution.

        Args:
            first: [in_channels, ranks[0]]
            core: [ranks[0], ranks[1]]
            last: [ranks[1], out_channels]

        """
        self.first_layer.weight.data = \
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        self.last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        self.core_layer.weight.data = core

    @classmethod
    def estimate_ranks(cls, layer):
        # weights = layer.weight.data
        # unfold_0 = tl.base.unfold(weights, 0)
        # unfold_1 = tl.base.unfold(weights, 1)
        # _, diag_0, _, _ = EVBMF(unfold_0)
        # _, diag_1, _, _ = EVBMF(unfold_1)
        # ranks = [diag_0.shape[0], diag_1.shape[1]]
        # return ranks
        raise NotImplementedError

    @classmethod
    def from_module(cls, layer):
        ranks = cls.estimate_ranks(layer)
        print(layer, "VBMF Estimated ranks", ranks)
        # [d, r]
        core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], ranks=ranks, init='svd')

        obj = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            rank=ranks,
            bias=layer.bias is not None,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation
        )

        cls.last_layer.bias.data = layer.bias.data
        cls.first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        cls.last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        cls.core_layer.weight.data = core

        return obj

    def forward(self, x):
        x = self.first_layer(x)
        x = self.core_layer(x)
        x = self.last_layer(x)
        return x
