import torch
import torch.nn as nn
from typing import Union, Tuple


class CPConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            rank: int,
            bias: bool = True,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            init: str = 'zero',  # 'zero', 'svd', 'random',
            *args, **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.init = init

        if self.init.lower() == 'zero':
            last = torch.zeros(self.out_channels, self.rank)
            first = torch.ones(self.in_channels, self.rank)
            vertical = torch.ones(self.kernel_size[0], self.rank)
            horizontal = torch.ones(self.kernel_size[1], self.rank)
        elif self.init.lower() == 'svd':
            raise NotImplementedError
        elif self.init.lower() == 'random':
            last = torch.rand(self.out_channels, self.rank)
            first = torch.rand(self.in_channels, self.rank)
            vertical = torch.rand(self.kernel_size[0], self.rank)
            horizontal = torch.rand(self.kernel_size[1], self.rank)
        else:
            raise ValueError(f"init should be one of ['zero', 'svd', 'random'], received {self.init}")

        self.factors = nn.ParameterList([
            nn.Parameter(last),      # last
            nn.Parameter(first),        # first
            nn.Parameter(vertical),     # vertical
            nn.Parameter(horizontal)      # horizontal
        ])

        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=bias) if bias else None

    def from_module(self, conv2d):
        raise NotImplementedError

    def reconstruct(self):
        factors = [f.data for f in self.factors]
        w = torch.einsum('ir,jr,kr,lr->ijkl', *factors)
        return w

    def forward(self, x):
        y = cp_conv_2d(x, self.factors, stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.bias is not None:
            y += self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return y


def cp_conv_2d(x, factors, stride, padding, dilation):
    # https://github.com/ruihangdu/Decompose-CNN/tree/master
    last, first, vertical, horizontal = factors
    rank = last.shape[1]

    pointwise_s_to_r_layer = lambda x, w: nn.functional.conv2d(
        x, w, bias=None, stride=1, padding=0, dilation=1
    )
    depthwise_r_to_r_layer = lambda x, w: nn.functional.conv2d(
        x, w, bias=None, stride=stride, padding=padding, dilation=dilation, groups=rank
    )
    pointwise_r_to_t_layer = lambda x, w: nn.functional.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1)

    sr = first.t().unsqueeze(-1).unsqueeze(-1)
    rt = last.unsqueeze(-1).unsqueeze(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(
        0, i, 1) for i in range(rank)]).unsqueeze(1)

    # More efficient in-place version
    # sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    # rt = last.unsqueeze_(-1).unsqueeze_(-1)
    # rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(
    #     0, i, 1) for i in range(rank)]).unsqueeze_(1)

    # Compute outputs
    x = pointwise_s_to_r_layer(x, sr)
    x = depthwise_r_to_r_layer(x, rr)
    x = pointwise_r_to_t_layer(x, rt)
    return x


def cp_decomposition(w, rank):
    factors = parafac(w, rank=rank, init='random')
    return factors
