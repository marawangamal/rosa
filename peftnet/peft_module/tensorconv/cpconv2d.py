import torch
import torch.nn as nn


class CPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, rank=1, stride=1, padding=0, dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.factors = nn.ParameterList([
            nn.Parameter(torch.zeros(out_channels, rank)),      # last
            nn.Parameter(torch.ones(in_channels, rank)),        # first
            nn.Parameter(torch.ones(kernel_size[0], rank)),     # vertical
            nn.Parameter(torch.ones(kernel_size[1], rank))      # horizontal
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
