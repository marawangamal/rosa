import torch
import torch.nn as nn

from tuckerconv2d import TuckerConv2d


def main():

    # Parameters
    in_channels = 16
    out_channels = 16
    kernel_size = 3
    rank = 8

    tucker_factors = [
        torch.randn(out_channels, rank),
        torch.randn(rank, rank, kernel_size, kernel_size),
        torch.randn(in_channels, rank),
    ]

    w_true = torch.einsum("os,srhw,ir->oihw", *tucker_factors)

    # Initialize Conv2d using CP factors
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    conv2d.weight.data = w_true  # [out_channels, in_channels, kernel_size, kernel_size]

    # Initialize CPConv2d using CP factors
    tuckerconv2d = TuckerConv2d(in_channels, out_channels, kernel_size, rank=[rank, rank], bias=False)

    # Initialize TuckerConv2d using Tucker factors
    last, core, first = tucker_factors
    tuckerconv2d.first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    tuckerconv2d.last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    tuckerconv2d.core_layer.weight.data = core

    # Compare outputs
    x = torch.randn(1, in_channels, 10, 10)
    y_conv2d = conv2d(x)
    y_tuckerconv2d = tuckerconv2d(x)

    # Check if outputs are equal [pass/fail]
    diff = torch.norm(y_conv2d - y_tuckerconv2d)
    pass_fail = "PASS" if torch.allclose(y_conv2d, y_tuckerconv2d, atol=1e-3) else "FAIL"
    print(f"[{pass_fail}] Projections are equal (diff={diff})")


if __name__ == '__main__':
    main()