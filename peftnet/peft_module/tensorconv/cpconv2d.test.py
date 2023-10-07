import torch
import torch.nn as nn

from cpconv2d import CPConv2d


def main():

    # Parameters
    in_channels = 16
    out_channels = 16
    kernel_size = 3
    rank = 1

    cp_factors = [
        torch.randn(out_channels, rank),
        torch.randn(in_channels, rank),
        torch.randn(kernel_size, rank),
        torch.randn(kernel_size, rank)
    ]

    w_true = torch.einsum("ir,jr,kr,lr->ijkl", *cp_factors)

    # Initialize Conv2d using CP factors
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    conv2d.weight.data = w_true  # [out_channels, in_channels, kernel_size, kernel_size]

    # Initialize CPConv2d using CP factors
    cpconv2d = CPConv2d(in_channels, out_channels, kernel_size, bias=False)

    for i, factor in enumerate(cp_factors):
        cpconv2d.factors[i].data = factor.data

    # Compare outputs
    x = torch.randn(1, in_channels, 10, 10)
    y_conv2d = conv2d(x)
    y_cpconv2d = cpconv2d(x)

    # Check if outputs are equal [pass/fail]
    diff = torch.norm(y_conv2d - y_cpconv2d)
    pass_fail = "PASS" if torch.allclose(y_conv2d, y_cpconv2d, atol=1e-5) else "FAIL"
    print(f"[{pass_fail}] Projections are equal (diff={diff})")


if __name__ == '__main__':
    main()