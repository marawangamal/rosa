from typing import Union

import torch
import torch.nn as nn


class IA3Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            mode: str = 'in',
            bias: bool = False
    ):
        """ IA3 linear layer with multiplicative trainable parameters.

        Args:
            in_features: number of input features
            out_features: number of output features
            mode: which side to apply multiplicative parameters [in, out, in_out]
            bias: whether to include bias

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.mode = mode
        if self.mode == 'in':
            self.d = nn.Parameter(torch.zeros(in_features))
        elif self.mode == 'out':
            self.d = nn.Parameter(torch.zeros(out_features))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.register_buffer('merged', torch.tensor([False]))

    def initialize_weights(self, d_init: torch.Tensor = None, w_init: torch.Tensor = None, bias_init: torch.Tensor = None):
        """Initialize weights and biases with given tensors."""
        self.d.data = d_init if d_init is not None else self.d.data
        self.w.data = w_init if w_init is not None else self.w.data

        if bias_init is not None:
            assert self.bias.data.shape == bias_init.shape, "Bias shape mismatch"
            self.bias.data = bias_init

    @classmethod
    def from_module(cls, linear_layer: nn.Module, rank=1.0, fan_in_fan_out=True) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module"""

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), rank=rank, bias=bias is not None
        )
        d = torch.ones(obj.in_features if obj.mode == 'in' else obj.out_features, device=w.device)
        obj.initialize_weights(w_init=w, d_init=d, bias_init=bias)
        return obj

    def merge(self):
        """Merge `a` and `b` with `w` and make `w` trainable"""
        if not self.merged.item():
            # Merge w [out_f, in_f] with d [in_f] or [out_f]
            self.w.data = self.w.data * self.d.data.reshape(-1, 1) if self.mode == 'in' else self.w.data * self.d.data

            # Make d fixed and w trainable
            self.d.requires_grad = False
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = torch.tensor([True])
        return self

    def factorize(self, **kwargs):
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        return (f'{cls}('
                f'rank={self.rank}, '
                f'd={self.d.shape, self.d.requires_grad}, '
                f'w={self.w.shape, self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
                f')')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        if self.merged.item() and self.training:
            raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged.item():
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias.reshape(-1) if self.bias is not None else x @ self.w
        else:

            if self.mode == 'in':
                # [*, in_f] @ [1, in_f] @ [in_f, out_f] + [out_f] = [*, out_features]
                return x * self.d.reshape(-1, 1) @ self.w + self.bias.reshape(-1) if self.bias is not None \
                    else x @ self.d.reshape(-1, 1) @ self.w

            elif self.mode == 'out':
                # [*, in_f] @ [in_f, out_f] * [1, out_f] + [out_f] = [*, out_features]
                return x @ (self.w * self.d.reshape(1, -1)) + self.bias.reshape(-1) if self.bias is not None \
                    else x @ (self.w * self.d.reshape(1, -1))
