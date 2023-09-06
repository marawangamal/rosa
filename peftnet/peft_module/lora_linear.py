#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoraLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            init_w: nn.Module = None,
            init_bias: nn.Module = None,
            **kwargs
    ):
        super(LoraLinear, self).__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features

        if init_w is not None:
            self.weight = nn.Parameter(init_w)
            self.bias = nn.Parameter(init_bias) if init_bias is not None else None
        elif fan_in_fan_out:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Actual trainable parameters
        if rank > 0:
            # self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            # self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.lora_A = nn.Parameter(self.weight.new_zeros((in_features, rank)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((rank, out_features)))
            self.scaling = self.lora_alpha / self.rank

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @staticmethod
    def ratio2int(rank_ratio, max_rank, min_rank=1):
        """ Convert a ratio to an integer"""
        assert 0 < rank_ratio <= 1, "`rank_ratio` must be a float between 0 and 1"
        return max(int(rank_ratio * max_rank), min_rank)


    @classmethod
    def from_module(cls, linear_layer, rank=1, fan_in_fan_out=True):

        assert 0 < rank < 1 or isinstance(rank, int), "r must be a float between 0 and 1 or an integer"
        assert linear_layer.weight.data is not None, "The layer must have a weight matrix"

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None
        full_rank = min(w.size(0), w.size(1))
        rank = cls.ratio2int(rank, full_rank) if rank < 1 else rank

        return cls(
            in_features=w.size(0), out_features=w.size(1), init_w=w, init_bias=bias, fan_in_fan_out=fan_in_fan_out,
            rank=rank
        )

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(a={self.lora_A.shape if self.rank > 0 else None}, ' \
               f'b={self.lora_B.shape if self.rank > 0 else None}, ' \
               f'r={self.rank} ' \
               f'bias={self.bias.shape if self.bias is not None else None}, ' \
               f'w={[self.weight.shape, self.weight.requires_grad]})'

    def forward(self, x: torch.Tensor):
        # def T(w):
        #     return w.transpose(0, 1) if self.fan_in_fan_out else w
        #
        # if self.rank > 0 and not self.merged:
        #     result = F.linear(x, T(self.weight), bias=self.bias)
        #     if self.rank > 0:
        #         result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
        #                                                                                               1)) * self.scaling
        #     return result
        # else:
        #     return F.linear(x, T(self.weight), bias=self.bias)

        if self.rank > 0 and not self.merged:
            x_shape = x.shape
            # x = x.reshape(*x_shape[:-1], self.in_features) @ self.weight + \
            #     ((x.reshape(*x_shape[:-1], self.in_features) @ self.lora_A) @ self.lora_B) * self.scaling
            x = x.reshape(*x_shape[:-1], self.in_features) @ self.weight + \
                (x.reshape(*x_shape[:-1], self.in_features) @ self.lora_A) @ self.lora_B
            return x
        else:
            x_shape = x.shape
            x = x.reshape(*x_shape[:-1], self.in_features) @ self.weight
            return x.reshape(*x_shape[:-1], self.out_features) if self.bias is None else x + self.bias


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                      1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(T(delta_w * self.scaling))
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels // self.groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)


class LoraLinear2(nn.Module):
    def __init__(self, in_features, out_features, rank=1.0, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank if isinstance(rank, int) else int(rank * min(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 1)) if bias else None
        self.a = nn.Parameter(torch.zeros(in_features, self.rank))
        self.b = nn.Parameter(torch.randn(self.rank, out_features))
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.merged = True

    def initialize_weights(self, a_init: torch.Tensor = None, b_init: torch.Tensor = None, w_init: torch.Tensor = None):
        self.a.data = a_init if a_init is not None else self.a.data
        self.b.data = b_init if b_init is not None else self.b.data
        self.w.data = w_init if w_init is not None else self.w.data

    @classmethod
    def from_module(cls, linear_layer: nn.Module, rank=1.0, fan_in_fan_out=True) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module """

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), rank=rank, bias=bias is not None
        )
        obj.initialize_weights(w_init=w)
        return obj

    def merge(self):
        """Convert to merged format"""
        # Merge w and ab
        self.w.data = self.a.data @ self.b.data + self.w.data

        # Make a, b fixed and w trainable
        self.a.requires_grad = False
        self.b.requires_grad = False
        self.w.requires_grad = True

        # Toggle merged flag
        self.merged = True

    def factorize(self):
        if not self.merged:
            self.merge()

        # Factorize
        u, s, vt = torch.linalg.svd(self.w.data, full_matrices=False)  # [in_f, r],[r,],[r, out_f]
        a = s.reshape(1, -1) * u
        b = vt
        w_hat = a @ b

        # Check reconstruction error
        assert torch.allclose(self.w.data, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"

        # Sample trainable params (Random)
        full_rank = min(self.w.shape)
        rank = self.rank
        start_idx = torch.randint(0, full_rank - rank, (1,)).item()
        end_idx = start_idx + rank
        grad_indices = torch.arange(start_idx, end_idx)
        non_grad_indices = [i for i in range(full_rank) if i not in grad_indices]

        # Mask gradients
        init_a_trainable = a[:, grad_indices]  # [in_f, r']
        init_b_trainable = b[grad_indices, :]  # [r', out_f]
        init_a_fixed = a[:, non_grad_indices]
        init_b_fixed = b[non_grad_indices, :]
        init_w = init_a_fixed @ init_b_fixed

        # Initialize
        self.a.data = init_a_trainable
        self.b.data = init_b_trainable
        self.w.data = init_w

        # Make a, b trainable and w fixed
        self.a.requires_grad = True
        self.b.requires_grad = True
        self.w.requires_grad = False

        # Toggle merged flag
        self.merged = False

    def forward(self, x):
        if self.merged:
            return x @ self.w + self.bias
        else:
            return x @ self.a @ self.b + x @ self.w + self.bias