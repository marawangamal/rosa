from typing import Union

import torch
import torch.nn as nn


class RosaLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False
    ):
        """ROSA PEFT linear layer with trainable and fixed parameters

        Args:
            in_features: number of input features
            out_features: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = self._integer_rank(rank, full_rank=min(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 1)) if bias else None
        self.a = nn.Parameter(torch.zeros(in_features, self.rank))
        self.b = nn.Parameter(torch.randn(self.rank, out_features))
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.merged = False

    def initialize_weights(self, a_init: torch.Tensor = None, b_init: torch.Tensor = None, w_init: torch.Tensor = None):
        self.a.data = a_init if a_init is not None else self.a.data
        self.b.data = b_init if b_init is not None else self.b.data
        self.w.data = w_init if w_init is not None else self.w.data

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
        a = torch.zeros(obj.in_features, obj.rank, device=w.device)
        b = torch.randn(obj.rank, obj.out_features, device=w.device)
        obj.initialize_weights(w_init=w, a_init=a, b_init=b)
        return obj

    def merge(self):
        """Merge `a` and `b` with `w` and make `w` trainable"""
        if not self.merged:
            # Merge w and ab
            self.w.data = self.a.data @ self.b.data + self.w.data

            # todo: empty a and b tensors to save memory
            # Make a, b fixed and w trainable
            self.a.requires_grad = False
            self.b.requires_grad = False
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = True
        return self

    def factorize(self):
        """Factorize `w` into `a` and `b` and make a portion of `a` and `b` trainable"""
        if not self.merged:
            self.merge()

        # Factorize
        u, s, vt = torch.linalg.svd(self.w.data, full_matrices=False)  # [in_f, r],[r,],[r, out_f]
        a = s.reshape(1, -1) * u
        b = vt
        w_hat = a @ b

        # Check reconstruction error
        assert torch.allclose(self.w.data, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"

        rank_upper_bound = min(self.w.shape)
        trainable_indices, fixed_indices = self._select_k_from_n(self.rank, rank_upper_bound, mode='random')

        # Mask gradients
        init_a_trainable = a[:, trainable_indices]  # [in_f, r']
        init_b_trainable = b[trainable_indices, :]  # [r', out_f]
        init_a_fixed = a[:, fixed_indices]
        init_b_fixed = b[fixed_indices, :]
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
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        return (f'{cls}('
                f'rank={self.rank}, '
                f'a={self.a.shape, self.a.requires_grad}, '
                f'b={self.b.shape, self.b.requires_grad}, '
                f'w={self.w.shape, self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
                f')')

    @staticmethod
    def _integer_rank(rank, full_rank):
        """ Convert a ratio to an integer"""
        return rank if rank > 1 else max(int(rank * full_rank), 1)

    def _select_k_from_n(self, k: int, n: int, mode: str = 'random'):
        """Choose k indices from n indices"""

        assert 0 < k <= n, "k must be an integer between 0 and n"
        assert isinstance(k, int) and isinstance(n, int), "k and n must be integers"

        if mode.lower() == 'random':
            start_idx = torch.randint(0, n - k, (1,)).item()
            end_idx = start_idx + k
            chosen_ids = torch.arange(start_idx, end_idx)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'top':
            raise NotImplementedError
        elif mode.lower() == 'bottom':
            raise NotImplementedError
        else:
            raise AttributeError(f"Unknown mode: {mode}. Mode must be one of ['random', 'top', 'bottom']")

        return chosen_ids, remaining_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x: x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        if self.merged and self.training:
            raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged:
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias if self.bias.reshape(-1) is not None else x @ self.w
        else:
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.a @ self.b + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else x @ self.a @ self.b + x @ self.w


class RosaLinearOld(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            rank=1,
            init_a_trainable=None,
            init_b_trainable=None,
            init_a_fixed=None,
            init_b_fixed=None,
            init_w_fixed=None,
            init_bias=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank if init_a_trainable is None else init_a_trainable.size(1)

        assert (init_a_trainable is None) == (init_b_trainable is None), \
            "`init_a_trainable` or `init_b_trainable` must be jointly provided/omitted"
        assert (init_a_fixed is None) == (init_b_fixed is None), \
            "`init_a_fixed` or `init_b_fixed` must be jointly provided/omitted"

        if init_a_fixed is not None or init_w_fixed is not None:
            assert (init_a_fixed is not None) != (init_w_fixed is not None), \
                "Must provide either `init_a_fixed` or `init_w_fixed`"

        bias = True if init_bias is not None else bias
        init_bias = init_bias if init_bias is not None else torch.Tensor(out_features, 1)

        if init_a_trainable is not None:

            self.rosa_a_trainable = nn.Parameter(init_a_trainable)
            self.rosa_b_trainable = nn.Parameter(init_b_trainable)

            self.rosa_w_fixed = nn.Parameter(init_w_fixed, requires_grad=False) if init_w_fixed is not None else None
            self.rosa_a_fixed = nn.Parameter(init_a_fixed, requires_grad=False) if init_a_fixed is not None else None
            self.rosa_b_fixed = nn.Parameter(init_b_fixed, requires_grad=False) if init_b_fixed is not None else None

        else:
            self.rosa_a_trainable = nn.Parameter(torch.randn((in_features, self.rank)))
            self.rosa_b_trainable = nn.Parameter(torch.randn((self.rank, out_features)))
            self.rosa_a_fixed, self.rosa_b_fixed, self.rosa_w_fixed = None, None, None

        if init_bias is not None:
            self.rosa_bias = nn.Parameter(init_bias)
        elif bias:
            self.rosa_bias = nn.Parameter(torch.zeros((out_features, 1)))

    @classmethod
    def from_module(cls, linear_layer, fan_in_fan_out=True):
        """ Initialize from a nn.Linear/Conv1D module """

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        u, s, vt = torch.linalg.svd(
            w, full_matrices=False
        )  # [in_f, r],[r,],[r, out_f]

        a = s.reshape(1, -1) * u
        b = vt
        w_hat = a @ b

        # Check reconstruction error
        assert torch.allclose(w, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"

        return cls(
            in_features=a.size(0), out_features=b.size(1), init_a_trainable=a, init_b_trainable=b, init_bias=bias
        )

    def from_state_dict(self, state_dict):
        """ Create a FactorizedLinearMaskedGradient layer from a state dict

        Args:
            state_dict: State dict of a FactorizedLinearMaskedGradient layer

        Returns:
            A FactorizedLinearMaskedGradient layer
        """

        init_a_trainable = state_dict['rosa_a_trainable']
        init_b_trainable = state_dict['rosa_b_trainable']
        in_feat, out_feat = init_a_trainable.size(0), init_b_trainable.size(1)
        init_a_fixed = state_dict['rosa_a_fixed'] if "rosa_a_fixed" in state_dict is not None else None
        init_b_fixed = state_dict['rosa_b_fixed'] if "rosa_b_fixed" in state_dict is not None else None
        init_w_fixed = state_dict['rosa_w_fixed'] if "rosa_w_fixed" in state_dict is not None else None
        init_bias = state_dict['rosa_bias'] if "rosa_bias" in state_dict is not None else None

        return self.__class__(
            in_features=in_feat,
            out_features=out_feat,
            init_a_trainable=init_a_trainable,
            init_b_trainable=init_b_trainable,
            init_a_fixed=init_a_fixed,
            init_b_fixed=init_b_fixed,
            init_w_fixed=init_w_fixed,
            init_bias=init_bias,
        )

    @property
    def a_weight(self):
        return torch.cat([self.rosa_a_trainable, self.rosa_a_fixed], dim=1) if self.rosa_a_fixed \
                                                                               is not None else self.rosa_a_trainable

    @property
    def b_weight(self):
        try:
            return torch.cat([self.rosa_b_trainable, self.rosa_b_fixed], dim=0) if self.rosa_b_fixed \
                                                                                   is not None else self.rosa_b_trainable
        except:
            import pdb;
            pdb.set_trace()

    @staticmethod
    def ratio2int(rank_ratio, max_rank, min_rank=1):
        """ Convert a ratio to an integer"""
        assert 0 < rank_ratio <= 1, "`rank_ratio` must be a float between 0 and 1"
        return max(int(rank_ratio * max_rank), min_rank)

    def sample_trainable(self, rank=0, method='random', collapse_fixed=True):
        """ Mask gradients of the trainable parameters

        Args:
            rank: Ratio of gradients to be masked
            method: Method to mask gradients (one of ['random', 'top', 'bottom'])
            collapse_fixed: Whether to collapse the fixed parameters into single matrix

        Returns:
            FactorizedLinear layer with low rank trainable matrix
        """
        assert 0 < rank < 1 or isinstance(rank, int), "r must be a float between 0 and 1 or an integer"

        with torch.no_grad():
            # Sample gradient indices
            if method == 'random':

                if collapse_fixed:
                    w_tot = self.rosa_a_trainable @ self.rosa_b_trainable
                    w_tot = w_tot + self.rosa_w_fixed if self.rosa_w_fixed is not None else w_tot
                    u, s, vt = torch.linalg.svd(w_tot, full_matrices=False)
                    a = s.reshape(1, -1) * u  # [in_f, full_rank]
                    b = vt  # [full_rank, out_f]

                else:
                    a = self.a_weight.data  # [in_f, full_rank]
                    b = self.b_weight.data  # [full_rank, out_f]

                full_rank = a.size(1)
                bias = self.rosa_bias.data if self.rosa_bias is not None else None

                if rank < 1:  # rank interpreted as a ratio
                    start_idx = torch.randint(0, full_rank - self.ratio2int(rank, full_rank), (1,)).item()
                    end_idx = start_idx + self.ratio2int(rank, full_rank)

                else:  # rank interpreted as an integer
                    start_idx = torch.randint(0, full_rank - rank, (1,)).item()
                    end_idx = start_idx + rank

                grad_indices = torch.arange(start_idx, max(end_idx, start_idx + 1))
                non_grad_indices = [i for i in range(full_rank) if i not in grad_indices]

            elif method in ['top', 'bottom']:

                if collapse_fixed:
                    w_tot = self.rosa_a_trainable @ self.rosa_b_trainable
                    w_tot = w_tot + self.rosa_w_fixed if self.rosa_w_fixed is not None else w_tot
                    u, s, vt = torch.linalg.svd(w_tot, full_matrices=False)
                    a = s.reshape(1, -1) * u  # [in_f, full_rank]
                    b = vt  # [full_rank, out_f]

                else:
                    a = self.a_weight.data  # [in_f, r_full]
                    b = self.b_weight.data  # [r_full, out_f]
                    w_tot = a @ b
                    u, s, vt = torch.linalg.svd(w_tot, full_matrices=False)
                    a = torch.sqrt(s).reshape(1, -1) * u  # [in_f, r]
                    b = torch.sqrt(s).reshape(-1, 1) * vt  # [r, out_f]

                full_rank = a.size(1)
                bias = self.rosa_bias.data if self.rosa_bias is not None else None

                if method == 'bottom':
                    start_idx = max(len(s) - self.ratio2int(rank, full_rank) - 1, 0) if rank < 1 else max(len(s) - rank,
                                                                                                          0)
                    grad_indices = torch.arange(start_idx, len(s))
                    non_grad_indices = [i for i in range(full_rank) if i not in grad_indices]
                else:
                    start_idx = 0
                    end_idx = self.ratio2int(rank, full_rank) if rank < 1 else rank
                    grad_indices = torch.arange(start_idx, end_idx)
                    non_grad_indices = [i for i in range(full_rank) if i not in grad_indices]

            else:
                raise NotImplementedError(f"Unknown grad_sample_method: {method}")

            # Mask gradients
            init_a_trainable = a[:, grad_indices]  # [in_f, r']
            init_b_trainable = b[grad_indices, :]  # [r', out_f]

            init_a_fixed = a[:, non_grad_indices]
            init_b_fixed = b[non_grad_indices, :]
            init_w_fixed = None

            if collapse_fixed:
                init_w_fixed = init_a_fixed @ init_b_fixed
                init_a_fixed = None
                init_b_fixed = None

            return self.__class__(
                in_features=a.size(0), out_features=b.size(1),
                init_a_trainable=init_a_trainable, init_b_trainable=init_b_trainable, init_bias=bias,
                init_a_fixed=init_a_fixed, init_b_fixed=init_b_fixed, init_w_fixed=init_w_fixed
            )

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(rank={self.rank}, at={self.rosa_a_trainable.shape}, bt={self.rosa_b_trainable.shape}, ' \
               f'af={[self.rosa_a_fixed.shape, self.rosa_a_fixed.requires_grad] if self.rosa_a_fixed is not None else None}, ' \
               f'bf={[self.rosa_b_fixed.shape, self.rosa_b_fixed.requires_grad] if self.rosa_b_fixed is not None else None}, ' \
               f'wf={[self.rosa_w_fixed.shape, self.rosa_w_fixed.requires_grad] if self.rosa_w_fixed is not None else None}, ' \
               f'bias={[self.rosa_bias.shape, self.rosa_bias.requires_grad] if self.rosa_bias is not None else None}) '

    def forward(self, x):
        """ Forward pass

        Args:
            x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        x_shape = x.shape
        if self.rosa_w_fixed is not None:  # collapsed fixed weights
            x = x.reshape(*x_shape[:-1], self.in_features) @ self.rosa_w_fixed + \
                (x.reshape(*x_shape[:-1], self.in_features) @ self.rosa_a_trainable) @ self.rosa_b_trainable

        elif self.rosa_a_fixed is not None:
            x = (x.reshape(*x_shape[:-1], self.in_features) @ self.rosa_a_fixed) @ self.rosa_b_fixed \
                + (x.reshape(*x_shape[:-1], self.in_features) @ self.rosa_a_trainable) @ self.rosa_b_trainable

        else:  # no fixed weights
            x = (x.reshape(*x_shape[:-1], self.in_features) @ self.rosa_a_trainable) @ self.rosa_b_trainable

        return x.reshape(*x_shape[:-1], self.out_features) if self.rosa_bias is None else x + self.rosa_bias
