import torch
import torch.nn as nn

from .factorized_layer import FactorizedLayer


class RosaLinear(FactorizedLayer):
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

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f]
        bias = linear_layer.bias.data if linear_layer.bias is not None else None
        u, s, vt = torch.linalg.svd(w, full_matrices=False)  # [out_f, r],[r,],[r, in_f] or [in_f, r],[r,],[r, out_f]

        if fan_in_fan_out:  # Conv1D
            a = torch.sqrt(s).reshape(1, -1) * u  # [in_f, r]
            b = torch.sqrt(s).reshape(-1, 1) * vt  # [r, out_f]
            w_hat = a @ b
        else:
            b = (torch.sqrt(s).reshape(1, -1) * u).T  # [out_f, r] => [r, out_f]
            a = (torch.sqrt(s).reshape(-1, 1) * vt).T  # [r, in_f] => [in_f, r]
            w_hat = b @ a

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
    def ratio2int(rank_ratio, max_rank):
        """ Convert a ratio to an integer"""
        assert 0 < rank_ratio <= 1, "r must be a float between 0 and 1"
        return int(rank_ratio * max_rank)

    def sample_trainable(self, rank=0, method='random', collapse_fixed=True):
        """ Mask gradients of the trainable parameters

        Args:
            rank: Ratio of gradients to be masked
            method: Method to mask gradients. 'random' or 'top'
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
                    a = torch.sqrt(s).reshape(1, -1) * u  # [in_f, full_rank]
                    b = torch.sqrt(s).reshape(-1, 1) * vt  # [full_rank, out_f]

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

            elif method == 'bottom':

                if collapse_fixed:
                    w_tot = self.rosa_a_trainable @ self.rosa_b_trainable
                    w_tot = w_tot + self.rosa_w_fixed if self.rosa_w_fixed is not None else w_tot
                    u, s, vt = torch.linalg.svd(w_tot, full_matrices=False)
                    a = torch.sqrt(s).reshape(1, -1) * u  # [in_f, r]
                    b = torch.sqrt(s).reshape(-1, 1) * vt  # [r, out_f]

                else:
                    a = self.a_weight.data  # [in_f, r_full]
                    b = self.b_weight.data  # [r_full, out_f]
                    w_tot = a @ b
                    u, s, vt = torch.linalg.svd(w_tot, full_matrices=False)
                    a = torch.sqrt(s).reshape(1, -1) * u  # [in_f, r]
                    b = torch.sqrt(s).reshape(-1, 1) * vt  # [r, out_f]

                full_rank = a.size(1)
                bias = self.rosa_bias.data if self.rosa_bias is not None else None

                start_idx = max(len(s) - self.ratio2int(rank, full_rank) - 1, 0) if rank < 1 else max(len(s) - rank, 0)
                grad_indices = torch.arange(start_idx, len(s))
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
