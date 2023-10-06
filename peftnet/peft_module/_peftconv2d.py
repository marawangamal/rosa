from typing import Union

import torch
import torch.nn as nn
import tltorch


class PeftConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 0,
            rank: Union[int, float] = 1.0,
            bias: bool = False,
            use_scale: bool = False,
            alpha: float = 32.0,
            adapt_method: str = 'ab',  # 'ab', 'a', 'b'
            sample_method: str = 'random',
            factorize_method: str = 'equal',  # 'equal', 'add'
            init_method: str = 'zero',  # 'zero', 'random'
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ PEFT linear layer with trainable and fixed parameters in parallel.

        Args:
            in_channels: number of input channels
            in_channels: number of output channels
            rank: {'same', float, int}, default is same
                way to determine the rank, by default 'same'
                if 'same': rank is computed to keep the number of parameters (at most) the same
                if float, computes a rank so as to keep rank percent of the original number of parameters
                if int, just returns rank
            bias: whether to include bias
            use_scale: whether to use scale factor
            alpha: scale factor
            adapt_method: which parameters to adapt [`ab`, `a`, `b`] (default: `ab`)
            sample_method: sample method [`random`, `top`, `bottom`]
            factorize_method: factorize method `w` \gets usv_1 + usv_2  (equal) or `w` \gets w + usv_2 (add)
            init_method: initialization method for `a` [`zero`, `random`]
            debug: whether to use debug mode

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()

        # Input validation
        assert isinstance(in_channels, int) and isinstance(out_channels, int), \
            "in_channels and out_channels must be integers"
        assert isinstance(rank, (int, float)), "rank must be an integer or a float"
        assert isinstance(bias, bool), "bias must be a boolean"
        assert isinstance(use_scale, bool), "use_scale must be a boolean"
        assert isinstance(alpha, (int, float)), "alpha must be an integer or a float"
        assert factorize_method in ['equal', 'add'], "factorize_method must be one of ['equal', 'add']"
        assert sample_method in ['random', 'top', 'bottom'], \
            "sample_method must be one of ['random', 'top', 'bottom']"
        assert init_method in ['zero', 'random'], "init_method must be one of ['zero', 'random']"
        assert adapt_method in ['ab', 'a', 'b'], "adapt_method must be one of ['ab', 'a', 'b']"
    
        # Convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # PEFT parameters
        self.rank = self._integer_rank(rank, full_rank=min(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=bias_requires_grad) if bias else None
        self.use_scale = use_scale
        self.alpha = alpha
        self.adapt_method = adapt_method
        self.sample_method = sample_method
        self.factorize_method = factorize_method
        self.init_method = init_method
        self.debug = debug
        self.fast_mode = fast_mode

        # Set requires_grad for a and b
        self.requires_grad_a = True if self.adapt_method in ['ab', 'a'] else False
        self.requires_grad_b = True if self.adapt_method in ['ab', 'b'] else False

        self.w_hat_conv = tltorch.FactorizedConv(
            self.in_channels, self.out_channels, self.kernel_size, order=2, rank=rank, factorization='cp',
            bias=False, padding=self.padding, stride=self.stride
        )

        self.w_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, bias=False, padding=self.padding, stride=self.stride
        )

        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=bias_requires_grad) if bias else None
        self.register_buffer('merged', torch.tensor([False]))

    def initialize_weights(self, w_init: torch.Tensor = None, bias_init: torch.Tensor = None):
        """Initialize weights and biases with given tensors."""
        self.w_conv.weight.data = w_init if w_init is not None else self.w_conv.weight.data
        self.bias.data = bias_init if bias_init is not None else self.bias.data

    @classmethod
    def from_module(cls, conv_layer: nn.Module, rank=1.0, *args, **kwargs) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module

        Args:
            conv_layer: linear layer to initialize from
            rank: rank of factorized matrices
            *args:
            **kwargs:

        Returns:
            obj: initialized PEFTConv2d object

        """

        in_c = conv_layer.in_channels
        out_c = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size
        w = conv_layer.weight.data
        bias = conv_layer.bias.data if conv_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            rank=rank,
            bias=bias is not None,
            *args, **kwargs
        )
        obj.initialize_weights(w_init=w, bias_init=bias if bias is not None else None)
        return obj

    def merge(self):
        raise NotImplementedError

    def factorize(self):
        raise NotImplementedError

    # def __repr__(self):
    #     cls = self.__class__.__name__
    #     return (f'{cls}('
    #             f'rank={self.rank}, '
    #             f'a={self.a.shape} grad={self.a.requires_grad} scale={self.use_scale}, alpha={self.alpha}, '
    #             f'b={self.b.shape} grad={self.b.requires_grad}, '
    #             f'w={self.w.shape} grad={self.w.requires_grad}, '
    #             f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
    #             f')')

    @staticmethod
    def _integer_rank(rank, full_rank):
        """Convert a ratio to an integer"""
        return rank if isinstance(rank, int) else max(int(rank * full_rank), 1)

    @staticmethod
    def _select_k_from_n(k: int, n: int, mode: str = 'random'):
        """Choose `k` indices from `n` indices"""

        assert 0 < k < n, f"k must be an integer between 0 and n, received k={k}, n={n}"
        assert isinstance(k, int) and isinstance(n, int), "k and n must be integers"

        if mode.lower() == 'random':
            # Select k random indices from n indices
            start_idx = torch.randint(0, n - k, (1,)).item()
            end_idx = start_idx + k
            chosen_ids = torch.arange(start_idx, end_idx)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'top':
            # Select top k indices from n indices
            chosen_ids = torch.arange(0, min(k, n - k))
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'bottom':
            # Select bottom k indices from n indices
            chosen_ids = torch.arange(max(0, n - k), n)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        else:
            raise AttributeError(f"Unknown mode: {mode}. Mode must be one of ['random', 'top', 'bottom']")

        return chosen_ids, remaining_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. Assumes `w` is fan_in x fan_out.

        Args:
            x: [N, C, H, W] input tensor

        Returns:
            y: [N, F, H, W] output tensor
        """

        # if self.merged.item() and self.training:
        #     raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged.item():
            raise NotImplementedError("Merged forward pass not implemented yet.")
        elif self.debug:  # retain intermediate gradient (for plotting purposes)
            raise NotImplementedError("Debug mode not implemented yet.")
        else:
            # [*, in_channels] @ [in_channels, rank] @ [rank, out_channels] + [out_channels, 1] = [*, out_channels]
            y = self.w_hat_conv(x) + self.w_conv(x)
            return y
