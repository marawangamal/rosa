import torch.nn as nn


class FactorizedLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_module(cls, linear_layer):
        raise NotImplementedError

    @classmethod
    def compression_step(cls, factorized_layer):
        raise NotImplementedError
