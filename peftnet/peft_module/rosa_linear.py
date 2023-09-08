from typing import Union
from ._peft_linear import PeftLinear


class RosaLinear(PeftLinear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias
        )
