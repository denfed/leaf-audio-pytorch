"""Initializer classes for each layer of the learnable frontend."""

import torch
from torch import Tensor
import numpy as np

def PreempInit(tensor: Tensor, alpha: float=0.97) -> Tensor:
    """Pytorch initializer for the pre-emphasis.

    Modifies conv weight Tensor to initialize the pre-emphasis layer of a Leaf instance.

    Attributes:
        alpha: parameter that controls how much high frequencies are emphasized by
        the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
        alpha < 1 (higher alpha boosts high frequencies)
    """

    shape = tensor.shape
    assert shape == (1,1,2), f"Cannot initialize preemp layer of size {shape}"

    with torch.no_grad():
        tensor[0, 0, 0] = -alpha
        tensor[0, 0, 1] = 1
    
        return tensor