"""Initializer classes for each layer of the learnable frontend."""

import torch
from torch import Tensor

from leaf_audio_pytorch import melfilters

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

def GaborInit(tensor: Tensor, **kwargs) -> Tensor:
    kwargs.pop('n_filters', None)

    shape = tensor.shape

    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = melfilters.Gabor(
        n_filters=n_filters, window_len=window_len, **kwargs)
    if len(shape) == 2:
        with torch.no_grad():
            tensor = gabor_filters.gabor_params_from_mels
            return tensor
    else:
        # TODO: FINISH
        pass

def ConstantInit(tensor: Tensor) -> Tensor:
    tensor[:,:,:,:] = 0.4
    return tensor
