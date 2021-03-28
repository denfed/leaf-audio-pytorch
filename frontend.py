import torch
from torch import nn
from typing import Callable, Optional

import initializers
import convolution
import pooling
import postprocessing

# TODO: implement weight freezing if learn_filters is False

class SquaredModulus(nn.Module):
    """Squared modulus layer.

    Returns a keras layer that implements a squared modulus operator.
    To implement the squared modulus of C complex-valued channels, the expected
    input dimension is N*1*W*(2*C) where channels role alternates between
    real and imaginary part.
    The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
    - squared operator on real and imag
    - average pooling to compute (real ** 2 + imag ** 2) / 2
    - multiply by 2

    Attributes:
        pool: average-pooling function over the channel dimensions
    """

    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        output = 2 * self._pool(x**2)
        return torch.transpose(output, 2, 1)


class Leaf(nn.Module):
    """Pytorch layer that implements time-domain filterbanks.

    Creates a LEAF frontend, a learnable front-end that takes an audio
    waveform as input and outputs a learnable spectral representation. This layer
    can be initialized to replicate the computation of standard mel-filterbanks.
    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596.

    """
    def __init__(
        self,
        learn_pooling: bool = True,
        learn_filters: bool = True,
        conv1d_cls=convolution.GaborConv1D,
        activation=SquaredModulus(),
        pooling_cls=pooling.GaussianLowpass,
        n_filters: int = 40,
        sample_rate: int = 16000,
        window_len: float = 25.,
        window_stride: float = 10.,
        # compression_fn=None,
        compression_fn=postprocessing.PCENLayer(
            alpha=0.96,
            smooth_coef=0.04,
            delta=2.0,
            floor=1e-12,
            trainable=True,
            learn_smooth_coef=True,
            per_channel_smooth_coef=True),
        preemp: bool = False,
        preemp_init=initializers.PreempInit,
        complex_conv_init=initializers.GaborInit,
        pooling_init=initializers.ConstantInit,
        # regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
        mean_var_norm: bool = False,
        spec_augment: bool = False,
        name='leaf'):

        super(Leaf, self).__init__()

        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)

        self._preemp = preemp
        
        if preemp:
            self._preemp_conv = torch.nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=1,
                padding=(2//2),
                bias=False,
            )
            preemp_init(self._preemp_conv.weight)

        self._complex_conv = conv1d_cls(
            filters=2 * n_filters,
            kernel_size=window_size,
            strides=1,
            padding=(window_size//2), # TODO: validate that this is correct
            use_bias=False,
            input_shape=(None, None, 1),
            kernel_initializer=complex_conv_init,
            # kernel_regularizer=regularizer_fn if learn_filters else None,
            kernel_regularizer=None,
            name='tfbanks_complex_conv',
            trainable=learn_filters)

        self._activation = activation
        self._pooling = pooling_cls(
            kernel_size=window_size,
            strides=window_stride,
            filter_size=n_filters,
            padding=(window_size//2),
            use_bias=False,
            kernel_initializer=pooling_init,
            # kernel_regularizer=regularizer_fn if learn_pooling else None,
            trainable=learn_pooling)

        self._compress_fn = compression_fn if compression_fn else nn.Identity()
        # Pass number of filters to PCEN layer for on-the-fly building.
        # We do this to avoid adding num_channels as an arg into the class itself to avoid double setting the same arg
        # when instantiating the Leaf class.
        if isinstance(self._compress_fn, postprocessing.PCENLayer):
            self._compress_fn.build(num_channels=n_filters)

    def forward(self, x):
        outputs = x.unsqueeze(1) if len(x.shape) < 2 else x # TODO: validate this
        if self._preemp:
            outputs = self._preemp_conv(x)
            # Pytorch padding trick needed because 'same' doesn't exist and kernel is even.
            # Remove the first value in the feature dim of the tensor to match tf's padding.
            outputs = outputs[:,:,1:]

        outputs = self._complex_conv(outputs)
        outputs = self._activation(outputs)
        outputs = self._pooling(outputs)
        # As far as I know, torch cannot perform element-wise maximum between a tensor and scalar, here is a workaround.
        output_copy = torch.zeros_like(outputs)
        output_copy[:,:,:] = 1e-5
        outputs = torch.maximum(outputs, output_copy)
        outputs = self._compress_fn(outputs)

        return outputs


if __name__ == "__main__": 
    print("done")