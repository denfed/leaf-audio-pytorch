import torch
from torch import nn
from typing import Callable, Optional

import initializers
import convolution


# TODO: implement weight freezing if learn_filters is False

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
        # activation=SquaredModulus(),
        # pooling_cls=pooling.GaussianLowpass,
        n_filters: int = 40,
        sample_rate: int = 16000,
        window_len: float = 25.,
        window_stride: float = 10.,
        # compression_fn: _TensorCallable = postprocessing.PCENLayer(
        #     alpha=0.96,
        #     smooth_coef=0.04,
        #     delta=2.0,
        #     floor=1e-12,
        #     trainable=True,
        #     learn_smooth_coef=True,
        #     per_channel_smooth_coef=True),
        preemp: bool = False,
        preemp_init=initializers.PreempInit,
        # complex_conv_init: _Initializer = initializers.GaborInit(
        #   sample_rate=16000, min_freq=60.0, max_freq=7800.0),
        # pooling_init: _Initializer = tf.keras.initializers.Constant(0.4),
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
            # kernel_initializer=complex_conv_init,
            kernel_initializer=None,
            # kernel_regularizer=regularizer_fn if learn_filters else None,
            kernel_regularizer=None,
            name='tfbanks_complex_conv',
            trainable=learn_filters)

    def forward(self, x):
        if self._preemp:
            outputs = self._preemp_conv(x)
            # Pytorch padding trick needed because 'same' doesn't exist and kernel is even.
            # Remove the first value in the feature dim of the tensor to match tf's padding.
            outputs = outputs[:,:,1:]
            
        return outputs


if __name__ == "__main__": 
    print("done")