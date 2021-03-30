import torch
from torch import nn
import torch.nn.functional as F

from leaf_audio_pytorch import impulse_responses


class GaussianLowpass(nn.Module):
    """Depthwise pooling (each input filter has its own pooling filter).

    Pooling filters are parametrized as zero-mean Gaussians, with learnable
    std. They can be initialized with tf.keras.initializers.Constant(0.4)
    to approximate a Hanning window.
    We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
    """

    def __init__(
        self,
        kernel_size,
        strides=1,
        filter_size=40,
        padding=0,
        use_bias=True,
        kernel_initializer=nn.init.xavier_uniform_,
        kernel_regularizer=None,
        trainable=False):

        super(GaussianLowpass, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.filter_size = filter_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.trainable = trainable

        initialized_kernel = self.kernel_initializer(torch.zeros(1, 1, self.filter_size, 1).type(torch.float32))
        self._kernel = nn.Parameter(initialized_kernel, requires_grad=self.trainable)

        # Register an initialization tensor here for creating the gaussian lowpass impulse response to automatically
        # handle cpu/gpu device selection.
        self.register_buffer("gaussian_lowpass_init_t", torch.arange(0, self.kernel_size, dtype=torch.float32))

    def forward(self, x):
        kernel = impulse_responses.gaussian_lowpass(self._kernel, self.kernel_size, self.gaussian_lowpass_init_t)
        kernel = kernel.squeeze(3)
        kernel = kernel.permute(2, 0, 1)

        outputs = F.conv1d(x, kernel, stride=self.strides, groups=self.filter_size, padding=self.padding)
        return outputs

