import torch
from torch import nn
import torch.nn.functional as F
import math

from leaf_audio_pytorch import impulse_responses


class GaborConstraint(nn.Module):
    """Constraint mu and sigma, in radians.

    Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
    gaussian response is in [1,pi/2]. The full-width at half maximum of the
    Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
    https://arxiv.org/pdf/1711.01161.pdf for more details.
    """

    def __init__(self, kernel_size):
        """Initialize kernel size.

        Args:
        kernel_size: the length of the filter, in samples.
        """
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = torch.clamp(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = torch.clamp(kernel[:, 1], sigma_lower, sigma_upper)
        return torch.stack([clipped_mu, clipped_sigma], dim=1)


class GaborConv1D(nn.Module):
    """Implements a convolution with filters defined as complex Gabor wavelets.

    These filters are parametrized only by their center frequency and
    the full-width at half maximum of their frequency response.
    Thus, for n filters, there are 2*n parameters to learn.
    """

    def __init__(self, filters, kernel_size, strides, padding, use_bias,
               input_shape, kernel_initializer, kernel_regularizer, name,
               trainable, sort_filters=False):

        super(GaborConv1D, self).__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters

        initialized_kernel = kernel_initializer(torch.zeros(self._filters, 2), sample_rate=16000, min_freq=60.0, max_freq=7800.0)
        self._kernel = nn.Parameter(initialized_kernel, requires_grad=trainable)
        # TODO: implement kernel regularizer here
        self._kernel_constraint = GaborConstraint(self._kernel_size)
        if self._use_bias:
            self._bias = nn.Parameter(torch.zeros(self.filters*2,), requires_grad=trainable) # TODO: validate that requires grad is the same as trainable

        # Register an initialization tensor here for creating the gabor impulse response to automatically handle cpu/gpu
        # device selection.
        self.register_buffer("gabor_filter_init_t",
                             torch.arange(-(self._kernel_size // 2), (self._kernel_size + 1) // 2, dtype=torch.float32))
        
    def forward(self, x):
        kernel = self._kernel_constraint(self._kernel)
        if self._sort_filters:
            # TODO: validate this
            filter_order = torch.argsort(kernel[:, 0])
            kernel = torch.gather(kernel, dim=0, index=filter_order)

        filters = impulse_responses.gabor_filters(kernel, self._kernel_size, self.gabor_filter_init_t)
        real_filters = torch.real(filters)
        img_filters = torch.imag(filters)
        stacked_filters = torch.stack([real_filters, img_filters], dim=1)
        stacked_filters = stacked_filters.view(2*self._filters, self._kernel_size)
        stacked_filters = stacked_filters.unsqueeze(1)
        output = F.conv1d(x, stacked_filters,
                          bias=self._bias if self._use_bias else None, stride=self._strides,
                          padding=self._padding)
        return output
