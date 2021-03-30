import torch
from torch import Tensor
import math
import numpy as np


def gabor_impulse_response(t: Tensor, center: Tensor,
                           fwhm: Tensor) -> Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (np.sqrt(2.0 * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm**2), -t**2, dims=0)) # TODO: validate the dims here
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        1j * torch.tensordot(center_frequency_complex, t_complex, dims=0))
        #continue down
    # denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
    denominator = denominator.type(torch.complex64).unsqueeze(1) # this should be the above line
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401, t_tensor=None) -> Tensor:
    """Computes the gabor filters from its parameters for a given size.

    Args:
    kernel: Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

    Returns:
    A Tensor<float>[filters, size].
    """
    return gabor_impulse_response(
        t_tensor,
        center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma: Tensor, filter_size: int, t_tensor: Tensor):
    """Generates gaussian windows centered in zero, of std sigma.

    Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

    Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
    """

    sigma = torch.clamp(sigma, (2. / filter_size), 0.5)
    t = t_tensor.view(1, filter_size, 1, 1)
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator)**2)
