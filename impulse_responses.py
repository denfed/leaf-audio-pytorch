import torch
from torch import Tensor
import math


def gabor_impulse_response(t: Tensor, center: Tensor,
                           fwhm: Tensor) -> Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (torch.sqrt(2.0 * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm**2), -t**2, axes=0)) # TODO: validate the dims here
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        1j * torch.tensordot(center_frequency_complex, t_complex, axes=0))
        #continue down
    # denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
    denominator = denominator.type(torch.complex64).unsqueeze(1) # this should be the above line
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401) -> Tensor:
  """Computes the gabor filters from its parameters for a given size.

  Args:
    kernel: Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

  Returns:
    A Tensor<float>[filters, size].
  """
  return gabor_impulse_response(
      torch.range(-(size // 2), (size + 1) // 2, dtype=tf.float32),
      center=kernel[:, 0], fwhm=kernel[:, 1])