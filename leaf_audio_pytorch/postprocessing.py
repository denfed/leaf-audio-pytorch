import torch
from torch import nn
from torch import Tensor

def log_compression(inputs: Tensor, epsilon=1e-6) -> Tensor:
    "Log compression"
    return torch.log(inputs+epsilon)


class ExponentialMovingAverage(nn.Module):
    """Computes of an exponential moving average of an sequential input."""
    def __init__(self,
        coeff_init: float,
        per_channel: bool = False, trainable: bool = False):
        """Initializes the ExponentialMovingAverage.

        Args:
          coeff_init: the value of the initial coeff.
          per_channel: whether the smoothing should be different per channel.
          trainable: whether the smoothing should be trained or not.
        """
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self._trainable = trainable

    def build(self, num_channels):
        ema_tensor = torch.zeros((num_channels)).type(torch.float32) if self._per_channel else torch.zeros((1))
        ema_tensor[:] = self._coeff_init
        self._weights = nn.Parameter(ema_tensor, requires_grad=self._trainable)

    def forward(self, x, initial_state):
        w = torch.clamp(self._weights, 0.0, 1.0)
        func = lambda a, y: w * y + (1.0 - w) * a

        def scan(foo, x):
            res = []
            res.append(x[0].unsqueeze(0))
            a_ = x[0].clone()

            for i in range(1, len(x)):
                res.append(foo(a_, x[i]).unsqueeze(0))
                a_ = foo(a_, x[i])

            return torch.cat(res)

        res = scan(func, x.permute(2,0,1))
        return res.permute(1,0,2)
        # for i in x.permute(2,0,1):
        #     print(i)

    # def call(self, inputs: Tensor, initial_state: Tensor):
    #     """Inputs is of shape [batch, seq_length, num_filters]."""
    #     w = torch.clamp(self._weights, 0.0, 1.0)
    #     result = tf.scan(lambda a, x: w * x + (1.0 - w) * a,
    #                      tf.transpose(inputs, (1, 0, 2)),
    #                      initializer=initial_state)
    #     return tf.transpose(result, (1, 0, 2))


class PCENLayer(nn.Module):
    """Per-Channel Energy Normalization.

    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    See https://arxiv.org/abs/1607.05666 for more details.
    """

    def __init__(self,
                 alpha: float = 0.96,
                 smooth_coef: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6,
                 trainable: bool = False,
                 learn_smooth_coef: bool = False,
                 per_channel_smooth_coef: bool = False,
                 name='PCEN'):
        """PCEN constructor.

        Args:
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
          trainable: bool, False means fixed_pcen, True is trainable_pcen
          learn_smooth_coef: bool, True means we also learn the smoothing
            coefficient
          per_channel_smooth_coef: bool, True means each channel has its own smooth
            coefficient
          name: str, name of the layer
        """
        super(PCENLayer, self).__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

    def build(self, num_channels):
        alpha_tensor = torch.zeros((num_channels)).type(torch.float32)
        alpha_tensor[:] = self._alpha_init
        self.alpha = nn.Parameter(alpha_tensor, requires_grad=self._trainable)

        delta_tensor = torch.zeros((num_channels)).type(torch.float32)
        delta_tensor[:] = self._delta_init
        self.delta = nn.Parameter(delta_tensor, requires_grad=self._trainable)

        root_tensor = torch.zeros((num_channels)).type(torch.float32)
        root_tensor[:] = self._root_init
        self.root = nn.Parameter(root_tensor, requires_grad=self._trainable)

        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(
                coeff_init=self._smooth_coef,
                per_channel=self._per_channel_smooth_coef,
                trainable=True)
            self.ema.build(num_channels)
        else:
            # TODO: implement simple RNN here
            pass

    def forward(self, x):
        alpha = torch.minimum(self.alpha, torch.ones_like(self.alpha))
        root = torch.maximum(self.root, torch.ones_like(self.root))

        ema_smoother = self.ema(x, x[:,:,0])
        one_over_root = 1. / root
        output = (x.permute(0,2,1) / (self._floor + ema_smoother) ** alpha + self.delta)\
                ** one_over_root - self.delta ** one_over_root
        return output.permute(0,2,1)




  # def call(self, inputs):
  #   alpha = tf.math.minimum(self.alpha, 1.0)
  #   root = tf.math.maximum(self.root, 1.0)
  #   ema_smoother = self.ema(inputs, initial_state=tf.gather(inputs, 0, axis=1))
  #   one_over_root = 1. / root
  #   output = ((inputs / (self._floor + ema_smoother)**alpha + self.delta)
  #             **one_over_root - self.delta**one_over_root)
  #   return output