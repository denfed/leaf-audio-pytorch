import convolution as torch_convolution
import frontend as torch_frontend

import sys
sys.path.append('tf/leaf-audio/')

import leaf_audio.convolution as tf_convolution
import leaf_audio.frontend as tf_frontend

import numpy as np
import tensorflow as tf
import torch
from torch import Tensor
np.random.seed(0)
if __name__ == "__main__":
    tf_leaf = tf_frontend.Leaf()

    # (batch_size, num_samples, 1)
    test_audio = np.random.random((8,15000,1)).astype(np.float32)

    # convert to channel first for pytorch
    tf_audio = tf.convert_to_tensor(test_audio, dtype=tf.float32)
    print(tf_audio)

    print(tf_leaf(tf_audio))
