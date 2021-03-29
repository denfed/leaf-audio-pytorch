import numpy as np
import torch

import leaf_audio_pytorch.frontend as torch_frontend

import logging
logging.getLogger('tensorflow').disabled = True
np.random.seed(0)
if __name__ == "__main__":
    py_leaf = torch_frontend.Leaf()

    # (batch_size, num_samples, 1)
    test_audio = np.random.random((8,15000,1)).astype(np.float32)

    # convert to channel first for pytorch
    t_audio = torch.Tensor(test_audio).permute(0,2,1)
    print(t_audio)

    # print("after preemp")
    # print(tf_leaf(tf_audio))
    # print(py_leaf(t_audio), py_leaf(t_audio).shape)

    # print("Comparing preemp conv weights")
    # print(py_leaf._preemp_conv.weight.shape)

    # print(py_leaf(t_audio))

    test_pcen = np.random.random((8,94,40)).astype(np.float32)
    test_pcen_t = torch.Tensor(test_pcen).permute(0,2,1)

    from leaf_audio_pytorch import postprocessing as torch_postprocessing

    pcen = torch_postprocessing.PCENLayer(
        alpha=0.96,
        smooth_coef=0.04,
        delta=2.0,
        floor=1e-12,
        trainable=True,
        learn_smooth_coef=True,
        per_channel_smooth_coef=True
    )
    pcen.build(40)

    pcen(test_pcen_t)

