# leaf-audio-pytorch
Pytorch port of Google Research's [LEAF Audio paper published at ICLR 2021](https://openreview.net/forum?id=jM76BCb6F9m).

This port is not completely finished, but the `Leaf()` frontend is fully ported over, functional and validated to have 
similar outputs to the original tensorflow implementation. A few small things are missing, such as the `SincNet` and 
`SincNet+` implementations, a few different pooling layers, etc. 

**PLEASE leave issues, pull requests, comments, or anything you find in using this repository that may be of value to 
others who will try to use this.**

### NOTE: I just noticed there is an issue with running this on gpus. Some tensors that are internally created within the model are not being transferred to the GPU when calling model.cuda(). Looking at a fix currently.

## Installation

From the root directory of this repo, run:

```
pip install -e .
```

## Usage

`leaf_audio_pytorch` mirrors it's original respository; imports and arguments are the same.

```python
import leaf_audio_pytorch.frontend as frontend

leaf = frontend.Leaf()
```

## Installation for Developing 
If you are looking to develop on this repo, the `requirements.txt` contains everything needed to run the torch and tf
implementations of leaf audio simultaneously.

**NOTE: There is some weird dependency stuff going on with the original `leaf-audio` repo. Seems like its a 
dependency issue with `lingvo` and `waymo-open-dataset`. These below commands are a workaround.**

Install the packages required:

```
pip install -r requirements.txt --no-deps
```

Install the `leaf-audio` repo from Git SSH:

```
pip install git+ssh://git@github.com/google-research/leaf-audio.git --no-deps
```

Then add the `leaf_audio_pytorch` package as well

```
python setup.py develop
```

At this point everything should be good to go! The scripts in `test/` contains some testing code to validate the torch 
implementation mirrors tf.

## Some Things to Keep in Mind (PLEASE READ)
* When writing this port, I ran a debugger of the torch and tf implementations side by side and validated that each layer
and operation mirrors the tensorflow implementation (to within a few significant digits, i.e. a tensor's values may variate 
  by 0.001). _There is one notable exception:_ The depthwise convolution within the `GaussianLowpass` pooling layer has 
  a larger variation in tensor values, but the ported operation still produces similar outputs. I'm not sure why this 
  operation is producing different values, but i'm currently 
  looking into it. **Please do your own due diligence** in using this port and making sure this works as expected. 
* As of `March 29`, I finished the initial version of the port, but I have not tested `Leaf()` in a traning setting yet. 
Calling `.backward()` on `Leaf()` throws no errors, meaning backprop works as expected. However, I do not
  yet know how this will function during training.
  
* As PyTorch and Tensorflow follow different tensor ordering conventions, `Leaf()` does all of its operations and 
  outputs tensors with channels _first_. 
  
## Reference

All credit and attribution goes to [Neil Zeghidour](https://research.google/people/106382/) and the [Google Research](https://research.google/) team who wrote the paper and created the 
Tensorflow implementation.

Please visit their [GitHub repository](https://github.com/google-research/leaf-audio) and review their [ICLR publication](https://openreview.net/forum?id=jM76BCb6F9m).
