# Warp CTC

(_linux only_) PyTorch bindings for WarpCTC (supporting `0.4.1`, `1.3.1`, `1.4.0`)

Baidu's [WarpCTC](https://github.com/baidu-research/warp-ctc), is a fast parallel implementation of
Connectionist Temporal Classification (CTC), on both CPU and GPU, written in C++ and CUDA.

It should be noted that from PyTorch 1.0, PyTorch have an
[officially supported CTCLoss function](https://pytorch.org/docs/stable/nn.html#ctcloss). While this is great, we have noticed that it does not converge quite as well as WarpCTC. To make things easier switching between WarpCTC and PyTorch's own CTCLoss, we have made the bindings operate in exactly the same way.

## Why different bindings for different versions

1. PyTorch <= 0.4.1 uses `torch.utils.ffi.create_extension` to build extensions, where the bindings
are written in C. Also, PyTorch 0.4.1 has non-staticmethod `forward` functions defined on autograd functions.
1. Pytorch >=1.0.0 uses `torch.utils.cpp_extension.{BuildExtension,CppExtension}` to build extensions,
where the bindings are written in C++. Also, Pytorch 1.3.1 has staticmethod `forward` functions defined on
autograd functions.

## Docker Images

I have taken offical pytorch images and pre-installed the warp-ctc functionality:

```txt
pytorch/pytorch             =>  asciialex/pytorch
---------------                 -----------------
:1.4-cuda10.1-cudnn7-devel  =>  :1.4-cuda10.1-cudnn7-devel-warp-ctc
```

## Building & Installing

We only consider supporting Linux distributions, and you will need `cmake` installed. This code was tested on Python 3.6.8.

You will first need to build WarpCTC:

```bash
cd warp-ctc
mkdir build && cd build
cmake .. && make
```

Depending on your version of PyTorch, `cd` into the correct bindings directory (`vX_X_X`):

### PyTorch 0.4.1

```bash
cd warp-ctc/bindings/v0_4_1
python3 build.py
```

### PyTorch 1.3.1 / 1.4.0

```bash
cd warp-ctc/bindings/v1_4_0  # or /v1_3_1
python3 setup.py install --user
```

## Running

To use the bindings, add the location of the binding to the `PYTHONPATH`, and export `LD_LIBRARY_PATH`:

```bash
# Bash (remember to replace vX_X_X with the appropriate version)
export PYTHONPATH=/path/to/warp-ctc/warp-ctc/bindings/vX_X_X:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/warp-ctc/warp-ctc/build
```

```python
# python
import pytorch_warpctc

loss_fn = pytorch_warpctc.CTCLoss()
loss = loss_fn(out, y, x_lengths, y_lengths)

# The loss will be a vector of length `batch-size`.
average_loss = grads = grads / grads.size()[0]
```

The following table shows the supported arguments of `pytorch_warpctc.CTCLoss()`, although it mostly follows the official PyTorch documentation, but with one very distinct difference: _WarpCTC accepts the output probabilities, NOT the log softmax that the official PyTorch version does, as WarpCTC does this inside the C++ implementation._

|Argument |Description |Type |Allowed Values | Default |
|---      |---         |---  |---            |---      |
|blank    |The integer representing the blank label.     |`int`|`x>=0`         |\<The size of the output dimension\> + 1|
|reduction |A transformation to be applied on the batch. `"none"` will return an array of length `batch_size`, `"mean"` will return the average of the losses from the loss vector, and `"sum"` will return the sum.|`str` |`"none"`, `"mean"` and `"sum"` |`"mean"`|
|zero_infinity |Whether to zero infinite losses and the associated gradients. Infinite losses mainly occur when the inputs are too short to be aligned to the targets. |`bool` | `True`, `False` | `False`|

CTCLoss(blank=0, reduction='mean', zero_infinity=False)

## Tests

I will create test scripts to test multiple versions, right now only tested with 0.4.1 or 1.3.1.
