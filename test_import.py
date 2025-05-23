import torch
from quatnet_pytorch_layer import QuatNetPytorchDenseLayer


def test_forward_backward():
    layer = QuatNetPytorchDenseLayer(3, 2, activation="split_tanh")
    x = torch.randn(4, 3, 4, requires_grad=True)
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert out.shape == (4, 2, 4)
    assert x.grad is not None
