import torch
from softplus_power import SoftplusPowerTorch, CalibratedSoftplusPowerTorch, softplus_power_torch

def test_forward_and_grad():
    x = torch.linspace(-5, 5, 101, dtype=torch.float32, requires_grad=True)
    y = softplus_power_torch(x, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0)
    y.sum().backward()
    assert x.grad is not None

def test_module_script():
    m = SoftplusPowerTorch(1.0, 1.0, 0.0, 1.0, 0.0)
    sm = torch.jit.script(m)  # should script
    x = torch.randn(8, 16)
    y = sm(x)
    assert y.shape == x.shape

def test_calibrated_learns():
    m = CalibratedSoftplusPowerTorch()
    x = torch.randn(32, 8)
    y = m(x).sum()
    y.backward()
    for p in m.parameters():
        assert p.grad is not None
