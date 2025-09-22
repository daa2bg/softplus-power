import numpy as np
from softplus_power import softplus_power_np

def test_shapes_and_monotonicity():
    x = np.linspace(-5, 5, 101).astype(np.float32)
    y = softplus_power_np(x, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0)
    assert y.shape == x.shape
    # Softplus is monotonic; raising to p>=0.5 preserves monotonicity
    assert (np.diff(y) >= -1e-6).all()
