from __future__ import annotations
import math

def inv_softplus(y: float) -> float:
    # inverse of softplus for y>0; guarded near 0
    return math.log(max(y, 1e-12)) if y <= 1e-6 else math.log(math.exp(y) - 1.0)

def softplus_stable(x):
    """
    Framework-agnostic numerically-stable softplus: log(1+exp(x)).
    Works for NumPy, Torch, TF, JAX if they provide:
      - abs, maximum, exp, log1p
    We keep this as a reference utility; framework-specific wrappers re-implement
    to keep dtypes/devices native.
    """
    # This file hosts only reference math; actual framework versions live in their modules.
    raise NotImplementedError("Use framework-specific softplus in torch/keras/jax modules.")
