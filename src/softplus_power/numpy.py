from __future__ import annotations
import numpy as np
from .common import inv_softplus

def softplus_np(x: np.ndarray) -> np.ndarray:
    # stable log(1+exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def softplus_power_np(
    x: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    theta: float = 0.0,
    p: float = 1.0,
    delta: float = 0.0,
) -> np.ndarray:
    z = beta * (x - theta)
    sp = softplus_np(z)
    return alpha * (sp ** p) - delta

def calibrated_params(alpha0=1.0, beta0=1.0, p0=1.0, theta0=0.0, delta0=0.0):
    # For doc/tests: raw params that map to constrained {alpha>0,beta>0,p>=0.5}
    return dict(
        a_raw=inv_softplus(alpha0),
        b_raw=inv_softplus(beta0),
        p_raw=p0 - 0.5,
        theta=theta0,
        delta=delta0,
    )
