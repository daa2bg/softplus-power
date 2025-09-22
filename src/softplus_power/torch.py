from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .common import inv_softplus

_PRESETS = {
    "generic":      dict(alpha0=1.0, beta0=1.0, theta0=0.0, p0=1.0, delta0=0.0),
    "retinal_on":   dict(alpha0=1.2, beta0=1.0, theta0=-0.2, p0=1.0, delta0=0.0),
    "retinal_off":  dict(alpha0=1.2, beta0=1.0, theta0=+0.2, p0=1.0, delta0=0.1),
    "lgn_opponent": dict(alpha0=1.0, beta0=1.4, theta0= 0.0, p0=1.2, delta0=0.0),
    "v1_simple":    dict(alpha0=1.0, beta0=1.6, theta0= 0.1, p0=1.1, delta0=0.0),
    "v1_complex":   dict(alpha0=1.0, beta0=1.2, theta0= 0.0, p0=0.8, delta0=0.0),
    "inhibitory":   dict(alpha0=0.9, beta0=1.5, theta0= 0.2, p0=1.0, delta0=0.2),
}

def _apply_jitter(d: dict, jitter: float, torch_random=True):
    if jitter <= 0: return d
    import random
    out = d.copy()
    # multiplicative jitter for alpha,beta; additive for others
    mul = lambda v: v * (1.0 + (random.uniform(-jitter, jitter)))
    add = lambda v: v + (random.uniform(-jitter, jitter))
    out["alpha0"] = mul(out["alpha0"])
    out["beta0"]  = mul(out["beta0"])
    out["theta0"] = add(out["theta0"])
    out["p0"]     = max(0.5, add(out["p0"]))   # keep p>=0.5
    out["delta0"] = max(0.0, add(out["delta0"]))
    return out


def softplus_torch(z: Tensor) -> Tensor:
    # stable log(1+exp(z))
    return torch.log1p(torch.exp(-torch.abs(z))) + torch.maximum(z, torch.zeros_like(z))

@torch.jit.script
def _softplus_power_kernel(x: Tensor, alpha: float, beta: float, theta: float, p: float, delta: float) -> Tensor:
    z = beta * (x - theta)
    sp = torch.log1p(torch.exp(-torch.abs(z))) + torch.maximum(z, torch.zeros_like(z))
    return alpha * (sp.pow(p)) - delta

def softplus_power_torch(x: Tensor, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0) -> Tensor:
    return _softplus_power_kernel(x, float(alpha), float(beta), float(theta), float(p), float(delta))

class SoftplusPowerTorch(nn.Module):
    """
    Fixed-parameter Softplus^p activation (no parameter learning).
    """
    def __init__(self, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("beta",  torch.tensor(float(beta)))
        self.register_buffer("theta", torch.tensor(float(theta)))
        self.register_buffer("p",     torch.tensor(float(p)))
        self.register_buffer("delta", torch.tensor(float(delta)))

    def forward(self, x: Tensor) -> Tensor:
        return _softplus_power_kernel(x, float(self.alpha.item()), float(self.beta.item()),
                                         float(self.theta.item()), float(self.p.item()),
                                         float(self.delta.item()))

class CalibratedSoftplusPowerTorch(nn.Module):
    """
    Trainable, parameter-safe variant with optional neuron presets.
    preset: one of {generic, retinal_on, retinal_off, lgn_opponent, v1_simple, v1_complex, inhibitory}
    jitter: float in [0, 0.5] to randomize initial params (population heterogeneity)
    """
    def __init__(self, alpha0=1.0, beta0=1.0, theta0=0.0, p0=1.0, delta0=0.0,
                 preset: str = "generic", jitter: float = 0.0):
        super().__init__()
        base = _PRESETS.get(preset, _PRESETS["generic"]).copy()
        # allow explicit overrides to win if user passes non-defaults
        if (alpha0, beta0, theta0, p0, delta0) != (1.0, 1.0, 0.0, 1.0, 0.0):
            base.update(dict(alpha0=alpha0, beta0=beta0, theta0=theta0, p0=p0, delta0=delta0))
        base = _apply_jitter(base, jitter)

        self.a_raw = nn.Parameter(torch.tensor(inv_softplus(base["alpha0"]), dtype=torch.float32))
        self.b_raw = nn.Parameter(torch.tensor(inv_softplus(base["beta0"]),  dtype=torch.float32))
        self.p_raw = nn.Parameter(torch.tensor(base["p0"] - 0.5, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(base["theta0"], dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(base["delta0"], dtype=torch.float32))


    def forward(self, x: Tensor) -> Tensor:
        alpha = F.softplus(self.a_raw)
        beta  = F.softplus(self.b_raw)
        p     = 0.5 + F.softplus(self.p_raw)
        z = beta * (x - self.theta)
        sp = softplus_torch(z)
        return alpha * (sp.pow(p)) - self.delta

    def extra_repr(self) -> str:
        with torch.no_grad():
            alpha = F.softplus(self.a_raw).item()
            beta  = F.softplus(self.b_raw).item()
            p     = (0.5 + F.softplus(self.p_raw)).item()
            theta = float(self.theta.item())
            delta = float(self.delta.item())
        return f"alpha={alpha:.4g}, beta={beta:.4g}, theta={theta:.4g}, p={p:.4g}, delta={delta:.4g}"
