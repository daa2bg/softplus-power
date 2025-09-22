# scripts/plot_activations.py
# Compare Softplus-Power presets vs ReLU / SiLU / Softplus with matplotlib.
# Usage:
#   python scripts/plot_activations.py --xlim -4 4 --jitter 0.10 --preset v1_simple --save figs

from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Reference activations (NumPy) ----------
def softplus_np(z: np.ndarray) -> np.ndarray:
    # numerically stable log(1+exp(z))
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def silu(x: np.ndarray) -> np.ndarray:
    # swish/silu: x * sigmoid(x)
    return x / (1.0 + np.exp(-x))

# ---------- Softplus-Power (NumPy) ----------
def softplus_power_np(x: np.ndarray, alpha: float, beta: float, theta: float, p: float, delta: float) -> np.ndarray:
    z  = beta * (x - theta)
    sp = softplus_np(z)
    return alpha * (sp ** p) - delta

# Presets (initializations). These are *starts*; in training they’d be learnable.
PRESETS = {
    "generic":      dict(alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0),
    "retinal_on":   dict(alpha=1.2, beta=1.0, theta=-0.2, p=1.0, delta=0.0),
    "retinal_off":  dict(alpha=1.2, beta=1.0, theta=+0.2, p=1.0, delta=0.1),
    "lgn_opponent": dict(alpha=1.0, beta=1.4, theta= 0.0, p=1.2, delta=0.0),
    "v1_simple":    dict(alpha=1.0, beta=1.6, theta= 0.1, p=1.1, delta=0.0),
    "v1_complex":   dict(alpha=1.0, beta=1.2, theta= 0.0, p=0.8, delta=0.0),
    "inhibitory":   dict(alpha=0.9, beta=1.5, theta= 0.2, p=1.0, delta=0.2),
}

def apply_jitter(params: dict, jitter: float, rng: np.random.Generator) -> dict:
    """Small heterogeneity at init: mult. noise on alpha/beta; add. noise on theta/p/delta.
       Keeps p >= 0.5 and delta >= 0."""
    if jitter <= 0:
        return params
    out = params.copy()
    mul = lambda v: v * (1.0 + rng.uniform(-jitter, jitter))
    add = lambda v: v + rng.uniform(-jitter, jitter)
    out["alpha"] = mul(out["alpha"])
    out["beta"]  = mul(out["beta"])
    out["theta"] = add(out["theta"])
    out["p"]     = max(0.5, add(out["p"]))
    out["delta"] = max(0.0, add(out["delta"]))
    return out

# ---------- Plot helpers ----------
def plot_presets(ax, xs: np.ndarray):
    for name, prm in PRESETS.items():
        ys = softplus_power_np(xs, **prm)
        ax.plot(xs, ys, label=name)
    ax.set_title("Softplus-Power presets")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()

def plot_derivatives(ax, xs: np.ndarray, eps: float = 1e-3):
    for name, prm in PRESETS.items():
        y1 = softplus_power_np(xs + eps, **prm)
        y0 = softplus_power_np(xs - eps, **prm)
        dydx = (y1 - y0) / (2 * eps)
        ax.plot(xs, dydx, label=name)
    ax.set_title("Derivatives (gain curves)")
    ax.set_xlabel("x"); ax.set_ylabel("df/dx"); ax.grid(True); ax.legend()

def plot_baselines(ax, xs: np.ndarray):
    ax.plot(xs, relu(xs), label="ReLU")
    ax.plot(xs, silu(xs), label="SiLU/Swish")
    ax.plot(xs, softplus_np(xs), label="Softplus")
    ax.set_title("Common activations")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()

def plot_overlay(ax, xs: np.ndarray, preset_name: str, n_jitter: int, jitter: float, seed: int = 123):
    rng = np.random.default_rng(seed)
    base = PRESETS[preset_name]
    for i in range(n_jitter):
        prm = apply_jitter(base, jitter=jitter, rng=rng)
        ax.plot(xs, softplus_power_np(xs, **prm), alpha=0.6)
    ax.plot(xs, softplus_power_np(xs, **base), "k--", linewidth=2, label=f"base {preset_name}")
    ax.set_title(f"Effect of jitter on {preset_name} (jitter={jitter})")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, default=-4.0)
    ap.add_argument("--xmax", type=float, default=+4.0)
    ap.add_argument("--points", type=int, default=400)
    ap.add_argument("--preset", type=str, default="v1_simple",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--n_jitter", type=int, default=6)
    ap.add_argument("--save", type=str, default="", help="Folder to save PNGs; if empty, just show()")
    args = ap.parse_args()

    xs = np.linspace(args.xmin, args.xmax, args.points)

    # 1) Presets vs each other
    fig1, ax1 = plt.subplots(figsize=(9,6))
    plot_presets(ax1, xs)

    # 2) Derivatives (gain)
    fig2, ax2 = plt.subplots(figsize=(9,6))
    plot_derivatives(ax2, xs)

    # 3) Baselines (ReLU/SiLU/Softplus)
    fig3, ax3 = plt.subplots(figsize=(9,6))
    plot_baselines(ax3, xs)

    # 4) Overlay: chosen preset + jittered variants
    fig4, ax4 = plt.subplots(figsize=(9,6))
    plot_overlay(ax4, xs, args.preset, n_jitter=args.n_jitter, jitter=args.jitter)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        fig1.savefig(os.path.join(args.save, "softplus_power_presets.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(os.path.join(args.save, "softplus_power_derivatives.png"), dpi=150, bbox_inches="tight")
        fig3.savefig(os.path.join(args.save, "common_activations.png"), dpi=150, bbox_inches="tight")
        fig4.savefig(os.path.join(args.save, f"jitter_{args.preset}.png"), dpi=150, bbox_inches="tight")
        print(f"Saved figures to: {os.path.abspath(args.save)}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
