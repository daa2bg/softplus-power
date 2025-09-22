from __future__ import annotations
import jax
import jax.numpy as jnp

def softplus_jax(z):
    return jnp.log1p(jnp.exp(-jnp.abs(z))) + jnp.maximum(z, 0.0)

def softplus_power_jax(x, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0):
    z = beta * (x - theta)
    sp = softplus_jax(z)
    return alpha * (sp ** p) - delta

def calibrated_softplus_power_params(alpha0=1.0, beta0=1.0, p0=1.0, theta0=0.0, delta0=0.0):
    # Return raw params suitable for optimization; map to constrained form in the fn closure
    def inv_softplus(y):
        return jnp.log(jnp.maximum(y, 1e-12)) if y <= 1e-6 else jnp.log(jnp.exp(y) - 1.0)
    return dict(
        a_raw=inv_softplus(alpha0),
        b_raw=inv_softplus(beta0),
        p_raw=p0 - 0.5,
        theta=theta0,
        delta=delta0,
    )

def softplus_power_jax_fn(params):
    """Return a function f(x) that uses constrained parameters."""
    a_raw=params["a_raw"]; b_raw=params["b_raw"]; p_raw=params["p_raw"]
    theta=params["theta"]; delta=params["delta"]
    def f(x):
        alpha = jax.nn.softplus(a_raw)
        beta  = jax.nn.softplus(b_raw)
        p     = 0.5 + jax.nn.softplus(p_raw)
        z = beta * (x - theta)
        sp = softplus_jax(z)
        return alpha * (sp ** p) - delta
    return f
