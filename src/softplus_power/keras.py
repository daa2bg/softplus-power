from __future__ import annotations
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras import layers
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

def _apply_jitter(d: dict, jitter: float):
    if jitter <= 0: return d
    import random
    out = d.copy()
    mul = lambda v: v * (1.0 + (random.uniform(-jitter, jitter)))
    add = lambda v: v + (random.uniform(-jitter, jitter))
    out["alpha0"] = mul(out["alpha0"])
    out["beta0"]  = mul(out["beta0"])
    out["theta0"] = add(out["theta0"])
    out["p0"]     = max(0.5, add(out["p0"]))
    out["delta0"] = max(0.0, add(out["delta0"]))
    return out


def softplus_tf(z):
    return tfm.log1p(tfm.exp(-tfm.abs(z))) + tfm.maximum(z, 0.0)

def softplus_power_keras(x, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0):
    z = beta * (x - theta)
    sp = softplus_tf(z)
    return alpha * tf.pow(sp, p) - delta

class SoftplusPowerKeras(layers.Layer):
    """Fixed Softplus^p as a Keras layer."""
    def __init__(self, alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha=float(alpha); self.beta=float(beta); self.theta=float(theta); self.p=float(p); self.delta=float(delta)

    def call(self, inputs):
        return softplus_power_keras(inputs, self.alpha, self.beta, self.theta, self.p, self.delta)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(alpha=self.alpha, beta=self.beta, theta=self.theta, p=self.p, delta=self.delta))
        return cfg
class CalibratedSoftplusPowerKeras(layers.Layer):
    """
    Trainable, parameter-safe: alpha=softplus(a_raw), beta=softplus(b_raw), p=0.5+softplus(p_raw).
    preset: {generic, retinal_on, retinal_off, lgn_opponent, v1_simple, v1_complex, inhibitory}
    jitter: float in [0, 0.5] for initial heterogeneity.
    """
    def __init__(self, alpha0=1.0, beta0=1.0, theta0=0.0, p0=1.0, delta0=0.0, **kwargs):

        preset = kwargs.pop("preset", "generic")
        jitter = float(kwargs.pop("jitter", 0.0))

        super().__init__(**kwargs)  

        base = _PRESETS.get(preset, _PRESETS["generic"]).copy()
        if (alpha0, beta0, theta0, p0, delta0) != (1.0, 1.0, 0.0, 1.0, 0.0):
            base.update(dict(alpha0=alpha0, beta0=beta0, theta0=theta0, p0=p0, delta0=delta0))
        base = _apply_jitter(base, jitter)

        # Save for get_config (so layer is serializable & reproducible)
        self._preset = preset
        self._jitter = jitter
        self._init_vals = base

        self.a_raw_init = inv_softplus(base["alpha0"])
        self.b_raw_init = inv_softplus(base["beta0"])
        self.p_raw_init = base["p0"] - 0.5
        self.theta_init = base["theta0"]
        self.delta_init = base["delta0"]

    def build(self, _):
        self.a_raw = self.add_weight(name="a_raw", shape=(), dtype=self.dtype if hasattr(self,"dtype") else tf.float32,
                                     initializer=tf.keras.initializers.Constant(self.a_raw_init), trainable=True)
        self.b_raw = self.add_weight(name="b_raw", shape=(), dtype=self.dtype if hasattr(self,"dtype") else tf.float32,
                                     initializer=tf.keras.initializers.Constant(self.b_raw_init), trainable=True)
        self.p_raw = self.add_weight(name="p_raw", shape=(), dtype=self.dtype if hasattr(self,"dtype") else tf.float32,
                                     initializer=tf.keras.initializers.Constant(self.p_raw_init), trainable=True)
        self.theta = self.add_weight(name="theta", shape=(), dtype=self.dtype if hasattr(self,"dtype") else tf.float32,
                                     initializer=tf.keras.initializers.Constant(self.theta_init), trainable=True)
        self.delta = self.add_weight(name="delta", shape=(), dtype=self.dtype if hasattr(self,"dtype") else tf.float32,
                                     initializer=tf.keras.initializers.Constant(self.delta_init), trainable=True)

    def call(self, inputs):
        alpha = tf.nn.softplus(self.a_raw)
        beta  = tf.nn.softplus(self.b_raw)
        p     = 0.5 + tf.nn.softplus(self.p_raw)
        z = beta * (inputs - self.theta)
        sp = softplus_tf(z)
        return alpha * tf.pow(sp, p) - self.delta

    def get_config(self):
        cfg = super().get_config()
        # Reconstruct a config that round-trips via from_config
        if self.built:
            alpha0 = float(tf.nn.softplus(self.a_raw).numpy())
            beta0  = float(tf.nn.softplus(self.b_raw).numpy())
            p0     = float((0.5 + tf.nn.softplus(self.p_raw)).numpy())
            theta0 = float(self.theta.numpy())
            delta0 = float(self.delta.numpy())
        else:
            alpha0 = self._init_vals["alpha0"]
            beta0  = self._init_vals["beta0"]
            p0     = self._init_vals["p0"]
            theta0 = self._init_vals["theta0"]
            delta0 = self._init_vals["delta0"]

        cfg.update(dict(
            alpha0=alpha0, beta0=beta0, p0=p0, theta0=theta0, delta0=delta0,
            preset=self._preset, jitter=self._jitter,
        ))
        return cfg
