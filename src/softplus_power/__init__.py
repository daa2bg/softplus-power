# src/softplus_power/__init__.py
from .numpy import softplus_power_np
from .torch import (
    SoftplusPowerTorch,
    CalibratedSoftplusPowerTorch,
    softplus_power_torch,
)
# Keras/TensorFlow (optional dependency, but we export it if TF is present)
try:
    from .keras import (
        SoftplusPowerKeras,
        CalibratedSoftplusPowerKeras,
        softplus_power_keras,
    )
except Exception as _KERAS_ERR:  # keep the package importable even if TF missing
    SoftplusPowerKeras = None
    CalibratedSoftplusPowerKeras = None
    softplus_power_keras = None
    __KERAS_IMPORT_ERROR__ = _KERAS_ERR  # for debugging

# JAX
from .jax import (
    softplus_power_jax,
    calibrated_softplus_power_params,
    softplus_power_jax_fn,
)

__all__ = [
    # NumPy
    "softplus_power_np",
    # Torch
    "SoftplusPowerTorch",
    "CalibratedSoftplusPowerTorch",
    "softplus_power_torch",
    # Keras (may be None if TF missing)
    "SoftplusPowerKeras",
    "CalibratedSoftplusPowerKeras",
    "softplus_power_keras",
    # JAX
    "softplus_power_jax",
    "calibrated_softplus_power_params",
    "softplus_power_jax_fn",
]
