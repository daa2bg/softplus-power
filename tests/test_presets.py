import torch
import tensorflow as tf
from softplus_power import CalibratedSoftplusPowerTorch, CalibratedSoftplusPowerKeras

def test_torch_presets_forward():
    x = torch.randn(8, 16)
    for preset in ["generic","retinal_on","retinal_off","lgn_opponent","v1_simple","v1_complex","inhibitory"]:
        act = CalibratedSoftplusPowerTorch(preset=preset, jitter=0.05)
        y = act(x)
        assert y.shape == x.shape

def test_keras_presets_forward():
    x = tf.random.normal([8, 16])
    for preset in ["generic","retinal_on","retinal_off","lgn_opponent","v1_simple","v1_complex","inhibitory"]:
        act = CalibratedSoftplusPowerKeras(preset=preset, jitter=0.05)
        y = act(x)
        assert y.shape == x.shape
