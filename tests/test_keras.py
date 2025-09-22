import tensorflow as tf
from softplus_power import SoftplusPowerKeras, CalibratedSoftplusPowerKeras

def test_keras_layers_build_and_call():
    x = tf.random.normal([4, 10])
    layer = SoftplusPowerKeras(alpha=1.0, beta=1.0, theta=0.0, p=1.0, delta=0.0)
    y = layer(x)
    assert y.shape == x.shape

    layer2 = CalibratedSoftplusPowerKeras()
    y2 = layer2(x)
    assert y2.shape == x.shape

def test_keras_serialization():
    layer = SoftplusPowerKeras(alpha=1.2, beta=0.5, theta=0.1, p=1.3, delta=0.2)
    cfg = layer.get_config()
    new_layer = SoftplusPowerKeras.from_config(cfg)
    x = tf.ones([2, 3])
    _ = new_layer(x)
