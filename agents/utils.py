import numpy as np
import jax
import chex
from flax import nnx


def layer_init(module: nnx.Module, key: chex.PRNGKey, std: float = np.sqrt(2), bias_value: float = 0):
    """layer init for Conv and Linear"""

    for _, layer in module.iter_modules():
        if not isinstance(layer, (nnx.Conv, nnx.Linear)):
            continue

        key, key1, key2 = jax.random.split(key, 3)

        layer.kernel.value = nnx.initializers.orthogonal(scale=std)(
            key1,
            layer.kernel.value.shape,
            layer.kernel.value.dtype
        )

        layer.bias.value = nnx.initializers.constant(bias_value)(
            key2,
            layer.bias.value.shape,
            layer.bias.value.dtype
        )