# Stock python packages
import functools
import os
from typing import Tuple

# Third party packages
from clu import metric_writers
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp

import h5py
import natsort
import tensorflow as tf
from scipy.ndimage import geometric_transform
from scipy.ndimage import gaussian_filter

from swirl_dynamics import templates
from swirl_dynamics.lib import metrics
from pysteps.utils.spectral import rapsd

# Local repo code
from src import models
from src import trainers
from WideBNetModel import WideBNet, morton


def compile_widebnet(
    L: int,
    s: int,
    r: int,
    input_shape: Tuple[int],
    num_resnet: int = 3,
    num_cnn: int = 3,
) -> Tuple[WideBNet.WideBNetModel, models.DeterministicModel]:
    """Compile a WideBNet model with the given parameters.

    Args:
        L (int): Number of levels.
        s (int): Other data size parameter. Computational domain has (2^L) * s points along one edge.
        r (int): Rank of the butterfly factorization.
        input_shape (Tuple[int]): Shape of the input data. Does not include the batch dimension.
        num_resnet (int, Optional): Number of resnet blocks in the model. Defaults to 3.
        num_cnn (int, Optional): Number of CNN blocks in the model. Defaults to 3.

    Returns:
        models.DeterministicModel: The compiled model.
    """
    idx_morton_to_flatten = morton.morton_to_flatten_indices(L, s)

    core_module = WideBNet.WideBNetModel(
        L=L,
        s=s,
        r=r,
        NUM_RESNET=num_resnet,
        NUM_CNN=num_cnn,
        idx_morton_to_flatten=idx_morton_to_flatten,
    )

    Model = models.DeterministicModel(input_shape=input_shape, core_module=core_module)

    return core_module, Model
