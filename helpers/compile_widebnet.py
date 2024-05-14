# Stock python packages
import functools
import os
from typing import Tuple, List

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

# Extra helper functions from Olivia
# Calculates nfreqs_per_partition automatically
def calc_partition_ranges(L, nu_low, nu_high):
    """Calculates the ranges for each frequency partition of the interval [nu_low, nu_high]
    Caution: the bottom 1/2^(L/2+1) fraction of the overall range will not be included in any partition
    (for L=2 this is 1/4; L=4 this is 1/8; for L=6 this is 1/16 of the interval)

    Supposes that the frequencies lies within [nu_low, nu_high]
    """
    nu_range = nu_high - nu_low
    nu_step  = nu_range / 2**L
    nu_partitions = [
        (nu_low + nu_step * 2**(L-ell-1), nu_low + nu_step * 2**(L-ell))
        for ell in range(0,L//2+1)
    ]
    # Let the lowest partition to extend all the way to nu_low
    # even though technically that's not the recommendation
    nu_lowest_ptn = nu_partitions[-1]
    nu_partitions[-1] = (nu_low, nu_lowest_ptn[1])
    return nu_partitions

def find_nfreqs_per_partition(nu_vals, nu_partitions):
    """Finds the number of frequencies per partition"""
    # helper fn 1
    def is_val_in_range(val, range_min, range_max):
        """Indicate if val in (range_min, range_max].
        Uses an exclusive lower bound and inclusive upper bound (to match the behavior in frequency bands)
        """
        return (val > range_min) and (val <= range_max)
    # helper fn 2
    def num_vals_in_range(vals, range_min, range_max):
        """Finds how many values in a list are contained within a given range"""
        return sum(is_val_in_range(val, range_min, range_max) for val in vals)
    # main portion
    return [
        num_vals_in_range(nu_vals, nu_ptn[0], nu_ptn[1])
        for nu_ptn in nu_partitions
    ]

# Main function
def compile_widebnet(
    L: int,
    s: int,
    r: int,
    input_shape: Tuple[int],
    wavenumber_list_desc: List[float],
    wavenumber_low: float,
    wavenumber_high: float,
    num_resnet: int = 3,
    num_cnn: int = 3,
) -> Tuple[WideBNet.WideBNetModel, models.DeterministicModel]:
    """Compile a WideBNet model with the given parameters.

    Args:
        L (int): Number of levels.
        s (int): Other data size parameter. Computational domain has (2^L) * s points along one edge.
        r (int): Rank of the butterfly factorization.
        input_shape (Tuple[int]): Shape of the input data. Does not include the batch dimension.
        wavenumber_list_desc (List[float]): a list of the input (non-angular) wavenumbers in descending order
        wavenumber_low (float): the lowest wavenumber for the partitioning scheme
        wavenumber_highest (float): the highest wavenumber for the partitioning scheme
        num_resnet (int, Optional): Number of resnet blocks in the model. Defaults to 3.
        num_cnn (int, Optional): Number of CNN blocks in the model. Defaults to 3.

    Returns:
        models.DeterministicModel: The compiled model.
    """
    idx_morton_to_flatten = morton.morton_to_flatten_indices(L, s)
    wn_partitions = calc_partition_ranges(L, wavenumber_low, wavenumber_high)
    nfreqs_per_partition = find_nfreqs_per_partition(wavenumber_list_desc, wn_partitions)

    print(f"Frequencies per partition: {nfreqs_per_partition}")

    core_module = WideBNet.WideBNetModel(
        L=L,
        s=s,
        r=r,
        NUM_RESNET=num_resnet,
        NUM_CNN=num_cnn,
        idx_morton_to_flatten=idx_morton_to_flatten,
        nfreq_ptn=np.array(nfreqs_per_partition),
    )

    Model = models.DeterministicModel(input_shape=input_shape, core_module=core_module)

    return core_module, Model
