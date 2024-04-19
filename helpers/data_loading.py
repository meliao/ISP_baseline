# Stock python packages
import functools
import os
from typing import Tuple
import logging

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


def _load_eta_scatter_from_dir_Borong(
    dir: str, L: int, s: int, truncate_num: int = None
) -> Tuple[np.ndarray]:
    """Load eta and scatter data from a given directory.

    Args:
        dir (str): Directory to load data from.
        L (int): Number of levels.
        s (int): Other data size parameter. Computational domain has (2^L) * s points along one edge.

    Returns:
        Tuple[np.ndarray]: Tuple of the loaded data. Has the following arrays:
            scatter: Input data. Has shape (n_samples, 2, n_pixels * n_pixels, n_freqs)
            eta: Output data. Has shape (n_samples, n_pixels, n_pixels)
    """

    idx_flatten_to_morton = morton.flatten_to_morton_indices(L, s)
    neta = (2**L) * s

    # If truncate_num is not specified, use the number of samples in the data
    if truncate_num is None:
        with h5py.File(f"{dir}/eta.h5", "r") as f:
            eta = f[list(f.keys())[0]][:].reshape(-1, neta, neta).astype("float32")
            truncate_num = eta.shape[0]

    with h5py.File(f"{dir}/eta.h5", "r") as f:
        # Read eta data, apply Gaussian blur, and reshape
        eta_re = f[list(f.keys())[0]][:truncate_num, :].reshape(-1, neta, neta)

    with h5py.File(f"{dir}/scatter.h5", "r") as f:
        keys = natsort.natsorted(f.keys())

        # Process real part of scatter data
        tmp1 = f[keys[3]][:truncate_num, :]
        tmp2 = f[keys[4]][:truncate_num, :]
        tmp3 = f[keys[5]][:truncate_num, :]
        scatter_re = np.stack((tmp3, tmp2, tmp1), axis=-1)
        scatter_re = scatter_re[:, idx_flatten_to_morton, :]

        # Process imaginary part of scatter data
        tmp1 = f[keys[0]][:truncate_num, :]
        tmp2 = f[keys[1]][:truncate_num, :]
        tmp3 = f[keys[2]][:truncate_num, :]
        scatter_im = np.stack((tmp3, tmp2, tmp1), axis=-1)
        scatter_im = scatter_im[:, idx_flatten_to_morton, :]

        # Combine real and imaginary parts
        scatter = np.stack((scatter_re, scatter_im), axis=1).astype("float32")

    return (scatter, eta_re)


def load_data_from_dir(
    dir: str,
    standardize_eta_bool: bool,
    L: int,
    s: int,
    blur_sigma: float = None,
    eta_mean: np.ndarray = None,
    eta_std: np.ndarray = None,
    scatter_means: np.ndarray = None,
    scatter_stds: np.ndarray = None,
    truncate_num: int = None,
) -> Tuple[np.ndarray]:
    """Load data from a given directory.
    The inputs (scatter) will always be standardized.
    The outputs (eta) will be standardized if standardize_eta_bool is True.
    The means and standard deviations used for standardization can be specified. If not specified,
    they will be computed from the data.

    Args:
        dir (str): Directory to load data from.
        standardize_eta_bool (bool): Whether to standardize the eta.
        L (int): Number of levels.
        s (int): Other data size parameter. Computational domain has (2^L) * s points along one edge.
        eta_mean (np.ndarray, Optional): Mean of the eta data. Defaults to None.
        eta_std (np.ndarray, Optional): Standard deviation of the eta data. Defaults to None.
        scatter_means (np.ndarray, Optional): Mean of the scatter data. Should have shape (n_freqs).
            Defaults to None.
        scatter_stds (np.ndarray, Optional): Standard deviation of the scatter data. Should have shape (n_freqs).
            Defaults to None.

    Returns:
        Tuple[np.ndarray]: Tuple of the loaded data. Has the following arrays:
            scatter: Input data. Has shape (n_samples, 2, n_pixels * n_pixels, n_freqs)
            eta: Output data. Has shape (n_samples, n_pixels, n_pixels)
            eta_mean: Mean of the eta data. Has shape (1,)
            eta_std: Standard deviation of the eta data. Has shape (1,)
            scatter_means: Mean of the scatter data. Has shape ( n_freqs,)
            scatter_stds: Standard deviation of the scatter data. Has shape (n_freqs,)
    """
    print("Loading data from directory: ", dir)

    if "eta.h5" in os.listdir(dir):
        # Borong's directory structure
        scatter_pre, eta_pre = _load_eta_scatter_from_dir_Borong(
            dir, L, s, truncate_num
        )
    else:
        raise ValueError("TODO: implement loading for our directory structure")

    n_freqs = scatter_pre.shape[-1]

    # Blur the eta if a blur parameter is specified
    if blur_sigma is not None:
        print("Blurring eta data with sigma = ", blur_sigma)
        blur_fn = lambda x: gaussian_filter(x, sigma=blur_sigma)
        eta_pre = np.stack(
            [blur_fn(eta_pre[i, :, :]) for i in range(eta_pre.shape[0])]
        ).astype("float32")

    # Compute the means and standard deviations, and standardize the data if applicable
    eta_mean_out = np.mean(eta_pre)
    eta_std_out = np.std(eta_pre)

    if eta_mean is None and eta_std is None:
        # If means and stds are not specified, set them to the ones we just measured.
        eta_mean = eta_mean_out
        eta_std = eta_std_out

    if standardize_eta_bool:
        eta_pre -= eta_mean
        eta_pre /= eta_std

    # Compute the means and standard deviations of the scatter data
    scatter_means_out = []
    scatter_stds_out = []
    for i in range(n_freqs):
        scatter_means_out.append(np.mean(scatter_pre[:, :, :, i]))
        scatter_stds_out.append(np.std(scatter_pre[:, :, :, i]))
    scatter_means_out = np.array(scatter_means_out)
    scatter_stds_out = np.array(scatter_stds_out)

    assert scatter_means_out.shape == (n_freqs,)

    if scatter_means is None and scatter_stds is None:
        # If means and stds are not specified, set them to the ones we just measured.
        scatter_means = scatter_means_out
        scatter_stds = scatter_stds_out

    assert scatter_means.shape == (
        n_freqs,
    ), f"scatter_means.shape: {scatter_means.shape} vs. (n_freqs,): {n_freqs}"

    # Standardize the scatter data
    for i in range(n_freqs):
        scatter_pre[:, :, :, i] -= scatter_means[i]
        scatter_pre[:, :, :, i] /= scatter_stds[i]

    return (
        scatter_pre,
        eta_pre,
        eta_mean_out,
        eta_std_out,
        scatter_means_out,
        scatter_stds_out,
    )
