# Stock python packages
import functools
import os
from typing import Tuple
import logging
import time

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


# Define a custom sorting key function
def _get_number_from_filename(filename: str) -> int:
    """Assumes a file has format .*_{number}.h5 and extracts the number"""
    f = filename.split("_")[-1]
    return int(f.split(".")[0])


def load_field_in_hdf5(
    key: str, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> np.ndarray:
    """Loads an individual field to the specified field in a given hdf5 file"""
    if not os.path.exists(fp_out):
        raise FileNotFoundError("Can't load field %s from %s" % (key, fp_out))
    if retries >= 10:
        raise IOError(f"(lfih) Couldn't open file after 10 tries")
    try:
        with h5py.File(fp_out, "a") as hf:
            data_loaded = hf[key][()]
            data = data_loaded[idx_slice]

        return data

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(30)
        return load_field_in_hdf5(key, fp_out, idx_slice, retries + 1)


D_RS = "d_rs"
Q_CART = "q_cart"
SCATTER = "scatter"
ETA = "eta"
KEYS_OUT = [SCATTER, ETA]


def _concat_all_files_in_dir_ours(
    dir: str, truncate_num: int | None = None
) -> Tuple[np.ndarray]:
    """Loads the data from a directory of hdf5 files that we assume to have the same fields,
      as specified by the generate_measurement_files.py script.

    Args:
        dir_name (str): Directory containing all of the files
        truncate_num (int | None, optional): How many samples to load. If set to None, all samples are loaded.
            Defaults to None.

    Returns:
        Tuple[np.ndarray]: Tuple of the loaded data. Has the following arrays:
            scatter: Input data. Has shape (n_samples, n_pixels, n_pixels, n_freqs)
            eta: Output data. Has shape (n_samples, n_pixels, n_pixels)
    """

    # Get the list of files
    file_list = os.listdir(dir)
    file_list = sorted(file_list, key=_get_number_from_filename)

    n_files = len(file_list)

    # Load the first file to get the shapes
    fp_0_meas = os.path.join(dir, file_list[0])
    out_dd = {
        SCATTER: load_field_in_hdf5(D_RS, fp_0_meas),
        ETA: load_field_in_hdf5(Q_CART, fp_0_meas),
    }
    n_samples_0 = out_dd[ETA].shape[0]

    # set truncate_num to infinity if not specified.
    truncate_num = np.inf if truncate_num is None else truncate_num

    # Early exit if we already have enough samples
    if n_samples_0 > truncate_num:
        for k in KEYS_OUT:
            out_dd[k] = out_dd[k][:truncate_num]
        return out_dd[SCATTER], out_dd[ETA]

    for i in range(1, n_files):
        break_bool = False
        fname = file_list[i]
        fp_meas = os.path.join(dir, fname)
        dd_new = {
            SCATTER: load_field_in_hdf5(D_RS, fp_meas),
            ETA: load_field_in_hdf5(Q_CART, fp_meas),
        }

        new_n_samples = dd_new[ETA].shape[0]

        if out_dd[ETA].shape[0] + new_n_samples > truncate_num:
            # In the case that we have to truncate, we first compute
            # how many samples to keep, and then concatenate the contents
            # of dd_new into out_dd
            n_samples_to_keep = truncate_num - out_dd[ETA].shape[0]
            dd_new = {i: dd_new[i][:n_samples_to_keep] for i in KEYS_OUT}
            break_bool = True

        for i in KEYS_OUT:
            out_dd[i] = np.concatenate([out_dd[i], dd_new[i]])
        if break_bool:
            # Break out of the for loop if we have to truncate
            break

    return out_dd[SCATTER], out_dd[ETA]


def _load_eta_scatter_from_dir_ours(
    dir_frmt: str, L: int, s: int, truncate_num: int, wavenumbers: Tuple[str]
) -> Tuple[np.ndarray]:

    idx_flatten_to_morton = morton.flatten_to_morton_indices(L, s)

    n_freqs = len(wavenumbers)

    # OOT 2024-04-19: Sort the wavenumbers to ensure they are entered in decreasing order
    wavenumbers = sorted(wavenumbers, key=float, reverse=True)

    scatter_eta_tuples = [
        _concat_all_files_in_dir_ours(dir_frmt.format(w), truncate_num)
        for w in wavenumbers
    ]
    (
        n_samples,
        n_pixels,
        _,
    ) = scatter_eta_tuples[
        0
    ][0].shape

    # Scatter_all should have shape (n_samples, n_pixels, n_pixels, n_freqs)
    scatter_all = np.concatenate(
        [x[0][:, :, :, None] for x in scatter_eta_tuples], axis=-1
    ).transpose(0,2,1,3) # OOT 2024-04-19: flip the s and r dimensions since we save (r,s) but need (s,r)

    # Eta_all should have shape (n_samples, n_pixels, n_pixels)
    eta_all = scatter_eta_tuples[0][1]

    # Reshape the scatter_all array to have shape (n_samples, 2, n_pixels * n_pixels, n_freqs).
    # Also, we need to re-arrange the flattened axis to follow the "morton" ordering
    scatter_all_out = np.empty(
        (n_samples, 2, n_pixels * n_pixels, n_freqs), dtype=np.float32
    )
    scatter_all_out[:, 0, :, :] = np.real(scatter_all).reshape(
        n_samples, n_pixels * n_pixels, n_freqs
    )
    # Re-arrange the flattened axis to follow the "morton" ordering
    scatter_all_out[:, 0, :, :] = scatter_all_out[:, 0, idx_flatten_to_morton, :]
    scatter_all_out[:, 1, :, :] = np.imag(scatter_all).reshape(
        n_samples, n_pixels * n_pixels, n_freqs
    )
    # Re-arrange the flattened axis to follow the "morton" ordering
    scatter_all_out[:, 1, :, :] = scatter_all_out[:, 1, idx_flatten_to_morton, :]

    return scatter_all_out, eta_all


def load_data_from_dir(
    dir: str,
    standardize_eta_bool: bool,
    L: int,
    s: int,
    wavenumbers: Tuple[int] = None,
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
        wavenumbers (Tuple[int]): Wavenumbers to load.
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

    # Both of these code paths should return scatter_pre with shape (n_samples, 2, n_pixels * n_pixels, n_freqs)
    # and eta_pre with shape (n_samples, n_pixels, n_pixels)
    try:
        if "eta.h5" in os.listdir(dir):
            # Borong's directory structure
            scatter_pre, eta_pre = _load_eta_scatter_from_dir_Borong(
                dir, L, s, truncate_num
            )
        else:
            raise ValueError(
                "Expected directory to contain eta.h5 file or be formattable. Here is dir: ",
                dir,
            )
    except FileNotFoundError:
        scatter_pre, eta_pre = _load_eta_scatter_from_dir_ours(
            dir, L, s, truncate_num, wavenumbers
        )

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
