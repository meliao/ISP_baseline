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


def imagewise_rel_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred has shape (batch, n_eta, n_eta)
    target has shape (batch, n_eta, n_eta)
    return value has shape (batch,)
    """
    target_nrms = np.square(np.linalg.norm(target, axis=(-2, -1)))
    diff_nrms = np.square(np.linalg.norm(target - pred, axis=(-2, -1)))
    return diff_nrms / target_nrms


def pixelwise_sq_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred has shape (batch, n_eta, n_eta)
    target has shape (batch, n_eta, n_eta)
    return value has shape (batch,)
    """
    diff_nrms = np.square(np.linalg.norm(target - pred, axis=(-2, -1)))
    return diff_nrms


def test_model(
    scatter_test: np.ndarray,
    eta_test: np.ndarray,
    workdir: str,
    core_module: WideBNet.WideBNetModel,
    std_eta: float,
    mean_eta: float,
) -> None:

    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{workdir}/checkpoints", step=None
    )

    inference_fn = trainers.DeterministicTrainer.build_inference_fn(
        trained_state, core_module
    )

    test_batch = 100
    test_dataset = tf.data.Dataset.from_tensor_slices((scatter_test, eta_test))
    test_dataset = test_dataset.batch(test_batch)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.as_numpy_iterator()

    validation_errors_rrmse = []
    validation_errors_rapsd = []
    pixelwise_sq_loss_vals = []
    imagewise_rel_loss_vals = []
    eta_pred = np.zeros(eta_test.shape)

    rrmse = functools.partial(
        metrics.mean_squared_error,
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )

    b = 0
    for batch in test_dataset:
        pred = inference_fn(batch[0]) * std_eta + mean_eta
        eta_pred[b * test_batch : (b + 1) * test_batch, :, :] = pred
        b += 1
        true = batch[1]
        ## OJM CHANGE: Added the next 4 lines.
        xx = pixelwise_sq_loss(pred, true)
        pixelwise_sq_loss_vals.append(xx)
        yy = imagewise_rel_loss(pred, true)
        imagewise_rel_loss_vals.append(yy)
        validation_errors_rrmse.append(rrmse(pred=pred, true=true))
        for i in range(true.shape[0]):
            validation_errors_rapsd.append(
                np.abs(
                    np.log(
                        rapsd(pred[i], fft_method=np.fft)
                        / rapsd(true[i], fft_method=np.fft)
                    )
                )
            )

    pixelwise_sq_loss_vec = np.concatenate(pixelwise_sq_loss_vals).flatten()
    print(
        "Mean pixelwise squared loss on test dataset:", np.mean(pixelwise_sq_loss_vec)
    )

    imagewise_rel_loss_vec = np.concatenate(imagewise_rel_loss_vals).flatten()
    print(
        "Mean imagewise relative loss on test dataset:", np.mean(imagewise_rel_loss_vec)
    )
