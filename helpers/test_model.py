# Stock python packages
import functools
import os
from typing import Tuple
import yaml

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


def relative_l2_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred has shape (batch, n_eta, n_eta)
    target has shape (batch, n_eta, n_eta)
    return value has shape (batch,)
    """
    target_nrms = np.linalg.norm(target, axis=(-2, -1))
    diff_nrms = np.linalg.norm(target - pred, axis=(-2, -1))
    return diff_nrms / target_nrms


def parse_val(text_val):
    """Parses a text to int, float, or bool if possible"""
    try:
        return int(text_val)
    except:
        pass
    try:
        return float(text_val)
    except:
        pass
    if text_val in ["True", "true"]:
        return True
    elif text_val in ["False", "false"]:
        return False

    return text_val  # unchanged if no other conversions are possible


def extract_line_by_field(
    file_name: str,
    field: str,
    selection_mode: str = "min",
    verbosity_level: int = 0,
) -> tuple[dict, float]:
    """Take a given field in a file and use it to extract the line containing the min/max value
    Parameters:
        file_name (string/file path): name of the relevant file to retrieve
        field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
        verbosity_level (int): indicate a relative level of outputs
    Return Value:
        line_entry (dict): a lookup-table of the contents in this particular line (to avoid
            concerns about ordering within the header)
        field_value_selected (int/float most likely): the relevant min/max value of the field in question
    """
    with open(file_name, "r") as file:
        file_contents = [line.strip().split("\t") for line in file]
    header = file_contents[0]
    contents = file_contents[1:]

    try:
        field_idx = header.index(field)
    except:
        raise KeyError(f"Unable to locate field '{field}' in the header {header}")
    field_arr = np.array([parse_val(entry[field_idx]) for entry in contents])

    if selection_mode.lower() == "min":
        line_idx = np.argmin(field_arr)
    elif selection_mode.lower() == "max":
        line_idx = np.argmax(field_arr)
    else:
        raise ValueError(
            f"Expected mode keyword as one of ['min', 'max'] to choose the selection direction"
        )
    field_val_selected = field_arr[line_idx]
    line_entry = {
        key: parse_val(contents[line_idx][ki]) for ki, key in enumerate(header)
    }

    return line_entry, field_val_selected


def test_model(
    scatter_test: np.ndarray,
    eta_test: np.ndarray,
    workdir: str,
    results_fp_eval: str,
    core_module: WideBNet.WideBNetModel,
    std_eta: float,
    mean_eta: float,
) -> None:

    ## Find the checkpoint with the lowest validation error.
    best_line, best_val = extract_line_by_field(
        results_fp_eval, "eval_rrmse_mean", selection_mode="min"
    )
    best_step = best_line["step"]
    print(
        f"On the validation set, found best step {best_step} with relative RMSE {best_val}"
    )

    ## Re-load the model with these weights.
    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{workdir}/checkpoints", step=best_step
    )

    inference_fn = trainers.DeterministicTrainer.build_inference_fn(
        trained_state, core_module
    )

    ## Set up test dataset
    test_batch = 100
    test_dataset = tf.data.Dataset.from_tensor_slices((scatter_test, eta_test))
    test_dataset = test_dataset.batch(test_batch)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.as_numpy_iterator()

    validation_errors_rrmse = []
    validation_errors_rapsd = []
    pixelwise_sq_loss_vals = []
    imagewise_rel_loss_vals = []
    relative_l2_error_vals = []
    eta_pred = np.zeros(eta_test.shape)

    ## Compute test statistics
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
        zz = relative_l2_error(pred, true)
        relative_l2_error_vals.append(zz)
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
    imagewise_rel_loss_vec = np.concatenate(imagewise_rel_loss_vals).flatten()
    relative_l2_error_vec = np.concatenate(relative_l2_error_vals).flatten()

    ## Print and save test statistics
    print(
        "Mean pixelwise squared loss on test dataset:", np.mean(pixelwise_sq_loss_vec)
    )

    print(
        "Mean imagewise relative loss on test dataset:", np.mean(imagewise_rel_loss_vec)
    )

    print("Mean relative L2 error on test dataset:", np.mean(relative_l2_error_vec))
    print("Relative L2 error std on test dataset:", np.std(relative_l2_error_vec))

    summary_dict = {
        "mean_pixelwise_sq_loss": np.mean(pixelwise_sq_loss_vec),
        "stddev_pixelwise_sq_loss": np.std(pixelwise_sq_loss_vec),
        "mean_imagewise_rel_loss": np.mean(imagewise_rel_loss_vec),
        "stddev_imagewise_rel_loss": np.std(imagewise_rel_loss_vec),
        "mean_relative_l2_error": np.mean(relative_l2_error_vec),
        "stddev_relative_l2_error": np.std(relative_l2_error_vec),
        "best_step": best_step,
    }

    test_fp_out = os.path.join(workdir, "test_results.yaml")
    with open(test_fp_out, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)

    ## Save the predicted eta field
    np.save(os.path.join(workdir, "eta_pred.npy"), eta_pred)
