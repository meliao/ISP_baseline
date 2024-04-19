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


def _get_trainer(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    model: models.DeterministicModel,
) -> trainers.DeterministicTrainer:
    """Initializes a trainer object for the given model.

    Args:
        init_value (float): Initial learning rate.
        transition_steps (int): Number of steps over which the learning rate decays.
        decay_rate (float): Rate at which the learning rate decays.
        model (models.DeterministicModel): The model to train.
    """
    trainer = trainers.DeterministicTrainer(
        model=model,
        rng=jax.random.PRNGKey(42),
        optimizer=optax.adam(
            learning_rate=optax.exponential_decay(
                init_value=init_value,
                transition_steps=transition_steps,
                decay_rate=decay_rate,
                staircase=True,
            ),
        ),
    )

    return trainer


def train_model(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    batch_size: int,
    model: models.DeterministicModel,
    num_train_steps: int,
    workdir: str,
    eta_train: np.ndarray,
    scatter_train: np.ndarray,
    eta_eval: np.ndarray = None,
    scatter_eval: np.ndarray = None,
) -> None:
    """This function is responsible for taking pre-loaded and standardized data, taking pre-compiled model, and training the model.

    Args:
        init_value (float): Initial learning rate.
        transition_steps (int): Number of steps before the next learning rate decay.
        decay_rate (float): Rate at which the learning rate decays.
        batch_size (int): Batch size for training.
        model (models.DeterministicModel): Wide-band butterfly network.
        num_train_steps (int): Number of training steps. Should = n_train / batch_size * desired_num_epochs.
        workdir (str): Directory to save training checkpoints.
        eta_train (np.ndarray): Training data for eta.
        scatter_train (np.ndarray): Training data for scatter.
        eta_eval (np.ndarray, Optional): Evaluation data for eta. If not specified, validation is performed on train set.
        scatter_eval (np.ndarray, Optional): Evaluation data for scatter. If not specified, validation is performed on train set.
    """

    # Training dataset
    dict_data = {"eta": eta_train}
    dict_data["scatter"] = scatter_train
    dataset = tf.data.Dataset.from_tensor_slices(dict_data)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.as_numpy_iterator()

    # Validation dataset
    if eta_eval is not None:
        dict_data = {"eta": eta_eval}
        dict_data["scatter"] = scatter_eval
        eval_dataset = tf.data.Dataset.from_tensor_slices(dict_data)
        eval_dataset = eval_dataset.shuffle(buffer_size=1000)
        eval_dataset = eval_dataset.batch(batch_size)
        eval_dataset = eval_dataset.repeat()
        eval_dataset = eval_dataset.prefetch(tf.data.AUTOTUNE)
        eval_dataset = eval_dataset.as_numpy_iterator()
    else:
        eval_dataset = dataset

    ckpt_interval = 2000  # @param
    max_ckpt_to_keep = 3  # @param

    trainer = _get_trainer(
        init_value=init_value,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        model=model,
    )

    templates.run_train(
        train_dataloader=dataset,
        trainer=trainer,
        workdir=workdir,
        total_train_steps=num_train_steps,
        metric_writer=metric_writers.create_default_writer(workdir, asynchronous=False),
        metric_aggregation_steps=100,
        eval_dataloader=eval_dataset,
        eval_every_steps=1000,
        num_batches_per_eval=2,
        callbacks=(
            templates.TqdmProgressBar(
                total_train_steps=num_train_steps,
                train_monitors=("train_loss",),
                eval_monitors=("eval_rrmse_mean",),
            ),
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=ckpt_interval, max_to_keep=max_ckpt_to_keep
                ),
            ),
        ),
    )
