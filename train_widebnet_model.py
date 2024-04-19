# Stock python packages
import argparse
import os
import logging
import sys

# Third party packages
import numpy as np
import jax

# Local repo code
from helpers.compile_widebnet import compile_widebnet
from helpers.train_model import train_model
from helpers.test_model import test_model
from helpers.data_loading import load_data_from_dir


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument(
        "-train_data_dir", help="Directory containing training data", required=True
    )
    parser.add_argument(
        "-val_data_dir", help="Directory containing validation data", required=False
    )
    parser.add_argument(
        "-test_data_dir", help="Directory containing test data", required=True
    )

    # Model architecture parameters
    parser.add_argument("-L", help="Number of levels in the model", default=4, type=int)
    parser.add_argument("-s", help="Size parameter for the model", default=5, type=int)
    parser.add_argument(
        "-r", help="Rank of the butterfly factorization", default=3, type=int
    )

    # Optimization parameters
    parser.add_argument(
        "-init_value", help="Initial learning rate", type=float, default=5e-3
    )
    parser.add_argument(
        "-transition_steps",
        help="Number of steps before the learning rate decays",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "-decay_rate",
        help="Rate at which the learning rate decays",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "-batch_size", help="Batch size for training", type=int, default=32
    )
    parser.add_argument(
        "-num_train_steps", help="Number of training steps", type=int, default=98_438
    )

    # Other hyperparams
    parser.add_argument(
        "-blur_sigma", help="Sigma for Gaussian blur", type=float, default=0.75
    )
    parser.add_argument(
        "--blur_test_eta",
        help="Whether to blur test eta",
        action="store_true",
        default=False,
    )

    # Output options
    parser.add_argument(
        "-workdir", help="Directory to save model checkpoints and logs", required=True
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:

    # Set up workdir
    workdir = os.path.abspath(args.workdir)
    print("Workdir:", workdir)
    os.makedirs(workdir, exist_ok=True)

    # Load train data
    print("Loading train data")
    (
        scatter_train,
        eta_train,
        eta_mean_train,
        eta_std_train,
        scatter_means_train,
        scatter_stds_train,
    ) = load_data_from_dir(
        dir=args.train_data_dir,
        standardize_eta_bool=True,
        L=args.L,
        s=args.s,
        blur_sigma=args.blur_sigma,
    )
    print("Scatter train shape:", scatter_train.shape)
    print("Eta train shape:", eta_train.shape)

    # # pickle scatter_train and eta_train into the workdir
    # np.save(os.path.join(workdir, "scatter_train.npy"), scatter_train)
    # np.save(os.path.join(workdir, "eta_train.npy"), eta_train)

    # Load validation data if path is specified
    if args.val_data_dir is not None:
        print("Loading validation data")
        (scatter_val, eta_val, _, _, _, _) = load_data_from_dir(
            dir=args.val_data_dir,
            standardize_eta_bool=True,
            L=args.L,
            s=args.s,
            blur_sigma=args.blur_sigma,
            eta_mean=eta_mean_train,
            eta_std=eta_std_train,
            scatter_means=scatter_means_train,
            scatter_stds=scatter_stds_train,
        )
    else:
        scatter_val = None
        eta_val = None

    # Load test data
    print("Loading test data")
    sigma_test = args.blur_sigma if args.blur_test_eta else None
    (scatter_test, eta_test, _, _, _, _) = load_data_from_dir(
        dir=args.test_data_dir,
        standardize_eta_bool=False,
        L=args.L,
        s=args.s,
        blur_sigma=sigma_test,
        eta_mean=eta_mean_train,
        eta_std=eta_std_train,
        scatter_means=scatter_means_train,
        scatter_stds=scatter_stds_train,
    )

    # save scatter_test and eta_test into the workdir
    # np.save(os.path.join(workdir, "scatter_test.npy"), scatter_test)
    # np.save(os.path.join(workdir, "eta_test.npy"), eta_test)
    # Compile the model
    core_module, Model = compile_widebnet(
        L=args.L, s=args.s, r=args.r, input_shape=scatter_train[0].shape
    )
    rng = jax.random.PRNGKey(888)
    params = Model.initialize(rng)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of trainable parameters:", param_count)

    # Train the model
    train_model(
        init_value=args.init_value,
        transition_steps=args.transition_steps,
        decay_rate=args.decay_rate,
        batch_size=args.batch_size,
        model=Model,
        num_train_steps=args.num_train_steps,
        workdir=workdir,
        eta_train=eta_train,
        scatter_train=scatter_train,
        eta_eval=eta_val,
        scatter_eval=scatter_val,
    )

    # Test the model
    test_model(
        scatter_test=scatter_test,
        eta_test=eta_test,
        workdir=workdir,
        core_module=core_module,
        std_eta=eta_std_train,
        mean_eta=eta_mean_train,
    )


if __name__ == "__main__":
    args = setup_args()
    main(args)
