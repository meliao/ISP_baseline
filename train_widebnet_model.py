# Stock python packages
import argparse
import os
import logging
import sys
from typing import Any, Dict
import hashlib

# Third party packages
import numpy as np
import jax
import wandb

# Local repo code
from helpers.compile_widebnet import compile_widebnet
from helpers.train_model import train_model
from helpers.test_model import test_model
from helpers.data_loading import load_data_from_dir


def hash_dict(dictionary: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Data paths and options
    parser.add_argument(
        "-train_data_dir", help="Directory containing training data", required=True
    )
    parser.add_argument(
        "-val_data_dir", help="Directory containing validation data", required=False
    )
    parser.add_argument(
        "-test_data_dir", help="Directory containing test data", required=True
    )
    parser.add_argument("-wavenumbers", nargs="+", help="Wavenumbers to use")
    parser.add_argument("-wavenumber_low", type=float, default=0)
    parser.add_argument("-wavenumber_high", type=float)
    parser.add_argument(
        "-truncate_num", help="Number of samples to truncate to", type=int, default=None
    )
    parser.add_argument(
        "-truncate_num_val",
        help="Number of samples to truncate the validation set to",
        type=int,
        default=None,
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
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use wandb for logging"
    )
    parser.add_argument(
        "-wandb_project_name",
        help="Name of the wandb project",
        default="2024-04-22_widebnet_hyperparam_experiments",
    )
    parser.add_argument(
        "-wandb_entity", help="Wandb entity", default="recursive-linearization"
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:

    # Set up workdir
    workdir = os.path.abspath(args.workdir)
    print("Workdir:", workdir)
    os.makedirs(workdir, exist_ok=True)

    fp_results = os.path.join(workdir, "results.txt")

    args.wavenumber_high = args.wavenumber_high \
        if args.wavenumber_high is not None else max(map(float, args.wavenumbers))
    # Sort the wavenumbers in descending order
    wavenumber_list_desc = sorted(list(map(float, args.wavenumbers)), key=float, reverse=True)

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
        wavenumbers=args.wavenumbers,
        truncate_num=args.truncate_num,
    )
    print("Scatter train shape:", scatter_train.shape)
    print("Eta train shape:", eta_train.shape)

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
            wavenumbers=args.wavenumbers,
            truncate_num=args.truncate_num_val,
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
        wavenumbers=args.wavenumbers,
        truncate_num=args.truncate_num,
    )

    # Compile the model
    core_module, Model = compile_widebnet(
        L=args.L, s=args.s, r=args.r, input_shape=scatter_train[0].shape,
        wavenumber_list_desc=wavenumber_list_desc,
        wavenumber_low=args.wavenumber_low,
        wavenumber_high=args.wavenumber_high,
        # NOTE: num_resnet and num_cnn will default to 3 each
    )
    rng = jax.random.PRNGKey(888)
    params = Model.initialize(rng)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Number of trainable parameters:", param_count)

    # Train the model
    print(f"Training the model...")
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
        use_wandb=args.use_wandb,
        results_fp=fp_results,
    )

    # Test the model
    print(f"Testing the model...")
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

    id_hash = hash_dict(vars(args))
    args.hash = id_hash

    if args.use_wandb:
        # print(vars(a))
        # print(id_hash)
        with wandb.init(
            id=id_hash,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            mode="online",
            reinit=True,
            resume=None,
            settings=wandb.Settings(start_method="fork"),
        ) as wandbrun:
            main(args)

    else:
        main(args)
