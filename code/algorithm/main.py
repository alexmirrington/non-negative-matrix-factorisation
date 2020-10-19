"""Main entrypoint for model training and evaluation."""
import argparse
import hashlib
import json
import os.path
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import wandb
from datasets import load_data
from factories import ModelFactory, PreprocessorFactory
from loggers import JSONLLogger, StreamLogger, WandbLogger
from termcolor import colored
from utilities import rescale

from config import Dataset, Model, Noise


def main(config: argparse.Namespace):
    """Train or evaluate a dictionary learning model."""
    # Set prng seeds
    np.random.seed(config.seed)

    # Determine dataset directory
    if config.dataset == Dataset.ORL:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "ORL")
        original_shape = (112, 92)
    elif config.dataset == Dataset.YALEB:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "CroppedYaleB")
        original_shape = (192, 168)
    else:
        raise NotImplementedError()
    original_shape = tuple([s // config.reduce for s in original_shape])

    # Create loggers
    loggers = [StreamLogger(), JSONLLogger(Path(config.results_dir) / f"{config.id}.jsonl")]
    if config.wandb:
        loggers.append(WandbLogger())

    # Create preprocessor (noise function)
    print(colored("preprocessor:", attrs=["bold"]))
    preprocessor_factory = PreprocessorFactory()
    preprocessor = preprocessor_factory.create(config)
    print(config.noise)

    # Load dataset
    print(colored("dataset:", attrs=["bold"]))
    noisy_images, clean_images, labels = load_data(
        root=data_dir, reduce=config.reduce, preprocessor=preprocessor
    )
    print(config.dataset)
    if config.scale != 255:
        noisy_images = noisy_images * config.scale / 255
        clean_images = clean_images * config.scale / 255
    print(f"full: {clean_images.shape}")

    # Generate indices to train and test on
    indices = np.random.permutation(labels.shape[0])
    take = int(config.subset * labels.shape[0])
    train_data = noisy_images[:, indices][:, :take]
    test_data = clean_images[:, indices][:, :take]
    test_labels = labels[indices][:take]
    print(
        f"subset: {train_data.shape} using indices",
        f"[{', '.join([str(val) for val in indices[:min(10, take)]])}, ...]",
    )
    # Create model
    print(colored("model:", attrs=["bold"]))
    model_factory = ModelFactory()
    model = model_factory.create(train_data, config)
    print(config.model)

    # Train and evaluate model
    print(colored("training:", attrs=["bold"]))
    model.fit(
        max_iter=config.iterations,
        tol=config.tolerance,
        callback_freq=config.log_step,
        callbacks=loggers,
        clean_data=test_data,
        true_labels=test_labels,
    )

    # Log data samples and model dictionaries
    if config.wandb:
        img_count = 8
        logger = WandbLogger(commit=False)
        test_samples = test_data.T[:img_count].reshape((img_count, *original_shape))
        logger(
            {
                "original": [
                    wandb.Image(sample, caption=f"id: {indices[i]}, class: {test_labels[i]}")
                    for i, sample in enumerate(test_samples)
                ]
            }
        )
        train_samples = train_data.T[:img_count].reshape((img_count, *original_shape))
        logger(
            {
                "noisy": [
                    wandb.Image(sample, caption=f"id: {indices[i]}, class: {test_labels[i]}")
                    for i, sample in enumerate(train_samples)
                ]
            }
        )
        reconstructed_samples = (
            model.reconstructed_data().T[:img_count].reshape((img_count, *original_shape))
        )
        logger(
            {
                "reconstructed": [
                    wandb.Image(sample, caption=f"id: {indices[i]}, class: {test_labels[i]}")
                    for i, sample in enumerate(reconstructed_samples)
                ]
            }
        )
        w_components = model.W.reshape((*original_shape, config.components))[:, :, :img_count]
        logger(
            {
                "w": [
                    wandb.Image(rescale(w_components[:, :, i]), caption=f"component: {i}")
                    for i in range(w_components.shape[2])
                ]
            }
        )
        # Plot the response of the images to each component in W as a heatmap
        h_y, h_x = np.meshgrid(np.arange(model.H.shape[0]), np.arange(img_count))
        h_x = list(h_x.T.flatten())
        h_y = list(h_y.T.flatten())
        h_values = list(model.H[:, :img_count].flatten())
        h_labels = list(test_labels[:img_count]) * model.H.shape[0]
        # Assert matrix reshapes match up
        for row in range(model.H.shape[0]):
            for col in range(img_count):
                assert abs(float(h_values[row * img_count + col]) - float(model.H[row][col])) < 1e-4
                assert col == int(h_x[row * img_count + col])
                assert row == int(h_y[row * img_count + col])
        h = wandb.Table(
            data=[list(row) for row in zip(h_x, h_y, h_values, h_labels)],
            columns=["x", "y", "value", "label"],
        )
        logger({"h": h})


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse a list of command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data_parser = parser.add_argument_group("data")
    data_parser.add_argument(
        "--dataset",
        type=Dataset,
        default=Dataset.ORL,
        choices=list(iter(Dataset)),
        metavar=str({str(dataset.value) for dataset in iter(Dataset)}),
        help="The dataset to use.",
    )
    data_parser.add_argument(
        "--reduce",
        type=int,
        default=1,
        help="Factor by which to reduce the width and height of each input image.",
    )
    data_parser.add_argument(
        "--subset",
        type=float,
        default=0.9,
        help="A float between 0 and 1, the amount of training data to train on.",
    )
    data_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use when choosing a subset of the data to train on.",
    )
    data_parser.add_argument(
        "--scale",
        type=float,
        default=255,
        help="Scale of the input data, i.e. the maximum value the data can take.",
    )
    noise_parser = parser.add_argument_group("noise")
    noise_parser.add_argument(
        "--noise",
        type=Noise,
        choices=list(iter(Noise)),
        metavar=str({str(noise.value) for noise in iter(Noise)}),
        help="The noise function to use",
    )
    noise_parser.add_argument(
        "--noise_p",
        type=float,
        help="The proportion of pixels that should be made either white or black"
        + f"when using '{Noise.SALT_AND_PEPPER.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise_r",
        type=float,
        help="The proportion of the corrupted pixels that are white when using"
        + f"'{Noise.SALT_AND_PEPPER.value}' noise. Conversely (1-r) is the "
        + "proportion of corrupted pixels that are black.",
    )
    noise_parser.add_argument(
        "--noise_mean",
        type=float,
        help=f"The mean value of the noise when using '{Noise.GAUSSIAN.value}' "
        + f"or '{Noise.UNIFORM.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise_std",
        type=float,
        help=f"The standard deviation of the noise when using '{Noise.GAUSSIAN.value}' "
        + f"or '{Noise.UNIFORM.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise_blocksize",
        type=int,
        help=f"The size of the blocks to remove when using '{Noise.MISSING_BLOCK.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise_blocks",
        type=int,
        help=f"The number of blocks to remove when using '{Noise.MISSING_BLOCK.value}' noise.",
    )
    model_parser = parser.add_argument_group("model")
    model_parser.add_argument(
        "--model",
        type=Model,
        default=Model.STANDARD,
        choices=list(iter(Model)),
        metavar=str({str(model.value) for model in iter(Model)}),
        help="The model to use.",
    )
    model_parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="The number of components of the learned dictionary."
        + "Defaults to the number of unique dataset classes.",
    )
    model_parser.add_argument(
        "--iterations",
        type=int,
        default=8192,
        help="The maximum number of dictionary update iterations to perform.",
    )
    model_parser.add_argument(
        "--tolerance",
        type=int,
        default=1e-4,
        help="The maximum number of dictionary update iterations to perform.",
    )
    model_parser.add_argument(
        "--lam",
        type=float,
        default=0.3,
        help="Regularisation coefficient for the L1 robust NMF model.",
    )
    logging_parser = parser.add_argument_group("logging")
    logging_parser.add_argument(
        "--id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="A unique name to identify the run.",
    )
    logging_parser.add_argument(
        "--log_step",
        type=int,
        default=16,
        help="The number of iterations between logging metrics.",
    )
    logging_parser.add_argument(
        "--results_dir",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "results")),
        help="The directory to save results to.",
    )
    logging_parser.add_argument(
        "--wandb", action="store_true", help="Sync results to wandb if specified."
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.components is None:
        if parsed_args.dataset == Dataset.ORL:
            parsed_args.components = 40
        elif parsed_args.dataset == Dataset.YALEB:
            parsed_args.components = 38
    # Perform post-parse validation
    if parsed_args.noise is not None:
        if parsed_args.noise in (Noise.UNIFORM, Noise.GAUSSIAN):
            assert parsed_args.noise_mean is not None
            assert parsed_args.noise_std is not None
            assert parsed_args.noise_r is None
            assert parsed_args.noise_p is None
            assert parsed_args.noise_blocksize is None
            assert parsed_args.noise_blocks is None
        elif parsed_args.noise == Noise.MISSING_BLOCK:
            assert parsed_args.noise_mean is None
            assert parsed_args.noise_std is None
            assert parsed_args.noise_r is None
            assert parsed_args.noise_p is None
            assert parsed_args.noise_blocksize is not None
            assert parsed_args.noise_blocks is not None
        elif parsed_args.noise == Noise.SALT_AND_PEPPER:
            assert parsed_args.noise_mean is None
            assert parsed_args.noise_std is None
            assert parsed_args.noise_r is not None
            assert parsed_args.noise_p is not None
            assert parsed_args.noise_blocksize is None
            assert parsed_args.noise_blocks is None
        else:
            raise NotImplementedError()
    return parsed_args


if __name__ == "__main__":
    # Parse arguments
    config = parse_args(sys.argv[1:])
    # Create folders for results if they do not exist
    if not Path(config.results_dir).exists():
        Path(config.results_dir).mkdir()
    # Inject a `seed_group` field into the config, which is identical for runs
    # that have the same config ignoring `config.seed` and other irrelevant fields.
    # This allows us to report mean and variance of runs with different training data
    # across separate processes.
    config_dict = copy(config.__dict__)
    del config_dict["id"]
    del config_dict["seed"]
    del config_dict["results_dir"]
    # CAVEAT: The following assumes the config is not nested.
    config.__setattr__(
        "seed_group",
        hashlib.sha256(
            bytes(json.dumps(config_dict, sort_keys=True, default=str), encoding="UTF-8")
        ).hexdigest(),
    )
    # Set up wandb if specified
    if config.wandb:
        wandb.init(project="non-negative-matrix-factorisation", dir=config.results_dir)
        config.id = wandb.run.id
        # Serialise and deserialise config to convert enums to strings before
        # sending to wandb
        wandb_config = json.dumps(config.__dict__, sort_keys=True, default=lambda x: x.value)
        wandb_config = json.loads(wandb_config)
        del wandb_config["id"]
        wandb.config.update(wandb_config)
    # Run the model
    main(config)
