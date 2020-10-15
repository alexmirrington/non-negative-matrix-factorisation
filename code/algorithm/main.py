"""Main entrypoint for model training and evaluation."""
import argparse
import os.path
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from config import Dataset, Model, Noise
from datasets import load_data
from factories import ModelFactory, PreprocessorFactory
from loggers import JSONLLogger, StreamLogger, WandbLogger
from termcolor import colored

import wandb


def main(config: argparse.Namespace):
    """Train or evaluate a dictionary learning model."""
    # Determine dataset directory
    if config.dataset == Dataset.ORL:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "ORL")
    elif config.dataset == Dataset.YALEB:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "CroppedYaleB")
    else:
        raise NotImplementedError()

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
    print(f"{config.dataset}: {clean_images.shape}")

    # Create model
    print(colored("model:", attrs=["bold"]))
    model_factory = ModelFactory()
    model = model_factory.create(noisy_images, config)
    print(config.model)

    # Train model
    print(colored("training:", attrs=["bold"]))
    model.fit(
        max_iter=config.iterations,
        tol=config.tolerance,
        callback_freq=config.log_step,
        callbacks=loggers,
        clean_data=clean_images,
        true_labels=labels,
    )

    # Evaluate model
    results = model.evaluate(clean_images, labels)
    results = {f"test/{key}": val for key, val in results.items()}
    for logger in loggers:
        logger(results)

    # Log data samples

    # Log model dictionaries


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
    noise_parser = parser.add_argument_group("noise")
    noise_parser.add_argument(
        "--noise",
        type=Noise,
        choices=list(iter(Noise)),
        metavar=str({str(noise.value) for noise in iter(Noise)}),
        help="The noise function to use",
    )
    noise_parser.add_argument(
        "--noise-p",
        type=float,
        help="The proportion of pixels that should be made either white or black"
        + f"when using '{Noise.SALT_AND_PEPPER.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise-r",
        type=float,
        help="The proportion of the corrupted pixels that are white when using"
        + f"'{Noise.SALT_AND_PEPPER.value}' noise. Conversely (1-r) is the "
        + "proportion of corrupted pixels that are black.",
    )
    noise_parser.add_argument(
        "--noise-mean",
        type=float,
        help=f"The mean value of the noise when using '{Noise.GAUSSIAN.value}' "
        + f"or '{Noise.UNIFORM.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise-std",
        type=float,
        help=f"The standard deviation of the noise when using '{Noise.GAUSSIAN.value}' "
        + f"or '{Noise.UNIFORM.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise-blocksize",
        type=float,
        help=f"The size of the blocks to remove when using '{Noise.MISSING_BLOCK.value}' noise.",
    )
    noise_parser.add_argument(
        "--noise-blocks",
        type=float,
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
        default=40,
        help="The number of components of the learned dictionary.",
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
    logging_parser = parser.add_argument_group("logging")
    logging_parser.add_argument(
        "--id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="A unique name to identify the run.",
    )
    logging_parser.add_argument(
        "--log-step",
        type=int,
        default=16,
        help="The number of iterations between logging metrics.",
    )
    logging_parser.add_argument(
        "--results-dir",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "results")),
        help="The directory to save results to.",
    )
    logging_parser.add_argument(
        "--wandb", action="store_true", help="Sync results to wandb if specified."
    )
    parsed_args = parser.parse_args(args)
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
    # Set up wandb if specified
    if config.wandb:
        wandb.init(project="non-negative-matrix-factorisation", dir=config.results_dir)
        config.id = wandb.run.id
        wandb.config.update(config)
    # Run the model
    main(config)
