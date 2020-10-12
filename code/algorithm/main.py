"""Main entrypoint for model training and evaluation."""
import argparse
import os.path
import sys
from typing import List

from config import Dataset, NMFModel
from datasets import load_data
from factories import ModelFactory
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

    # Load dataset
    print(colored("dataset:", attrs=["bold"]))
    images, ids = load_data(root=data_dir, reduce=2)
    print(f"{config.dataset}: {images.shape}")

    # Create model
    print(colored("model:", attrs=["bold"]))
    model_factory = ModelFactory()
    model = model_factory.create(images, config)
    print(f"{config.model}: {model}")

    # Train model
    print(colored("training:", attrs=["bold"]))
    model.fit(max_iter=200)


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse a list of command line arguments."""
    parser = argparse.ArgumentParser()
    data_parser = parser.add_argument_group("data")
    data_parser.add_argument(
        "--dataset",
        type=Dataset,
        required=True,
        choices=list(iter(Dataset)),
        metavar=str({str(dataset.value) for dataset in iter(Dataset)}),
        help="The dataset to use.",
    )
    model_parser = parser.add_argument_group("model")
    model_parser.add_argument(
        "--model",
        type=NMFModel,
        required=True,
        choices=list(iter(NMFModel)),
        metavar=str({str(model.value) for model in iter(NMFModel)}),
        help="The model to use.",
    )
    logging_parser = parser.add_argument_group("logging")
    logging_parser.add_argument(
        "--wandb", action="store_true", help="Sync results to wandb if specified."
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    if config.wandb:
        wandb.init(project="non-negative-matrix-factorisation")
    main(config)
