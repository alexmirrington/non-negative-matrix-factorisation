"""Main entrypoint for model training and evaluation."""
import argparse
import os.path
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import wandb
from config import Dataset, NMFModel
from datasets import load_data
from factories import ModelFactory
from loggers import JSONLLogger, StreamLogger, WandbLogger
from termcolor import colored


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

    # Load dataset
    print(colored("dataset:", attrs=["bold"]))
    images, ids = load_data(root=data_dir, reduce=2)
    print(f"{config.dataset}: {images.shape}")

    # Create model
    print(colored("model:", attrs=["bold"]))
    model_factory = ModelFactory()
    model = model_factory.create(images, config)
    print(config.model)

    # Train model
    print(colored("training:", attrs=["bold"]))
    model.fit(max_iter=200)


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse a list of command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        "--id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="A unique name to identify the run.",
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

    return parser.parse_args(args)


if __name__ == "__main__":
    # Parse arguments
    config = parse_args(sys.argv[1:])
    # Create folders for results if they do not exist
    if not Path(config.results_dir).exists():
        Path(config.results_dir).mkdir()
    # Set up wandb if specified
    if config.wandb:
        wandb.init(
            project="non-negative-matrix-factorisation", dir=config.results_dir, config=config
        )
        config.name = wandb.run.id
    # Run the model
    main(config)
