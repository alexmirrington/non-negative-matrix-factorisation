"""Main entrypoint for model training and evaluation."""
import argparse
import os.path
import sys
from typing import List

from config import Dataset
from datasets import load_data
from models import MultiplicativeUpdateNMF


def main(config: argparse.Namespace):
    """Train or evaluate a dictionary learning model."""
    # Determine dataset directory.
    if config.dataset == Dataset.ORL:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "ORL")
    elif config.dataset == Dataset.YALEB:
        data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "CroppedYaleB")
    else:
        raise NotImplementedError()

    # Load dataset.
    X, Y = load_data(root=data_dir, reduce=2)
    print("ORL dataset: X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))

    mur = MultiplicativeUpdateNMF(X, n_components=len(set(Y)))
    mur.fit(max_iter=200)


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
    parser.add_argument("--sync", action="store_true", help="Sync results to wandb if specified.")
    return parser.parse_args(args)


if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    main(config)
