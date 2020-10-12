"""Main entrypoint for model training and evaluation."""
from pathlib import Path

from .datasets import load_data
from .models import MultiplicativeUpdateNMF


def main():
    """Train or evaluate a dictionary learning model."""
    # Load ORL dataset.
    X, Y = load_data(root=Path(__name__) / Path.parent / "data" / "ORL", reduce=2)
    print("ORL dataset: X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))

    mur = MultiplicativeUpdateNMF(X, n_components=len(set(Y)))
    mur.fit(max_iter=200)


if __name__ == "__main__":
    main()
