"""This module contains code to load and interact with the datasets."""
import os
import numpy as np
from PIL import Image


def load_data(root, reduce=4):
    """Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.

            # reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])

            # TODO: preprocessing.

            # convert image to numpy array.
            img = np.asarray(img).reshape((-1, 1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


def test():
    # Load ORL dataset.
    X, Y = load_data(root='../data/ORL', reduce=2)
    print('ORL dataset: X.shape = {}, Y.shape = {}'.format(X.shape, Y.shape))

    # Load Extended YaleB dataset.
    X, Y = load_data(root='../data/CroppedYaleB', reduce=4)
    print('Extended YalB dataset: X.shape = {}, Y.shape = {}'.format(X.shape, Y.shape))

if __name__ == "__main__":
    test()
