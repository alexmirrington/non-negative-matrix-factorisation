"""Utilities for loading and interacting with the datasets."""
import os
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image


def load_data(
    root: str, reduce: int = 4, preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ORL or Extended YaleB dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
    """
    processed_images, images, ids = [], [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith("Ambient.pgm"):
                continue

            if not fname.endswith(".pgm"):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.
            
            # reduce computation complexity.
            img = img.resize([s // reduce for s in img.size])


            img = np.asarray(img)
            # APPLY NOISE FUNCTION TO IMAGE HERE
            # Not sure how you want to set up config with this,
            # as the different noises have different parameters.
            # To discuss.
            pimg = img
            if preprocessor is not None:
                pimg = preprocessor(img)
            # convert image to numpy array.
            pimg = pimg.reshape((-1, 1))
            img = img.reshape((-1, 1))
            # collect data and label.
            processed_images.append(pimg)
            images.append(img)
            ids.append(i)

    # concate all images and labels.
    processed_images = np.concatenate(processed_images, axis=1)
    images = np.concatenate(images, axis=1)
    ids = np.array(ids)

    return processed_images, images, ids


def salt_and_pepper(data: np.ndarray, p: float, r: float) -> np.ndarray:
    """Corrupt image with salt and pepper noise.

    This noise randomly changes a chosen proportion of pixels to either white or black.
    The ratio of white to black pixels can also be controlled.

    Args
    ---
    data: The input image to be corrupted. Shape: (x_pixels, y_pixels).
    p: The proportion of pixels that should be made either white or black
    r: The proportion of the corrupted pixels that are white. Conversely (1-r) is the
        proportion of corrupted pixels that are black.
    """
    num_corrupt = int(round(data.size * p))
    num_white = int(round(num_corrupt) * r)

    # Create a mask of randomly shuffled white (255), black (0),
    # and unchanged (-1) values which can be applied to the image
    corrupt_mask = np.full(data.size, -1)
    corrupt_mask[:num_white] = np.full(num_white, 255)
    corrupt_mask[num_white: num_corrupt] = np.full(num_corrupt - num_white, 0)
    np.random.shuffle(corrupt_mask)
    corrupt_mask = corrupt_mask.reshape(data.shape)

    # Add the noise to the data
    data = np.where(corrupt_mask == -1, data, corrupt_mask)

    return data

# def salt_and_pepper(data: np.ndarray, p: float, r: float) -> np.ndarray:
#     """Corrupt input data with salt and pepper noise.
#
#     This noise randomly changes a chosen proportion of pixels to either white or black.
#     The ratio of white to black pixels can also be controlled.
#
#     Each column (ie. each image) is corrupted separately.
#
#     Args
#     ---
#     data: The input data to be corrupted. Shape: (n_features, n_samples).
#     p: The proportion of pixels that should be made either white or black
#     r: The proportion of the corrupted pixels that are white. Conversely (1-p) is the
#         proportion of corrupted pixels that are black.
#
#     """
#     # Loop through and noise each sample separately, since p and r values are per sample
#     for col in range(data.shape[1]):
#         sample = data[:, col]
#         num_corrupt = int(round(len(sample) * p))
#         num_white = int(round(num_corrupt) * r)
#
#         # Create a mask of randomly shuffled white (255), black (0),
#         # and unchanged (-1) values which can be applied to the image
#         corrupt_mask = np.full(sample.shape, -1)
#         corrupt_mask[:num_white] = np.full(num_white, 255)
#         corrupt_mask[num_white: num_corrupt] = np.full(num_corrupt - num_white, 0)
#         np.random.shuffle(corrupt_mask)
#
#         # Add the noise to the data
#         data[:, col] = np.where(corrupt_mask == -1, sample, corrupt_mask)
#
#     return data


def missing_block(data: np.ndarray, block_size: int, num_blocks: int = 1) -> np.ndarray:
    """Corrupt image by removing a block/s of pixels, and setting to white.

    If there is more than one block, the blocks may overlap.

    Args
    ---
    data: The input data to be corrupted. Shape: (x_pixels, y_pixels).
    block_size: Size of the block/s of pixels to be removed from each image.
    num_blocks: How many blocks to remove.

    """

    # Get a random pixel value between (0, 0) and (x_pixels-block_size, y_pixels-block_size)
    rand_x = np.random.randint(data.shape[0] - block_size + 1)
    rand_y = np.random.randint(data.shape[1] - block_size + 1)

    for i in range(block_size):
        for j in range(block_size):
            # Corrupt pixel
            data[rand_x + i, rand_y + j] = 255
    return data


def gaussian(data: np.ndarray, mean: float = 0, std: float = 1):
    """Corrupt image by adding Gaussian noise to pixel values.

    Args
    ---
    data: The input image to be corrupted. Shape: (x_pixels, y_pixels)
    mean: The mean of the noise.
    std: Standard deviation of the noise.
    """
    noise = np.random.normal(mean, std, data.shape)
    print(f"{np.mean(noise)=}")
    print(f"{np.std(noise)=}")
    # Add noise, ensuring values remain between 0 and 255
    return np.clip(data + noise, 0, 255)


def uniform(data: np.ndarray, mean: float = 0, std: float = 1):
    """Corrupt image with uniform distributed noise.

    The resultant noise will lie in [mean - sqrt(3) * std, mean + sqrt(3) * std].

    Args
    ---
    data: The input image to be corrupted. Shape: (x_pixels, y_pixels)
    mean: The mean of the noise.
    scale: Scales max magnitude of the noise.
    """
    # std = (max - min) / sqrt(12)
    # std = ((mean + spread) - (mean - spread)) / sqrt(12) where spread = (max - min)/2
    # std = (2 * spread) / sqrt(12)
    # spread = std * sqrt(3)
    noise = np.random.uniform(mean - std * 3 ** 0.5, mean + std * 3 ** 0.5, data.shape)
    print(f"{np.mean(noise)=}")
    print(f"{np.std(noise)=}")

    # Add noise, ensuring values remain between 0 and 255
    return np.clip(data + noise, 0, 255)
