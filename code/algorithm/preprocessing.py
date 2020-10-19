"""Utilities for preprocessing data."""
import numpy as np


def salt_and_pepper(data: np.ndarray, scale: float, p: float, r: float) -> np.ndarray:
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
    corrupt_mask[:num_white] = np.full(num_white, scale)
    corrupt_mask[num_white:num_corrupt] = np.full(num_corrupt - num_white, 0)
    np.random.shuffle(corrupt_mask)
    corrupt_mask = corrupt_mask.reshape(data.shape)

    # Add the noise to the data
    data = np.where(corrupt_mask == -1, data, corrupt_mask)

    return data


def missing_block(
    data: np.ndarray, block_size: int, num_blocks: int = 1, fill: int = 1
) -> np.ndarray:
    """Corrupt image by removing a block/s of pixels, and setting to white.

    If there is more than one block, the blocks may overlap.

    Args
    ---
    data: The input data to be corrupted. Shape: (x_pixels, y_pixels).
    block_size: Size of the block/s of pixels to be removed from each image.
    num_blocks: How many blocks to remove.
    """
    data = data.copy()
    for _ in range(num_blocks):
        # Get a random pixel value between (0, 0) and (x_pixels-block_size, y_pixels-block_size)
        rand_x = np.random.randint(data.shape[0] - block_size + 1)
        rand_y = np.random.randint(data.shape[1] - block_size + 1)

        for i in range(block_size):
            for j in range(block_size):
                # Corrupt pixel
                data[rand_x + i, rand_y + j] = fill

    return data


def gaussian(data: np.ndarray, mean: float = 0, std: float = 1, min: float = 0, max: float = 1):
    """Corrupt image by adding Gaussian noise to pixel values.

    Args
    ---
    data: The input image to be corrupted. Shape: (x_pixels, y_pixels)
    mean: The mean of the noise.
    std: Standard deviation of the noise.
    """
    # Add noise, ensuring values remain between 0 and 255
    noise = np.random.normal(mean, std, data.shape)
    return np.clip(data + noise, min, max)


def uniform(data: np.ndarray, mean: float = 0, std: float = 1, min: float = 0, max: float = 1):
    """Corrupt image with uniform distributed noise.

    The resultant noise will lie in [mean - sqrt(3) * std, mean + sqrt(3) * std].

    Args
    ---
    data: The input image to be corrupted. Shape: (x_pixels, y_pixels)
    mean: The mean of the noise.
    scale: Scales max magnitude of the noise.
    """
    # Math:
    # std = (max - min) / sqrt(12)
    # std = ((mean + spread) - (mean - spread)) / sqrt(12) where spread = (max - min)/2
    # std = (2 * spread) / sqrt(12)
    # spread = std * sqrt(3)

    # Add noise, ensuring values remain between 0 and 255
    noise = np.random.uniform(mean - std * 3 ** 0.5, mean + std * 3 ** 0.5, data.shape)
    return np.clip(data + noise, min, max)
