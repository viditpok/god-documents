import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    pixel_data = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            if file.lower().endswith(("jpeg", "png", "jpg", "bmp", "tif", "tiff")):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    gray_img = img.convert("L")
                    img_array = np.asarray(gray_img) / 255.0
                    pixel_data.append(img_array.flatten())
    all_pixels = np.concatenate(pixel_data)
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    return mean, std
