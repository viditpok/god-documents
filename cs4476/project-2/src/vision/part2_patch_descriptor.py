import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Args:
        image_bw: array of shape (M,N) representing grayscale image.
        X: array of shape (K,) representing x-coordinate of keypoints.
        Y: array of shape (K,) representing y-coordinate of keypoints.
        feature_width: size of the square window.

    Returns:
        fvs: array of shape (K,D) representing feature descriptors.
    """

    fvs = []

    for x, y in zip(X, Y):

        x_start = max(x - feature_width // 2 + 1, 0)
        y_start = max(y - feature_width // 2 + 1, 0)
        x_end = x_start + feature_width - 1
        y_end = y_start + feature_width - 1

        patch = image_bw[y_start : y_end + 1, x_start : x_end + 1]

        patch_l2 = np.linalg.norm(patch)
        patch_normalized = patch / patch_l2

        fvs.append(patch_normalized.flatten())

    fvs_np = np.array(fvs)
    return fvs_np
