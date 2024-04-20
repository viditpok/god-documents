"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """

    mean_coords = np.mean(points, axis=0)
    c_u, c_v = mean_coords

    centered_points = points - mean_coords

    std_dev = np.std(centered_points, axis=0)
    s_u, s_v = 1 / std_dev

    s_u = s_u if s_u != np.inf else 1.0
    s_v = s_v if s_v != np.inf else 1.0

    scale_matrix = np.diag([s_u, s_v, 1])
    offset_matrix = np.array([[1, 0, -c_u], [0, 1, -c_v], [0, 0, 1]])

    T = scale_matrix @ offset_matrix

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized_homogeneous = (T @ points_homogeneous.T).T

    points_normalized = (
        points_normalized_homogeneous[:, :2] / points_normalized_homogeneous[:, [2]]
    )

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """

    F_orig = T_b.T @ F_norm @ T_a
    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    points_a_norm, T_a = normalize_points(points_a)
    points_b_norm, T_b = normalize_points(points_b)

    N = len(points_a)
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = points_a_norm[i]
        x2, y2 = points_b_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U_F, S_F, Vt_F = np.linalg.svd(F)
    S_F[2] = 0
    F_rank2 = U_F @ np.diag(S_F) @ Vt_F

    F_unnorm = T_b.T @ F_rank2 @ T_a

    return F_unnorm
