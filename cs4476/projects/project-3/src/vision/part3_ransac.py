import math
import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    prob_all_inliers = ind_prob_correct**sample_size

    num_samples = np.log(1 - prob_success) / np.log(1 - prob_all_inliers)

    num_samples = int(np.ceil(num_samples))

    return num_samples


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """

    prob_success = 0.999
    sample_size = 8
    ind_prob_success = 0.5
    error_threshold = 0.1
    num_iterations = calculate_num_ransac_iterations(
        prob_success, sample_size, ind_prob_success
    )

    max_inliers = 0
    best_F = None
    inliers_a = None
    inliers_b = None

    for _ in range(num_iterations):

        indices = np.random.choice(matches_a.shape[0], sample_size, replace=False)
        sample_a = matches_a[indices]
        sample_b = matches_b[indices]

        F = estimate_fundamental_matrix(sample_a, sample_b)

        ones = np.ones((matches_a.shape[0], 1))
        points_a_homog = np.hstack((matches_a, ones))
        points_b_homog = np.hstack((matches_b, ones))

        Fx1 = np.dot(F, points_a_homog.T)
        Fx2 = np.dot(F.T, points_b_homog.T)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        errors = (np.sum(points_b_homog.T * Fx1, axis=0) ** 2) / denom

        current_inliers = errors < error_threshold
        num_inliers = np.sum(current_inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            inliers_a = matches_a[current_inliers]
            inliers_b = matches_b[current_inliers]

    return best_F, inliers_a[:30], inliers_b[:30]
