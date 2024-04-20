#!/usr/bin/python3

import copy
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vision.part1_harris_corner import get_harris_interest_points
from src.vision.part3_feature_matching import match_features_ratio_test
from src.vision.part4_sift_descriptor import (
    get_magnitudes_and_orientations,
    get_gradient_histogram_vec_from_patch,
    get_SIFT_descriptors,
    get_feat_vec,
    rotate_image,
    get_correlation_coeff,
    get_intensity_based_matches
)
from src.vision.utils import load_image, evaluate_correspondence, rgb2gray, PIL_resize

ROOT = Path(__file__).resolve().parent.parent  # ../..

def test_get_magnitudes_and_orientations():
    """ Verify gradient magnitudes and orientations are computed correctly"""
    Ix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Iy = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    # there are 3 vectors -- (1,0) at 0 deg, (0,1) at 90 deg, and (-1,1) and 135 deg
    expected_magnitudes = np.array([[1, 1, 1], [1, 1, 1], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])
    expected_orientations = np.array(
        [[0, 0, 0], [np.pi / 2, np.pi / 2, np.pi / 2], [3 * np.pi / 4, 3 * np.pi / 4, 3 * np.pi / 4]]
    )

    assert np.allclose(magnitudes, expected_magnitudes)
    assert np.allclose(orientations, expected_orientations)


def test_get_gradient_histogram_vec_from_patch():
    """ Check if weighted gradient histogram is computed correctly """
    window_magnitudes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
    )

    wgh = get_gradient_histogram_vec_from_patch(window_magnitudes, window_orientations)

    expected_wgh = np.array(
        [
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4, magnitude 1
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4, magnitude 0
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], 
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], # bin 5, magnitude 2
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.]
        ]
    ).reshape(128, 1)

    assert np.allclose(wgh, expected_wgh, atol=1e-1)


def test_get_feat_vec():
    """ Check if feature vector for a specific interest point is returned correctly """
    window_magnitudes = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]
    C = 5/8 * np.pi # squarely in bin [pi/2, 3pi/4]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ]
        ]
    )

    feature_width = 16

    x, y = 7, 8

    fv = get_feat_vec(x, y, window_magnitudes, window_orientations, feature_width)

    expected_fv = np.array(
        [
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ]
        ]
    ).reshape(128, 1)

    assert np.allclose(fv, expected_fv, atol=1e-2)


def test_get_SIFT_descriptors():
    """ Check if the 128-d SIFT feature vector computed at each of the input points is returned correctly """

    image1 = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ]
    ).astype(np.float32)

    X1, Y1 = np.array([8, 9]), np.array([8, 9])

    SIFT_descriptors = get_SIFT_descriptors(image1, X1, Y1)

    expected_SIFT_descriptors = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.499],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.547],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.379, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.329, 0.0, 0.592],
                [0.0, 0.0, 0.499, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.0, 0.547, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.0, 0.592, 0.0, 0.0, 0.329, 0.0, 0.0],
                [0.0, 0.332, 0.544, 0.0, 0.0, 0.285, 0.0, 0.544],
            ],
        ]
    ).reshape(2, 128)

    assert np.allclose(SIFT_descriptors, expected_SIFT_descriptors, atol=1e-1)


def test_feature_matching_speed():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must run in under 90 seconds.
    """
    start = time.time()
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)
    print("{:d} matches from {:d} corners".format(len(matches), len(X1)))

    end = time.time()
    duration = end - start
    print(f"Your Feature matching pipeline takes {duration:.2f} seconds to run on Notre Dame")

    MAX_ALLOWED_TIME = 90  # sec
    assert duration < MAX_ALLOWED_TIME


def test_feature_matching_accuracy():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must achieve at least 80% accuracy.
    """
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)

    acc, _ = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        X1[matches[:, 0]],
        Y1[matches[:, 0]],
        X2[matches[:, 1]],
        Y2[matches[:, 1]],
    )

    print(f"Your Feature matching pipeline achieved {100 * acc:.2f}% accuracy to run on Notre Dame")

    MIN_ALLOWED_ACC = 0.80  # 80 percent
    assert acc > MIN_ALLOWED_ACC

def test_corr():
    """
    Tests implementation of get_correlation_coeff()
    """
    arr1 = np.array([1,0.5,0.5,0.2,0.3,0,1])
    arr2 = np.array([0.1,1,0.5,0.6,0.2,0,1])
    expected_corr = 0.7674981642057755
    corr = get_correlation_coeff(arr1, arr2)
    assert abs(corr-expected_corr) < 1e-2, "Correlation calculation incorrect"

def test_get_intensity_based_matches():
    """
    Tests implementation of get_intensity_based_matches on Mount Rushmore images
    """
    image1 = load_image(f"{ROOT}/data/2a_rushmore.jpg")
    image2 = load_image(f"{ROOT}/data/2b_rushmore.jpg")
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    intensity_matches = get_intensity_based_matches(image1, image2)
    expected_intensity_matches= np.array([[[0, 0], [768, 256]], [[128, 0], [768, 256]], [[256, 0], [768, 256]], [[384, 0], [768, 256]], [[512, 0], [768, 256]], [[640, 0], [768, 256]], [[768, 0], [768, 256]], [[896, 0], [768, 256]], [[1024, 0], [768, 256]], [[1152, 0], [768, 256]], [[0, 128], [896, 640]], [[128, 128], [768, 256]], [[256, 128], [0, 640]], [[384, 128], [128, 128]], [[512, 128], [768, 256]], [[640, 128], [768, 256]], [[768, 128], [768, 256]], [[896, 128], [768, 256]], [[1024, 128], [768, 256]], [[1152, 128], [768, 256]], [[0, 256], [256, 256]], [[128, 256], [1152, 512]], [[256, 256], [640, 384]], [[384, 256], [256, 128]], [[512, 256], [1152, 512]], [[640, 256], [768, 256]], [[768, 256], [768, 256]], [[896, 256], [768, 256]], [[1024, 256], [768, 256]], [[1152, 256], [768, 256]], [[0, 384], [896, 640]], [[128, 384], [256, 512]], [[256, 384], [384, 384]], [[384, 384], [896, 256]], [[512, 384], [768, 512]], [[640, 384], [256, 256]], [[768, 384], [128, 128]], [[896, 384], [640, 384]], [[1024, 384], [256, 384]], [[1152, 384], [896, 640]], [[0, 512], [1152, 512]], [[128, 512], [1152, 512]], [[256, 512], [896, 640]], [[384, 512], [1152, 512]], [[512, 512], [1152, 512]], [[640, 512], [768, 512]], [[768, 512], [768, 512]], [[896, 512], [640, 384]], [[1024, 512], [1024, 256]], [[1152, 512], [896, 640]], [[0, 640], [896, 640]], [[128, 640], [1152, 512]], [[256, 640], [384, 256]], [[384, 640], [1152, 512]], [[512, 640], [1152, 512]], [[640, 640], [256, 512]], [[768, 640], [768, 512]], [[896, 640], [1152, 512]], [[1024, 640], [384, 512]], [[1152, 640], [1152, 512]], [[0, 768], [1024, 640]], [[128, 768], [384, 128]], [[256, 768], [384, 512]], [[384, 768], [896, 640]], [[512, 768], [0, 640]], [[640, 768], [1152, 512]], [[768, 768], [896, 640]], [[896, 768], [384, 512]], [[1024, 768], [256, 256]], [[1152, 768], [1152, 512]], [[0, 896], [1152, 512]], [[128, 896], [256, 512]], [[256, 896], [1152, 512]], [[384, 896], [1152, 512]], [[512, 896], [896, 640]], [[640, 896], [1152, 512]], [[768, 896], [896, 640]], [[896, 896], [768, 512]], [[1024, 896], [1024, 768]], [[1152, 896], [768, 512]]])  
    print("{:d} intensity-based matches in Mount Rushmore".format(len(intensity_matches)))
    assert np.allclose(intensity_matches.shape, expected_intensity_matches.shape, atol=1e-1), "Match shape incorrect"
    assert np.allclose(intensity_matches, expected_intensity_matches, atol=1e-3), "Match values incorrect"

def test_rotate_image():
    image = np.array([[1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]])
    # Test case 1: Rotation by 0 degrees
    rotated_image_0 = rotate_image(image, 0)
    assert np.array_equal(rotated_image_0, image), "Rotation by 0 degrees failed."
    # Test case 3: Rotation by 180 degrees
    expected_image_180 = np.array([[5, 4, 3, 2, 1],
                                [5, 4, 3, 2, 1],
                                [5, 4, 3, 2, 1],
                                [5, 4, 3, 2, 1],
                                [5, 4, 3, 2, 1]])
    rotated_image_180 = rotate_image(image, 180)
    assert np.array_equal(rotated_image_180, expected_image_180), "Rotation by 180 degrees failed."
    print("All test cases passed!")
