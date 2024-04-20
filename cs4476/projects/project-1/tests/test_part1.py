#!/usr/bin/python3

import copy
from pathlib import Path

import numpy as np
from vision.part1 import (
    create_Gaussian_kernel_1D,
    create_Gaussian_kernel_2D,
    separate_Gaussian_kernel_2D,
    create_hybrid_image,
    my_conv2d_numpy,
)
from vision.utils import (
    load_image,
)

ROOT = Path(__file__).resolve().parent.parent  # ../..

"""
Even size kernels are not required for this project, so we exclude this test case.
"""


def test_create_Gaussian_kernel_1D():
    """Check that a few values are correct inside 1d kernel"""
    ksize = 29
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)

    assert kernel.shape == (29, 1), "The kernel is not the correct size"
    kernel = kernel.squeeze()

    # peak should be at center
    gt_kernel_crop = np.array(
        [0.05405, 0.05688, 0.05865, 0.05925, 0.05865, 0.05688, 0.05405]
    )
    h_center = ksize // 2
    student_kernel_crop = kernel[h_center - 3 : h_center + 4]

    assert np.allclose(
        gt_kernel_crop, student_kernel_crop, atol=1e-5
    ), "Values dont match"
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_1D_sumsto1():
    """Verifies that generated 1d Gaussian kernel sums to 1."""
    ksize = 29
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_2D_sumsto1():
    """Verifies that generated 2d Gaussian kernel sums to 1."""
    cutoff_frequency = 7
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_1D_peak():
    """Ensure peak of 1d kernel is at center, and dims are correct"""
    ksize = 29
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)

    # generated Gaussian kernel should have odd dimensions
    assert kernel.shape[0] % 1 == 0
    assert kernel.shape[1] % 1 == 0
    assert kernel.ndim == 2

    center_idx = kernel.shape[0] // 2

    assert kernel.squeeze().argmax() == center_idx, "Peak is not at center index"

    coords = np.where(kernel == kernel.max())
    coords = np.array(coords).T

    # should be only 1 peak
    assert coords.shape == (1, 2), "Peak is not unique"


def test_create_Gaussian_kernel_2D_peak():
    """Ensure peak of 2d kernel is at center, and dims are correct"""
    cutoff_frequency = 7
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)

    # generated Gaussian kernel should have odd dimensions
    assert kernel.shape[0] % 1 == 0
    assert kernel.shape[1] % 1 == 0
    assert kernel.ndim == 2

    center_row = kernel.shape[0] // 2
    center_col = kernel.shape[1] // 2

    coords = np.where(kernel == kernel.max())
    coords = np.array(coords).T

    # should be only 1 peak
    assert coords.shape == (1, 2), "Peak is not unique"
    assert coords[0, 0] == center_row, "Peak is not at center row"
    assert coords[0, 1] == center_col, "Peak is not at center column"


def test_gaussian_kernel_2D() -> None:
    """Verify values of inner 5x5 patch of 29x29 Gaussian kernel."""
    cutoff_frequency = 7
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)
    assert kernel.shape == (29, 29), "The kernel is not the correct size"

    # peak should be at center
    gt_kernel_crop = np.array(
        [
            [0.00324, 0.00334, 0.00337, 0.00334, 0.00324],
            [0.00334, 0.00344, 0.00348, 0.00344, 0.00334],
            [0.00337, 0.00348, 0.00351, 0.00348, 0.00337],
            [0.00334, 0.00344, 0.00348, 0.00344, 0.00334],
            [0.00324, 0.00334, 0.00337, 0.00334, 0.00324],
        ]
    )

    kernel_h, kernel_w = kernel.shape
    h_center = kernel_h // 2
    w_center = kernel_w // 2
    student_kernel_crop = kernel[
        h_center - 2 : h_center + 3, w_center - 2 : w_center + 3
    ]

    assert np.allclose(
        gt_kernel_crop, student_kernel_crop, atol=1e-5
    ), "Values dont match"
    assert np.allclose(kernel.sum(), 1.0, atol=1e-3), "Values don't sum to 1"


kernel_to_separate = np.array([[0.00034388, 0.00063359, 0.00104462, 0.00154117, 0.00203464,
        0.00240364, 0.00254095, 0.00240364, 0.00203464, 0.00154117,
        0.00104462, 0.00063359, 0.00034388],
    [0.00063359, 0.00116738, 0.00192468, 0.00283956, 0.00374877,
        0.00442865, 0.00468165, 0.00442865, 0.00374877, 0.00283956,
        0.00192468, 0.00116738, 0.00063359],
    [0.00104462, 0.00192468, 0.00317327, 0.00468165, 0.00618068,
        0.00730161, 0.00771874, 0.00730161, 0.00618068, 0.00468165,
        0.00317327, 0.00192468, 0.00104462],
    [0.00154117, 0.00283956, 0.00468165, 0.00690703, 0.00911861,
        0.01077237, 0.01138777, 0.01077237, 0.00911861, 0.00690703,
        0.00468165, 0.00283956, 0.00154117],
    [0.00203464, 0.00374877, 0.00618068, 0.00911861, 0.01203832,
        0.0142216 , 0.01503405, 0.0142216 , 0.01203832, 0.00911861,
        0.00618068, 0.00374877, 0.00203464],
    [0.00240364, 0.00442865, 0.00730161, 0.01077237, 0.0142216 ,
        0.01680083, 0.01776063, 0.01680083, 0.0142216 , 0.01077237,
        0.00730161, 0.00442865, 0.00240364],
    [0.00254095, 0.00468165, 0.00771874, 0.01138777, 0.01503405,
        0.01776063, 0.01877525, 0.01776063, 0.01503405, 0.01138777,
        0.00771874, 0.00468165, 0.00254095],
    [0.00240364, 0.00442865, 0.00730161, 0.01077237, 0.0142216 ,
        0.01680083, 0.01776063, 0.01680083, 0.0142216 , 0.01077237,
        0.00730161, 0.00442865, 0.00240364],
    [0.00203464, 0.00374877, 0.00618068, 0.00911861, 0.01203832,
        0.0142216 , 0.01503405, 0.0142216 , 0.01203832, 0.00911861,
        0.00618068, 0.00374877, 0.00203464],
    [0.00154117, 0.00283956, 0.00468165, 0.00690703, 0.00911861,
        0.01077237, 0.01138777, 0.01077237, 0.00911861, 0.00690703,
        0.00468165, 0.00283956, 0.00154117],
    [0.00104462, 0.00192468, 0.00317327, 0.00468165, 0.00618068,
        0.00730161, 0.00771874, 0.00730161, 0.00618068, 0.00468165,
        0.00317327, 0.00192468, 0.00104462],
    [0.00063359, 0.00116738, 0.00192468, 0.00283956, 0.00374877,
        0.00442865, 0.00468165, 0.00442865, 0.00374877, 0.00283956,
        0.00192468, 0.00116738, 0.00063359],
    [0.00034388, 0.00063359, 0.00104462, 0.00154117, 0.00203464,
        0.00240364, 0.00254095, 0.00240364, 0.00203464, 0.00154117,
        0.00104462, 0.00063359, 0.00034388]])


def test_separate_Gaussian_kernel_2D():
    """Verify values of separated 29x1 1D Gaussian kernels."""
    student_v, student_h = separate_Gaussian_kernel_2D(kernel_to_separate)
    assert student_v.shape == (13, 1), "The vertical kernel is not the correct size"
    assert student_h.shape == (13, 1), "The horizontal kernel is not the correct size"

    gt_v = np.array([[-0.01854402],
       [-0.03416694],
       [-0.05633176],
       [-0.08310854],
       [-0.10971929],
       [-0.12961803],
       [-0.13702282],
       [-0.12961803],
       [-0.10971929],
       [-0.08310854],
       [-0.05633176],
       [-0.03416694],
       [-0.01854402]])
    
    gt_h = np.array([[-0.01854402],
       [-0.03416694],
       [-0.05633176],
       [-0.08310854],
       [-0.10971929],
       [-0.12961803],
       [-0.13702282],
       [-0.12961803],
       [-0.10971929],
       [-0.08310854],
       [-0.05633176],
       [-0.03416694],
       [-0.01854402]])

    assert np.allclose(
        gt_v, student_v, atol=1e-6
    ), "Vertical kernel values dont match"
    assert np.allclose(
        gt_h, student_h, atol=1e-6
    ), "Horizontal kernel values dont match"


def test_separate_Gaussian_kernel_2D_product() -> None:
    """Verify values of the dot product of separated 29x1 1D Gaussian kernels."""
    student_v, student_h = separate_Gaussian_kernel_2D(kernel_to_separate)
    student_kernel = student_v * student_h.T
    assert student_kernel.shape == (13, 13), "The v*h.T is not the correct size"

    # peak should be at center
    gt_kernel_crop = np.array([[0.01203832, 0.0142216 , 0.01503405, 0.0142216 , 0.01203832],
                            [0.0142216 , 0.01680083, 0.01776063, 0.01680083, 0.0142216 ],
                            [0.01503405, 0.01776063, 0.01877525, 0.01776063, 0.01503405],
                            [0.0142216 , 0.01680083, 0.01776063, 0.01680083, 0.0142216 ],
                            [0.01203832, 0.0142216 , 0.01503405, 0.0142216 , 0.01203832]])

    kernel_h, kernel_w = student_kernel.shape
    h_center = kernel_h // 2
    w_center = kernel_w // 2
    student_kernel_crop = student_kernel[
        h_center - 2 : h_center + 3, w_center - 2 : w_center + 3
    ]

    assert np.allclose(
        gt_kernel_crop, student_kernel_crop, atol=1e-5
    ), "The v*h.T values dont match"
    assert np.allclose(student_kernel.sum(), 1.0, atol=1e-3), "The v*h.T values don't sum to 1"


def test_my_conv2d_numpy_identity():
    """Check identity filter works correctly on all channels"""
    print("eee")
    filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    channel_img = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    )

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 0] = channel_img
    img[:, :, 1] = channel_img
    img[:, :, 2] = channel_img
    filtered_img = my_conv2d_numpy(copy.deepcopy(img), filter)
    print("results:")
    print(filtered_img)
    print(img)
    assert np.allclose(filtered_img, img)


def test_my_conv2d_numpy_ones_filter():
    """Square filter of all 1s"""
    filter = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    channel_img = np.array([[0, 1], [2, 3]])

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    for i in range(3):
        img[:, :, i] = channel_img

    filtered_img = my_conv2d_numpy(copy.deepcopy(img), filter)

    # 0 + 1 + 2 + 3 = 6
    gt_filtered_channel_img = np.array([[6, 6], [6, 6]])
    gt_filtered_img = np.zeros((2, 2, 3), dtype=np.uint8)
    for i in range(3):
        gt_filtered_img[:, :, i] = gt_filtered_channel_img

    assert np.allclose(filtered_img, gt_filtered_img)


def test_my_conv2d_numpy_nonsquare_filter():
    """ """

    filter = np.array([[1, 1, 1]])

    channel_img = np.array([[0, 1, 3], [4, 5, 6]])

    img = np.zeros((2, 3, 3), dtype=np.uint8)
    for i in range(3):
        img[:, :, i] = channel_img

    filtered_img = my_conv2d_numpy(copy.deepcopy(img), filter)

    gt_filtered_channel_img = np.array([[1, 4, 4], [9, 15, 11]])
    gt_filtered_img = np.zeros((2, 3, 3), dtype=np.uint8)
    for i in range(3):
        gt_filtered_img[:, :, i] = gt_filtered_channel_img

    assert np.allclose(filtered_img, gt_filtered_img)


def test_hybrid_image_np() -> None:
    """Verify that hybrid image values are correct."""
    image1 = load_image(f"{ROOT}/data/1a_dog.bmp")
    image2 = load_image(f"{ROOT}/data/1b_cat.bmp")
    kernel = create_Gaussian_kernel_2D(7)
    _, _, hybrid_image = create_hybrid_image(image1, image2, kernel)

    img_h, img_w, _ = image2.shape
    k_h, k_w = kernel.shape
    # Exclude the border pixels.
    hybrid_interior = hybrid_image[k_h : img_h - k_h, k_w : img_w - k_w]
    correct_sum = np.allclose(158339.52, hybrid_interior.sum())

    # ground truth values
    gt_hybrid_crop = np.array(
        [
            [[0.5429589, 0.55373234, 0.5452099], [0.5290553, 0.5485607, 0.545738]],
            [[0.55020595, 0.55713284, 0.5457024], [0.5368045, 0.5603536, 0.5505791]],
        ],
        dtype=np.float32,
    )

    # H,W,C order in Numpy
    correct_crop = np.allclose(
        hybrid_image[100:102, 100:102, :], gt_hybrid_crop, atol=1e-3
    )
    assert (
        correct_sum and correct_crop
    ), "Hybrid image values are not correct, please double check your implementation."

    ## Purely for debugging/visualization ##
    # plt.imshow(hybrid_image)
    # plt.show()
    ########################################
