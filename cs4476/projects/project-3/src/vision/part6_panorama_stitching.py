import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental_matrix


def compute_homography(points_src, points_dst):
    """
    Computes the homography from points_src to points_dst.
    """
    A = []
    for i in range(len(points_src)):
        x, y = points_src[i][0], points_src[i][1]
        u, v = points_dst[i][0], points_dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = H / H[-1, -1]
    return H


def ransac_homography(pointsA, pointsB, iterations=1000, threshold=5.0):
    """
    Apply RANSAC to find the best homography that maps pointsA to pointsB.
    """
    max_inliers = []
    best_H = None
    for _ in range(iterations):
        idx = np.random.choice(np.arange(len(pointsA)), 4, replace=False)
        sample_pointsA = pointsA[idx]
        sample_pointsB = pointsB[idx]
        H = compute_homography(sample_pointsA, sample_pointsB)
        pointsA_homog = np.concatenate(
            (pointsA, np.ones((pointsA.shape[0], 1))), axis=1
        )
        estimated_pointsB_homog = np.dot(H, pointsA_homog.T).T
        estimated_pointsB = (
            estimated_pointsB_homog[:, :2] / estimated_pointsB_homog[:, 2:]
        )
        errors = np.linalg.norm(pointsB - estimated_pointsB, axis=1)
        inliers = errors < threshold
        if sum(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H
    if best_H is not None:
        best_H = compute_homography(pointsA[max_inliers], pointsB[max_inliers])
    return best_H, pointsA[max_inliers], pointsB[max_inliers]


def panorama_stitch(imageA, imageB):
    if isinstance(imageA, str):
        imageA = cv.imread(imageA)
    if isinstance(imageB, str):
        imageB = cv.imread(imageB)
    sift = cv.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptorsA, descriptorsB, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    pointsA = np.float32([keypointsA[m.queryIdx].pt for m in good_matches])
    pointsB = np.float32([keypointsB[m.trainIdx].pt for m in good_matches])

    H, _, _ = ransac_homography(pointsA, pointsB)
    if H is None:
        raise Exception("Homography could not be computed with the given matches.")

    corners = np.float32(
        [
            [0, 0, 1],
            [imageA.shape[1], 0, 1],
            [0, imageA.shape[0], 1],
            [imageA.shape[1], imageA.shape[0], 1],
        ]
    ).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2] / warped_corners[2]

    min_x = min(min(warped_corners[0]), 0)
    max_x = max(max(warped_corners[0]), imageB.shape[1])
    min_y = min(min(warped_corners[1]), 0)
    max_y = max(max(warped_corners[1]), imageB.shape[0])

    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))

    panorama = cv.warpPerspective(
        imageA, translation_matrix @ H, (output_width, output_height)
    )

    panorama[
        int(-min_y) : int(-min_y) + imageB.shape[0],
        int(-min_x) : int(-min_x) + imageB.shape[1],
    ] = imageB

    return panorama
