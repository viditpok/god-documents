import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_keypoints_and_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts

def warp_image(image, H):
    h, w = image.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    new_corners = cv2.perspectiveTransform(corners, H)
    [xmin, ymin] = np.int32(new_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(new_corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-xmin, -ymin]
    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )
    warped_image = cv2.warpPerspective(
        image, H_translation.dot(H), (xmax - xmin, ymax - ymin)
    )
    return warped_image


def image_rectification(img1, img2, points1, points2):
    points1 = np.float32(points1).reshape(-1, 1, 2)
    points2 = np.float32(points2).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    if F is None or F.shape != (3, 3):
        raise ValueError(
            "Fundamental matrix calculation failed or returned unexpected shape."
        )
    h, w = img1.shape[:2]
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(
        points1[mask.ravel() == 1], points2[mask.ravel() == 1], F, img1.shape[:2]
    )
    if not ret:
        raise ValueError("Stereo rectification failed.")
    rectified1 = warp_image(img1, H1)
    rectified2 = warp_image(img2, H2)
    return rectified1, rectified2, F


def plot_epipolar_lines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c = img1.shape[:2]
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        pt1 = tuple(np.int32(pt1.ravel()))
        pt2 = tuple(np.int32(pt2.ravel()))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, pt1, 5, color, -1)
        img2 = cv2.circle(img2, pt2, 5, color, -1)
    return img1, img2


img1 = cv2.imread("../../data/pic1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("../../data/pic2.jpg", cv2.IMREAD_COLOR)
points1, points2 = find_keypoints_and_matches(img1, img2)
rectified1, rectified2, F = image_rectification(img1, img2, points1, points2)
lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
epi_img1, epi_img2 = plot_epipolar_lines(
    rectified1, rectified2, lines1, points1, points2
)
plt.figure(figsize=(15, 10))
plt.subplot(121), plt.imshow(cv2.cvtColor(epi_img1, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv2.cvtColor(epi_img2, cv2.COLOR_BGR2RGB))
plt.suptitle("Epipolar line plots after image rectification")
plt.show()
