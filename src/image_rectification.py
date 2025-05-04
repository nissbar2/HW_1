import os
import sys

from scipy.ndimage import map_coordinates

import env
import src.utils.utils as utils

from PIL import Image

import numpy as np
import cv2
from src.fundamental_matrix import *
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# NOTICE!! (I think the comment is wrong, because in main() it calculates F as p'Fp=0, so I will treat it that way)
def compute_epipole(points1: np.array, 
                    points2: np.array, 
                    F: np.array) -> np.array:
    '''
    Computes the epipole in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        epipole - the homogenous coordinates [x y 1] of the epipole in the image
    '''
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]  # last row of Vt = last column of V

    # Normalize so last coordinate is 1
    if e[-1] != 0:
        e /= e[-1]

    return e


def compute_matching_homographies(e2: np.ndarray,
                                  F: np.ndarray,
                                  im2: np.ndarray,
                                  points1: np.ndarray,
                                  points2: np.ndarray) -> tuple:
    """
    Determines homographies H1 and H2 such that they rectify a pair of images.
    Follows the algorithm described in the course notes (Hartley & Zisserman approach).

    Arguments:
        e2 (np.ndarray): The epipole in the second image (e'), shape (3,).
                         Assumed normalized (last coordinate is 1).
        F (np.ndarray): The Fundamental matrix, shape (3, 3).
        im2 (np.ndarray): The second image (used for dimensions), shape (H, W, C) or (H, W).
        points1 (np.ndarray): N points in the first image, shape (N, 2).
        points2 (np.ndarray): N points in the second image, shape (N, 2).

    Returns:
        H1 (np.ndarray): The homography for the first image, shape (3, 3).
        H2 (np.ndarray): The homography for the second image, shape (3, 3).
    """
    if points1.shape[1] == 3:
        # aviod div by zero
        points1[points1[:, 2] == 0, 2] = 1e-8
        points1 = points1[:, :2] / points1[:, 2][:, np.newaxis]
    if points2.shape[1] == 3:
        # aviod div by zero
        points2[points2[:, 2] == 0, 2] = 1e-8
        points2 = points2[:, :2] / points2[:, 2][:, np.newaxis]

    if e2[2] != 0 and not np.isclose(e2[2], 1.0):
        e2 = e2 / e2[2]

    height, width = im2.shape[:2]

    # === Compute H2 ===
    # H2 = T_inv @ G @ R @ T

    # = Translation =
    T = np.array([
        [1, 0, -width / 2.0],
        [0, 1, -height / 2.0],
        [0, 0, 1]
    ])

    e2_translated_hom = T @ e2

    # = Rotation =
    e1_t, e2_t, e3_t = e2_translated_hom

    # Calculate rotation angle components
    d = np.sqrt(e1_t ** 2 + e2_t ** 2)
    if np.isclose(d, 0):
        cos_theta = 1.0
        sin_theta = 0.0
    else:
        cos_theta = e1_t / d
        sin_theta = e2_t / d

    R = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # = Projective transformation =
    e2_rotated_hom = R @ T @ e2
    f = e2_rotated_hom[0] / e2_rotated_hom[
        2]

    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / f, 0, 1]
    ])

    # = Inverse translation =
    T_inv = np.array([
        [1, 0, width / 2.0],
        [0, 1, height / 2.0],
        [0, 0, 1]
    ])

    # -- Compute H2 --
    H2 = T_inv @ G @ R @ T

    # === Compute H1 ===
    # H1 = HA @ H2 @ M

    # Compute the matrix M = [e2]_x F + e2 v^T
    e2_reshaped = e2.reshape(3, 1)
    e2_x = np.array([
        [0, -e2_reshaped[2, 0], e2_reshaped[1, 0]],
        [e2_reshaped[2, 0], 0, -e2_reshaped[0, 0]],
        [-e2_reshaped[1, 0], e2_reshaped[0, 0], 0]
    ])

    # Use v = [1, 1, 1]^T
    v = np.ones((1, 3))

    M = e2_x @ F + e2_reshaped @ v

    n_points = points1.shape[0]
    points1_hom = np.hstack((points1, np.ones((n_points, 1))))
    points2_hom = np.hstack((points2, np.ones((n_points, 1))))

    p1_transformed_hom = (H2 @ M @ points1_hom.T).T
    p2_transformed_hom = (H2 @ points2_hom.T).T

    # Normalize the transformed points (divide by the third coordinate)
    p1_w = p1_transformed_hom[:, 2].reshape(-1, 1)
    p2_w = p2_transformed_hom[:, 2].reshape(-1, 1)

    p1_w[np.isclose(p1_w, 0)] = 1e-8  # Avoid division by zero
    p2_w[np.isclose(p2_w, 0)] = 1e-8

    p1_transformed = p1_transformed_hom[:, :2] / p1_w
    p2_transformed = p2_transformed_hom[:, :2] / p2_w

    # Extract coordinates
    x_hat = p1_transformed[:, 0]
    y_hat = p1_transformed[:, 1]
    x_prime_hat = p2_transformed[:, 0]
    y_prime_hat = p2_transformed[:,
                  1]

    W = np.vstack([x_hat, y_hat, np.ones(n_points)]).T  # Shape (N, 3)
    b = x_prime_hat  # Shape (N,)

    # Solve the least squares problem Wa = b for 'a'
    a, residuals, rank, s = np.linalg.lstsq(W, b, rcond=None)
    a1, a2, a3 = a

    # Construct matrix HA
    HA = np.array([
        [a1, a2, a3],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # -- Compute H1 --
    H1 = HA @ H2 @ M

    return H1, H2


def compute_rectified_image(im: np.array, H: np.array) -> tuple:
    """
    Rectifies an image using a homography matrix.

    Arguments:
        im - an image represented as a NumPy array (H x W or H x W x C)
        H - a 3x3 homography matrix that rectifies the image

    Returns:
        new_image - a new image matrix after applying the homography
        offset - a tuple (ox, oy) representing the translation offset
                 of the top-left corner of the original image bounds
                 within the new rectified coordinate system.
    """

    h, w = im.shape[:2]
    is_color = len(im.shape) == 3
    channels = im.shape[2] if is_color else 1

    corners = np.array([
        [0, 0, 1],  # Top-left
        [w - 1, 0, 1],  # Top-right
        [0, h - 1, 1],  # Bottom-left
        [w - 1, h - 1, 1]  # Bottom-right
    ]).T

    # Transform corners using homography H
    new_corners_h = H @ corners

    # Convert back to Cartesian
    new_corners_h[2, :] = np.where(np.abs(new_corners_h[2, :]) < 1e-8, 1e-8,
                                   new_corners_h[2, :])
    new_corners = new_corners_h[:2, :] / new_corners_h[2, :]

    # Determine the bounds of the new image
    min_x = np.floor(np.min(new_corners[0, :])).astype(int)
    max_x = np.ceil(np.max(new_corners[0, :])).astype(int)
    min_y = np.floor(np.min(new_corners[1, :])).astype(int)
    max_y = np.ceil(np.max(new_corners[1, :])).astype(int)

    new_w = max_x - min_x
    new_h = max_y - min_y
    offset = (min_x,
              min_y)  # Offset represents the top-left corner in the *rectified* coordinate system

    H_inv = np.linalg.inv(H)

    # 7. Create the output image canvas
    #    Initialize with zeros (black) or another background color if desired
    if is_color:
        new_image = np.zeros((new_h, new_w, channels), dtype=im.dtype)
    else:
        new_image = np.zeros((new_h, new_w), dtype=im.dtype)

    # inverse mapping

    x_new_coords = np.arange(new_w) + min_x
    y_new_coords = np.arange(new_h) + min_y
    xx_new, yy_new = np.meshgrid(x_new_coords, y_new_coords)

    pts_new_h = np.vstack((
        xx_new.ravel(),
        yy_new.ravel(),
        np.ones(new_h * new_w)
    ))

    pts_orig_h = H_inv @ pts_new_h

    # Convert back to Cartesian coordinates
    w_orig = pts_orig_h[2, :]
    valid_w = np.abs(w_orig) >= 1e-8  # Mask for valid points
    pts_orig = np.zeros((2, new_h * new_w))  # Initialize with zeros

    # Calculate coordinates only for valid points
    pts_orig[0, valid_w] = pts_orig_h[0, valid_w] / w_orig[valid_w]  # x_orig
    pts_orig[1, valid_w] = pts_orig_h[1, valid_w] / w_orig[valid_w]  # y_orig

    # Reshape to image dimensions
    x_orig = pts_orig[0, :].reshape(new_h, new_w)
    y_orig = pts_orig[1, :].reshape(new_h, new_w)
    valid_mask = valid_w.reshape(new_h, new_w)  # Also reshape the validity mask

    # sample pixel values using bilinear interpolation
    x1 = np.floor(x_orig).astype(int)
    y1 = np.floor(y_orig).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = x_orig - x1
    dy = y_orig - y1

    # Create boundary masks to check if the 4 neighbors are within the original image bounds
    mask11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
    mask12 = (x1 >= 0) & (x1 < w) & (y2 >= 0) & (y2 < h)
    mask21 = (x2 >= 0) & (x2 < w) & (y1 >= 0) & (y1 < h)
    mask22 = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)

    # Combine masks
    valid_interp_mask = mask11 & mask12 & mask21 & mask22 & valid_mask

    y1_valid = y1[valid_interp_mask]
    x1_valid = x1[valid_interp_mask]
    y2_valid = y2[valid_interp_mask]
    x2_valid = x2[valid_interp_mask]
    dx_valid = dx[valid_interp_mask]
    dy_valid = dy[valid_interp_mask]

    # Get neighbor pixel values for valid points
    if is_color:
        dx_valid = dx_valid[:, np.newaxis]
        dy_valid = dy_valid[:, np.newaxis]

        Q11 = im[y1_valid, x1_valid, :]
        Q12 = im[y2_valid, x1_valid, :]
        Q21 = im[y1_valid, x2_valid, :]
        Q22 = im[y2_valid, x2_valid, :]
    else:
        Q11 = im[y1_valid, x1_valid]
        Q12 = im[y2_valid, x1_valid]
        Q21 = im[y1_valid, x2_valid]
        Q22 = im[y2_valid, x2_valid]

    # Perform bilinear interpolation only for valid points
    R1 = (1 - dx_valid) * Q11 + dx_valid * Q21
    R2 = (1 - dx_valid) * Q12 + dx_valid * Q22
    interpolated_values = (1 - dy_valid) * R1 + dy_valid * R2

    new_image[valid_interp_mask] = interpolated_values.astype(im.dtype)
    return new_image, offset



def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Find matches between two images using SIFT
    Arguments:
        img1 - the first image
        img2 - the second image
    Returns:
        kp1 - the keypoints of the first image
        kp2 - the keypoints of the second image
        matches - the matches between the keypoints
    """
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # k-NN matching (k=2 for Lowe's ratio test)
    matches = flann.knnMatch(des1, des2, k=2)

    #  Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches

def show_matches(img1: np.array, 
                 img2: np.array, 
                 kp1: list, 
                 kp2: list, 
                 matches: list) -> np.array:
    result_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(result_img)
    plt.title("SIFT Matches")
    plt.show()
    return result_img


if __name__ == '__main__':
    if not os.path.exists(env.p6.output):
        os.makedirs(env.p6.output)
    expected_e1, expected_e2 = np.load(env.p6.expected_e1), np.load(env.p6.expected_e2)
    expected_H1, expected_H2 = np.load(env.p6.expected_H1), np.load(env.p6.expected_H2)
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)

    points1 = utils.load_points(env.p5.pts_1)
    points2 = utils.load_points(env.p5.pts_2)
    assert (points1.shape == points2.shape)
    F = normalized_eight_point_alg(points1, points2)

    # Part 6.a
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)
    assert np.allclose(e1, expected_e1, rtol=1e-2), f"e1 does not match this expected value:\n{expected_e1}"
    assert np.allclose(e2, expected_e2, rtol=1e-2), f"e2 does not match this expected value:\n{expected_e2}"
    np.save(env.p6.e1, e1)
    np.save(env.p6.e2, e2)

    # Part 6.b
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)
    H1 = H1 / H1[2, 2]
    H2 = H2 / H2[2, 2]
    assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    np.save(env.p6.H1, H1)
    np.save(env.p6.H2, H2)

    # Part 6.c
    rectified_im1, offset1 = compute_rectified_image(im1, expected_H1)
    rectified_im2, offset2 = compute_rectified_image(im2, expected_H2)

    new_points1 = expected_H1.dot(points1.T)
    new_points2 = expected_H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)
    total_offset_y = np.mean(new_points1[:, 1] - new_points2[:, 1]).round()

    F_new = normalized_eight_point_alg(new_points1, new_points2)
    lines1 = compute_epipolar_lines(new_points2, F_new.T)
    lines2 = compute_epipolar_lines(new_points1, F_new)
    aligned_img = show_epipolar_imgs(rectified_im1, rectified_im2, lines1, lines2, new_points1, new_points2, offset=int(total_offset_y))
    Image.fromarray(aligned_img).save(env.p6.aligned_epipolar)

    # Part 6.d
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)
    kp1, kp2, good_matches = find_matches(im1, im2)
    cv_matches = show_matches(im1, im2, kp1, kp2, good_matches)
    Image.fromarray(cv_matches).save(env.p6.cv_matches)
