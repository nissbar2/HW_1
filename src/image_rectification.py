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
    

def compute_matching_homographies(e2: np.array,
                                  F: np.array,
                                  im2: np.array,
                                  points1: np.array,
                                  points2: np.array) -> tuple:
    '''
    Determines homographies H1 and H2 such that they
    rectify a pair of images
    Arguments:
        e2 - the second epipole
        F - the Fundamental matrix
        im2 - the second image
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
    Returns:
        H1 - the homography associated with the first image
        H2 - the homography associated with the second image
    '''
    # === Step 1: Compute H2 ===

    # Move image center to origin
    h, w = im2.shape[:2]
    T = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0,    1  ]
    ])

    e = e2 / e2[2]  # normalize epipole

    # Rotate epipole to lie on x-axis
    ex, ey = e[0], e[1]
    r = np.sqrt(ex**2 + ey**2)
    sin_theta = ey / r
    cos_theta = ex / r
    R = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Send epipole to infinity along x-axis
    f = np.sqrt(ex**2 + ey**2)
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / f, 0, 1],
    ])

    H2 = G @ R @ T  # final homography for image 2

    # === Step 2: Compute H1 using least squares ===

    # Transform points2 with H2
    def to_homog(p):
        if p.shape[1] == 2:
            return np.hstack([p, np.ones((p.shape[0], 1))])
        elif p.shape[1] == 3:
            return p
        else:
            raise ValueError("Point array must have shape (N,2) or (N,3)")

    def from_homog(p): return (p[:, :2].T / p[:, 2]).T

    # Convert 2D points to homogeneous coordinates
    p2_h = to_homog(points2)  # shape: (N, 3)

    # Transform points2 using H2
    p2_prime_h = (H2 @ p2_h.T).T  # shape: (N, 3)
    p2_prime = p2_prime_h[:, :2] / p2_prime_h[:, 2:]

    # Now estimate H1 that maps points1 to p2_prime
    H1, _ = cv2.findHomography(points1[:, :2], p2_prime)

    # Normalize
    H1 = H1 / H1[2, 2]
    H2 = H2 / H2[2, 2]

    return H1, H2


def compute_rectified_image(im: np.array,
                            H: np.array) -> tuple:
    '''
    Rectifies an image using a homography matrix
    Arguments:
        im - an image
        H - a homography matrix that rectifies the image
    Returns:
        new_image - a new image matrix after applying the homography
        offset - the offest in the image.
    '''

    # TODO: Implement this method!

    H_inv = np.linalg.inv(H)

    h, w = im.shape[:2]

    # Step 1: Map the corners of the image to get output bounds
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ])  # (4, 3)

    warped_corners = (H @ corners.T).T  # (4, 3)
    warped_corners /= warped_corners[:, 2][:, None]

    x_coords = warped_corners[:, 0]
    y_coords = warped_corners[:, 1]

    x_min, x_max = np.floor(x_coords.min()), np.ceil(x_coords.max())
    y_min, y_max = np.floor(y_coords.min()), np.ceil(y_coords.max())

    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    # Offset to shift all coords into positive bounds
    x_offset = -x_min
    y_offset = -y_min
    offset = (y_offset, x_offset)

    # Step 2: Create meshgrid for new image
    xx, yy = np.meshgrid(np.arange(new_w), np.arange(new_h))
    ones = np.ones_like(xx)
    grid = np.stack([xx + x_min, yy + y_min, ones], axis=-1)  # (H, W, 3)

    # Step 3: Map these new pixels back to original image using H⁻¹
    flat_grid = grid.reshape(-1, 3).T  # (3, H*W)
    orig_coords = (H_inv @ flat_grid).T  # (H*W, 3)
    orig_coords /= orig_coords[:, 2][:, None]
    x_src = orig_coords[:, 0]
    y_src = orig_coords[:, 1]

    # Step 4: Interpolate using scipy.ndimage.map_coordinates
    if im.ndim == 2:
        # Grayscale
        warped = map_coordinates(im, [y_src, x_src], order=1, mode='reflect')
        new_image = warped.reshape((new_h, new_w))
    else:
        # Color image: warp each channel
        warped_channels = []
        for c in range(im.shape[2]):
            warped = map_coordinates(im[:, :, c], [y_src, x_src], order=1, mode='reflect')
            warped_channels.append(warped.reshape((new_h, new_w)))
        new_image = np.stack(warped_channels, axis=2)

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
    # Step 1: Create SIFT detector
    sift = cv2.SIFT_create()

    # Step 2: Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Step 3: FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Step 4: Perform k-NN matching (k=2 for Lowe's ratio test)
    matches = flann.knnMatch(des1, des2, k=2)

    # Step 5: Apply Lowe's ratio test
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
