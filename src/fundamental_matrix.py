import os
import sys
import env
import src.utils.utils as utils
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import numpy as np


def lstsq_eight_point_alg(points1: np.array, points2: np.array) -> np.array:
    '''
    Computes the fundamental matrix from matching points using 
    linear least squares eight point algorithm
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        F - the fundamental matrix such that (points2)^T * F * points1 = 0
    '''
    # Convert to homogeneous coordinates (N, 3)
    ones = np.ones((points1.shape[0], 1))
    x1 = np.hstack([points1, ones])
    x2 = np.hstack([points2, ones])

    # Construct matrix W (N, 9)
    W = np.zeros((points1.shape[0], 9))
    for i in range(points1.shape[0]):
        X = x1[i]
        Xp = x2[i]
        W[i] = [
            Xp[0]*X[0], Xp[0]*X[1], Xp[0]*X[2],
            Xp[1]*X[0], Xp[1]*X[1], Xp[1]*X[2],
            Xp[2]*X[0], Xp[2]*X[1], Xp[2]*X[2]
        ]

    # Solve Wf = 0 using SVD -> f = last column of V
    _, _, Vt = np.linalg.svd(W)
    f = Vt[-1]
    F = f.reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # smallest singular value to 0
    F_rank2 = U @ np.diag(S) @ Vt

    return F_rank2



def compute_normalization_matrix(points_h: np.array) -> np.array:
    """
    Computes the normalization matrix T for a set of points given in
    homogeneous coordinates.

    Args:
        points_h (np.array): Nx3 array of homogeneous points.

    Returns:
        np.array: 3x3 normalization matrix T.
    """
    if points_h.shape[1] != 3:
        raise ValueError("Input points must be Nx3 homogeneous coordinates.")
    if points_h.shape[0] == 0:
        raise ValueError("Points array cannot be empty")

    # Convert to non-homogeneous coordinates for centroid/scale calculation
    # Handle potential division by zero if w is zero, although unlikely for image points
    w = points_h[:, 2]
    # Create a mask for non-zero w values
    valid_w_mask = np.abs(w) > np.finfo(float).eps
    if not np.all(valid_w_mask):
        print(f"Warning: {np.sum(~valid_w_mask)} points have near-zero w coordinate. Excluding them from normalization calculation.")
        # Consider raising an error or handling these points specifically if needed
        points_h_valid = points_h[valid_w_mask]
        if points_h_valid.shape[0] < 2: # Need at least 2 points for std dev / scaling
             raise ValueError("Not enough valid points (w!=0) for normalization.")
        w_valid = points_h_valid[:, 2]
        points_nh = points_h_valid[:, :2] / w_valid[:, np.newaxis] # Use np.newaxis for broadcasting
    else:
        # If all w are valid (non-zero)
        points_nh = points_h[:, :2] / w[:, np.newaxis]

    if points_nh.shape[0] == 0:
         raise ValueError("No valid points with non-zero w to compute normalization.")


    # 1. Compute centroid of non-homogeneous points
    centroid = np.mean(points_nh, axis=0)
    cx, cy = centroid[0], centroid[1]

    # 2. Translate points so centroid is at the origin
    translated_points = points_nh - centroid

    # 3. Compute mean squared distance from the origin
    squared_distances = np.sum(translated_points**2, axis=1)
    mean_squared_distance = np.mean(squared_distances)

    # Handle case where all points are the same (mean_squared_distance is 0)
    if mean_squared_distance <= np.finfo(float).eps:
         # If all points map to the same non-homogeneous point, just translate
         scale = 1.0
    else:
        # 4. Calculate scaling factor so mean squared distance is 2
        # We want s^2 * mean_squared_distance = 2
        scale = np.sqrt(2.0 / mean_squared_distance)

    # 5. Construct the normalization matrix T = T_scale @ T_translate
    # Translation matrix to move centroid to origin
    T_translate = np.array([[1, 0, -cx],
                            [0, 1, -cy],
                            [0, 0,  1]])

    # Scaling matrix
    T_scale = np.array([[scale, 0,     0],
                        [0,     scale, 0],
                        [0,     0,     1]])

    # Combined normalization matrix
    T = T_scale @ T_translate

    return T

def normalized_eight_point_alg(points1_h: np.array, points2_h: np.array) -> np.array:
    '''
    Computes the fundamental matrix from matching points in homogeneous coordinates
    using the normalized eight point algorithm.

    Arguments:
        points1_h (np.array): N x 3 array of homogeneous points in the first image.
        points2_h (np.array): N x 3 array of homogeneous points in the second image.
                             Assumes points1_h[i] corresponds to points2_h[i].

    Returns:
        F (np.array): 3x3 fundamental matrix such that (points2_h)^T * F * points1_h = 0.
    '''
    if points1_h.shape[0] != points2_h.shape[0]:
        raise ValueError("Number of points in points1 and points2 must be the same.")
    if points1_h.shape[0] < 8:
        raise ValueError("At least 8 points are required for the eight-point algorithm.")
    if points1_h.shape[1] != 3 or points2_h.shape[1] != 3:
        raise ValueError("Input points arrays must be Nx3 homogeneous coordinates.")

    N = points1_h.shape[0]

    # 1. Normalize the points
    # compute_normalization_matrix expects homogeneous coords as input
    T1 = compute_normalization_matrix(points1_h)
    T2 = compute_normalization_matrix(points2_h)

    # Apply normalization transformations directly to homogeneous coordinates
    # Resulting shapes are (N, 3)
    # Formula: p'_h = T @ p_h
    # Implementation: norm_points_h = (T @ points_h.T).T
    norm_points1_h = (T1 @ points1_h.T).T
    norm_points2_h = (T2 @ points2_h.T).T

    # Convert normalized homogeneous coordinates to normalized non-homogeneous
    # coordinates for constructing matrix A.
    w1_norm = norm_points1_h[:, 2]
    w2_norm = norm_points2_h[:, 2]

    # --- Robustness Check (Optional but recommended) ---
    # Check if any normalized w are close to zero, which could indicate issues.
    if np.any(np.abs(w1_norm) < np.finfo(float).eps) or \
       np.any(np.abs(w2_norm) < np.finfo(float).eps):
        # This case is less likely with standard normalization but could happen
        # with ill-conditioned data or different normalization schemes.
        raise ValueError("Normalized points resulted in zero homogeneous coordinate (w). Cannot proceed.")
        # Or handle by removing these points if appropriate for the application.
    # --- End Check ---

    norm_points1 = norm_points1_h[:, :2] / w1_norm[:, np.newaxis]
    norm_points2 = norm_points2_h[:, :2] / w2_norm[:, np.newaxis]


    # 2. Construct the constraint matrix A using normalized non-homogeneous points
    # For each pair (x1', y1') and (x2', y2'), the equation is:
    # x2'*x1'*f11 + x2'*y1'*f12 + x2'*f13 + y2'*x1'*f21 + y2'*y1'*f22 + y2'*f23 + x1'*f31 + y1'*f32 + f33 = 0
    # A is an N x 9 matrix
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = norm_points1[i]
        x2, y2 = norm_points2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # 3. Solve Af = 0 using SVD
    U, S, Vh = np.linalg.svd(A)
    # The solution f_norm is the last column of V, which is the last row of Vh
    f_norm = Vh[-1, :]

    # Reshape f into the 3x3 fundamental matrix F_norm
    F_norm_initial = f_norm.reshape(3, 3)

    # 4. Enforce the rank-2 constraint using SVD
    U_f, S_f, Vh_f = np.linalg.svd(F_norm_initial)
    # Set the smallest singular value to zero
    S_f[-1] = 0
    # Reconstruct F_norm with rank 2
    F_norm = U_f @ np.diag(S_f) @ Vh_f

    # 5. Denormalize the fundamental matrix
    # F = T2^T * F_norm * T1
    F = T2.T @ F_norm @ T1

    return F

def compute_epipolar_lines(points: np.array, F: np.array) -> np.array:
    """
    Computes the epipolar lines in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points - N points in the first image that match with points2
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0    
    Returns:
        lines - the epipolar lines in homogenous coordinates
    """
    num_points = points.shape[0]
    lines = np.zeros((num_points, 2))

    for i in range(num_points):
        x, y, _ = points[i]
        p_homog = np.array([x, y, 1])
        l = F @ p_homog  # epipolar line coefficients [A, B, C]

        A, B, C = l
        if B != 0:
            m = -A / B
            b = -C / B
        else:
            # Handle vertical lines (undefined slope)
            m = np.inf
            b = np.nan  # or some fallback value

        lines[i] = [m, b]

    return lines


def show_epipolar_imgs(img1: np.ndarray, 
                       img2: np.ndarray, 
                       lines1: np.ndarray, 
                       lines2: np.ndarray, 
                       pts1: np.ndarray, 
                       pts2: np.ndarray, 
                       offset: int=0) -> np.ndarray:
    epi_img1 = get_epipolar_img(img1, lines1, pts1)
    epi_img2 = get_epipolar_img(img2, lines2, pts2)

    if offset < 0:
        h1, w1, c1 = epi_img1.shape
        padding = np.zeros((-offset, w1, c1), dtype=epi_img1.dtype)
        epi_img1 = np.vstack((padding, epi_img1))
    else:
        h2, w2, c2 = epi_img2.shape
        padding = np.zeros((offset, w2, c2), dtype=epi_img1.dtype)
        epi_img2 = np.vstack((padding, epi_img2))
    
    h1, w1, c1 = epi_img1.shape
    h2, w2, c2 = epi_img2.shape

    max_h = max(h1, h2)

    if h1 < max_h:
        pad_height = max_h - h1
        padding = np.zeros((pad_height, w1, c1), dtype=epi_img1.dtype)
        epi_img1 = np.vstack((padding, epi_img1))

    if h2 < max_h:
        pad_height = max_h - h2
        padding = np.zeros((pad_height, w2, c2), dtype=epi_img2.dtype)
        epi_img2 = np.vstack((epi_img2, padding))

    combined_img = np.hstack((epi_img1, epi_img2))
    plt.imshow(combined_img)
    plt.title("Epipolar Lines")
    plt.show()

    return combined_img   

def draw_points(img: np.ndarray, 
                points: np.ndarray, 
                color: tuple=(0, 255, 0), 
                radius: int=5) -> np.ndarray:
    img_with_corners = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_corners)

    for (x, y, _) in points:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)
    
    return np.array(img_with_corners)

def draw_lines(img: np.ndarray, 
               lines: np.ndarray, 
               color: tuple=(255, 0, 0), 
               thickness: int=3) -> np.ndarray:
    from PIL import Image, ImageDraw
    import numpy as np

    img_with_lines = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_lines)
    width, _ = img_with_lines.size

    for (m, b) in lines:
        # Compute two endpoints using x = 0 and x = width.
        x1 = 0
        y1 = m * x1 + b
        x2 = width
        y2 = m * x2 + b

        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

    return np.array(img_with_lines)


def compute_distance_to_epipolar_lines(points1: np.array, 
                                       points2: np.array, 
                                       F: np.array) -> float:
    l = F.T.dot(points2.T)
    # distance from point(x0, y0) to line: Ax + By + C = 0 is
    # |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    d = np.mean(np.abs(np.sum(l * points1.T, axis=0)) / np.sqrt(l[0, :] ** 2 + l[1, :] ** 2))
    return d


def get_epipolar_img(img: np.ndarray, 
                     lines: np.ndarray, 
                     points: np.ndarray) -> np.ndarray:
    lines_img = draw_lines(img, lines)
    points_img = draw_points(lines_img, points)
    return points_img 

if __name__ == '__main__':
    if not os.path.exists(env.p5.output):
        os.makedirs(env.p5.output)
    expected_F_LLS = np.load(env.p5.expected_F_LLS)
    expected_dist_im1_LLS, expected_dist_im2_LLS = np.load(env.p5.expected_dist_LLS)

    expected_F_normalized = np.load(env.p5.expected_F_normalized)
    expected_dist_im1_normalized, expected_dist_im2_normalized = np.load(env.p5.expected_dist_normalized)

    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)

    points1 = utils.load_points(env.p5.pts_1)
    points2 = utils.load_points(env.p5.pts_2)
    assert (points1.shape == points2.shape)

    # Part 5.a
    F_lls = lstsq_eight_point_alg(points1, points2)
    print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
    assert np.allclose(F_lls, expected_F_LLS, atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_LLS}"
    np.save(env.p5.F_LLS, F_lls)

    dist_im1_LLS = compute_distance_to_epipolar_lines(points1, points2, F_lls)
    dist_im2_LLS = compute_distance_to_epipolar_lines(points2, points1, F_lls.T)
    print("Distance to lines in image 1 for LLS:", \
        dist_im1_LLS)
    print("Distance to lines in image 2 for LLS:", \
        dist_im2_LLS)
    assert np.allclose(dist_im1_LLS, expected_dist_im1_LLS, atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_LLS}"
    assert np.allclose(dist_im2_LLS, expected_dist_im2_LLS, atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_LLS}"
    np.save(env.p5.dist_LLS, np.array([dist_im1_LLS, dist_im2_LLS]))

    # Part 5.b
    F_normalized = normalized_eight_point_alg(points1, points2)
    print("Fundamental Matrix from normalized 8-point algorithm:\n", \
        F_normalized)
    assert np.allclose(F_normalized, expected_F_normalized, atol=1e-2), f"Fundamental matrix does not match this expected matrix:\n{expected_F_normalized}"

    dist_im1_normalized = compute_distance_to_epipolar_lines(points1, points2, F_normalized)
    dist_im2_normalized = compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)
    print("Distance to lines in image 1 for normalized:", \
        dist_im1_normalized)
    print("Distance to lines in image 2 for normalized:", \
        dist_im2_normalized)
    assert np.allclose(dist_im1_normalized, expected_dist_im1_normalized, atol=1e-2), f"Distance to lines in image 1 does not match this expected distance: {expected_dist_im1_normalized}"
    assert np.allclose(dist_im2_normalized, expected_dist_im2_normalized, atol=1e-2), f"Distance to lines in image 2 does not match this expected distance: {expected_dist_im2_normalized}"
    np.save(env.p5.dist_normalized, np.array([dist_im1_normalized, dist_im2_normalized]))

    # Part 5.c
    lines1 = compute_epipolar_lines(points2, F_lls.T)
    lines2 = compute_epipolar_lines(points1, F_lls)
    lls_img = show_epipolar_imgs(im1, im2, lines1, lines2, points1, points2)
    Image.fromarray(lls_img).save(env.p5.lls_img)

    lines1 = compute_epipolar_lines(points2, F_normalized.T)
    lines2 = compute_epipolar_lines(points1, F_normalized)
    norm_img = show_epipolar_imgs(im1, im2, lines1, lines2, points1, points2)
    Image.fromarray(norm_img).save(env.p5.norm_img)
