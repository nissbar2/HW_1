import os
import sys

from scipy.optimize import least_squares

import env
import src.utils.engine
import src.utils.utils as utils
import numpy as np

from src.calibrate_camera import *
from src.reconstruction_3D import *
from src.image_rectification import *


def reprojection_error(camera_params, camera_indices, point_indices, observed_2d):
    """
    Compute the reprojection error for the bundle adjustment problem.
    """
    # TODO: Extra credit!
    return NotImplementedError

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, observed_2d, camera_matrix):
    """
    Refines camera parameters and 3D points using bundle adjustment.
    """

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    def pack(cam_params, pts_3d):
        return np.hstack((cam_params.ravel(), pts_3d.ravel()))

    def unpack(x):
        cam_params = x[:n_cameras * 6].reshape((n_cameras, 6))
        pts_3d = x[n_cameras * 6:].reshape((n_points, 3))
        return cam_params, pts_3d

    def reprojection_error(packed_params):
        cam_params, pts_3d = unpack(packed_params)
        num_obs = camera_indices.shape[0]
        errors = np.zeros((num_obs, 2))

        for i in range(num_obs):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]

            rvec = cam_params[cam_idx, :3]
            tvec = cam_params[cam_idx, 3:6]
            pt3d = pts_3d[pt_idx].reshape(1, 3)

            proj_2d, _ = cv2.projectPoints(pt3d, rvec, tvec, camera_matrix, distCoeffs=None)
            errors[i] = proj_2d.ravel() - observed_2d[i]

        return errors.ravel()

    x0 = pack(camera_params, points_3d)

    result = least_squares(reprojection_error, x0, verbose=2, method='trf', ftol=1e-6, xtol=1e-6)

    camera_params_opt, points_3d_opt = unpack(result.x)
    return camera_params_opt, points_3d_opt


def main():
    import cv2
    import numpy as np
    from src.SfM_pipeline import bundle_adjustment
    global points_3d, camera_matrix

    if not os.path.exists(env.p8.output):
        os.makedirs(env.p8.output)

    chessboard_size = (16, 10)
    images_folder = env.p8.statue_images
    chessboard_path = env.p7.chessboard

    # Intrinsics
    camera_matrix = np.eye(3)
    focal_length = 719.5459
    camera_matrix[0, 0] = focal_length
    camera_matrix[1, 1] = focal_length
    camera_matrix[0, 2] = 640
    camera_matrix[1, 2] = 480

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    global_points_3D = []
    R_global_prev = np.eye(3, dtype=np.float64)
    t_global_prev = np.zeros((3,), dtype=np.float64)

    camera_params = []
    observed_2d = []
    camera_indices = []
    point_indices = []
    all_3d_points = []
    point_counter = 0

    camera_params.append(np.hstack((cv2.Rodrigues(R_global_prev)[0].ravel(), t_global_prev.ravel())))

    for i in range(len(image_files) - 1):
        img_path1 = os.path.join(images_folder, image_files[i])
        img_path2 = os.path.join(images_folder, image_files[i + 1])

        im1 = utils.load_image(img_path1)
        im2 = utils.load_image(img_path2)

        kp1, kp2, good_matches = find_matches(im1, im2)
        F, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, good_matches)
        inlier_pts1, inlier_pts2 = get_inliers(mask, pts1, pts2)
        E = compute_essential_matrix(camera_matrix, F)
        R_candidates, t_candidates = estimate_initial_RT(E)
        R_local, t_local = find_best_RT(R_candidates, t_candidates, inlier_pts1, inlier_pts2, camera_matrix)
        P1_local = get_identity_projection_matrix(camera_matrix)
        P2_local = get_local_projection_matrix(camera_matrix, R_local, t_local)

        pts4D_h = cv2.triangulatePoints(P1_local, P2_local, inlier_pts1, inlier_pts2)
        pts3D_local = (pts4D_h[:3] / pts4D_h[3]).T

        R_global_curr = R_global_prev @ R_local
        t_global_curr = R_global_prev @ t_local + t_global_prev

        pts3D_global = (R_global_prev @ pts3D_local.T).T + t_global_prev

        global_points_3D.append(pts3D_global)
        all_3d_points.append(pts3D_global)

        # Store new camera extrinsics
        rvec_curr, _ = cv2.Rodrigues(R_global_curr)
        camera_params.append(np.hstack((rvec_curr.ravel(), t_global_curr.ravel())))

        # Fill reprojection data
        for j in range(pts3D_global.shape[1] - 1):
            observed_2d.append(inlier_pts1[j])
            camera_indices.append(i)
            point_indices.append(point_counter)

            observed_2d.append(inlier_pts2[j])
            camera_indices.append(i + 1)
            point_indices.append(point_counter)

            point_counter += 1

        R_global_prev = R_global_curr
        t_global_prev = t_global_curr

        print(f"Pair {i}->{i+1}: Triangulated {len(pts3D_local)} local points; "
              f"transformed to global. Pose R=\n{R_global_curr}\nt={t_global_curr}\n")

    if global_points_3D:
        points_3d = np.vstack(all_3d_points)
        observed_2d = np.vstack(observed_2d)
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        camera_params = np.array(camera_params)

        print(f"\nRunning bundle adjustment on {points_3d.shape[0]} points and {camera_params.shape[0]} cameras...\n")

        camera_params_opt, points_3d_opt = bundle_adjustment(
            camera_params,
            points_3d,
            camera_indices,
            point_indices,
            observed_2d,
            camera_matrix
        )

        print("Bundle adjustment complete.")
        show_points_matplotlib(points_3d_opt)
    else:
        print("No points were accumulated.")

if __name__ == '__main__':
    main()

        
        
