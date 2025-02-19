import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer
from mpl_toolkits.mplot3d import Axes3D
from rtc_vision_toolbox.calibration.calibrations import (
    solve_rigid_transformation,
)

from rpad.calibration.estimate_tag_and_cam import jointly_optimize_gripper_cam
from rpad.calibration.utils import plot_3d_frame


# From https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better/75871586#75871586
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the
    old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []

    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points,
            c,
            mtx,
            distortion,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def computer_errs(T1, T2):
    t_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    r, _ = cv2.Rodrigues(T1[:3, :3].T @ T2[:3, :3])
    r_err_deg = np.rad2deg(np.linalg.norm(r))
    return t_err, r_err_deg


def calibrate(
    output_dir: Path = Path.cwd() / "captures/output_20250217_203007",
    debug: bool = False,
    nsteps: int = 10000,
    alpha: float = 1000,
):
    # Load in intrinsics from numpytxt
    intrinsics = np.loadtxt(str(output_dir / "intrinsics.txt"))

    # Load camera poses
    poses = pickle.load(open(str(output_dir / "poses.pkl"), "rb"))

    # Get the first dict item
    first_poses = list(poses.values())[0]
    # Check if there's "ground-truth" available.
    has_gt_camera = "T_world_from_camera" in first_poses
    has_gt_tag = "T_world_from_tag" in first_poses

    # Gather some static transformations.
    T_world_from_base_link = list(poses.values())[0]["T_world_from_base_link"]
    if has_gt_camera:
        T_world_from_camera_gt = list(poses.values())[0]["T_world_from_camera"]
        # Rotate T_world_from_camera around the x axis 180 degrees, due
        # to mujoco conventions.
        T_world_from_camera_gt = T_world_from_camera_gt @ np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])

    # Load image names, with keys from poses.
    image_names = list(poses.keys())

    T_base_to_target_set = []
    Ts_world_to_gripper_base = []
    Ts_camera_from_tag_est = []

    # Perform tag detection, and estimate tag pose in the camera frame.
    for img_name in image_names:
        # Load image
        image = cv2.imread(str(output_dir / img_name))

        # CV2 tag detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            print(f"No tags detected in image {img_name}")
            continue

        # Size in meters.
        marker_length = 0.05
        K = intrinsics
        dist_coeffs = np.zeros(5)

        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            corners, marker_length, K, dist_coeffs
        )
        assert len(ids) == 1
        rvec, tvec = rvecs[0], tvecs[0]

        # Draw the tags on there.
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.drawFrameAxes(image, K, dist_coeffs, rvec, tvec, marker_length * 0.5)

        # Compute tag pose in camera frame using cv2.
        R_tag_from_camera, _ = cv2.Rodrigues(rvec)

        # Construct the homogeneous transformation matrix
        T_camera_from_tag_est = np.eye(4)
        T_camera_from_tag_est[:3, :3] = R_tag_from_camera
        T_camera_from_tag_est[:3, 3] = tvec.flatten()

        # Now compare against the poses.
        # Set numpy print option to only have 3 decimal places
        np.set_printoptions(precision=3, suppress=True)

        T_world_from_base_link = poses[img_name]["T_world_from_base_link"]
        T_world_from_gripper_base = poses[img_name]["T_world_from_gripper_base"]
        T_tag_from_camera_est = np.linalg.inv(T_camera_from_tag_est)

        # Keep track of all estimates.
        Ts_camera_from_tag_est.append(T_camera_from_tag_est)
        Ts_world_to_gripper_base.append(T_world_from_gripper_base)

        if has_gt_tag:
            T_world_from_tag = poses[img_name]["T_world_from_tag"]
            T_world_from_camera_est = T_world_from_tag @ T_tag_from_camera_est
            T_base_to_target_set.append(T_world_from_tag)

            # If we have GT, we compute some error metrics.
            if has_gt_camera:
                t_err, r_err_deg = computer_errs(
                    T_world_from_camera_gt, T_world_from_camera_est
                )
                print(f"t error: {t_err:.4f} m, r error: {r_err_deg:.4f} deg")

        # Show the image
        if debug:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Per-sample, we can estimate.
        if debug and has_gt_tag:
            # Visualize reference frames with matplotlib.
            fig = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

            # Plot world frame at origin
            plot_3d_frame(np.eye(4), ax, label="World", scale=0.1)  # type: ignore
            plot_3d_frame(T_world_from_base_link, ax, label="Robot", scale=0.1)
            plot_3d_frame(T_world_from_gripper_base, ax, label="Gripper", scale=0.1)
            plot_3d_frame(T_world_from_tag, ax, label="Tag", scale=0.1)
            plot_3d_frame(T_world_from_camera_est, ax, label="Camera (est)", scale=0.1)

            if has_gt_camera:
                plot_3d_frame(T_world_from_camera_gt, ax, label="Camera", scale=0.1)
                T_world_from_tag_est = T_world_from_camera_gt @ T_camera_from_tag_est
                plot_3d_frame(T_world_from_tag_est, ax, label="Tag (est)", scale=0.1)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-0.4, 1])
            plt.show()

    # We can use Shobhit's code to solve for a rigid transformation,
    # but only if we have ground-truth tag poses.
    if has_gt_tag:
        T_world_from_camera_est = solve_rigid_transformation(
            T_base_to_target_set, Ts_camera_from_tag_est, method="SVD_ALGEBRAIC"
        )
        if T_world_from_camera_est is None:
            raise ValueError("Failed to solve rigid transformation")

        if has_gt_camera:
            t_err, r_err_deg = computer_errs(
                T_world_from_camera_gt, T_world_from_camera_est
            )
            print(f"SVD Opt: t error: {t_err:.4f} m, r error: {r_err_deg:.4f} deg")

    # Apply joint optimization.
    Ts_world_from_gripper_base = np.asarray(Ts_world_to_gripper_base)
    Ts_camera_from_tag_est = np.asarray(Ts_camera_from_tag_est)

    if has_gt_tag:
        T_gripper_base_from_tag_guess = np.stack([
            np.linalg.inv(poses[key]["T_world_from_gripper_base"])
            @ poses[key]["T_world_from_tag"]
            for key in poses.keys()
        ])[0]

    T_gripper_base_from_tag_est, T_world_from_camera_est = jointly_optimize_gripper_cam(
        Ts_world_from_gripper_base,
        # Ts_camera_from_tag_gt,
        Ts_camera_from_tag_est,
        lr=0.01,
        nsteps=nsteps,
        alpha=alpha,
        # T_gripper_base_from_tag_init=T_gripper_base_from_tag_guess
    )

    print(f"Pred T_world_from_camera_est:\n{T_world_from_camera_est}")
    print(f"Pred T_gripper_base_from_tag_est:\n{T_gripper_base_from_tag_est}")

    # Joint optimization error.
    if has_gt_camera:
        print(f"GT T_world_from_camera: {T_world_from_camera_gt}")
        print(f"GT T_gripper_base_from_tag: {T_gripper_base_from_tag_guess}")

        t_err, r_err_deg = computer_errs(
            T_world_from_camera_gt, np.array(T_world_from_camera_est)
        )

        print(f"Joint opt. t error: {t_err} m, r error: {r_err_deg} deg")

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    # Plot world frame at origin
    plot_3d_frame(np.eye(4), ax, label="World", scale=0.1)  # type: ignore
    plot_3d_frame(T_world_from_base_link, ax, label="Robot", scale=0.1)
    if has_gt_camera:
        plot_3d_frame(T_world_from_camera_gt, ax, label="Camera", scale=0.1)
    plot_3d_frame(T_world_from_camera_est, ax, label="Camera (est)", scale=0.1)  # type: ignore

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.4, 1])
    plt.show()


if __name__ == "__main__":
    typer.run(calibrate)
