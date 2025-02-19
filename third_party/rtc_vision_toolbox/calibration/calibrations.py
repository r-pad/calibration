"""Borrowed from
https://github.com/cmu-mfi/rtc_vision_toolbox/blob/412e31982bc303a7df7929d70e800fb9a2b58754/calibration/calibrations.py

Changes:
* Comment out open3d import, as it is not used in this file.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
# import open3d as o3d

from datetime import datetime
#from autolab_core import RigidTransform
import cv2
import time

MIN_TRIALS = 3
MAX_TRIALS = 50

def get_camera_marker_tf(camera, marker):

    image = camera.get_rgb_image()
    depth = camera.get_raw_depth_data(use_new_frame=False)
    #convert depth data in mm to meters
    depth = depth/1000.0

    '''
    print("DISPLAY IMAGES")
    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.imshow(depth)
    fig.show()

    fig = px.imshow(image)
    fig.show()
    '''

    # get camera intrinsics
    camera_matrix = camera.get_rgb_intrinsics()
    print(camera_matrix.astype(np.float32))
    camera_distortion = camera.get_rgb_distortion()

    # get aruco marker poses w.r.t. camera
    transforms, ids = marker.get_center_poses(input_image = image,
                                              camera_matrix = camera_matrix,
                                              camera_distortion = camera_distortion,
                                              depth_image = depth,
                                              debug=True)

    # print the transformation
    for i in range(len(ids)):
        print("Transform for id{} with depth image:\n {}".format(ids[i], transforms[i]))

    return transforms[0]

def get_robot_camera_tf(camera, robot, marker, T_eef2marker, method='JOG', num_trials=None, verbose=True, use_depth=True):
    '''
    Collects and solves for the transformation between the robot base and the camera

    Parameters:
        camera: Camera object
        robot: Robot object
        marker: Marker object
        method: Method to collect data
            * JOG: Manually move the robot
            * PLAY: Randomly move the robot
        num_trials: Number of trials to collect data
        verbose: Print debug information
        use_depth: Use depth data for calibration

    Returns:
        T_base2camera: Transformation matrix from robot base to camera

    Note:
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    '''

    # 1. Initialize the robot, camera and marker
    T_camera2marker_set = []
    T_base2eef_set = []

    # 2. Collect data
    T_base2eef_set, T_camera2marker_set = collect_data(camera, robot, marker, method, num_trials, verbose, use_depth)

    # 2.5. Save the data with data and timestamp for debugging
    print(datetime.now().strftime("%Y%m%d%H%M"))
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_camera2marker_set"+ now +".npy", T_camera2marker_set)
    np.save("T_base2eef_set"+ now +".npy", T_base2eef_set)

    # 3. Solve for the transformation
    # 3.1. Solve the extrinsic calibration between the marker and the base

    T_base2marker_set = [np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set]

    T_base2camera_set = []
    avg_error_set = []

    methods = ["ONE_SAMPLE_ESTIMATE", "SVD_ALGEBRAIC", "CALIB_HAND_EYE_TSAI", "CALIB_HAND_EYE_ANDREFF"]

    for method in methods:
        print(f"\nMETHOD: {method}")
        T_base2camera = solve_rigid_transformation(T_base2marker_set, T_camera2marker_set, method=method)
        avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)
        T_base2camera_set.append(T_base2camera)
        avg_error_set.append(avg_error)
        print(f"Transformation matrix T_base2camera:\n{T_base2camera}")
        print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    # 3.2. Save the best calibration and error for debugging
    T_base2camera = T_base2camera_set[np.argmin(avg_error_set)]
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_base2camera_"+ now +".npy", T_base2camera)

    return T_base2camera

def calculate_reprojection_error(T_a2t_set, T_b2t_set, T_a2b):
    '''
    Example use:
    avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)

    Note:
    T_a2b represents frame 'b' in frame 'a', or
    tranforms a point from frame 'b' to frame 'a'
    '''
    errors = []
    assert len(T_a2t_set) == len(T_b2t_set)

    # Calculate error using transformation projections
    # for i in range(len(T_b2t_set)):
    #     T_a2t_calc = np.dot(T_a2b, T_b2t_set[i])
    #     error = np.linalg.norm(T_a2t_set[i] - T_a2t_calc)
    #     errors.append(error)

    # Calculate error using pose projections
    for i in range(len(T_a2t_set)):
        pose_target_in_frame_b = T_b2t_set[i][:,3]
        pose_target_in_frame_a = np.dot(T_a2b, pose_target_in_frame_b)
        error = np.linalg.norm(T_a2t_set[i][:,3] - pose_target_in_frame_a)
        errors.append(error)

    # Compute average error
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    return avg_error, std_error

def solve_rigid_transformation(T_base2target_set, T_camera2target_set, method="ONE_SAMPLE_ESTIMATE"):
    """
    Solves for the rigid transformation between two sets of transformations

    Parameters:
        T_base2target_set: List of transformation matrices from base to target
        T_camera2target_set: List of transformation matrices from camera to target
        method: Method to use for solving the transformation
            * SVD_ALGEBRAIC: Algebraic method using SVD
            * ONE_SAMPLE_ESTIMATE: One sample estimate
            * CALIB_HAND_EYE_TSAI: Tsai's method using OpenCV library (T_base2target can be T_base2eef, error calc will be off though)
            * CALIB_HAND_EYE_ANDREFF: Andreff's method using OpenCV library (T_base2target can be T_base2eef, error calc will be off though)
            * CALIB_HAND_EYE_PARK: Park's method using OpenCV library (T_base2target can be T_base2eef, error calc will be off though)

    Returns:
        T_base2camera: Transformation matrix from base to camera

    Note:
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    """


    if method == "SVD_ALGEBRAIC":
        t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
        t_base2target = np.array([T[:3,3] for T in T_base2target_set])

        inpts = t_camera2target
        outpts = t_base2target

        assert inpts.shape == outpts.shape
        inpts, outpts = np.copy(inpts), np.copy(outpts)
        inpt_mean = inpts.mean(axis=0)
        outpt_mean = outpts.mean(axis=0)
        outpts -= outpt_mean
        inpts -= inpt_mean

        X = inpts.T
        Y = outpts.T

        covariance = np.dot(X, Y.T)
        U, s, V = np.linalg.svd(covariance)
        S = np.diag(s)

        assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
        V = V.T
        idmatrix = np.identity(3)
        idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
        R = np.dot(np.dot(V, idmatrix), U.T)
        t = outpt_mean.T - np.dot(R, inpt_mean)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        return T

    elif method == "ONE_SAMPLE_ESTIMATE":
        T_base2camera_set = []
        errors = []
        for i in range(len(T_base2target_set)):
            T_base2target = T_base2target_set[i]
            T_camera2target = T_camera2target_set[i]
            T_base2camera = np.dot(T_base2target, np.linalg.inv(T_camera2target))
            T_base2camera_set.append(T_base2camera)
            error, _ = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera)
            errors.append(error)

        min_error_index = np.argmin(errors)
        T_base2camera = T_base2camera_set[min_error_index]
        return T_base2camera

    elif method ==  "CALIB_HAND_EYE_TSAI":
        T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
        R_target2base = [T[:3,:3] for T in T_target2base_set]
        t_target2base = [T[:3,3] for T in T_target2base_set]

        R_camera2target = [T[:3,:3] for T in T_camera2target_set]
        t_camera2target = [T[:3,3] for T in T_camera2target_set]
        R_camera2base, t_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base,
                                                            R_camera2target, t_camera2target,
                                                            cv2.CALIB_HAND_EYE_TSAI)
        T_base2camera = np.eye(4)
        T_base2camera[:3,:3] = R_camera2base
        T_base2camera[:3,3] = np.squeeze(t_base2camera)
        return T_base2camera

    elif method == "CALIB_HAND_EYE_ANDREFF":
        T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
        R_target2base = [T[:3,:3] for T in T_target2base_set]
        t_target2base = [T[:3,3] for T in T_target2base_set]

        R_camera2target = [T[:3,:3] for T in T_camera2target_set]
        t_camera2target = [T[:3,3] for T in T_camera2target_set]
        R_camera2base, t_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base,
                                                            R_camera2target, t_camera2target,
                                                            cv2.CALIB_HAND_EYE_ANDREFF)
        T_base2camera = np.eye(4)
        T_base2camera[:3,:3] = R_camera2base
        T_base2camera[:3,3] = np.squeeze(t_base2camera)
        return T_base2camera

    elif method == "CALIB_HAND_EYE_PARK":
        T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
        R_target2base = [T[:3,:3] for T in T_target2base_set]
        t_target2base = [T[:3,3] for T in T_target2base_set]

        R_camera2target = [T[:3,:3] for T in T_camera2target_set]
        t_camera2target = [T[:3,3] for T in T_camera2target_set]
        R_camera2base, t_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base,
                                                            R_camera2target, t_camera2target,
                                                            cv2.CALIB_HAND_EYE_PARK)
        T_base2camera = np.eye(4)
        T_base2camera[:3,:3] = R_camera2base
        T_base2camera[:3,3] = np.squeeze(t_base2camera)
        return T_base2camera

    else:
        print("Invalid method. Aborting...")
        return None

def collect_data(camera, robot, marker, method='JOG', num_trials=None, verbose=True, use_depth=True):
    '''
    Collects data for robot-camera calibration

    Parameters:
        camera: Camera object
        robot: Robot object
        marker: Marker object
        method: Method to collect data
            * JOG: Manually move the robot
            * PLAY: Randomly move the robot
        num_trials: Number of trials to collect data
        verbose: Print debug information
        use_depth: Use depth data for calibration
    '''

    T_camera2marker_set = []
    T_base2eef_set = []

    if num_trials is None:
        num_trials = int(input("Enter the number of trials: "))

    if num_trials < MIN_TRIALS or num_trials > MAX_TRIALS:
        print("Invalid number of trials. Aborting...")
        return

    home = robot.get_eef_pose()
    home_pos = np.squeeze(home[:3,3])
    home_rot = home[:3,:3]
    home_quart = R.from_matrix(home_rot).as_quat()

    robot.move_to_pose(home_pos, home_quart)

    # Safety Check
    go_on = input("Ensure the robot is in safe position. Do you want to continue? (y/n): ")
    if(go_on != 'y'):
        return None

    for i in range(num_trials):
        retry = 'y'
        while retry=='y':

            if method == 'JOG':
                # 2.1. Move the robot with controller and collect data
                good_to_go = 'n'
                while good_to_go != 'y':
                    if i == num_trials - 1:
                        good_to_go = 'y'
                    else:
                        good_to_go = input("Jog the robot. Done? (y/n): ")


            # 2.1. Move the robot and collect data
            if method == 'PLAY':
                random_delta_pos = np.random.uniform(-0.05, 0.05, size=(3,))
                # random_delta_quart = np.random.uniform(-0.3, 0.3, size=(4,))
                random_delta_quart = np.random.uniform(-0.2, 0.2, size=(4,))
                robot_pose = robot.move_to_pose(position = home_pos + random_delta_pos,
                                                orientation = home_quart + random_delta_quart)
            if robot_pose is not None:
                # 2.2. Collect data from camera
                image = camera.get_rgb_image()
                if use_depth:
                    depth_data_mm = camera.get_raw_depth_data(use_new_frame=False)
                    depth_data = depth_data_mm/1000.0
                else:
                    depth_data = None
                camera_matrix = camera.get_rgb_intrinsics()
                camera_distortion = camera.get_rgb_distortion()
                transforms, ids = marker.get_center_poses(input_image = image,
                                                        camera_matrix = camera_matrix,
                                                        camera_distortion = camera_distortion,
                                                        depth_image = depth_data)

                if ids is None:
                    print("No markers found. Skipping this trial.")
                    retry = 'y'
                else:
                    if verbose:
                        print(f"Marker pose in camera frame:\n {transforms[0]}")
                        #if transform has nan values, retry
                    if np.isnan(transforms[0]).any():
                        print("Nan values found. Retrying...")
                        retry = 'y'
                    else:
                        time.sleep(1)
                        retry = 'N'
                        # retry = input("Retry? (y/N)")
            else:
                retry = 'y'

        # 2.3. Collect data from robot
        gripper_pose = robot.get_eef_pose()
        if verbose:
            print(f"Gripper pose in base frame:\n {gripper_pose}")

        # 2.4. Append data if valid
        if(gripper_pose.shape == (4,4) and transforms[0].shape == (4,4)):
            T_base2eef_set.append(gripper_pose)
            T_camera2marker_set.append(transforms[0])

        print(f"---TRIAL {i+1} COMPLETED---.")

    return T_base2eef_set, T_camera2marker_set


# WIP
'''def get_pcd_registration(source_pcd, target_pcd, method='ICP_P2P', verbose=True):
    ###
    Registers the source point cloud to the target point cloud.
    * Assuming that the source point cloud is in the target frame.
    * Use the method to fine-tune the transformation

    Parameters:
        source_pcd [numpy.ndarray or open3d.geometry.PointCloud]: Source point cloud
        target_pcd [numpy.ndarray or open3d.geometry.PointCloud]: Target point cloud
        method: Method to use for registration
            * ICP_P2Point: Iterative Closest Point with Point-to-Point correspondence
            * ICP_P2Plane: Iterative Closest Point with Point-to-Plane correspondence
    Returns:
        T_target2source: Transformation matrix to transform point in source frame to target frame

    Note:
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    ###

    # Convert numpy array to open3d point cloud
    if isinstance(source_pcd, np.ndarray):
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd)

    if isinstance(target_pcd, np.ndarray):
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pcd)

    # Initial Evaluation
    T_init = np.eye(4)
    threshold = 0.02
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, threshold, T_init)
    print("Initial evaluation:")
    print(evaluation)

    # Registration
    T_result = None
    if method == "ICP_P2Point":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        T_result = reg_p2p.transformation
        if verbose:
            print(reg_p2p)

    if method == "ICP_P2Plane":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T_result = reg_p2p.transformation
        if verbose:
            print(reg_p2p)

    else:
        print("Invalid method. Aborting...")
        return None

    # Final Evaluation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, threshold, T_result)
    print("Final evaluation:")
    print(evaluation)

    T_result = np.linalg.inv(T_result)
    return T_result
'''
