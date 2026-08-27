import os

import numpy as np


def planarPoses(positions, yaws):
    positions = np.asarray(positions)
    yaws = np.asarray(yaws).reshape(-1)
    poses = np.repeat(np.eye(4)[None], len(yaws), axis=0)
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    poses[:, 0, 0] = cos_yaw
    poses[:, 0, 1] = -sin_yaw
    poses[:, 1, 0] = sin_yaw
    poses[:, 1, 1] = cos_yaw
    poses[:, :2, 3] = positions
    return poses


def writeAzimuthOdometryArrays(
    output_path,
    frame_times_us,
    reference_poses,
    azimuth_times_us,
    frame_offsets,
    odom_transforms,
    frame_transforms,
    frame_body_velocities,
    velocity_start_us,
    velocity_end_us,
):
    if not np.allclose(
        frame_transforms[1:] @ reference_poses[:-1], reference_poses[1:]
    ):
        raise ValueError("Invalid inter-frame transform composition.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        frame_timestamps_us=frame_times_us,
        reference_poses=reference_poses,
        azimuth_timestamps_us=azimuth_times_us,
        frame_offsets=frame_offsets,
        odom_transforms=odom_transforms,
        frame_transforms=frame_transforms,
        frame_body_velocities=frame_body_velocities,
        velocity_start_us=velocity_start_us,
        velocity_end_us=velocity_end_us,
        odom_transform_convention=np.asarray("right"),
    )
