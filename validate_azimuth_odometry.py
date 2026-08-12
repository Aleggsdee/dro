import argparse
import os.path as osp

import numpy as np
from pyboreas import BoreasDataset
from pyboreas.utils.odometry import interpolate_poses
from scipy.spatial.transform import Rotation as R, Slerp


PERCENTILES = [50, 90, 95, 99, 100]


def pose_errors(estimate, ground_truth):
    error = estimate @ np.linalg.inv(ground_truth)
    translation_xyz = error[:, :3, 3]
    rotation_xyz_deg = np.rad2deg(R.from_matrix(error[:, :3, :3]).as_rotvec())
    return (
        np.linalg.norm(translation_xyz, axis=1),
        np.linalg.norm(rotation_xyz_deg, axis=1),
        translation_xyz,
        rotation_xyz_deg,
    )


def print_values(name, values, unit):
    stats = np.percentile(values, PERCENTILES)
    print(
        f"  {name} [{unit}] mean/rms/p50/p90/p95/p99/max: "
        f"{np.mean(values):.6g} / {np.sqrt(np.mean(values**2)):.6g} / "
        + " / ".join(f"{value:.6g}" for value in stats)
    )


def print_axis_values(name, values, labels, unit):
    rms = np.sqrt(np.mean(values**2, axis=0))
    p95 = np.percentile(np.abs(values), 95, axis=0)
    print(
        f"  {name} {labels} [{unit}] rms: {rms}; absolute p95: {p95}"
    )


def print_error_stats(name, errors, distances=None, min_distance=0.25):
    translation, rotation, translation_xyz, rotation_xyz = errors
    print(f"\n{name} ({len(translation)} comparisons)")
    print_values("translation", translation, "m")
    print_values("rotation", rotation, "deg")
    print_axis_values("translation", translation_xyz, "xyz", "m")
    print_axis_values("rotation", rotation_xyz, "xyz", "deg")
    if distances is not None:
        valid = distances > min_distance
        print(f"  normalized over {np.count_nonzero(valid)} motions > {min_distance:g} m")
        if np.any(valid):
            print_values("translation / distance", 100 * translation[valid] / distances[valid], "%")
            print_values("rotation / distance", rotation[valid] / distances[valid], "deg/m")


def load_applanix_ground_truth(sequence_root):
    path = osp.join(sequence_root, "applanix/gps_post_process.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    times_us = data[:, 0] * 1e6
    positions = data[:, 1:4]
    rotations = R.from_euler("zyx", -data[:, [9, 8, 7]])
    relative_times_s = (times_us - times_us[0]) * 1e-6
    slerp = Slerp(relative_times_s, rotations)

    T_applanix_lidar = np.loadtxt(
        osp.join(sequence_root, "calib/T_applanix_lidar.txt")
    )
    T_radar_lidar = np.loadtxt(
        osp.join(sequence_root, "calib/T_radar_lidar.txt")
    )
    T_applanix_radar = T_applanix_lidar @ np.linalg.inv(T_radar_lidar)
    return times_us, positions, slerp, T_applanix_radar


def interpolate_radar_poses(applanix_gt, query_times_us):
    times_us, positions, slerp, T_applanix_radar = applanix_gt
    query_times_us = np.asarray(query_times_us, dtype=np.float64)
    if query_times_us.min() < times_us[0] or query_times_us.max() > times_us[-1]:
        raise ValueError("Radar query timestamps exceed gps_post_process.csv coverage.")

    poses = np.zeros((len(query_times_us), 4, 4), dtype=np.float64)
    poses[:, 3, 3] = 1.0
    poses[:, :3, 3] = np.column_stack(
        [np.interp(query_times_us, times_us, positions[:, axis]) for axis in range(3)]
    )
    poses[:, :3, :3] = slerp((query_times_us - times_us[0]) * 1e-6).as_matrix()
    return poses @ T_applanix_radar


def selected_azimuth_data(sequence, frame_indices, azimuth_times, offsets, transforms):
    times, odometry, counts = [], [], []
    for frame_idx in frame_indices:
        radar_frame = sequence.get_radar(frame_idx)
        start, end = offsets[frame_idx : frame_idx + 2]
        frame_times = radar_frame.timestamps.flatten().astype(np.int64)
        if not np.array_equal(azimuth_times[start:end], frame_times):
            raise ValueError(f"Azimuth timestamps differ for frame {radar_frame.frame}.")
        times.append(frame_times)
        odometry.append(transforms[start:end])
        counts.append(end - start)
        radar_frame.unload_data()
    return np.concatenate(times), np.concatenate(odometry), np.asarray(counts)


def sparse_interpolation_floor(
    sequence, frame_indices, high_rate_odometry, counts, frame_stride
):
    sparse_odometry = []
    high_rate_inner = []
    cursor = 0
    for frame_idx, count in zip(frame_indices, counts):
        if (
            frame_idx % frame_stride == 0
            and 0 < frame_idx < len(sequence.radar_frames) - 1
        ):
            radar_frame = sequence.get_radar(frame_idx)
            neighbors = sequence.radar_frames[frame_idx - 1 : frame_idx + 2]
            poses = [np.linalg.inv(frame.pose) for frame in neighbors]
            times = [frame.timestamp_micro for frame in neighbors]
            azimuth_poses = interpolate_poses(
                poses, times, radar_frame.timestamps.flatten().tolist()
            )
            sparse_odometry.append(np.asarray(azimuth_poses) @ radar_frame.pose)
            high_rate_inner.append(high_rate_odometry[cursor : cursor + count])
            radar_frame.unload_data()
        cursor += count
    if not sparse_odometry:
        return None, None
    return np.concatenate(sparse_odometry), np.concatenate(high_rate_inner)


def print_worst_frames(frame_indices, counts, intra_translation, frame_translation):
    print("\nWorst inter-frame translation errors")
    for error_idx in np.argsort(frame_translation)[-5:][::-1]:
        print(f"  frame {error_idx + 1}: {frame_translation[error_idx]:.6g} m")

    starts = np.concatenate(([0], np.cumsum(counts)))
    maxima = np.asarray(
        [np.max(intra_translation[start:end]) for start, end in zip(starts[:-1], starts[1:])]
    )
    print("Worst selected-frame intra-frame translation errors")
    for selected_idx in np.argsort(maxima)[-5:][::-1]:
        print(f"  frame {frame_indices[selected_idx]}: {maxima[selected_idx]:.6g} m")


def main():
    parser = argparse.ArgumentParser(
        description="Validate 3DRO increments against 200 Hz Applanix radar ground truth."
    )
    parser.add_argument("data_root")
    parser.add_argument("sequence")
    parser.add_argument("azimuth_odometry")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sparse-floor-stride", type=int, default=50)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-distance-m", type=float, default=0.25)
    parser.add_argument("--translation-p95-m", type=float, default=0.05)
    parser.add_argument("--rotation-p95-deg", type=float, default=0.2)
    args = parser.parse_args()
    if args.frame_stride <= 0 or args.sparse_floor_stride <= 0:
        raise ValueError("Frame strides must be positive.")

    sequence = BoreasDataset(args.data_root, split=[[args.sequence]]).sequences[0]
    with np.load(args.azimuth_odometry) as data:
        if data["odom_transform_convention"].item() != "right":
            raise ValueError("Expected right-side pipeline_dfo odometry transforms.")
        frame_times = data["frame_timestamps_us"]
        reference_poses = data["reference_poses"]
        azimuth_times = data["azimuth_timestamps_us"]
        offsets = data["frame_offsets"]
        odom_transforms = data["odom_transforms"]
        frame_transforms = data["frame_transforms"]
        frame_body_velocities = data["frame_body_velocities"]
        velocity_start_us = data["velocity_start_us"]
        velocity_end_us = data["velocity_end_us"]

    frame_count = len(frame_times)
    if reference_poses.shape != (frame_count, 4, 4):
        raise ValueError(f"Invalid reference pose shape: {reference_poses.shape}.")
    if frame_transforms.shape != reference_poses.shape:
        raise ValueError(f"Invalid frame transform shape: {frame_transforms.shape}.")
    if frame_body_velocities.shape != (frame_count, 2):
        raise ValueError(f"Invalid body velocity shape: {frame_body_velocities.shape}.")
    if offsets.shape != (frame_count + 1,) or offsets[0] != 0:
        raise ValueError(f"Invalid frame offsets: {offsets.shape}.")
    if offsets[-1] != len(azimuth_times) or odom_transforms.shape != (
        len(azimuth_times),
        4,
        4,
    ):
        raise ValueError("Azimuth timestamp, offset, and transform lengths differ.")
    if not np.isfinite(reference_poses).all() or not np.isfinite(odom_transforms).all():
        raise ValueError("3DRO odometry contains non-finite poses.")
    if not np.isfinite(frame_transforms).all() or not np.isfinite(frame_body_velocities).all():
        raise ValueError("3DRO frame transforms or 2DRO velocities contain non-finite values.")
    if not np.allclose(frame_transforms[0], np.eye(4)) or not np.allclose(
        frame_transforms[1:] @ reference_poses[:-1], reference_poses[1:]
    ):
        raise ValueError("Stored inter-frame transforms do not compose with reference poses.")
    expected_start_us = np.asarray(
        [np.min(azimuth_times[start:end]) for start, end in zip(offsets[:-1], offsets[1:])]
    )
    expected_end_us = np.asarray(
        [np.max(azimuth_times[start:end]) for start, end in zip(offsets[:-1], offsets[1:])]
    )
    if not np.array_equal(velocity_start_us, expected_start_us) or not np.array_equal(
        velocity_end_us, expected_end_us
    ):
        raise ValueError("2DRO velocity and radar scan validity timestamps differ.")

    expected_frame_times = np.asarray(
        [frame.timestamp_micro for frame in sequence.radar_frames], dtype=np.int64
    )
    if not np.array_equal(frame_times, expected_frame_times):
        raise ValueError("3DRO and Boreas radar frame timestamps differ.")

    sequence_root = osp.join(args.data_root, args.sequence)
    applanix_gt = load_applanix_ground_truth(sequence_root)
    gt_world_reference = interpolate_radar_poses(applanix_gt, frame_times)
    csv_world_reference = np.asarray([frame.pose for frame in sequence.radar_frames])
    calibration_errors = pose_errors(
        np.linalg.inv(csv_world_reference), np.linalg.inv(gt_world_reference)
    )
    print_error_stats("200 Hz radar reconstruction vs radar_poses.csv", calibration_errors)
    if np.max(calibration_errors[0]) > 1e-3 or np.max(calibration_errors[1]) > 0.01:
        raise ValueError("Applanix-to-radar reconstruction failed its calibration sanity check.")

    gt_reference_poses = np.linalg.inv(gt_world_reference)
    dro_frame_delta = frame_transforms[1:]
    gt_frame_delta = gt_reference_poses[1:] @ np.linalg.inv(gt_reference_poses[:-1])
    frame_errors = pose_errors(dro_frame_delta, gt_frame_delta)
    frame_distances = np.linalg.norm(gt_frame_delta[:, :3, 3], axis=1)
    print_error_stats(
        "3DRO inter-frame increments vs 200 Hz GT",
        frame_errors,
        frame_distances,
        args.min_distance_m,
    )

    frame_indices = list(range(0, len(frame_times), args.frame_stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected for intra-frame validation.")

    selected_times, selected_odometry, counts = selected_azimuth_data(
        sequence, frame_indices, azimuth_times, offsets, odom_transforms
    )
    gt_world_azimuth = interpolate_radar_poses(applanix_gt, selected_times)
    gt_azimuth_poses = np.linalg.inv(gt_world_azimuth)
    repeated_gt_world_reference = np.repeat(
        gt_world_reference[frame_indices], counts, axis=0
    )
    selected_reference_poses = np.repeat(reference_poses[frame_indices], counts, axis=0)
    selected_left_odometry = (
        selected_reference_poses
        @ selected_odometry
        @ np.linalg.inv(selected_reference_poses)
    )
    gt_left_odometry = gt_azimuth_poses @ repeated_gt_world_reference
    intra_errors = pose_errors(selected_left_odometry, gt_left_odometry)
    intra_distances = np.linalg.norm(gt_left_odometry[:, :3, 3], axis=1)
    print_error_stats(
        "3DRO intra-frame increments vs 200 Hz GT",
        intra_errors,
        intra_distances,
        args.min_distance_m,
    )

    sparse_odometry, high_rate_inner = sparse_interpolation_floor(
        sequence,
        frame_indices,
        gt_left_odometry,
        counts,
        args.sparse_floor_stride,
    )
    if sparse_odometry is not None:
        print_error_stats(
            "4 Hz STEAM interpolation vs 200 Hz GT (evaluation floor)",
            pose_errors(sparse_odometry, high_rate_inner),
        )
    print_worst_frames(frame_indices, counts, intra_errors[0], frame_errors[0])

    if np.percentile(intra_errors[0], 95) > args.translation_p95_m:
        raise ValueError("Intra-frame translation p95 exceeds the acceptance threshold.")
    if np.percentile(intra_errors[1], 95) > args.rotation_p95_deg:
        raise ValueError("Intra-frame rotation p95 exceeds the acceptance threshold.")


if __name__ == "__main__":
    main()
