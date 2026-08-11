import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd
import pyboreas as pb
import boreas_eval as be

from pyboreas.utils.odometry import (
    read_traj_file_gt
)



def main():
    data_root = "/media/asrl/Extreme SSD/ASRL/boreas/data"
    velocities_root = "output_2d"
    sequences = ['boreas-2024-12-03-12-54']
    no_calib = False
    no_3d = False

    dro3DCalibration(data_root, velocities_root, sequences)


    sequences = be.sequence_type.keys()


    for i, sequence in enumerate(sequences):
        print(f"Processing sequence {sequence} ({i + 1}/{len(sequences)})...")
        try:
            data_path = os.path.join(data_root, sequence, "imu/dmu_imu.csv")
            velocities_path = os.path.join(velocities_root, sequence, "other_log/velocity.csv")
            T_applanix_dmu = np.loadtxt(os.path.join(data_root, sequence, "calib/T_applanix_dmu.txt"))


            T_applanix_lidar = np.loadtxt(os.path.join(data_root, sequence, "calib/T_applanix_lidar.txt"))
            T_radar_lidar = np.loadtxt(os.path.join(data_root, sequence, "calib/T_radar_lidar.txt"))
            T_radar_dmu = T_radar_lidar @ np.linalg.inv(T_applanix_lidar) @ T_applanix_dmu
            T_radar_applanix = T_radar_lidar @ np.linalg.inv(T_applanix_lidar)

            # Read the lidar times for the output (needed as the evaluation is done in the lidar frame).
            # _, lidar_times = readGTLidarBoreas(os.path.join(data_root, sequence, "applanix/lidar_poses.csv"))
            _, radar_times = readGTLidarBoreas(os.path.join(data_root, sequence, "applanix/radar_poses.csv"))

            # anchor the first radar pose in enu
            dataset = pb.BoreasDataset(data_root, split=[[sequence]])
            data_seq = dataset.sequences[0]
            first_radar_frame = data_seq.radar_frames[0]
            T_enu_radar0 = first_radar_frame.pose

            frame_times_us, azimuth_times_us, frame_offsets = readRadarTimestamps(data_seq)
            frame_body_velocities, velocity_start_us, velocity_end_us = readFrameVelocities(
                velocities_path,
                frame_times_us,
                azimuth_times_us,
                frame_offsets,
            )
            radar_times_us = np.rint(radar_times * 1e6).astype(np.int64)
            if not np.array_equal(frame_times_us, radar_times_us):
                raise ValueError("Dataset and radar ground-truth timestamps differ.")
            query_times_us = np.unique(np.concatenate((frame_times_us, azimuth_times_us)))

            if no_calib:
                scale_z = 0.0
            else:
                scale_z = np.loadtxt(os.path.join("calib", "radar_vel_z_scale.txt"))


            traj = droOdom3DOffline(
                data_path,
                velocities_path,
                T_radar_dmu,
                scale_z,
                query_times_us * 1e-6,
                yaw_only=no_3d,
            )  # T_radar(0)_radar(t)
            frame_indices = np.searchsorted(query_times_us, frame_times_us)
            anchor = T_enu_radar0 @ np.linalg.inv(traj[frame_indices[0]])
            reference_poses = np.linalg.inv(anchor @ traj[frame_indices])  # T_radar(t)_enu
            assert np.allclose(reference_poses[0], np.linalg.inv(T_enu_radar0), rtol=0, atol=1e-8)

            # traj = traj @ T_radar_applanix # T_radar(0)_applanix(t)
            # traj = np.linalg.inv(T_radar_applanix) @ traj # T_applanix(0)_applanix(t)
            
            output_dir = os.path.join("output", sequence, "odometry_result")
            writeOdom3DTrajectory(
                radar_times,
                np.linalg.inv(reference_poses),
                os.path.join(output_dir, sequence + ".txt"),
            )
            writeAzimuthOdometry(
                os.path.join(output_dir, "azimuth_odometry.npz"),
                query_times_us,
                traj,
                anchor,
                frame_times_us,
                reference_poses,
                azimuth_times_us,
                frame_offsets,
                frame_body_velocities,
                velocity_start_us,
                velocity_end_us,
            )
        except Exception as e:
            print(f"Error processing sequence {sequence}: {e}. Skipping.")
            continue



def readGTLidarBoreas(gt_traj_path):
    gt_traj_path = gt_traj_path.replace('\\ ', ' ')  # Remove \\ in the path
    poses, times = read_traj_file_gt(gt_traj_path, np.eye(4), dim=3)
    poses = np.array(poses) 
    times = np.array(times) * 1e-6  # Convert from microseconds to seconds
    poses = np.linalg.inv(poses)
    return poses, times

def writeOdom3DTrajectory(timestamps, poses, output_path):
    to_write = []
    for i in range(len(timestamps)):
        timestamp = round(timestamps[i] * 1e6)  # Convert seconds to microseconds
        pose = poses[i]
        pose_flat = np.linalg.inv(pose)[:3,:].flatten()
        to_write.append([timestamp] + pose_flat.tolist())
    to_write = np.array(to_write)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    to_write_pd = pd.DataFrame(to_write)
    # Set the first column to integer type
    to_write_pd[0] = to_write_pd[0].astype(int)
    to_write_pd.to_csv(output_path, index=False, header=False, sep=' ')


def readRadarTimestamps(sequence):
    frame_times = []
    azimuth_times = []
    frame_offsets = [0]
    for radar_frame in sequence.radar_frames:
        radar_frame.load_data()
        times = radar_frame.timestamps.flatten().astype(np.int64)
        if len(times) == 0:
            raise ValueError(f"Invalid azimuth timestamps for radar frame {radar_frame.frame}.")
        frame_times.append(radar_frame.timestamp_micro)
        azimuth_times.extend(times)
        frame_offsets.append(len(azimuth_times))
        radar_frame.unload_data()
    return (
        np.asarray(frame_times, dtype=np.int64),
        np.asarray(azimuth_times, dtype=np.int64),
        np.asarray(frame_offsets, dtype=np.int64),
    )


def readFrameVelocities(path, frame_times_us, azimuth_times_us, frame_offsets):
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    if data.shape != (len(frame_times_us), 5) or not np.isfinite(data).all():
        raise ValueError(f"Invalid 2DRO velocity data shape or values: {data.shape}.")

    velocity_frame_times_us = np.rint(data[:, 0] * 1e6).astype(np.int64)
    velocity_start_us = data[:, 1].astype(np.int64)
    velocity_end_us = data[:, 2].astype(np.int64)
    expected_start_us = np.asarray(
        [np.min(azimuth_times_us[start:end]) for start, end in zip(frame_offsets[:-1], frame_offsets[1:])],
        dtype=np.int64,
    )
    expected_end_us = np.asarray(
        [np.max(azimuth_times_us[start:end]) for start, end in zip(frame_offsets[:-1], frame_offsets[1:])],
        dtype=np.int64,
    )
    if not np.array_equal(velocity_frame_times_us, frame_times_us):
        raise ValueError("2DRO velocity and radar reference timestamps differ.")
    if not np.array_equal(velocity_start_us, expected_start_us):
        raise ValueError("2DRO velocity and radar scan start timestamps differ.")
    if not np.array_equal(velocity_end_us, expected_end_us):
        raise ValueError("2DRO velocity and radar scan end timestamps differ.")
    return data[:, 3:5], velocity_start_us, velocity_end_us


def writeAzimuthOdometry(
    output_path,
    query_times_us,
    traj,
    anchor,
    frame_times_us,
    reference_poses,
    azimuth_times_us,
    frame_offsets,
    frame_body_velocities,
    velocity_start_us,
    velocity_end_us,
):
    azimuth_indices = np.searchsorted(query_times_us, azimuth_times_us)
    odom_transforms = np.empty((len(azimuth_times_us), 4, 4), dtype=np.float32)
    for frame_idx in range(len(frame_times_us)):
        start, end = frame_offsets[frame_idx:frame_idx + 2]
        azimuth_poses = np.linalg.inv(anchor @ traj[azimuth_indices[start:end]])
        # Left increment: P(azimuth) = transform @ P(frame reference).
        transforms = azimuth_poses @ np.linalg.inv(reference_poses[frame_idx])
        if not np.allclose(transforms @ reference_poses[frame_idx], azimuth_poses, atol=1e-8):
            raise ValueError(f"Invalid azimuth transform composition for frame {frame_idx}.")
        odom_transforms[start:end] = transforms

    frame_transforms = np.empty_like(reference_poses)
    frame_transforms[0] = np.eye(4)
    frame_transforms[1:] = reference_poses[1:] @ np.linalg.inv(reference_poses[:-1])
    if not np.allclose(frame_transforms[1:] @ reference_poses[:-1], reference_poses[1:], atol=1e-8):
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
    )





def droOdom3DOffline(imu_path, velocities_path, T_sensor_imu, scale_z, times_output, imu_bias_estimation=False, imu_bias_prior=None, yaw_only=False):
    times_output = np.asarray(times_output, dtype=np.float64)
    if np.any(np.diff(times_output) <= 0):
        raise ValueError("Output timestamps must be strictly increasing.")
    imu_data = IMUData()
    imu_data.readFromBoreasFile(imu_path)
    if imu_bias_prior is not None:
        imu_data.setBias(imu_bias_prior)

    traj = []

    # Read velocities from velocities_path and store them in a list of tuples (timestamp, velocity).
    velocity_data = np.loadtxt(velocities_path, delimiter=',', skiprows=1)


    odom3D = GyrBasedOdom3D(T_sensor_imu, bias_estimation=imu_bias_estimation)

    previous_time = velocity_data[0, 1] * 1e-6
    first_index = np.searchsorted(times_output, previous_time, side="right")
    traj.extend(np.eye(4) for _ in range(first_index))
    for i in range(len(velocity_data)):
        end_time = velocity_data[i, 2] * 1e-6
        body_vel = velocity_data[i, 3:5]

        if yaw_only:
            body_vel = np.array([body_vel[0], body_vel[1], 0.0])
        else:
            vel_norm = np.linalg.norm(body_vel)
            body_vel = np.array([body_vel[0], body_vel[1], scale_z * vel_norm])  # Add a z component to the velocity, which is the norm of the x-y velocity multiplied by the scale factor.

        start = np.searchsorted(times_output, previous_time, side="right")
        end = np.searchsorted(times_output, end_time, side="right")
        temp_time_output = times_output[start:end]


        gyr_timestamps, gyr_data, _ = imu_data.getInInterval(previous_time, end_time)
        if yaw_only:
            gyr_data[:, 0] = 0.0
            gyr_data[:, 1] = 0.0
        poses = odom3D.updateWithVel(gyr_timestamps, gyr_data, body_vel, end_time, temp_time_output)
        if len(poses) != len(temp_time_output):
            raise ValueError(f"Number of output poses {len(poses)} does not match number of output times {len(temp_time_output)}.")
        for j in range(len(temp_time_output)):
            traj.append(poses[j])
        previous_time = end_time

        print(f"Processed velocity data point {i + 1}/{len(velocity_data)}", end='    \r')

    if len(traj) != len(times_output):
        raise ValueError(
            f"Number of output poses {len(traj)} does not match number of output times "
            f"{len(times_output)}. Last output time: {times_output[-1]}, last velocity "
            f"end time: {velocity_data[-1, 2] * 1e-6}."
        )


    traj = np.array(traj)
    return traj


def dro3DCalibration(data_root, velocities_root, sequences):

    # Check if the calibration folder exists, if not create it.
    calib_folder = os.path.join("calib")
    os.makedirs(calib_folder, exist_ok=True)

    kTimeMargin = 0.005

    gt_vels = []
    radar_vels = []

    for sequence in sequences:
        velocity_data = np.loadtxt(os.path.join(velocities_root, sequence, "other_log/velocity.csv"), delimiter=',', skiprows=1)

        dataset = pb.BoreasDataset(data_root, split=[[sequence]])
        data_seq = dataset.sequences[0]

        print("Number of radar scans: ", len(dataset.sequences[0].radar_frames), " for sequence ", sequence)

        if len(data_seq.radar_frames) != len(velocity_data):
            raise ValueError(f"Number of radar frames {len(data_seq.radar_frames)} does not match number of velocity data points {len(velocity_data)} for sequence {sequence}.")

        for i in range(len(data_seq.radar_frames)):
            radar_frame = data_seq.radar_frames[i]
            gt_vel = radar_frame.body_rate[:3].flatten()  # Get the ground truth velocity from the radar frame.
            gt_vels.append(gt_vel)

            radar_vel = np.array([velocity_data[i, 3], velocity_data[i, 4], 0])  # Get the velocity from the velocity data and add a zero for the z-axis.
            radar_vels.append(radar_vel)

            if(np.abs(radar_frame.timestamp - velocity_data[i, 0]) > kTimeMargin):
                raise ValueError(f"Timestamp of radar frame {radar_frame.timestamp} does not match timestamp of velocity data {velocity_data[i, 0]} for sequence {sequence}, frame {i + 1}.")
            

            radar_frame.unload_data()
        


    gt_vels = np.array(gt_vels)
    radar_vels = np.array(radar_vels)

    # Estimate a scale factor using the ratio of the x-y norms of the velocities.
    radar_vels_xy_norm = np.linalg.norm(radar_vels[:, :2], axis=1)
    gt_vels_z = gt_vels[:, 2]
    mask = radar_vels_xy_norm > 1.0
    z_scale = np.mean(gt_vels_z[mask] / radar_vels_xy_norm[mask])

    save_path = os.path.join(calib_folder, "radar_vel_z_scale.txt")
    np.savetxt(save_path, np.array([z_scale]))
    print(f"Estimated z scale factor: {z_scale} saved to {save_path}")


    # Plot for debugging
    fig_width = 10
    fig, axs = plt.subplots(3, 1, figsize=(fig_width, 10))
    axs[0].plot(gt_vels[:, 0], label='GT vel. x')
    axs[0].plot(radar_vels[:, 0], label='Radar vel. x (RMSE: ' + str(round(np.sqrt(np.mean((gt_vels[:, 0] - radar_vels[:, 0]) ** 2)), 2)) + ')')
    axs[0].legend()
    axs[1].plot(gt_vels[:, 1], label='GT vel. y')
    axs[1].plot(radar_vels[:, 1], label='Radar vel. y (RMSE: ' + str(round(np.sqrt(np.mean((gt_vels[:, 1] - radar_vels[:, 1]) ** 2)), 2)) + ')')
    axs[1].legend()
    axs[2].plot(gt_vels[:, 2], label='GT vel. z')
    axs[2].plot(radar_vels_xy_norm * z_scale, label='Radar vel. z approx (mean error: ' + str(round(np.mean((gt_vels[:, 2] - radar_vels_xy_norm * z_scale)), 3)) + ', RMSE: ' + str(round(np.sqrt(np.mean((gt_vels[:, 2] - radar_vels_xy_norm * z_scale) ** 2)), 3)) + ')')
    axs[2].plot(radar_vels[:, 2], label='No vel. z (mean error: ' + str(round(np.mean((gt_vels[:, 2])), 3)) + ', RMSE: ' + str(round(np.sqrt(np.mean((gt_vels[:, 2]) ** 2)), 3)) + ')')
    axs[2].set_xlabel('Frame index of calibration sequence')
    axs[2].set_ylabel('Velocity z [m/s]')
    # Show legend with opaque background
    axs[2].legend(loc='upper left', framealpha=1)
    axs[2].set_ylim(-0.2,0.5)
    plt.tight_layout()
    plt.show()

    


class IMUData:
    def __init__(self):
        self.timestamps = []
        self.angular_velocities = []
        self.linear_accelerations = []
        self.bias_gyr = np.zeros(3)
        self.bias_acc = np.zeros(3)

    def readFromBoreasFile(self, file_path):
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        self.timestamps = data[:, 0] * 1e-9  # Convert from nanoseconds to seconds.
        self.angular_velocities = data[:, 1:4]  # Shape (N, 3)
        self.linear_accelerations = data[:, 4:7]  # Shape (N, 3)
        return self.timestamps, self.angular_velocities, self.linear_accelerations

    # Returns the IMU data between start_time and end_time.
    # If ends_included is True, returns data that surrounds the interval, otherwise only data that is strictly within the interval (including the boundaries).
    def getInInterval(self, start_time, end_time, ends_included=True):
        if not ends_included:
            mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        else:
            last_before_start = np.where(self.timestamps < start_time)[0][-1] if np.any(self.timestamps < start_time) else None
            first_after_end = np.where(self.timestamps > end_time)[0][0] if np.any(self.timestamps > end_time) else None
            mask = np.zeros_like(self.timestamps, dtype=bool)
            if last_before_start is not None:
                mask[last_before_start] = True
            if first_after_end is not None:
                mask[first_after_end] = True
            mask |= (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return self.timestamps[mask], self.angular_velocities[mask,:] - self.bias_gyr, self.linear_accelerations[mask,:] - self.bias_acc


    def setBias(self, bias_gyr=None, bias_acc=None):
        if bias_gyr is not None:
            self.bias_gyr = bias_gyr
        if bias_acc is not None:
            self.bias_acc = bias_acc


def interpolate(t0, t1, v0, v1, t):
    if t1 == t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return (1 - alpha) * v0 + alpha * v1


class GyrBasedOdom3D:
    def __init__(self, T_sensor_imu, scale = 1.0, bias_estimation = False):
        self.T_sensor_imu = T_sensor_imu
        self.current_pose = np.eye(4)
        self.current_time = None
        self.period = None
        self.scale = scale
        self.bias_estimation = bias_estimation

    def updateWithVel(self, gyr_timestamps, gyr_data, body_vel, end_time, times_output):
        if(self.current_time is not None and end_time < self.current_time):
            raise ValueError("End time must be greater than current time."
                             f"Current time: {self.current_time}, end time: {end_time}")
        if(end_time < gyr_timestamps[0]):
            raise ValueError("End time must be greater than the first timestamp of the gyro data."
                             f"First gyro timestamp: {gyr_timestamps[0]}, end time: {end_time}")
        if(end_time > gyr_timestamps[-1]):
            raise ValueError("End time must be less than the last timestamp of the gyro data."
                             f"Last gyro timestamp: {gyr_timestamps[-1]}, end time: {end_time}")
        for time_output in times_output:
            if(time_output > end_time):
                raise ValueError("Time output must be less than or equal to end time."
                                f"Time output: {time_output}, end time: {end_time}")
            if(self.current_time is not None and time_output < self.current_time):
                raise ValueError("Time output must be greater than or equal to current time."
                                f"Time output: {time_output}, current time: {self.current_time}")
        if(gyr_data.shape[0] != len(gyr_timestamps)):
            raise ValueError("Number of gyro data points must match the number of gyro timestamps."
                             f"Number of gyro data points: {gyr_data.shape[1]}, number of gyro timestamps: {len(gyr_timestamps)}")
        if(len(gyr_timestamps) < 2):
            raise ValueError("At least two gyro timestamps are required.")
        
        if self.period is None:
            if len(gyr_timestamps) == 3:
                self.period = np.min(np.diff(gyr_timestamps))
            if len(gyr_timestamps) > 3:
                self.period = np.median(np.diff(gyr_timestamps))
            else:
                raise ValueError("Cannot estimate period from less than 3 gyro timestamps.")

        if body_vel.shape[0] == 2:
            body_vel = np.hstack((body_vel.flatten(), np.array([0])))


        if self.current_time is None:
            self.current_time = gyr_timestamps[0]

        gyr_sensor = self.T_sensor_imu[:3, :3] @ gyr_data.T
        output_poses = []


        # If there are gaps in the gyro data, we need to interpolate the angular velocity to fill those gaps.
        full_gyr_timestamps = []
        full_gyr_data = []
        for i in range(len(gyr_timestamps) - 1):
            full_gyr_timestamps.append(gyr_timestamps[i])
            full_gyr_data.append(gyr_sensor[:, i])
            gap = gyr_timestamps[i + 1] - gyr_timestamps[i]
            if gap > self.period * 1.5:  # If the gap is larger than 1.5 times the period, we consider it a gap that needs interpolation.
                num_interp_points = int(np.ceil(gap / self.period)) - 1
                for j in range(num_interp_points):
                    interp_time = gyr_timestamps[i] + (j + 1) * self.period
                    if interp_time < gyr_timestamps[i + 1]:  # Only add interpolation points that are within the next timestamp.
                        interp_data = interpolate(gyr_timestamps[i], gyr_timestamps[i + 1], gyr_sensor[:, i], gyr_sensor[:, i + 1], interp_time)
                        full_gyr_timestamps.append(interp_time)
                        full_gyr_data.append(interp_data)
        full_gyr_timestamps.append(gyr_timestamps[-1])
        full_gyr_data.append(gyr_sensor[:, -1])


        output_idx = 0
        gyr_id = 1
        while gyr_id < len(full_gyr_timestamps) and full_gyr_timestamps[gyr_id] < self.current_time:
            gyr_id += 1
        gyr_id_start = gyr_id
        while self.current_time < end_time and gyr_id < len(full_gyr_timestamps) and full_gyr_timestamps[gyr_id-1] <= end_time:
            t0 = full_gyr_timestamps[gyr_id - 1]
            t1 = full_gyr_timestamps[gyr_id]
            omega0 = full_gyr_data[gyr_id - 1]
            omega1 = full_gyr_data[gyr_id]
            if gyr_id == gyr_id_start:
                omega0 = interpolate(t0, t1, omega0, omega1, self.current_time)
                t0 = self.current_time

            while output_idx < len(times_output) and times_output[output_idx] <= t1:
                time_output = times_output[output_idx]
                if t0 <= time_output:
                    omega_out = interpolate(t0, t1, omega0, omega1, time_output)
                    dt = time_output - t0
                    omega = (omega0 + omega_out) / 2
                    delta_R = R.from_rotvec(omega * dt).as_matrix()
                    delta_pos = body_vel * dt
                    delta_T_out = np.eye(4)
                    delta_T_out[:3, :3] = delta_R
                    delta_T_out[:3, 3] = delta_pos
                    output_poses.append(self.current_pose @ delta_T_out)
                output_idx += 1


            if t1 > end_time:
                omega1 = interpolate(t0, t1, omega0, omega1, end_time)
                t1 = end_time
            


            dt = t1 - t0
            omega = (omega0 + omega1) / 2
            delta_R = R.from_rotvec(omega * dt).as_matrix()
            delta_pos = body_vel * dt

            delta_T = np.eye(4)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = delta_pos

            self.current_pose = self.current_pose @ delta_T
            self.current_time = t1
            gyr_id += 1



        
        return output_poses





if __name__ == "__main__":
    main()
