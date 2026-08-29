import tempfile
import unittest
from pathlib import Path

import numpy as np

from azimuth_odometry import planarPoses, writeAzimuthOdometryArrays


class TestAzimuthOdometry2D(unittest.TestCase):
    def test_left_increment_is_world_frame_invariant(self):
        reference, azimuth, world_change, candidate = planarPoses(
            [[4, -2], [4.3, -1.8], [10, 5], [-3, 7]],
            [0.4, 0.5, -0.7, 1.2],
        )
        increment = azimuth @ np.linalg.inv(reference)

        np.testing.assert_allclose(
            (azimuth @ world_change) @ np.linalg.inv(reference @ world_change),
            increment,
        )
        np.testing.assert_allclose(
            increment @ candidate,
            azimuth @ np.linalg.inv(reference) @ candidate,
        )

    def test_pipeline_schema_and_planarity(self):
        frame_times = np.array([10, 20], dtype=np.int64)
        azimuth_times = np.array([9, 10, 11, 19, 20, 21], dtype=np.int64)
        offsets = np.array([0, 3, 6], dtype=np.int64)
        world_frames = planarPoses([[0, 0], [1, 0]], [0, 0.1])
        reference = np.linalg.inv(world_frames)
        azimuth_world = planarPoses(
            [[-0.1, 0], [0, 0], [0.1, 0], [0.9, 0], [1, 0], [1.1, 0]],
            [-0.01, 0, 0.01, 0.09, 0.1, 0.11],
        )
        odometry = np.concatenate(
            [
                np.linalg.inv(azimuth_world[start:end]) @ world_frames[i]
                for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
            ]
        ).astype(np.float32)
        frame_transforms = np.stack(
            [np.eye(4), reference[1] @ np.linalg.inv(reference[0])]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "azimuth_odometry.npz"
            writeAzimuthOdometryArrays(
                path,
                frame_times,
                reference,
                azimuth_times,
                offsets,
                odometry,
                frame_transforms,
                np.array([[1.0, 0.0], [1.0, 0.0]]),
                np.array([9, 19]),
                np.array([11, 21]),
            )
            with np.load(path) as result:
                self.assertEqual(result["odom_transform_convention"].item(), "left")
                np.testing.assert_array_equal(result["frame_offsets"], offsets)
                np.testing.assert_allclose(
                    result["odom_transforms"][:, 2, :], [[0, 0, 1, 0]] * 6
                )
                np.testing.assert_allclose(
                    result["odom_transforms"][[1, 4]],
                    np.repeat(np.eye(4)[None], 2, axis=0),
                    atol=1e-7,
                )
                repeated_reference = np.repeat(reference, np.diff(offsets), axis=0)
                np.testing.assert_allclose(
                    result["odom_transforms"] @ repeated_reference,
                    np.linalg.inv(azimuth_world),
                )
                np.testing.assert_allclose(
                    result["frame_transforms"][1:]
                    @ result["reference_poses"][:-1],
                    result["reference_poses"][1:],
                )


if __name__ == "__main__":
    unittest.main()
