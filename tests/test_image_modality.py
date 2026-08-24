import os
import tempfile
import unittest

import numpy as np

from sousvide.synthesize import data_compress_helper as compression
from sousvide.synthesize.image_modality import (
    get_aligned_stack_files,
    prepare_rollout_images,
)


class CompressionTests(unittest.TestCase):
    def test_rgb_and_grayscale_png_round_trip(self):
        rng = np.random.default_rng(2)
        for key, shape in (
            ("rgb", (2, 5, 7, 3)),
            ("kronecker_delta", (2, 5, 7)),
            ("event_bin", (2, 5, 7)),
            ("event_eros", (2, 5, 7)),
            ("event_tos", (2, 5, 7)),
        ):
            original = rng.integers(0, 256, shape, dtype=np.uint8)
            data = [{key: original.copy(), "rollout_id": "001000"}]
            compression.compress_data(data, key=key)
            self.assertIsInstance(data[0][key][0], bytes)
            compression.decompress_data(data[0], key=key)
            np.testing.assert_array_equal(data[0][key], original)


class ImageModalityTests(unittest.TestCase):
    def test_each_event_modality_has_independent_storage_and_three_channels(self):
        gray = np.arange(40,dtype=np.uint8).reshape(2,4,5)
        trajectory = {"rollout_id":"001000","Ndata":2}
        with tempfile.TemporaryDirectory() as course:
            os.makedirs(os.path.join(course,"trajectories"))
            with open(os.path.join(
                    course,"trajectories","trajectories001.pt"),"wb"):
                pass
            for modality in ("event_bin","event_eros","event_tos"):
                os.makedirs(os.path.join(course,modality))
                path = os.path.join(course,modality,f"{modality}001.pt")
                with open(path,"wb"):
                    pass
                aligned = get_aligned_stack_files(course,modality)
                self.assertEqual(aligned,[(
                    os.path.join(course,"trajectories","trajectories001.pt"),
                    path)])
                prepared = prepare_rollout_images(
                    trajectory,
                    {"rollout_id":"001000",modality:gray.copy()},modality)
                self.assertEqual(prepared.shape,(2,4,5,3))
                np.testing.assert_array_equal(prepared[...,0],gray)

    def test_kronecker_loader_selects_only_aligned_file_and_repeats_channels(self):
        with tempfile.TemporaryDirectory() as course:
            for folder in ("trajectories", "images", "kronecker"):
                os.makedirs(os.path.join(course, folder))
            for path in (
                os.path.join(course, "trajectories", "trajectories001.pt"),
                os.path.join(course, "images", "images001.pt"),
                os.path.join(course, "kronecker", "kronecker001.pt"),
            ):
                with open(path, "wb"):
                    pass

            paths = get_aligned_stack_files(course, "kronecker_delta")
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0][1].endswith("kronecker001.pt"))
            self.assertNotIn("/images/", paths[0][1])

        gray = np.arange(40, dtype=np.uint8).reshape(2, 4, 5)
        trajectory = {"rollout_id": "001000", "Ndata": 2}
        image_data = {"rollout_id": "001000", "kronecker_delta": gray}
        prepared = prepare_rollout_images(
            trajectory, image_data, "kronecker_delta"
        )
        self.assertEqual(prepared.shape, (2, 4, 5, 3))
        np.testing.assert_array_equal(prepared[..., 0], gray)
        np.testing.assert_array_equal(prepared[..., 1], gray)
        np.testing.assert_array_equal(prepared[..., 2], gray)

    def test_alignment_errors_are_rejected(self):
        trajectory = {"rollout_id": "001000", "Ndata": 2}
        wrong_id = {
            "rollout_id": "001001",
            "kronecker_delta": np.zeros((2, 4, 5), dtype=np.uint8),
        }
        with self.assertRaises(ValueError):
            prepare_rollout_images(trajectory, wrong_id, "kronecker_delta")


if __name__ == "__main__":
    unittest.main()
