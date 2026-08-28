import os
import tempfile
import unittest

import numpy as np
import torch

from sousvide.synthesize import data_compress_helper as compression
from sousvide.synthesize.image_modality import (
    event_image_to_model_channels,
    get_aligned_stack_files,
    image_modality_channels,
    is_grayscale_modality,
    is_voxel_grid_modality,
    prepare_rollout_images,
    rgb_to_grayscale,
    uses_modality_observation_folder,
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

    def test_voxel_tensor_deserializes_to_numpy_without_copy(self):
        tensor = torch.arange(60,dtype=torch.float32).reshape(1,5,3,4)
        data = {"event_voxel_grid":tensor}
        compression.decompress_data(data,key="event_voxel_grid")
        self.assertIsInstance(data["event_voxel_grid"],np.ndarray)
        self.assertEqual(data["event_voxel_grid"].dtype,np.float32)
        np.testing.assert_array_equal(
            data["event_voxel_grid"],tensor.numpy())


class ImageModalityTests(unittest.TestCase):
    def test_grayscale_is_derived_from_rgb_and_repeated_for_rgb_backbones(self):
        rgb = np.array([[
            [[255,0,0],[0,255,0]],
            [[0,0,255],[10,20,30]],
        ]],dtype=np.uint8)
        trajectory = {"rollout_id":"001000","Ndata":1}
        image_data = {"rollout_id":"001000","rgb":rgb}

        prepared = prepare_rollout_images(
            trajectory,image_data,"grayscale")

        self.assertEqual(prepared.shape,(1,2,2,3))
        self.assertTrue(is_grayscale_modality("grayscale"))
        self.assertEqual(image_modality_channels("grayscale"),3)
        np.testing.assert_array_equal(prepared[...,0],prepared[...,1])
        np.testing.assert_array_equal(prepared[...,1],prepared[...,2])
        np.testing.assert_array_equal(
            prepared[...,0],np.array([[[76,150],[29,18]]],dtype=np.uint8))

    def test_grayscale_uses_rgb_rollouts_but_separate_observations(self):
        with tempfile.TemporaryDirectory() as course:
            os.makedirs(os.path.join(course,"trajectories"))
            os.makedirs(os.path.join(course,"images"))
            trajectory_path = os.path.join(
                course,"trajectories","trajectories001.pt")
            image_path = os.path.join(course,"images","images001.pt")
            open(trajectory_path,"wb").close()
            open(image_path,"wb").close()

            self.assertEqual(
                get_aligned_stack_files(course,"grayscale"),
                [(trajectory_path,image_path)])

        self.assertTrue(uses_modality_observation_folder("grayscale"))
        self.assertFalse(uses_modality_observation_folder("rgb"))

    def test_grayscale_conversion_accepts_single_channel_input(self):
        gray = np.arange(6,dtype=np.uint8).reshape(2,3,1)
        converted = rgb_to_grayscale(gray)
        self.assertEqual(converted.shape,(2,3,3))
        np.testing.assert_array_equal(converted[...,0],gray[...,0])

    def test_voxel_modalities_keep_float_channels_and_independent_storage(self):
        trajectory = {"rollout_id":"001000","Ndata":2}
        for modality,channels in (
                ("event_voxel_grid",5),
                ("event_voxel_grid_polarity",10)):
            voxel = np.arange(
                2*channels*4*5,dtype=np.float32).reshape(2,channels,4,5)
            prepared = prepare_rollout_images(
                trajectory,{"rollout_id":"001000",modality:voxel},modality)
            self.assertEqual(prepared.shape,(2,4,5,channels))
            self.assertEqual(prepared.dtype,np.float32)
            np.testing.assert_array_equal(prepared[...,0],voxel[:,0])
            self.assertTrue(is_voxel_grid_modality(modality))
            self.assertEqual(image_modality_channels(modality),channels)
            np.testing.assert_array_equal(
                event_image_to_model_channels(voxel[0],modality),prepared[0])

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
