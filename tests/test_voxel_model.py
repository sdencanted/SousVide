import json
import os
import tempfile
import unittest

import torch

import sousvide.control.network_factory as network_factory
import sousvide.control.network_helper as network_helper
from sousvide.control.pilot import _normalize_voxel_grid


class VoxelModelTests(unittest.TestCase):
    def _svnet_config(self):
        workspace = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(
                workspace,"configs","pilots","Maverick.json")) as config_file:
            return json.load(config_file)["networks"]["commNet"]

    def test_factory_builds_svnet_with_voxel_channel_count(self):
        for modality,channels in (
                ("event_voxel_grid",5),
                ("event_voxel_grid_polarity",10)):
            with tempfile.TemporaryDirectory() as pilot_path:
                network = network_factory.generate_network(
                    self._svnet_config(),"commNet",pilot_path,
                    image_modality=modality)

            self.assertEqual(
                len(network.io_idxs["xdp"]["rgb_image"][0]),channels)
            self.assertEqual(network.networks["feat"].features[0].in_channels,channels)
            inputs = network_helper.extract_io({
                "rgb_image":torch.zeros(1,channels,224,224),
                "current":torch.zeros(1,11),
                "feature_vector":torch.zeros(1,8),
            },network.io_idxs["xdp"])
            self.assertEqual(network(inputs)["command"].shape,(1,4))

    def test_factory_rejects_dino_voxel_input(self):
        with tempfile.TemporaryDirectory() as pilot_path:
            with self.assertRaisesRegex(ValueError,"supported only by SVNet"):
                network_factory.generate_network(
                    {"network_type":"dino"},"featNet",pilot_path,
                    image_modality="event_voxel_grid")

    def test_nonzero_voxel_normalization_is_joint_and_preserves_zeros(self):
        image = torch.tensor([
            [[0.0,1.0],[2.0,0.0]],
            [[3.0,0.0],[0.0,4.0]],
        ])
        normalized = _normalize_voxel_grid(image)
        values = normalized[normalized != 0]

        torch.testing.assert_close(values.mean(),torch.tensor(0.0),atol=1e-6,rtol=0)
        torch.testing.assert_close(
            torch.mean(values*values),torch.tensor(1.0),atol=1e-6,rtol=0)
        self.assertTrue(torch.all(normalized[image == 0] == 0))

    def test_zero_variance_voxel_normalization_is_stable(self):
        image = torch.tensor([[[0.0,2.0],[0.0,0.0]]])
        torch.testing.assert_close(_normalize_voxel_grid(image),image)


if __name__ == "__main__":
    unittest.main()
