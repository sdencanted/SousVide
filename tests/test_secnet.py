import copy
import os
import tempfile
import unittest

import torch

from sousvide.control.networks.secnet import SECNetEncoder
from sousvide.control.networks.svnet import SVNet


class SECNetTests(unittest.TestCase):
    def _encoder(self):
        return SECNetEncoder(
            num_points=16,feature_dims=(4,12,28,60),
            group_counts=(8,4,2),neighbors=2)

    def test_released_configuration_exposes_568_features(self):
        encoder = SECNetEncoder()
        self.assertEqual(encoder.out_features,568)
        self.assertEqual(encoder.num_points,4096)

    def test_output_gradients_seeded_training_and_deterministic_eval(self):
        encoder = self._encoder()
        event_cloud = torch.randn(2,16,4,requires_grad=True)
        torch.manual_seed(7)
        first = encoder(event_cloud)
        torch.manual_seed(7)
        repeated = encoder(event_cloud)
        torch.testing.assert_close(first,repeated)
        torch.manual_seed(8)
        different_seed = encoder(event_cloud)
        self.assertFalse(torch.equal(first,different_seed))
        self.assertEqual(first.shape,(2,120))
        first.square().mean().backward()
        self.assertIsNotNone(event_cloud.grad)
        self.assertTrue(any(
            parameter.grad is not None for parameter in encoder.parameters()))

        encoder.eval()
        with torch.inference_mode():
            evaluation1 = encoder(event_cloud.detach())
            evaluation2 = encoder(event_cloud.detach())
        torch.testing.assert_close(evaluation1,evaluation2)

    def test_checkpoint_round_trip(self):
        encoder = self._encoder().eval()
        event_cloud = torch.randn(1,16,4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"secnet.pt")
            torch.save(encoder.state_dict(),path)
            restored = self._encoder().eval()
            restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(
                encoder(event_cloud),restored(event_cloud))

    def _svnet_config(self):
        return {
            "inputs":{
                "rgb_image":[8,8,["r","g","b"]],
                "current1":[["pz"]],
                "current2":[["pz","vx"]],
                "feature_vector":[2],
            },
            "outputs":{"command":[["nf","wx"]]},
            "layers":{
                "dropout":0.0,"sqnet_size":1000,
                "visual_backbones":{
                    "default":"squeezenet1_1","event_cloud":"secnet"},
                "secnet":{
                    "num_points":16,"feature_dims":[4,12,28,60],
                    "group_counts":[8,4,2],"neighbors":2,
                },
                "hidden_sizes":[16,8],"cmd_aug_layer":1,
            },
        }

    def test_svnet_event_cloud_forward_and_invalid_pairings(self):
        config = self._svnet_config()
        network = SVNet(**config,image_modality="event_cloud").eval()
        self.assertEqual(network.visual_backbone,"secnet")
        self.assertEqual(network.networks["feat"].out_features,120)
        self.assertEqual(
            network.get_io_dims()["xdp"]["event_cloud"],[16,4])
        with torch.inference_mode():
            output = network({
                "event_cloud":torch.randn(2,16,4),
                "current1":torch.randn(2,1),
                "current2":torch.randn(2,2),
                "feature_vector":torch.randn(2,2),
            })
        self.assertEqual(output["command"].shape,(2,2))

        network.train()
        optimizer = torch.optim.Adam(network.parameters(),lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        training_output = network({
            "event_cloud":torch.randn(2,16,4),
            "current1":torch.randn(2,1),
            "current2":torch.randn(2,2),
            "feature_vector":torch.randn(2,2),
        })["command"]
        training_output.square().mean().backward()
        optimizer.step()
        self.assertTrue(any(
            parameter.grad is not None for parameter in network.parameters()))

        bad_cloud = copy.deepcopy(config)
        bad_cloud["layers"]["visual_backbones"]["event_cloud"] = (
            "squeezenet1_1")
        with self.assertRaisesRegex(ValueError,"require the SECNet"):
            SVNet(**bad_cloud,image_modality="event_cloud")

        bad_rgb = copy.deepcopy(config)
        bad_rgb["layers"]["visual_backbones"]["default"] = "secnet"
        with self.assertRaisesRegex(ValueError,"require the SqueezeNet"):
            SVNet(**bad_rgb,image_modality="rgb")


if __name__ == "__main__":
    unittest.main()
