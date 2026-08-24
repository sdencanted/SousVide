import os
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

import sousvide.control.network_factory as network_factory
import sousvide.control.network_helper as network_helper
from sousvide.control.policy import Policy


class ModalityNetworkWeightTests(unittest.TestCase):
    def test_collect_prediction_inputs_skips_final_target_forward(self):
        class HistoryNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.io_idxs = {
                    "xdp":network_helper.get_io_idxs(
                        {"current":[["t"]]})}

            def forward(self,inputs):
                return {"feature_vector":inputs["current"]+1}

        class CommandNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.io_idxs = {
                    "xdp":network_helper.get_io_idxs(
                        {"feature_vector":[1]})}
                self.forward_called = False

            def forward(self,inputs):
                self.forward_called = True
                return {"command":inputs["feature_vector"]}

        policy = Policy.__new__(Policy)
        nn.Module.__init__(policy)
        command = CommandNetwork()
        policy.networks = nn.ModuleDict({
            "histNet":HistoryNetwork(),"commNet":command})

        inputs = policy.collect_prediction_inputs(
            {"current":torch.zeros(2,11)},["commNet"])

        self.assertEqual(list(inputs),["commNet"])
        torch.testing.assert_close(
            inputs["commNet"]["feature_vector"],torch.ones(2,1))
        self.assertFalse(command.forward_called)

    def test_commnet_has_separate_canonical_paths(self):
        pilot_path = os.path.join("cohorts","test","roster","Maverick")
        self.assertEqual(
            network_factory.get_network_path(pilot_path,"commNet","rgb"),
            os.path.join(pilot_path,"commNet_rgb.pt"))
        self.assertEqual(
            network_factory.get_network_path(
                pilot_path,"commNet","kronecker_delta"),
            os.path.join(pilot_path,"commNet_kronecker_delta.pt"))
        for modality in ("event_bin","event_eros","event_tos"):
            self.assertEqual(
                network_factory.get_network_path(
                    pilot_path,"commNet",modality),
                os.path.join(pilot_path,f"commNet_{modality}.pt"))
        self.assertEqual(
            network_factory.get_network_path(
                pilot_path,"histNet","kronecker_delta"),
            os.path.join(pilot_path,"histNet.pt"))

    def test_rgb_loads_legacy_commnet_when_canonical_file_is_missing(self):
        with tempfile.TemporaryDirectory() as pilot_path:
            legacy_path = os.path.join(pilot_path,"commNet.pt")
            open(legacy_path,"wb").close()
            sentinel = object()
            with mock.patch.object(
                    network_factory.torch,"load",return_value=sentinel) as load:
                network = network_factory.generate_network(
                    {"network_type":"unused"},"commNet",pilot_path,
                    image_modality="rgb",require_commnet_weights=True)

            self.assertIs(network,sentinel)
            load.assert_called_once_with(legacy_path,weights_only=False)

    def test_kronecker_never_falls_back_to_legacy_rgb_commnet(self):
        with tempfile.TemporaryDirectory() as pilot_path:
            open(os.path.join(pilot_path,"commNet.pt"),"wb").close()
            expected_path = os.path.join(
                pilot_path,"commNet_kronecker_delta.pt")

            with self.assertRaisesRegex(
                    FileNotFoundError,"Train commNet with the same image_modality"):
                network_factory.generate_network(
                    {"network_type":"unused"},"commNet",pilot_path,
                    image_modality="kronecker_delta",
                    require_commnet_weights=True)

            self.assertEqual(
                network_factory.get_network_load_path(
                    pilot_path,"commNet","kronecker_delta"),expected_path)

    def test_legacy_commnet_uses_recorded_kronecker_modality(self):
        with tempfile.TemporaryDirectory() as pilot_path:
            legacy_path = os.path.join(pilot_path,"commNet.pt")
            open(legacy_path,"wb").close()
            torch.save(
                {"log":{"image_modality":"kronecker_delta"}},
                os.path.join(pilot_path,"losses_commNet.pt"))

            self.assertEqual(
                network_factory.get_network_load_path(
                    pilot_path,"commNet","kronecker_delta"),legacy_path)
            self.assertEqual(
                network_factory.get_network_load_path(
                    pilot_path,"commNet","rgb"),
                os.path.join(pilot_path,"commNet_rgb.pt"))

    def test_policy_records_selected_commnet_save_path(self):
        class FakeNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.Nhy = 1
                self.io_idxs = {"xdp":{},"ypd":{}}
                self.use_deploy = False

        profile = {"networks":{"commNet":{"network_type":"unused"}}}
        with tempfile.TemporaryDirectory() as pilot_path:
            with mock.patch.object(
                    network_factory,"generate_network",
                    return_value=FakeNetwork()) as generate:
                policy = Policy(
                    profile,"Maverick",pilot_path,
                    image_modality="kronecker_delta")

            self.assertEqual(
                policy.network_paths["commNet"],
                os.path.join(pilot_path,"commNet_kronecker_delta.pt"))
            generate.assert_called_once_with(
                profile["networks"]["commNet"],"commNet",pilot_path,
                image_modality="kronecker_delta",
                require_commnet_weights=False)


if __name__ == "__main__":
    unittest.main()
