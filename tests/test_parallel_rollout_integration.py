import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from sousvide.synthesize.rollout_generator import generate_rollouts


class FakeEventSimulator:
    conFiG = {"rollout": {"frequency": 100}}

    def update_frame(self, frame):
        self.frame = frame

    def simulate_with_events(
        self,controller,t0,tf,x0,event_callback,warmup_steps
    ):
        dark = np.zeros((8,8,3),dtype=np.uint8)
        bright = np.full((8,8,3),255,dtype=np.uint8)
        event_callback(dark,0.0,False)
        event_callback(bright,0.05,True)
        tro = np.array([t0,tf])
        xro = np.zeros((2,10))
        uro = np.zeros((1,4))
        wro = np.zeros((1,6))
        rgb = bright[None,...]
        depth = np.zeros((1,8,8,1),dtype=np.uint8)
        tsol = np.zeros(1)
        return tro,xro,uro,wro,rgb,depth,tsol


class FakeController:
    hz = 20

    def update_frame(self,frame):
        self.frame = frame


class ParallelRolloutIntegrationTests(unittest.TestCase):
    def test_parallel_results_remain_in_accepted_rollout_order(self):
        frames = [{"index":0},{"index":1}]
        perturbations = [
            {"t0":0.0,"x0":np.zeros(10)},
            {"t0":0.1,"x0":np.zeros(10)},
        ]
        desired = np.zeros((1,15))

        with (
            tempfile.TemporaryDirectory() as staging,
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_prms",
                return_value=np.zeros(1),
            ),
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_Wrs",
                return_value=np.zeros((1,6)),
            ),
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_FOro",
                return_value=np.zeros((1,4)),
            ),
        ):
            trajectories,images,kronecker,event_paths = generate_rollouts(
                FakeEventSimulator(),FakeController(),desired,{},
                frames,perturbations,0.05,np.inf,0,
                generate_events=True,event_staging_dir=staging,event_workers=2,
            )

            expected_ids = ["001000","001001"]
            self.assertEqual(
                [item["rollout_id"] for item in trajectories],expected_ids)
            self.assertEqual(
                [item["rollout_id"] for item in images],expected_ids)
            self.assertEqual(
                [item["rollout_id"] for item in kronecker],expected_ids)
            self.assertEqual(
                [item["kronecker_delta"].shape for item in kronecker],
                [(1,8,8),(1,8,8)],
            )
            self.assertTrue(all(os.path.isfile(path) for path in event_paths))
            self.assertFalse(any(name.endswith(".frames.npy") for name in os.listdir(staging)))
            self.assertFalse(any(name.endswith(".kronecker.npy") for name in os.listdir(staging)))


if __name__ == "__main__":
    unittest.main()
