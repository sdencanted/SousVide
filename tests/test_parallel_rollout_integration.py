import errno
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import sousvide.synthesize.rollout_generator as rollout_generator
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
    def test_save_rollouts_writes_independent_event_artifacts(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_module_path = os.path.join(
                workspace,"a","b","c","rollout_generator.py")
            staged_paths = []
            for rollout_id in ("001000","001001"):
                path = os.path.join(workspace,rollout_id+".h5")
                with open(path,"wb") as output:
                    output.write(b"events")
                staged_paths.append(path)
            trajectories = [
                {"rollout_id":rollout_id,"Ndata":1}
                for rollout_id in ("001000","001001")]
            images = [
                {
                    "rollout_id":rollout_id,
                    "rgb":np.zeros((1,2,3,3),dtype=np.uint8),
                    "depth":np.zeros((1,2,3,1),dtype=np.uint8),
                }
                for rollout_id in ("001000","001001")]
            event_specs = {
                "event_bin":((1,2,3),np.uint8),
                "event_eros":((1,2,3),np.uint8),
                "event_tos":((1,2,3),np.uint8),
                "event_voxel_grid":((1,5,2,3),np.float32),
                "event_voxel_grid_polarity":((1,10,2,3),np.float32),
            }
            event_images = {
                modality:[
                    {
                        "rollout_id":rollout_id,
                        modality:np.zeros(shape,dtype=dtype),
                        "event_surface_config":{},
                    }
                for rollout_id in ("001000","001001")]
                for modality,(shape,dtype) in event_specs.items()}
            for modality in (
                    "event_voxel_grid","event_voxel_grid_polarity"):
                for index,event_image in enumerate(event_images[modality]):
                    mapped = np.lib.format.open_memmap(
                        os.path.join(workspace,f"{modality}-{index}.npy"),
                        mode="w+",dtype=np.float32,
                        shape=event_specs[modality][0])
                    mapped[...] = event_image[modality]
                    event_image[modality] = mapped

            real_replace = os.replace

            def replace_across_devices(source,destination):
                if source in staged_paths:
                    raise OSError(
                        errno.EXDEV,"Invalid cross-device link",
                        source,destination)
                return real_replace(source,destination)

            with (
                mock.patch.object(
                    rollout_generator,"__file__",fake_module_path),
                mock.patch.object(
                    rollout_generator.os,"replace",
                    side_effect=replace_across_devices),
            ):
                rollout_generator.save_rollouts(
                    "cohort","course",trajectories,images,0,use_compress=True,
                    EventPaths=staged_paths,
                    EventImagesByModality=event_images)

            course = os.path.join(
                workspace,"cohorts","cohort","rollout_data","course")
            for modality in event_images:
                artifact = os.path.join(course,modality,f"{modality}001.pt")
                self.assertTrue(os.path.isfile(artifact))
                stored = torch.load(artifact,weights_only=False)[0][modality]
                if modality.startswith("event_voxel_grid"):
                    self.assertIsInstance(stored,torch.Tensor)
                    self.assertEqual(stored.dtype,torch.float32)
                else:
                    self.assertIsInstance(stored[0],bytes)
            self.assertEqual(
                sorted(os.listdir(os.path.join(course,"events"))),
                ["001000.h5","001001.h5"])
            self.assertFalse(any(os.path.exists(path) for path in staged_paths))

    def _inputs(self):
        return (
            [{"index":0},{"index":1}],
            [
                {"t0":0.0,"x0":np.zeros(10)},
                {"t0":0.1,"x0":np.zeros(10)},
            ],
            np.zeros((1,15)),
        )

    def test_parallel_rollout_generates_selected_modalities_together(self):
        frames,perturbations,desired = self._inputs()
        patches = (
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_prms",
                return_value=np.zeros(1)),
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_Wrs",
                return_value=np.zeros((1,6))),
            mock.patch(
                "sousvide.synthesize.rollout_generator.svu.compute_FOro",
                return_value=np.zeros((1,4))),
        )
        with tempfile.TemporaryDirectory() as staging, patches[0],patches[1],patches[2]:
            trajectories,images,event_images,event_paths = generate_rollouts(
                FakeEventSimulator(),FakeController(),desired,{},
                frames,perturbations,0.05,np.inf,0,
                event_staging_dir=staging,event_workers=2,
                event_modalities=(
                    "event_bin","event_eros","event_tos",
                    "event_voxel_grid","event_voxel_grid_polarity"))

            expected_ids = ["001000","001001"]
            self.assertEqual(set(event_images),
                             {"event_bin","event_eros","event_tos",
                              "event_voxel_grid","event_voxel_grid_polarity"})
            for modality,rollouts in event_images.items():
                self.assertEqual(
                    [item["rollout_id"] for item in rollouts],expected_ids)
                channels = (
                    10 if modality == "event_voxel_grid_polarity"
                    else 5 if modality == "event_voxel_grid" else None)
                expected_shape = (
                    (1,channels,8,8) if channels is not None else (1,8,8))
                self.assertTrue(all(
                    item[modality].shape == expected_shape for item in rollouts))
                self.assertTrue(all(
                    isinstance(item[modality],np.memmap) for item in rollouts))
            self.assertEqual(len(event_paths),2)
            self.assertEqual(
                len([name for name in os.listdir(staging)
                     if name.endswith(".npy")]),
                2*len(event_images))
            self.assertFalse(any(
                name.endswith(".frames.npy") for name in os.listdir(staging)))

    def test_parallel_results_remain_in_accepted_rollout_order(self):
        frames,perturbations,desired = self._inputs()

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
            self.assertEqual(
                len([name for name in os.listdir(staging)
                     if name.endswith(".kronecker_delta.npy")]),2)
            self.assertTrue(all(
                isinstance(item["kronecker_delta"],np.memmap)
                for item in kronecker))


if __name__ == "__main__":
    unittest.main()
