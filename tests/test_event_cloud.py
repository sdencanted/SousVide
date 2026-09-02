import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

import sousvide.synthesize.rollout_generator as rollout_generator
import sousvide.synthesize.observation_generator as observation_generator
import sousvide.control.network_helper as network_helper
from sousvide.synthesize.event_cloud import (
    event_cloud_metadata,
    events_to_event_cloud,
    resolve_event_cloud_options,
)
from sousvide.synthesize.event_generator import OnlineEventCloudGenerator
from sousvide.synthesize.image_modality import prepare_rollout_images
from sousvide.synthesize.parallel_event_generator import (
    process_event_stream_rollout,
)
from sousvide.instruct.synthesized_data import (
    OBSERVATION_FORMAT_VERSION,generate_dataset)


class _SequenceEmulator:
    outputs = []

    def __init__(self, **kwargs):
        self.index = 0

    def generate_events(self, image, timestamp):
        output = self.outputs[self.index]
        self.index += 1
        return output

    def cleanup(self):
        pass


class EventCloudSamplingTests(unittest.TestCase):
    def test_options_and_empty_window(self):
        self.assertEqual(
            resolve_event_cloud_options(), {"num_points":4096,"seed":0})
        cloud,count = events_to_event_cloud(
            None,10,8,stream_id="rollout",window_index=0,
            options={"num_points":7,"seed":3})
        self.assertEqual(cloud.shape,(7,4))
        self.assertEqual(cloud.dtype,np.float32)
        self.assertEqual(count,0)
        self.assertFalse(cloud.any())
        with self.assertRaises(ValueError):
            resolve_event_cloud_options({"num_points":0})
        with self.assertRaises(ValueError):
            resolve_event_cloud_options({"unknown":1})

    def test_short_exact_and_oversized_windows_are_stable_and_ordered(self):
        events = np.array([
            [10.0,0.0,0.0,0.0],
            [20.0,2.0,1.0,1.0],
            [30.0,4.0,2.0,0.0],
            [40.0,6.0,3.0,1.0],
            [50.0,8.0,4.0,0.0],
        ])
        short,count = events_to_event_cloud(
            events[:3],10,5,stream_id="r",window_index=0,
            options={"num_points":8,"seed":4})
        self.assertEqual(count,3)
        self.assertTrue(np.all(np.diff(short[:,0]) >= 0))
        self.assertTrue(set(np.unique(short[:,3])) <= {-1.0,1.0})
        self.assertGreaterEqual(short[:,0].min(),0.0)
        self.assertLessEqual(short[:,0].max(),1.0)

        exact,_ = events_to_event_cloud(
            events,10,5,stream_id="r",window_index=1,
            options={"num_points":5,"seed":4})
        np.testing.assert_allclose(exact[:,0],np.linspace(0,1,5))
        np.testing.assert_allclose(exact[:,1],events[:,1]/10)
        np.testing.assert_allclose(exact[:,2],events[:,2]/5)

        first,_ = events_to_event_cloud(
            events,10,5,stream_id="r",window_index=2,
            options={"num_points":3,"seed":4})
        repeated,_ = events_to_event_cloud(
            events,10,5,stream_id="r",window_index=2,
            options={"num_points":3,"seed":4})
        other_stream,_ = events_to_event_cloud(
            events,10,5,stream_id="other",window_index=2,
            options={"num_points":3,"seed":4})
        np.testing.assert_array_equal(first,repeated)
        self.assertFalse(np.array_equal(first,other_stream))
        self.assertTrue(np.all(np.diff(first[:,0]) >= 0))

    def test_constant_timestamps_normalize_to_zero(self):
        events = np.array([[5,1,2,0],[5,2,3,1]],dtype=np.float32)
        cloud,_ = events_to_event_cloud(
            events,4,4,stream_id="r",window_index=0,
            options={"num_points":2})
        np.testing.assert_array_equal(cloud[:,0],np.zeros(2,np.float32))


class EventCloudPipelineTests(unittest.TestCase):
    def _events(self):
        return np.array([
            [10_000,1,1,0], [20_000,2,2,1], [30_000,3,1,0],
            [60_000,4,2,1], [70_000,5,3,0], [80_000,6,4,1],
        ],dtype=np.uint32)

    def test_h5_generation_and_online_generation_are_byte_equivalent(self):
        raw = self._events()
        options = {"num_points":5,"seed":9}
        with tempfile.TemporaryDirectory() as folder:
            h5_path = os.path.join(folder,"stream.h5")
            cloud_path = os.path.join(folder,"cloud.npy")
            with h5py.File(h5_path,"w") as output:
                output.create_dataset("events",data=raw)
            _,counts = process_event_stream_rollout(
                h5_path,(0.05,0.10),8,10,("event_cloud",),{},
                {"event_cloud":cloud_path},options,"same-stream")
            offline = np.load(cloud_path,allow_pickle=False)

            _SequenceEmulator.outputs = [
                np.column_stack((
                    raw[:3,0]/1e6,raw[:3,1:3],
                    np.where(raw[:3,3] > 0,1,-1))),
                np.column_stack((
                    raw[3:,0]/1e6,raw[3:,1:3],
                    np.where(raw[3:,3] > 0,1,-1))),
            ]
            online = OnlineEventCloudGenerator(
                2,event_cloud_options=options,stream_id="same-stream",
                emulator_factory=_SequenceEmulator)
            image = np.zeros((8,10,3),dtype=np.uint8)
            clouds = np.stack([
                online.process_frame(image,0.05,True),
                online.process_frame(image,0.10,True),
            ])
            online.close()

            np.testing.assert_array_equal(counts,np.array([3,3]))
            self.assertEqual(offline.dtype,np.float32)
            self.assertEqual(offline.shape,(2,5,4))
            np.testing.assert_array_equal(offline,clouds)
            self.assertTrue(set(np.unique(offline[:,:,3])) <= {-1.0,1.0})

    def test_generate_event_representations_saves_tensor_and_metadata(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_module_path = os.path.join(
                workspace,"a","b","c","rollout_generator.py")
            course_path = os.path.join(
                workspace,"cohorts","cohort","rollout_data","course")
            for folder in ("trajectories","images","events"):
                os.makedirs(os.path.join(course_path,folder))
            rollout_id = "001000"
            torch.save([{
                "rollout_id":rollout_id,"Ndata":2,
                "Tro":np.array([10.0,10.05,10.10]),
            }],os.path.join(
                course_path,"trajectories","trajectories001.pt"))
            torch.save([{
                "rollout_id":rollout_id,
                "rgb":np.zeros((2,8,10,3),dtype=np.uint8),
                "depth":np.zeros((2,8,10,1),dtype=np.uint8),
            }],os.path.join(course_path,"images","images001.pt"))
            with h5py.File(os.path.join(
                    course_path,"events",rollout_id+".h5"),"w") as output:
                output.create_dataset("events",data=self._events())

            with mock.patch.object(
                    rollout_generator,"__file__",fake_module_path):
                rollout_generator.generate_event_representations(
                    "cohort",["course"],("event_cloud",),event_workers=1,
                    event_cloud_options={"num_points":5,"seed":9})

            artifact = torch.load(os.path.join(
                course_path,"event_cloud","event_cloud001.pt"),
                weights_only=False,mmap=True)[0]
            self.assertIsInstance(artifact["event_cloud"],torch.Tensor)
            self.assertEqual(tuple(artifact["event_cloud"].shape),(2,5,4))
            self.assertEqual(artifact["event_cloud"].dtype,torch.float32)
            self.assertEqual(artifact["rollout_id"],rollout_id)
            self.assertEqual(
                artifact["event_cloud_config"],
                event_cloud_metadata({"num_points":5,"seed":9}))
            np.testing.assert_array_equal(
                artifact["raw_event_counts"],np.array([3,3]))
            prepared = prepare_rollout_images(
                {"rollout_id":rollout_id,"Ndata":2},artifact,"event_cloud")
            self.assertEqual(prepared.shape,(2,5,4))
            self.assertEqual(prepared.dtype,np.float32)

    def test_event_cloud_observation_tensors_support_mapped_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder,"observations.pt")
            cloud = torch.randn(3,5,4)
            torch.save({
                "format_version":OBSERVATION_FORMAT_VERSION,
                "Xnn":{"event_cloud":cloud,"current1":torch.randn(3,1)},
                "Ynn":{"command":torch.randn(3,2)},
                "event_cloud_config":event_cloud_metadata(
                    {"num_points":5,"seed":9}),
            },path)

            dataset = generate_dataset(path,mmap=True)
            self.assertEqual(len(dataset),3)
            inputs,labels = dataset[1]
            self.assertEqual(inputs["event_cloud"].shape,(5,4))
            torch.testing.assert_close(inputs["event_cloud"],cloud[1])
            self.assertEqual(labels["command"].shape,(2,))

    def test_observation_generation_preserves_cloud_key_and_config(self):
        config = event_cloud_metadata({"num_points":5,"seed":9})
        cloud = np.arange(40,dtype=np.float32).reshape(2,5,4)

        class _Network:
            io_idxs = {"ypd":network_helper.get_io_idxs({
                "command":[["nf","wx","wy","wz"]]})}

        class _Policy:
            networks = {"commNet":_Network()}

            def __init__(self,pilot):
                self.pilot = pilot

            def collect_prediction_inputs(self,inputs,networks):
                return {"commNet":{
                    "event_cloud":self.pilot.cloud.unsqueeze(0),
                    "current1":torch.zeros(1,1),
                }}

        class _Pilot:
            name = "Maverick"
            da_cfg = {"type":"none","mean":np.zeros(10),"std":np.zeros(10)}

            def __init__(self):
                self.policy = _Policy(self)
                self.pch_cr = torch.zeros(1,1,1,1)
                self.cls_cr = torch.zeros(1,1)

            def observe(self,t,x,u,image,d,w):
                self.cloud = torch.as_tensor(image,dtype=torch.float32)

            def retain(self):
                pass

            def collate(self):
                return {"event_cloud":self.cloud.unsqueeze(0)}

        trajectory = [{
            "Tro":np.array([0.0,0.05,0.10]),
            "Xro":np.zeros((2,10)),"Uro":np.zeros((2,4)),
            "Wro":np.zeros((2,6)),"Wrs":np.zeros((2,6)),
            "Ndata":2,"rollout_id":"001000","frame":{},
            "params":np.zeros(2),
        }]
        images = [{
            "rollout_id":"001000","event_cloud":cloud,
            "event_cloud_config":config,
        }]
        observations,count = observation_generator.generate_observations(
            _Pilot(),trajectory,images,image_modality="event_cloud",
            networks=["commNet"])
        self.assertEqual(count,2)
        self.assertEqual(observations[0]["event_cloud_config"],config)
        self.assertEqual(
            tuple(observations[0]["Xnn"][0]["commNet"][
                "event_cloud"].shape),(1,5,4))

        with tempfile.TemporaryDirectory() as workspace:
            fake_module_path = os.path.join(
                workspace,"a","b","c","observation_generator.py")
            with mock.patch.object(
                    observation_generator,"__file__",fake_module_path):
                observation_generator.save_observations(
                    "cohort","course","Maverick",observations,0,
                    networks=["commNet"],image_modality="event_cloud")
            path = os.path.join(
                workspace,"cohorts","cohort","observation_data","Maverick",
                "event_cloud","commNet","course","observations001.pt")
            saved = torch.load(path,weights_only=False,mmap=True)
            self.assertEqual(saved["event_cloud_config"],config)
            self.assertEqual(saved["Xnn"]["event_cloud"].shape,(2,5,4))


if __name__ == "__main__":
    unittest.main()
