import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from sousvide.flight.deploy_figs import _DebugPolicyController
from sousvide.synthesize.event_generator import (
    OnlineEventImageGenerator,V2ERolloutRecorder)
from sousvide.synthesize.event_simulator import EventSimulator


class RecordingPolicy:
    hz = 20

    def __init__(self):
        self.images = []

    def control(self,t,x,u,rgb,depth,wrench):
        self.images.append(rgb.copy())
        return u,{"solve":0.0}


class OnlineEventDeploymentTests(unittest.TestCase):
    def test_online_generator_selects_each_python_surface(self):
        class FakeEmulator:
            def __init__(self,**kwargs):
                self.calls = 0

            def generate_events(self,frame,timestamp):
                self.calls += 1
                if self.calls == 1:
                    return None
                return np.array([
                    [timestamp,i%3,i%2,1 if i%2 else -1]
                    for i in range(10)
                ],dtype=np.float32)

            def cleanup(self):
                pass

        for modality in (
                "event_pseudo_gaussian","event_bilinear",
                "event_bin","event_eros","event_tos"):
            generator = V2ERolloutRecorder(
                None,1,emulator_factory=FakeEmulator,retain_images=False,
                event_modalities=(modality,))
            gray = np.zeros((2,3),dtype=np.uint8)
            self.assertIsNone(generator.process_gray_frame(gray,0.0,False))
            image = generator.process_gray_frame(gray,0.05,True)
            self.assertEqual(image.shape,(2,3))
            self.assertEqual(image.dtype,np.uint8)
            self.assertTrue(image.any())
            self.assertIsNone(generator.close())

    def test_debug_controller_reports_exact_policy_input_and_body_rates(self):
        class Controller:
            hz = 20
            name = "Maverick"

            def control(self,t,x,u,rgb,depth,wrench):
                return np.array([-2.0,0.1,-0.2,0.3]),{"solve":0.004}

        class DebugView:
            def __init__(self):
                self.updates = []

            def update(self,*args):
                self.updates.append(args)

        image = np.full((2,3,3),37,dtype=np.uint8)
        view = DebugView()
        controller = _DebugPolicyController(
            Controller(),"Maverick","kronecker_delta",view)

        command,timing = controller.control(
            1.25,np.zeros(10),np.zeros(4),image,
            np.zeros((2,3,1),dtype=np.uint8),np.zeros(6))

        self.assertEqual(controller.hz,20)
        np.testing.assert_array_equal(command,[-2.0,0.1,-0.2,0.3])
        self.assertEqual(timing,{"solve":0.004})
        self.assertEqual(len(view.updates),1)
        pilot,modality,timestamp,debug_image,debug_command = view.updates[0]
        self.assertEqual((pilot,modality,timestamp),
                         ("Maverick","kronecker_delta",1.25))
        self.assertIs(debug_image,image)
        self.assertIs(debug_command,command)

    def _simulator(self):
        simulator = EventSimulator.__new__(EventSimulator)
        simulator.conFiG = {
            "rollout": {
                "frequency":100,
                "noise":{"model":None,"sensor":None},
            },
            "frame":{},
            "forces":{},
        }

        class Solver:
            def simulate(self,x,u,p):
                result = x.copy()
                result[0] += 1
                return result

        class GSplat:
            def __init__(self):
                self.render_count = 0

            def generate_output_camera(self,camera):
                return object()

            def render_rgb(self,camera,transform):
                self.render_count += 1
                return (
                    np.full((2,3,3),11,dtype=np.uint8),
                    np.zeros((2,3,1),dtype=np.uint8),
                )

        simulator.solver = Solver()
        simulator.gsplat = GSplat()
        return simulator

    def _patch_dynamics(self):
        specification = {
            "nx":10,"nu":4,"m":1.0,"kt":1.0,"g":9.81,"Nrtr":4,
            "Tc2b":np.eye(4),"rgb_dim":(2,3,3),"dpt_dim":(2,3,1),
            "camera":{},
        }

        class Forces:
            def __init__(self,config):
                pass

            def get_forces(self,state,noisy=True):
                return np.zeros(3)

        return (
            mock.patch(
                "sousvide.synthesize.event_simulator.qs.generate_specifications",
                return_value=specification),
            mock.patch(
                "sousvide.synthesize.event_simulator.ExternalForces",Forces),
            mock.patch(
                "sousvide.synthesize.event_simulator.th.x_to_T",
                return_value=np.eye(4)),
            mock.patch(
                "sousvide.synthesize.event_simulator.oh.obedient_quaternion",
                side_effect=lambda current,previous:current),
        )

    def test_expert_controls_warmup_and_student_receives_boundary_images(self):
        simulator = self._simulator()
        expert = RecordingPolicy()
        student = RecordingPolicy()
        callbacks = []

        def event_callback(rgb,timestamp,close_window):
            callbacks.append((timestamp,close_window))
            if close_window:
                return np.full(rgb.shape[:2],37,dtype=np.uint8)
            return None

        patches = self._patch_dynamics()
        with patches[0],patches[1],patches[2],patches[3]:
            result = simulator.simulate_with_events(
                student,1.0,1.1,np.zeros(10),event_callback,5,
                warmup_policy=expert,image_modality="kronecker_delta")

        _,xro,uro,_,rgb,_,_ = result
        self.assertEqual(xro[0,0],5)
        self.assertEqual(len(uro),2)
        self.assertEqual(len(rgb),2)
        self.assertEqual(len(expert.images),1)
        self.assertTrue(np.all(expert.images[0] == 11))
        self.assertEqual(len(student.images),2)
        self.assertTrue(all(image.shape == (2,3,3) for image in student.images))
        self.assertTrue(all(np.all(image == 37) for image in student.images))
        self.assertEqual(
            [timestamp for timestamp,close in callbacks if close],
            [0.05,0.1])

    def test_student_receives_voxel_grids_in_hwc_layout(self):
        for modality,channels in (
                ("event_voxel_grid",5),
                ("event_voxel_grid_polarity",10)):
            simulator = self._simulator()
            expert = RecordingPolicy()
            student = RecordingPolicy()

            def event_callback(rgb,timestamp,close_window):
                if close_window:
                    return np.ones((channels,*rgb.shape[:2]),dtype=np.float32)
                return None

            patches = self._patch_dynamics()
            with patches[0],patches[1],patches[2],patches[3]:
                simulator.simulate_with_events(
                    student,1.0,1.1,np.zeros(10),event_callback,5,
                    warmup_policy=expert,image_modality=modality)

            self.assertEqual(len(student.images),2)
            self.assertTrue(all(
                image.shape == (2,3,channels) for image in student.images))
            self.assertTrue(all(
                image.dtype == np.float32 for image in student.images))

    def test_expert_baseline_skips_noncontrol_event_renders(self):
        simulator = self._simulator()
        expert = RecordingPolicy()
        patches = self._patch_dynamics()
        with patches[0],patches[1],patches[2],patches[3]:
            simulator.simulate_with_events(
                expert,1.0,1.1,np.zeros(10),None,5,
                warmup_policy=expert,image_modality="rgb")

        # One hidden control render plus two saved control renders.
        self.assertEqual(simulator.gsplat.render_count,3)
        self.assertEqual(len(expert.images),3)

    def test_real_v2e_cpu_generates_in_memory_without_files(self):
        with tempfile.TemporaryDirectory() as folder:
            previous_directory = os.getcwd()
            os.chdir(folder)
            try:
                generator = OnlineEventImageGenerator(1,device="cpu")
                dark = np.zeros((8,8,3),dtype=np.uint8)
                bright = np.full((8,8,3),255,dtype=np.uint8)
                self.assertIsNone(generator.process_frame(dark,0.0,False))
                image = generator.process_frame(bright,0.05,True)
                self.assertEqual(image.shape,(8,8))
                self.assertEqual(image.dtype,np.uint8)
                self.assertIsNone(generator.close())
                self.assertEqual(os.listdir(folder),[])
            finally:
                os.chdir(previous_directory)

    def test_abort_closes_online_emulator(self):
        class FakeEmulator:
            instance = None

            def __init__(self,**kwargs):
                self.cleaned = False
                FakeEmulator.instance = self

            def generate_events(self,frame,timestamp):
                return None

            def cleanup(self):
                self.cleaned = True

        generator = V2ERolloutRecorder(
            None,1,emulator_factory=FakeEmulator,retain_images=False)
        generator.process_frame(
            np.zeros((2,3,3),dtype=np.uint8),0.0,False)
        generator.abort()
        self.assertTrue(FakeEmulator.instance.cleaned)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is not available")
    def test_real_v2e_cuda_generates_in_memory(self):
        generator = OnlineEventImageGenerator(1,device="cuda")
        dark = np.zeros((8,8,3),dtype=np.uint8)
        bright = np.full((8,8,3),255,dtype=np.uint8)
        generator.process_frame(dark,0.0,False)
        image = generator.process_frame(bright,0.05,True)
        self.assertEqual(image.shape,(8,8))
        generator.close()


if __name__ == "__main__":
    unittest.main()
