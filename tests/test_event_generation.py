import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from sousvide.synthesize.event_generator import (
    V2ERolloutRecorder,
    events_to_kronecker,
)
from sousvide.synthesize.event_simulator import EventSimulator


class FakeEmulator:
    event_batches = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.cleaned = False

    def generate_events(self, frame, timestamp):
        self.calls.append((frame, timestamp))
        return self.event_batches[len(self.calls) - 1]

    def cleanup(self):
        self.cleaned = True


class EventImageTests(unittest.TestCase):
    def test_sparse_polarity_scaling_and_clipping(self):
        sparse = np.zeros((9, 4), dtype=np.float32)
        self.assertFalse(events_to_kronecker(sparse, 3, 4).any())

        events = np.array(
            [[i / 100, 1, 1, 1 if i % 2 else -1] for i in range(10)],
            dtype=np.float32,
        )
        positive = events.copy()
        positive[:, 3] = 1
        np.testing.assert_array_equal(
            events_to_kronecker(events, 3, 4),
            events_to_kronecker(positive, 3, 4),
        )
        self.assertEqual(events_to_kronecker(events, 3, 4)[1, 1], 255)

        counts = [(0, 0)] + [(1, 0)] * 2 + [(2, 0)] * 100
        clipped_events = np.array(
            [[0.0, x, y, 1] for x, y in counts], dtype=np.float32
        )
        image = events_to_kronecker(clipped_events, 1, 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.max(), 255)

    def test_upper_boundary_closes_preceding_window(self):
        first = np.array([[0.01, 1, 1, 1]] * 5, dtype=np.float32)
        boundary = np.array([[0.05, 1, 1, -1]] * 5, dtype=np.float32)
        FakeEmulator.event_batches = [None, first, boundary, None]

        with tempfile.TemporaryDirectory() as folder:
            recorder = V2ERolloutRecorder(
                os.path.join(folder, "001000.h5"),
                expected_windows=2,
                emulator_factory=FakeEmulator,
            )
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            recorder.process_frame(rgb, 0.00, False)
            recorder.process_frame(rgb, 0.01, False)
            recorder.process_frame(rgb, 0.05, True)
            recorder.process_frame(rgb, 0.10, True)
            images = recorder.close()

        self.assertEqual(images.shape, (2, 3, 4))
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(images[0, 1, 1], 255)
        self.assertFalse(images[1].any())

    def test_abort_removes_staged_h5(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "001000.h5")
            with open(path, "wb") as output:
                output.write(b"staged")
            recorder = V2ERolloutRecorder(path, expected_windows=1)
            recorder.abort()
            self.assertFalse(os.path.exists(path))


class EventSimulatorTests(unittest.TestCase):
    def test_warmup_is_one_control_interval_and_saved_lengths_are_unchanged(self):
        simulator = EventSimulator.__new__(EventSimulator)
        simulator.conFiG = {
            "rollout": {
                "frequency": 100,
                "noise": {"model": None, "sensor": None},
            },
            "frame": {},
            "forces": {},
        }

        class Solver:
            def simulate(self, x, u, p):
                result = x.copy()
                result[0] += 1
                return result

        class GSplat:
            def generate_output_camera(self, camera):
                return object()

            def render_rgb(self, camera, transform):
                return (
                    np.zeros((2, 3, 3), dtype=np.uint8),
                    np.zeros((2, 3, 1), dtype=np.uint8),
                )

        class Forces:
            def __init__(self, config):
                pass

            def get_forces(self, state, noisy=True):
                return np.zeros(3)

        class Policy:
            hz = 20

            def control(self, t, x, u, rgb, depth, wrench):
                return u, {"solve": 0.0}

        specification = {
            "nx": 10,
            "nu": 4,
            "m": 1.0,
            "kt": 1.0,
            "g": 9.81,
            "Nrtr": 4,
            "Tc2b": np.eye(4),
            "rgb_dim": (2, 3, 3),
            "dpt_dim": (2, 3, 1),
            "camera": {},
        }
        simulator.solver = Solver()
        simulator.gsplat = GSplat()
        callbacks = []

        with (
            mock.patch(
                "sousvide.synthesize.event_simulator.qs.generate_specifications",
                return_value=specification,
            ),
            mock.patch(
                "sousvide.synthesize.event_simulator.ExternalForces", Forces
            ),
            mock.patch(
                "sousvide.synthesize.event_simulator.th.x_to_T",
                return_value=np.eye(4),
            ),
            mock.patch(
                "sousvide.synthesize.event_simulator.oh.obedient_quaternion",
                side_effect=lambda current, previous: current,
            ),
        ):
            result = simulator.simulate_with_events(
                Policy(), 1.0, 1.1, np.zeros(10),
                lambda rgb, timestamp, close: callbacks.append((timestamp, close)),
                warmup_steps=5,
            )

        tro, xro, uro, _, rgb, _, _ = result
        self.assertEqual(len(tro), 3)
        self.assertEqual(len(xro), 3)
        self.assertEqual(len(uro), 2)
        self.assertEqual(len(rgb), 2)
        self.assertEqual(xro[0, 0], 5)
        self.assertEqual(len(callbacks), 11)
        self.assertEqual([t for t, close in callbacks if close], [0.05, 0.1])


if __name__ == "__main__":
    unittest.main()
