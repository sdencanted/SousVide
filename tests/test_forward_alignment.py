"""Run with PYTHONPATH=FiGS/src .venv/bin/python -m unittest discover -s tests -p 'test_*alignment.py'."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from figs.tsplines.min_time_snap import MinTimeSnap
from figs.utilities import transform_helper as th
from figs.utilities.course_editor import _normalise_course
from figs.utilities.forward_alignment import (
    forward_alignment_enabled, heading_keyframes, tangent_keyframes,
)
import test_course_timing as editor_tests


def waypoints(yaw=37.0, count=3):
    frames = {}
    for index in range(count):
        frames[f"fo{index}"] = {
            "t": float(index * 3),
            "fo": [[float(index), None, 0.0], [0.0, None, 0.0],
                   [0.0, 0.0, 0.0], [math.radians(yaw), 0.0, 0.0]],
        }
    return {"Nco": 6, "keyframes": frames, "timing": {"mode": "manual"}}


class ConversionTests(unittest.TestCase):
    def test_arbitrary_heading_speed_and_input_unchanged(self):
        for angle in (0, 90, 37, 143):
            for count in (2, 4):
                with self.subTest(angle=angle, count=count):
                    frames = waypoints(angle, count)["keyframes"]
                    original = copy.deepcopy(frames)
                    converted = tangent_keyframes(frames)
                    self.assertEqual(frames, original)
                    for frame in converted.values():
                        vx, vy = frame["fo"][0][1], frame["fo"][1][1]
                        yaw = frame["fo"][3][0]
                        self.assertAlmostEqual(-math.sin(yaw) * vx + math.cos(yaw) * vy, 0)
                        self.assertAlmostEqual(math.cos(yaw) * vx + math.sin(yaw) * vy, 1 / 3)

    def test_unequal_segment_speed_estimates(self):
        frames = waypoints()["keyframes"]
        frames["fo2"]["fo"][0][0] = 4.0
        converted = tangent_keyframes(frames)
        for frame, speed in zip(converted.values(), (1 / 3, 2 / 3, 1.0)):
            self.assertAlmostEqual(math.hypot(frame["fo"][0][1], frame["fo"][1][1]), speed)

    def test_vertical_coincident_short_segments_and_stops(self):
        for vertical in (False, True):
            frames = waypoints()["keyframes"]
            for index, frame in enumerate(frames.values()):
                frame["fo"][0][0] = 0.0
                frame["fo"][2][0] = float(index) if vertical else 0.0
            for frame in tangent_keyframes(frames).values():
                self.assertEqual([row[1] for row in frame["fo"][:2]], [0.0, 0.0])
        frames = waypoints()["keyframes"]
        frames["fo1"]["t"] = 1e-12
        for name in ("fo0", "fo2"):
            for row in frames[name]["fo"][:2]:
                row[1] = 0.0
        converted = tangent_keyframes(frames)
        self.assertLess(math.hypot(*[row[1] for row in converted["fo1"]["fo"][:2]]), 51.0)
        for name in ("fo0", "fo2"):
            self.assertEqual(converted[name], frames[name])

    def test_prescribed_velocity_compatibility_and_missing_values(self):
        frames = waypoints(37)["keyframes"]
        frames["fo1"]["fo"][0][1] = 2 * math.cos(math.radians(37))
        result = tangent_keyframes(frames)
        self.assertAlmostEqual(result["fo1"]["fo"][1][1], 2 * math.sin(math.radians(37)))
        frames["fo1"]["fo"][1][1] = -1.0
        with self.assertRaisesRegex(ValueError, "conflicts"):
            tangent_keyframes(frames)
        frames["fo1"]["fo"][3][0] = None
        self.assertEqual(tangent_keyframes(frames)["fo1"], frames["fo1"])
        frames = waypoints(0)["keyframes"]
        frames["fo1"]["fo"][0][1] = -1.0
        with self.assertRaisesRegex(ValueError, "conflicts"):
            tangent_keyframes(frames)
        frames = waypoints()["keyframes"]
        for frame in frames.values():
            frame["fo"][0] = [None]
            frame["fo"][1] = [0.0]
        for frame in tangent_keyframes(frames).values():
            self.assertEqual(frame["fo"][0], [None, 0.0])

    def test_wrapping_and_explicit_turns(self):
        frames = waypoints(179)["keyframes"]
        frames["fo1"]["fo"][3][0] = math.radians(-179)
        frames["fo2"]["fo"][3][0] = 4 * math.pi
        result = heading_keyframes(frames)
        self.assertAlmostEqual(result["fo1"]["fo"][3][0], math.radians(181))
        self.assertEqual(result["fo2"]["fo"][3][0], 4 * math.pi)
        self.assertEqual(frames["fo1"]["fo"][3][0], math.radians(-179))

    def test_invalid_settings_and_timings(self):
        self.assertFalse(forward_alignment_enabled({}))
        for bad in ("false", 1, None):
            with self.assertRaises(ValueError):
                forward_alignment_enabled({"forward_aligned_motion": bad})
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            frames = waypoints()["keyframes"]
            frames["fo1"]["t"] = bad
            with self.assertRaises(ValueError):
                tangent_keyframes(frames)


class SolverTests(unittest.TestCase):
    def test_off_is_identical_to_legacy_conversion_and_solution(self):
        wp = waypoints(179)
        wp["keyframes"]["fo1"]["fo"][3][0] = math.radians(-179)
        expected_times, expected_fo = th.KF_to_TpFO(wp["keyframes"], 5)
        original = MinTimeSnap(wp, 20, None)
        wp["forward_aligned_motion"] = False
        off = MinTimeSnap(wp, 20, None)
        np.testing.assert_array_equal(off.Tkf, expected_times)
        np.testing.assert_array_equal(off.FOkf, expected_fo)
        np.testing.assert_array_equal(off.Pnd, original.Pnd)
        self.assertEqual(off.keyframes, wp["keyframes"])

    def test_real_solver_tangents_in_all_timing_modes(self):
        for mode in ("manual", "automatic", "total_duration"):
            for angle in (0, 90, 37, 143):
                with self.subTest(mode=mode, angle=angle):
                    wp = waypoints(angle)
                    wp["forward_aligned_motion"] = True
                    wp["timing"] = {"mode": mode, "total_duration": 8.0}
                    for name in ("fo0", "fo2"):
                        for row in wp["keyframes"][name]["fo"][:2]:
                            row[1] = 0.0
                    original = copy.deepcopy(wp)
                    solved = MinTimeSnap(wp, 20, None)
                    self.assertEqual(wp, original)
                    values = th.dTPn_to_FO(solved.Tkf, solved.dTd, solved.Pnd, 5)
                    yaw = values[:, 3, 0]
                    forward = np.cos(yaw) * values[:, 0, 1] + np.sin(yaw) * values[:, 1, 1]
                    lateral = -np.sin(yaw) * values[:, 0, 1] + np.cos(yaw) * values[:, 1, 1]
                    np.testing.assert_allclose(lateral, 0, atol=1e-7)
                    self.assertTrue(np.all(forward >= -1e-7))
                    self.assertGreater(forward[1], 0)
                    np.testing.assert_allclose(values[[0, 2], :2, 1], 0, atol=1e-7)

    def test_real_solver_wrapping(self):
        wp = waypoints(179, 2)
        wp["keyframes"]["fo1"]["fo"][3][0] = math.radians(-179)
        wp["forward_aligned_motion"] = True
        solved = MinTimeSnap(wp, 20, None)
        self.assertAlmostEqual(np.diff(solved.FOkf[:, 3, 0])[0], math.radians(2))


class EditorTests(unittest.TestCase):
    def test_toggle_save_reload_and_restore(self):
        editor = editor_tests.EditorTimingTests().make_editor({"waypoints": waypoints()})
        self.assertFalse(editor.forward_aligned_gui.value)
        original = copy.deepcopy(editor.keyframes)
        baseline = MinTimeSnap(editor._validated_course()["waypoints"], 20, None)
        editor.forward_aligned_gui.value = True
        editor._set_forward_alignment()
        self.assertIsNone(editor.optimized_times)
        on = MinTimeSnap(editor._validated_course()["waypoints"], 20, None)
        self.assertFalse(np.allclose(on.Pnd, baseline.Pnd))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.json"
            editor._write_course(path)
            saved = json.loads(path.read_text())
            self.assertTrue(saved["waypoints"]["forward_aligned_motion"])
            self.assertEqual(saved["waypoints"]["keyframes"], original)
            reopened = editor_tests.EditorTimingTests().make_editor(saved)
            self.assertTrue(reopened.forward_aligned_gui.value)
        editor.forward_aligned_gui.value = False
        editor._set_forward_alignment()
        restored = MinTimeSnap(editor._validated_course()["waypoints"], 20, None)
        np.testing.assert_array_equal(restored.Pnd, baseline.Pnd)
        self.assertEqual(editor.keyframes, original)
        course, _ = _normalise_course({"waypoints": waypoints()})
        self.assertFalse(course["waypoints"]["forward_aligned_motion"])
