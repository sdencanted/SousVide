"""Run with PYTHONPATH=FiGS/src python -m unittest discover -s tests -p test_course_timing.py."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from figs.tsplines.min_time_snap import MinTimeSnap
from figs.utilities import transform_helper as th
from figs.utilities.course_editor import CourseEditor, _new_course, _normalise_course
from figs.utilities.course_timing import estimate_times, manual_times, timing_settings, TIMING_MODES
from figs.utilities.trajectory_diagnostics import TrajectoryTimingError


def sample_course():
    course = _new_course()
    course["waypoints"]["keyframes"]["fo1"]["fo"][0][0] = 3.0
    course["waypoints"]["keyframes"]["fo1"]["t"] = 6.0
    return course


class TimingTests(unittest.TestCase):
    def test_estimates_cover_distance_yaw_stops_and_coincident_points(self):
        frames = sample_course()["waypoints"]["keyframes"]
        original = copy.deepcopy(frames)
        stopped = estimate_times(frames, timing_settings())[-1]
        self.assertEqual(frames, original)
        for frame in frames.values():
            for row in frame["fo"]:
                row[1] = None
        moving = estimate_times(frames, timing_settings())[-1]
        self.assertGreater(stopped, moving)
        frames["fo1"]["fo"][3][0] = 6.0
        self.assertGreater(estimate_times(frames, timing_settings())[-1], moving)
        frames["fo1"]["fo"] = copy.deepcopy(frames["fo0"]["fo"])
        self.assertGreater(estimate_times(frames, timing_settings())[-1], 0)

    def test_total_time_estimate_respects_bounds_even_with_unequal_distances(self):
        frames = sample_course()["waypoints"]["keyframes"]
        frames["fo2"] = copy.deepcopy(frames["fo1"])
        frames["fo2"]["fo"][0][0] = 3000.0
        for duration in [0.02, 0.1, 59.99, 60.0]:
            times = estimate_times(frames, timing_settings({"mode": "total_duration", "total_duration": duration}))
            self.assertAlmostEqual(times[-1], duration)
            self.assertTrue(np.all(np.diff(times) >= 0.01 - 1e-12))
            self.assertTrue(np.all(np.diff(times) <= 30.0 + 1e-12))
        with self.assertRaisesRegex(ValueError, "between"):
            estimate_times(frames, {"mode": "total_duration", "total_duration": 61.0})

    def test_manual_times_are_validated_in_insertion_order(self):
        frames = sample_course()["waypoints"]["keyframes"]
        self.assertEqual(manual_times(frames), [0.0, 6.0])
        with self.assertRaisesRegex(ValueError, "first manual"):
            manual_times(dict(reversed(list(frames.items()))))
        for value in [0.0, -1.0, float("nan"), float("inf"), True]:
            frames["fo1"]["t"] = value
            with self.assertRaises(ValueError):
                manual_times(frames)
        with self.assertRaisesRegex(ValueError, "at least two"):
            manual_times({"fo0": frames["fo0"]})

    def test_settings_reject_invalid_numbers_and_modes(self):
        for value in [0, -1, True, float("nan"), float("inf"), "fast"]:
            with self.assertRaises(ValueError):
                timing_settings({"aggressiveness": value})
        with self.assertRaises(ValueError):
            timing_settings({"mode": "unknown"})


class SolverTimingTests(unittest.TestCase):
    def test_manual_overrides_pilot_and_legacy_policy_still_works(self):
        waypoints = sample_course()["waypoints"]
        waypoints["timing"]["mode"] = "manual"
        fixed = MinTimeSnap(waypoints, 20, 100.0)
        np.testing.assert_allclose(fixed.Tkf, [0, 6], atol=1e-10)
        del waypoints["timing"]
        legacy = MinTimeSnap(waypoints, 20, None)
        np.testing.assert_allclose(legacy.Pnd, fixed.Pnd)

    def test_automatic_ignores_seed_timestamps_and_responds_to_aggressiveness(self):
        waypoints = sample_course()["waypoints"]
        for frame in waypoints["keyframes"].values():
            del frame["t"]
        original = copy.deepcopy(waypoints)
        slow = MinTimeSnap(waypoints, 20, None)
        self.assertEqual(waypoints, original)
        waypoints["timing"]["aggressiveness"] = 4.0
        fast = MinTimeSnap(waypoints, 20, None)
        self.assertLess(fast.Tkf[-1], slow.Tkf[-1])
        endpoint_values = th.dTPn_to_FO(fast.Tkf, fast.dTd, fast.Pnd, 5)
        np.testing.assert_allclose(endpoint_values[:, 0, 0], [0, 3], atol=1e-8)
        np.testing.assert_allclose(endpoint_values[:, :3, 1:3], 0, atol=1e-8)

    def test_total_duration_optimizes_allocation_and_preserves_waypoints(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/courses/traverse.json").open() as file:
            waypoints = json.load(file)["waypoints"]
        waypoints["timing"] = timing_settings({"mode": "total_duration", "total_duration": 16.0})
        initial = estimate_times(waypoints["keyframes"], waypoints["timing"])
        solved = MinTimeSnap(waypoints, 20, 100.0, True)
        self.assertAlmostEqual(solved.Tkf[-1], 16.0, places=5)
        self.assertFalse(np.allclose(solved.Tkf, initial, atol=0.01))
        values = th.dTPn_to_FO(solved.Tkf, solved.dTd, solved.Pnd, 5)
        for index, frame in enumerate(waypoints["keyframes"].values()):
            for axis, row in enumerate(frame["fo"]):
                for derivative, value in enumerate(row):
                    if value is not None:
                        self.assertAlmostEqual(values[index, axis, derivative], value, places=6)

    def test_optimizer_failure_is_not_published_as_a_solution(self):
        with mock.patch("figs.tsplines.min_time_snap.minimize", return_value=SimpleNamespace(success=False, message="iteration limit")):
            with self.assertRaisesRegex(ValueError, "iteration limit"):
                MinTimeSnap(sample_course()["waypoints"], 20, 10.0)

    def test_failure_captures_actual_durations_constraints_and_costs(self):
        waypoints = sample_course()["waypoints"]
        waypoints["keyframes"]["fo1"]["fo"][3][0] = 4.0
        initial = estimate_times(waypoints["keyframes"], waypoints["timing"])
        failure = SimpleNamespace(success=False, message="Inequality constraints incompatible &#x20;", x=np.array([0.01]))
        with mock.patch("figs.tsplines.min_time_snap.minimize", return_value=failure):
            with self.assertRaises(TrajectoryTimingError) as caught:
                MinTimeSnap(waypoints, 20, 10.0)
        diagnostic = caught.exception
        segment = diagnostic.segments[0]
        self.assertEqual(segment["start"], "fo0")
        self.assertEqual(segment["end"], "fo1")
        self.assertEqual(segment["initial_dt"], initial[-1])
        self.assertEqual(segment["last_dt"], 0.01)
        self.assertEqual(segment["distance"], 3.0)
        self.assertEqual(segment["yaw_change"], 4.0)
        self.assertGreater(segment["cost"], 0)
        self.assertIn("at lower time bound", segment["notes"])
        self.assertIn("yaw change exceeds π", diagnostic.markdown())
        self.assertIn("suspects, not confirmed causes", diagnostic.markdown())
        self.assertIn("[4.0, 0.0, 0.0, null]", diagnostic.markdown())
        self.assertNotIn("&#x20;", str(diagnostic))
        waypoints["keyframes"]["fo1"]["fo"][0][0] = 999.0
        self.assertEqual(diagnostic.keyframes["fo1"]["fo"][0][0], 3.0)

    def test_failure_diagnostics_survive_nonfinite_iterate_and_cost_failure(self):
        failure = SimpleNamespace(success=False, message="Inequality constraints incompatible", x=np.array([np.nan]))
        with mock.patch("figs.tsplines.min_time_snap.minimize", return_value=failure), mock.patch.object(MinTimeSnap, "solve_uqp", side_effect=RuntimeError("singular matrix")):
            with self.assertRaises(TrajectoryTimingError) as caught:
                MinTimeSnap(sample_course()["waypoints"], 20, 10.0)
        diagnostic = caught.exception
        self.assertIn("invalid last duration", diagnostic.segments[0]["notes"])
        self.assertIsNone(diagnostic.segments[0]["cost"])
        self.assertIn("singular matrix", diagnostic.markdown())
        self.assertIn("No segment cost ranking", diagnostic.markdown())


class EditorTimingTests(unittest.TestCase):
    def make_editor(self, course=None):
        course, derivative_count = _normalise_course(course or sample_course())
        server = mock.MagicMock()

        def widget(label, *args, **kwargs):
            return mock.MagicMock(label=label, value=kwargs.get("initial_value"), content=label, **{
                "disabled": kwargs.get("disabled", False), "visible": True,
            })

        for method in ("add_dropdown", "add_button", "add_markdown", "add_number", "add_text", "add_checkbox"):
            getattr(server.gui, method).side_effect = widget
        with mock.patch.dict("sys.modules", {"viser": SimpleNamespace()}):
            return CourseEditor(course, derivative_count, Path("unused.json"), "127.0.0.1", 8080, server=server)

    def set_mode(self, editor, mode):
        editor.timing_mode_gui.value = TIMING_MODES[mode]
        editor._set_timing()

    def test_mode_controls_and_save_reload_preserve_planner_behavior(self):
        editor = self.make_editor()
        self.assertTrue(editor.time_gui.disabled)
        self.assertTrue(editor.aggressiveness_gui.visible)
        editor._set_time(999.0)
        self.assertEqual(editor.keyframes["fo0"]["t"], 0.0)
        self.set_mode(editor, "manual")
        self.assertFalse(editor.time_gui.disabled)
        self.assertFalse(editor.aggressiveness_gui.visible)
        editor._select("fo1")
        editor._set_time(8.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.json"
            editor._write_course(path)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["waypoints"]["timing"]["mode"], "manual")
            solved = MinTimeSnap(loaded["waypoints"], 20, 100.0)
            self.assertEqual(solved.Tkf[-1], 8.0)
            reopened = self.make_editor(loaded)
            self.assertFalse(reopened.time_gui.disabled)
            self.assertEqual(reopened.keyframes["fo1"]["t"], 8.0)
        self.set_mode(editor, "total_duration")
        self.assertTrue(editor.time_gui.disabled)
        self.assertTrue(editor.total_duration_gui.visible)
        self.assertAlmostEqual(editor.keyframes["fo1"]["t"], 10.0)

    def test_geometry_edit_retimes_and_invalidates_simulation_and_plan(self):
        editor = self.make_editor()
        old_time = editor.keyframes["fo1"]["t"]
        editor.optimized_times = [0.0, 4.0]
        editor.simulated_tro = np.arange(2)
        editor.simulated_xro = np.zeros((2, 10))
        editor.simulated_rgb = np.zeros((2, 1, 1, 3))
        editor.show_simulated_trajectory = True
        editor._select("fo1")
        editor.value_guis[0][0][1].value = 9.0
        editor._set_value(0, 0)
        self.assertGreater(editor.keyframes["fo1"]["t"], old_time)
        self.assertIsNone(editor.optimized_times)
        self.assertIsNone(editor.simulated_rgb)
        self.assertTrue(editor.export_video_button.disabled)
        self.assertTrue(editor.route_toggle_button.disabled)
        self.assertEqual(editor.feasibility_gui.content, "")

    def test_manual_reorder_errors_do_not_overwrite_saved_file(self):
        editor = self.make_editor()
        self.set_mode(editor, "manual")
        editor._select("fo1")
        editor._move_selected(0, first=True)
        self.assertIn("first manual", editor.arrival_times_gui.content)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.json"
            path.write_text("previous contents")
            with self.assertRaises(ValueError):
                editor._write_course(path)
            self.assertEqual(path.read_text(), "previous contents")

    def test_add_after_selecting_first_point_does_not_duplicate_manual_time(self):
        editor = self.make_editor()
        self.set_mode(editor, "manual")
        editor._add_keypoint()
        self.assertTrue(np.all(np.diff(manual_times(editor.keyframes)) > 0))

    def test_mode_switch_restores_manual_times_and_invalid_settings_restore_controls(self):
        editor = self.make_editor()
        self.set_mode(editor, "manual")
        editor._select("fo1")
        editor._set_time(12.5)
        self.set_mode(editor, "automatic")
        self.assertNotEqual(editor.keyframes["fo1"]["t"], 12.5)
        editor.aggressiveness_gui.value = -1.0
        editor._set_timing()
        self.assertEqual(editor.aggressiveness_gui.value, 1.0)
        self.assertIn("positive", editor.status.content)
        self.set_mode(editor, "manual")
        self.assertEqual(editor.keyframes["fo1"]["t"], 12.5)

    def test_solve_without_scene_displays_times_and_control_checks(self):
        editor = self.make_editor()
        self.assertTrue(editor.simulate_button.disabled)
        editor._solve_timing()
        self.assertIn("Trajectory solved", editor.status.content)
        self.assertEqual(len(editor.optimized_times), 2)
        self.assertIn("Peak sampled speed", editor.feasibility_gui.content)
        self.assertIn("Solved arrival", editor.arrival_times_gui.content)
        self.assertFalse(editor.solve_button.disabled)

    def test_control_limit_violation_and_nonfinite_reference_are_reported(self):
        editor = self.make_editor()
        fo = np.zeros((2, 4, 5))
        reference = np.zeros((2, 15))
        reference[0, -1] = 6.0
        report = editor._reference_report(fo, reference, [-1, -5, -5, -5], [0, 5, 5, 5])
        self.assertIn("Control limits exceeded", report)
        self.assertIn("yaw rate", report)
        reference[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            editor._reference_report(fo, reference, [-1, -5, -5, -5], [0, 5, 5, 5])

    def test_simulation_receives_timing_and_publishes_actual_arrivals(self):
        editor = self.make_editor()
        self.set_mode(editor, "manual")
        editor.gsplat = object()
        reference = np.zeros((2, 15))
        reference[:, 0] = [0.0, 6.0]
        controller = SimpleNamespace(
            FOd=np.zeros((2, 4, 5)), tXUd=reference, Tkf=np.array([0.0, 6.0]),
            lbu=np.array([-1, -5, -5, -5]), ubu=np.array([0, 5, 5, 5]), hz=20,
        )
        simulator = mock.MagicMock()
        simulator.simulate.return_value = (
            reference[:, 0], reference[:, 1:11], None, None, np.zeros((1, 1, 1, 3)), None, None,
        )
        factory = mock.Mock(return_value=controller)
        with mock.patch.dict("sys.modules", {
            "figs.control.vehicle_rate_mpc": SimpleNamespace(VehicleRateMPC=factory),
            "figs.simulator": SimpleNamespace(Simulator=mock.Mock(return_value=simulator)),
        }):
            editor._run_simulation()
        self.assertIn("Simulation complete", editor.status.content)
        self.assertEqual(editor.optimized_times, [0.0, 6.0])
        self.assertFalse(editor.export_video_button.disabled)
        snapshot = factory.call_args.args[1]
        self.assertEqual(snapshot["waypoints"]["timing"]["mode"], "manual")
        self.assertIsNot(snapshot, editor.course)

    def test_edit_during_solve_discards_old_results(self):
        editor = self.make_editor()
        self.set_mode(editor, "manual")

        def solve_then_edit(*args, **kwargs):
            result = MinTimeSnap(*args, **kwargs)
            editor._course_changed()
            return result

        with mock.patch("figs.tsplines.min_time_snap.MinTimeSnap", side_effect=solve_then_edit):
            editor._solve_timing()
        self.assertIsNone(editor.optimized_times)
        self.assertIn("Course changed during the solve", editor.status.content)

    def make_failure(self, editor):
        frames = editor.keyframes
        initial = np.diff([frame["t"] for frame in frames.values()])
        return TrajectoryTimingError("Inequality constraints incompatible &#x20;", frames, initial,
                                     np.full(len(initial), 0.01), (0.01, 30.0), costs=np.ones(len(initial)))

    def test_solve_failure_shows_values_and_selects_keypoint_then_clears_on_edit(self):
        editor = self.make_editor()
        failure = self.make_failure(editor)
        with mock.patch("figs.tsplines.min_time_snap.MinTimeSnap", side_effect=failure):
            editor._solve_timing()
        self.assertTrue(editor.diagnostics_folder.visible)
        self.assertIn("[3.0, 0.0, 0.0, null]", editor.diagnostics_gui.content)
        self.assertNotIn("&#x20;", editor.status.content)
        editor.diagnostic_keypoint_gui.value = "fo1"
        editor._inspect_failure_keypoint()
        self.assertEqual(editor.selection, "fo1")
        self.assertTrue(editor.diagnostics_folder.visible)
        editor._course_changed()
        self.assertFalse(editor.diagnostics_folder.visible)
        self.assertEqual(editor.diagnostics_gui.content, "")
        self.assertTrue(editor.inspect_keypoint_button.disabled)

    def test_simulation_failure_uses_same_diagnostics(self):
        editor = self.make_editor()
        editor.gsplat = object()
        failure = self.make_failure(editor)
        with mock.patch.dict("sys.modules", {
            "figs.control.vehicle_rate_mpc": SimpleNamespace(VehicleRateMPC=mock.Mock(side_effect=failure)),
            "figs.simulator": SimpleNamespace(Simulator=mock.MagicMock()),
        }):
            editor._run_simulation()
        self.assertIn("Simulation failed", editor.status.content)
        self.assertEqual(editor.diagnostics_gui.content, failure.markdown())
        self.assertFalse(editor.simulate_button.disabled)

    def test_failed_old_solve_does_not_diagnose_new_course(self):
        editor = self.make_editor()
        failure = self.make_failure(editor)
        revision = editor._revision
        editor._course_changed()
        editor._show_failure("Simulation", failure, revision)
        self.assertFalse(editor.diagnostics_folder.visible)
        self.assertIn("Course changed", editor.status.content)


if __name__ == "__main__":
    unittest.main()
