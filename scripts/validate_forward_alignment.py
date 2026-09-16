"""Lightweight real FiGS solve; optionally smoke-test an actual Viser server.

PYTHONPATH=FiGS/src .venv/bin/python scripts/validate_forward_alignment.py --viser
"""

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

import numpy as np

from figs.tsplines.min_time_snap import MinTimeSnap
from figs.utilities import transform_helper as th
from figs.utilities.course_editor import CourseEditor, _normalise_course


def validate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viser", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    course = json.loads((root / "configs/courses/traverse.json").read_text())
    # Use the example's authored arrival times, making this a quick repeatable solve.
    course["waypoints"]["timing"] = {"mode": "manual"}
    wp = course["waypoints"]
    baseline = MinTimeSnap(wp, 100, None)
    wp["forward_aligned_motion"] = True
    solved = MinTimeSnap(wp, 100, None)
    values = th.dTPn_to_FO(solved.Tkf, solved.dTd, solved.Pnd, 5)
    print("keypoint  position (world)       yaw          vx          vy     forward     lateral")
    for name, value in zip(wp["keyframes"], values):
        yaw, vx, vy = value[3, 0], value[0, 1], value[1, 1]
        forward = np.cos(yaw) * vx + np.sin(yaw) * vy
        lateral = -np.sin(yaw) * vx + np.cos(yaw) * vy
        print(f"{name:8} {str(np.round(value[:3, 0], 3)):22} {yaw:8.4f} {vx:11.6f} {vy:11.6f} {forward:11.6f} {lateral:11.3e}")
        assert abs(lateral) < 1e-7 and forward >= -1e-7
    assert not np.allclose(solved.Pnd, baseline.Pnd)
    wp["forward_aligned_motion"] = False
    np.testing.assert_array_equal(MinTimeSnap(wp, 100, None).Pnd, baseline.Pnd)
    print("Real MinTimeSnap: ON tangents verified; OFF restores identical coefficients.")
    if args.viser:
        import viser

        server = viser.ViserServer(host="127.0.0.1", port=8899)
        try:
            with tempfile.TemporaryDirectory() as directory:
                course, derivatives = _normalise_course(copy.deepcopy(course))
                editor = CourseEditor(course, derivatives, Path(directory) / "course.json",
                                      "127.0.0.1", server.get_port(), server=server)
                with urlopen(f"http://127.0.0.1:{server.get_port()}", timeout=5) as response:
                    assert response.status == 200
                assert editor.forward_aligned_gui.label == "Forward-aligned motion"
                for enabled in (False, True, False):
                    editor.forward_aligned_gui.value = enabled
                    deadline = time.monotonic() + 5
                    while editor.course["waypoints"]["forward_aligned_motion"] != enabled:
                        if time.monotonic() >= deadline:
                            raise AssertionError("Viser toggle callback did not update the model")
                        time.sleep(0.01)
                    editor._solve_timing()
                    assert editor.optimized_times is not None, editor.status.content
                    print(f"Viser toggle {enabled}: {editor.status.content}")
                editor._write_course(editor.output_path)
                assert json.loads(editor.output_path.read_text())["waypoints"]["forward_aligned_motion"] is False
                print("Real Viser server/widget/callback smoke test passed (no browser visual inspection).")
        finally:
            server.stop()


if __name__ == "__main__":
    validate()
