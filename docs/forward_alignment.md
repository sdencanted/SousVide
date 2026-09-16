# Forward-aligned keypoint motion

In the Viser editor's **Course** folder, enable **Forward-aligned motion**,
then use **Solve trajectory** or **Run FiGS simulation** to regenerate.
Changing the checkbox invalidates previous solve/simulation results. It defaults
to OFF, including when opening old JSON files. Save JSON persists
`waypoints.forward_aligned_motion`; generated derivatives never overwrite the
authored keyframes. Disabling the checkbox restores the original constraints.

## Implementation and repository findings

FiGS is a Git submodule at `FiGS/`, pointing to `sdencanted/FiGS`. This fork
already includes the Viser editor, course timing settings, and optimizer failure
diagnostics. It is not vendored Python code or merely a package dependency.
Changes inside it must be committed in the FiGS repository before updating the
parent repository's submodule pointer when publishing.

`figs.utilities.course_editor.CourseEditor.keyframes` references
`course["waypoints"]["keyframes"]`. Each frame has an arrival time `t` and `fo`
rows for world x/y/z/yaw, with columns for derivatives d0, d1, etc. `null` leaves
a derivative free. The editor already exposes derivative controls and serializes
the model to JSON. Its display converts canonical FLU coordinates to rendered
FRD coordinates; alignment operates on canonical world coordinates, not rendered
poses. Positions are never rotated.

The editor calls `MinTimeSnap` directly for Solve trajectory. Simulation,
SousVide rollout generation, deployment, and FiGS reference planning also use
that class. `transform_helper.KF_to_TpFO` converts the JSON matrices to solver
arrays, padding unspecified derivatives with NaN. Opt-in preprocessing at this
single entry point makes saved settings work consistently across these callers,
without requiring FiGS to depend on SousVide. The OFF branch retains its original
conversion, timing, yaw, and optimization behavior.

Inspection of the [upstream MinTimeSnap implementation](https://github.com/StanfordMSL/FiGS/blob/main/src/figs/tsplines/min_time_snap.py)
and the local fork showed block-diagonal per-output constraint matrices and a
fixed/free scalar-derivative mapping. The solver exposes no coupled equality or
inequality API. Option B would require changing that mapping or introducing an
equality-constrained solve, with additional rank and conditioning tests. Option A
therefore provides a localized implementation without changing the QP.

Existing automated coverage lives in `tests/test_course_timing.py` (timing,
solver constraints, editor state, JSON, mocked Viser interactions). New coverage
is in `tests/test_forward_alignment.py`.

## Tangent and speed rules

At each keypoint with specified yaw ψ, the prepared constraints are
`vx = s cos(ψ)` and `vy = s sin(ψ)`, with `s >= 0`. Thus lateral velocity
`-sin(ψ) vx + cos(ψ) vy` is zero, and forward velocity is nonnegative there.
This uses auto-speed tangent constraints, not a direction-only coupled solve
or an inequality added to the optimizer. z and all higher derivatives retain
their original constraints. Unspecified yaw is skipped.

For unconstrained horizontal velocity, speed is the mean of the adjacent
horizontal chord lengths divided by their segment durations. Endpoints use the
single adjacent segment. Durations come from the existing manual, automatic,
or fixed-total-duration initialization, and speeds remain fixed while the time
optimizer adjusts durations. The estimate floors durations at FiGS' existing
0.01-second lower optimization bound; this does not change manual timestamps or
make arbitrarily short trajectories physically feasible. Nonpositive/nonfinite
durations and nonfinite speed estimates raise errors.

Coincident and vertical-only segments contribute zero horizontal speed. A
segment with missing x/y position constraints contributes zero because its chord
cannot be estimated. Other valid adjacent segments can still contribute speed.
Explicit d1 components are preserved and determine speed where possible. Both
vx=vy=0 preserve a stop, at endpoints or interior points. A partial zero velocity
constraint can also force a stop when the heading has a nonzero component on
that axis. A component perpendicular to the heading must be zero. Conflicting
or negative-forward constraints raise an actionable error; clear the relevant
x/y d1 controls or revise yaw to proceed. No speed cap or continuous forward
guarantee is implied. Debug logging from `figs.utilities.forward_alignment`
reports chosen speeds and tangents.

The existing editor's nearest-equivalent-angle helper is shared with ON-mode
preparation. Principal-range yaw values are unwrapped before timing estimation
and solving, so 179° → -179° becomes 179° → 181°. Values outside [-π, π] are
treated as explicitly unwrapped turns. Authored yaw and the OFF behavior remain
unchanged. At zero horizontal speed, a direction of travel is undefined, but
both forward/lateral constraints are satisfied.

## Validation

```bash
PYTHONPATH=FiGS/src .venv/bin/python -m unittest discover -s tests -p 'test_course_timing.py'
PYTHONPATH=FiGS/src .venv/bin/python -m unittest discover -s tests -p 'test_forward_alignment.py'
PYTHONPATH=FiGS/src .venv/bin/python scripts/validate_forward_alignment.py --viser
```

The script uses the existing `traverse.json` example with its authored arrival
times. It evaluates the actual polynomial at solved keypoint times, prints
position/yaw/vx/vy/forward/lateral velocity, asserts alignment, and verifies exact
OFF coefficient restoration. `--viser` additionally launches a real local
Viser server, requests its HTTP page, checks the checkbox handle, drives its
callbacks through OFF/ON/OFF solves, and saves JSON. This is a server/widget
smoke test, not browser visual verification or a physics simulation.

Alignment is **keypoint-only**. Motion may still travel sideways or backward
relative to yaw between keypoints, and tangent constraints can create loops or
large acceleration demands. The next improvement is sampled cross-axis
constraints for a fixed yaw reference, followed by dense residual validation.
Jointly optimizing yaw makes these constraints nonlinear; finite collocation
alone would still not prove continuous zero lateral velocity.
