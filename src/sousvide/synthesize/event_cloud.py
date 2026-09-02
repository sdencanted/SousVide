"""Deterministic preprocessing for SECNet event-cloud inputs.

Event rows use the released SECNet implementation's ``[t, x, y, p]``
ordering.  Sampling follows SECNet's random-downsample-then-sort rule while
deriving a stable per-window seed so offline generation and deployment are
reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np


EVENT_CLOUD_FORMAT_VERSION = 1
DEFAULT_EVENT_CLOUD_NUM_POINTS = 4096
DEFAULT_EVENT_CLOUD_SEED = 0


class EventCloudOptions(TypedDict):
    num_points: int
    seed: int


def resolve_event_cloud_options(
        options: Mapping[str, Any] | None = None) -> EventCloudOptions:
    """Validate and populate event-cloud preprocessing options."""
    supplied = {} if options is None else dict(options)
    unknown = set(supplied)-{"num_points", "seed"}
    if unknown:
        raise ValueError(
            f"Unsupported event_cloud options: {sorted(unknown)}.")

    num_points = supplied.get("num_points", DEFAULT_EVENT_CLOUD_NUM_POINTS)
    seed = supplied.get("seed", DEFAULT_EVENT_CLOUD_SEED)
    if (isinstance(num_points, bool) or not isinstance(num_points, int)
            or num_points <= 0):
        raise ValueError("event_cloud num_points must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("event_cloud seed must be an integer.")
    return {"num_points": num_points, "seed": seed}


def event_cloud_metadata(
        options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the complete, serializable event-cloud format description."""
    resolved = resolve_event_cloud_options(options)
    return {
        "format_version": EVENT_CLOUD_FORMAT_VERSION,
        "num_points": resolved["num_points"],
        "seed": resolved["seed"],
        "column_order": ["t", "x", "y", "p"],
        "time_normalization": "sample_minmax",
        "spatial_normalization": "x/width,y/height",
        "polarity": "-1,+1",
        "sampling": "seeded_random_sorted",
    }


def _window_seed(seed: int, stream_id: str, window_index: int) -> int:
    payload = f"{seed}:{stream_id}:{window_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def events_to_event_cloud(
        events: np.ndarray | None,
        width: int,
        height: int,
        *,
        stream_id: str,
        window_index: int,
        options: Mapping[str, Any] | None = None,
        polarity_signed: bool = False,
) -> tuple[np.ndarray, int]:
    """Sample and normalize one variable-length ``[t,x,y,p]`` event window."""
    resolved = resolve_event_cloud_options(options)
    num_points = resolved["num_points"]
    if width <= 0 or height <= 0:
        raise ValueError("Event-cloud width and height must be positive.")
    if window_index < 0:
        raise ValueError("Event-cloud window_index must be non-negative.")

    if events is None:
        event_array = np.empty((0, 4), dtype=np.float64)
    else:
        event_array = np.asarray(events)
        if event_array.ndim != 2 or event_array.shape[1] < 4:
            raise ValueError(
                "Events must have shape (N, >=4) with columns [t,x,y,p].")
        event_array = event_array[:, :4]

    raw_count = len(event_array)
    if raw_count == 0:
        return np.zeros((num_points, 4), dtype=np.float32), 0
    if not np.all(np.isfinite(event_array)):
        raise ValueError("Event-cloud inputs must contain only finite values.")
    if np.any(np.diff(event_array[:, 0].astype(np.float64)) < 0):
        raise ValueError("Event-cloud timestamps must be monotonic.")
    if (np.any(event_array[:, 1] < 0) or np.any(event_array[:, 1] >= width)
            or np.any(event_array[:, 2] < 0)
            or np.any(event_array[:, 2] >= height)):
        raise ValueError("Event-cloud coordinates must be within sensor bounds.")
    valid_polarities = (-1,1) if polarity_signed else (0,1)
    if not np.all(np.isin(event_array[:,3],valid_polarities)):
        raise ValueError(
            f"Event-cloud polarity must use {valid_polarities} values.")

    rng = np.random.default_rng(
        _window_seed(resolved["seed"], str(stream_id), window_index))
    indices = rng.choice(
        raw_count, size=num_points, replace=raw_count < num_points)
    indices.sort()
    cloud = event_array[indices].astype(np.float64, copy=True)

    timestamp_min = float(cloud[:, 0].min())
    timestamp_range = float(cloud[:, 0].max())-timestamp_min
    if timestamp_range > np.finfo(np.float64).eps:
        cloud[:, 0] = (cloud[:, 0]-timestamp_min)/timestamp_range
    else:
        cloud[:, 0] = 0.0
    cloud[:, 1] /= float(width)
    cloud[:, 2] /= float(height)
    cloud[:, 3] = (
        np.where(cloud[:, 3] > 0, 1.0, -1.0)
        if polarity_signed else 2.0*cloud[:, 3]-1.0)
    return cloud.astype(np.float32, copy=False), raw_count
