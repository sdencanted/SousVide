"""NumPy event-surface accumulators derived from robotology/event-driven.

The EROS and TOS update rules are Python adaptations of the BSD-3-Clause
``surface.h`` implementation from the Event Driven Perception for Robotics
project:
https://github.com/robotology/event-driven/blob/main/ev2/event-driven/algs/surface.h

Copyright (c) 2021, Event Driven Perception for Robotics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np


EventModality = Literal[
    "kronecker_delta", "event_bin", "event_eros", "event_tos"
]

EVENT_MODALITIES: tuple[EventModality, ...] = (
    "kronecker_delta",
    "event_bin",
    "event_eros",
    "event_tos",
)

DEFAULT_EVENT_SURFACE_OPTIONS: dict[EventModality, dict[str, float | int]] = {
    "kronecker_delta": {},
    "event_bin": {},
    "event_eros": {"kernel_size": 7, "decay": 0.3},
    "event_tos": {"kernel_size": 5, "parameter": 2.0},
}


def validate_event_modalities(modalities) -> tuple[EventModality, ...]:
    """Validate, deduplicate, and preserve the order of event modalities."""
    if isinstance(modalities, str):
        modalities = (modalities,)
    try:
        requested = tuple(modalities)
    except TypeError as error:
        raise TypeError("event_modalities must be an iterable of strings.") from error
    if not requested:
        raise ValueError("event_modalities must contain at least one modality.")

    unknown = [item for item in requested if item not in EVENT_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown event modalities {unknown}; expected one of {EVENT_MODALITIES}."
        )
    return tuple(dict.fromkeys(requested))


def resolve_event_surface_options(
    modalities,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[EventModality, dict[str, float | int]]:
    """Return validated, fully populated parameters for each modality."""
    modalities = validate_event_modalities(modalities)
    supplied = {} if options is None else dict(options)
    unknown = set(supplied) - set(modalities)
    if unknown:
        raise ValueError(
            "event_surface_options contains unrequested modalities: "
            f"{sorted(unknown)}."
        )

    resolved: dict[EventModality, dict[str, float | int]] = {}
    for modality in modalities:
        defaults = DEFAULT_EVENT_SURFACE_OPTIONS[modality].copy()
        overrides = dict(supplied.get(modality, {}))
        unexpected = set(overrides) - set(defaults)
        if unexpected:
            raise ValueError(
                f"Unsupported {modality} options: {sorted(unexpected)}."
            )
        defaults.update(overrides)
        _validate_surface_options(modality, defaults)
        resolved[modality] = defaults
    return resolved


def _validate_surface_options(
    modality: EventModality, options: Mapping[str, float | int]
) -> None:
    if modality in ("kronecker_delta", "event_bin"):
        if options:
            raise ValueError(f"{modality} does not accept surface options.")
        return

    kernel_size = options.get("kernel_size")
    if (isinstance(kernel_size, bool) or not isinstance(kernel_size, int)
            or kernel_size <= 0 or kernel_size % 2 == 0):
        raise ValueError("kernel_size must be a positive odd integer.")

    if modality == "event_eros":
        decay = options.get("decay")
        if (isinstance(decay, bool) or not isinstance(decay, (int, float))
                or not 0.0 <= float(decay) <= 1.0):
            raise ValueError("EROS decay must be in the closed interval [0, 1].")
    elif modality == "event_tos":
        parameter = options.get("parameter")
        if (isinstance(parameter, bool)
                or not isinstance(parameter, (int, float))
                or float(parameter) <= 0.0
                or kernel_size * float(parameter) > 255.0):
            raise ValueError(
                "TOS parameter must be positive and kernel_size * parameter "
                "must not exceed 255."
            )


def _valid_xy(events: np.ndarray | None, height: int, width: int):
    if events is None or len(events) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError(
            "Events must have shape (N, >=3) with columns [t, x, y, ...]."
        )
    xs = events[:, 1].astype(np.int64, copy=False)
    ys = events[:, 2].astype(np.int64, copy=False)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return xs[valid], ys[valid]


class EventSurface:
    """Common interface for stateful event-to-image accumulators."""

    reset_after_snapshot = False

    def __init__(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("Event surface dimensions must be positive.")
        self.height = int(height)
        self.width = int(width)

    def update(self, events: np.ndarray | None) -> None:
        raise NotImplementedError

    def snapshot(self) -> np.ndarray:
        raise NotImplementedError


class BinaryEventSurface(EventSurface):
    """Binary event-presence image reset at every snapshot."""

    reset_after_snapshot = True

    def __init__(self, height: int, width: int) -> None:
        super().__init__(height, width)
        self.surface = np.zeros((height, width), dtype=np.uint8)

    def update(self, events: np.ndarray | None) -> None:
        xs, ys = _valid_xy(events, self.height, self.width)
        self.surface[ys, xs] = 255

    def snapshot(self) -> np.ndarray:
        image = self.surface.copy()
        self.surface.fill(0)
        return image


class EROSEventSurface(EventSurface):
    """Exponentially Reduced Ordinal Surface with local event decay."""

    def __init__(
        self, height: int, width: int, kernel_size: int = 7, decay: float = 0.3
    ) -> None:
        super().__init__(height, width)
        options = {"kernel_size": kernel_size, "decay": decay}
        _validate_surface_options("event_eros", options)
        self.kernel_size = kernel_size
        self.half_kernel = kernel_size // 2
        self.decay = float(decay)
        self.event_decay = self.decay ** (1.0 / kernel_size)
        self.surface = np.zeros(
            (height + 2 * self.half_kernel, width + 2 * self.half_kernel),
            dtype=np.float32,
        )

    def update(self, events: np.ndarray | None) -> None:
        xs, ys = _valid_xy(events, self.height, self.width)
        for x, y in zip(xs, ys):
            self.surface[
                y:y + self.kernel_size, x:x + self.kernel_size
            ] *= self.event_decay
            self.surface[y + self.half_kernel, x + self.half_kernel] = 1.0

    def snapshot(self) -> np.ndarray:
        h = self.half_kernel
        visible = self.surface[h:h + self.height, h:h + self.width]
        return np.clip(visible * 255.0, 0.0, 255.0).astype(np.uint8)


class TOSEventSurface(EventSurface):
    """Threshold-Ordinal Surface retaining locally recent event order."""

    def __init__(
        self, height: int, width: int, kernel_size: int = 5,
        parameter: float = 2.0,
    ) -> None:
        super().__init__(height, width)
        options = {"kernel_size": kernel_size, "parameter": parameter}
        _validate_surface_options("event_tos", options)
        self.kernel_size = kernel_size
        self.half_kernel = kernel_size // 2
        self.parameter = float(parameter)
        self.threshold = 255.0 - kernel_size * self.parameter
        self.surface = np.zeros(
            (height + 2 * self.half_kernel, width + 2 * self.half_kernel),
            dtype=np.float32,
        )

    def update(self, events: np.ndarray | None) -> None:
        xs, ys = _valid_xy(events, self.height, self.width)
        for x, y in zip(xs, ys):
            region = self.surface[
                y:y + self.kernel_size, x:x + self.kernel_size
            ]
            retained = region >= self.threshold
            region[~retained] = 0.0
            region[retained] -= 1.0
            self.surface[y + self.half_kernel, x + self.half_kernel] = 255.0

    def snapshot(self) -> np.ndarray:
        h = self.half_kernel
        visible = self.surface[h:h + self.height, h:h + self.width]
        return np.clip(visible, 0.0, 255.0).astype(np.uint8)


def create_event_surface(
    modality: EventModality,
    height: int,
    width: int,
    options: Mapping[str, float | int] | None = None,
) -> EventSurface:
    """Construct one non-Kronecker event surface with resolved options."""
    if modality == "kronecker_delta":
        raise ValueError("Kronecker accumulation is handled by its count converter.")
    resolved = resolve_event_surface_options(
        (modality,), {modality: options or {}}
    )[modality]
    if modality == "event_bin":
        return BinaryEventSurface(height, width)
    if modality == "event_eros":
        return EROSEventSurface(height, width, **resolved)
    if modality == "event_tos":
        return TOSEventSurface(height, width, **resolved)
    raise ValueError(f"Unsupported event modality: {modality}.")
