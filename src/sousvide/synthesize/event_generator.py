"""Utilities for producing aligned v2e and Kronecker-delta rollout data."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np


MIN_EVENTS_PER_IMAGE = 10


def events_to_kronecker(
    events: np.ndarray | None,
    height: int,
    width: int,
    min_events: int = MIN_EVENTS_PER_IMAGE,
) -> np.ndarray:
    """Convert ``[t, x, y, polarity]`` events to the reference count image."""
    image = np.zeros((height, width), dtype=np.uint8)
    if events is None or len(events) < min_events:
        return image

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError("Events must have shape (N, >=3) with columns [t, x, y, ...].")

    xs = events[:, 1].astype(np.int64, copy=False)
    ys = events[:, 2].astype(np.int64, copy=False)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if np.count_nonzero(valid) < min_events:
        return image

    counts = np.zeros((height, width), dtype=np.uint32)
    np.add.at(counts, (ys[valid], xs[valid]), 1)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return image

    event_min = max(0.0, float(nonzero.min()) - 1.0)
    scale = 255.0 / max(1.0, float(np.percentile(nonzero, 90)) - event_min)
    return np.clip(counts.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert FiGS RGB uint8 output to the grayscale input expected by v2e."""
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {rgb.shape}.")
    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


class V2ERolloutRecorder:
    """Stream rendered frames through v2e and collect fixed-window images."""

    def __init__(
        self,
        h5_path: str | None,
        expected_windows: int,
        emulator_factory: Callable[..., object] | None = None,
        device: str | None = None,
        retain_images: bool = True,
    ) -> None:
        self.h5_path = h5_path
        self.expected_windows = expected_windows
        self.emulator_factory = emulator_factory
        self.device = device
        self.retain_images = retain_images
        self.emulator = None
        self.height = None
        self.width = None
        self.window_events: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self.window_count = 0
        self.closed = False

    def _start(self, rgb: np.ndarray) -> None:
        self.height, self.width = rgb.shape[:2]
        if self.h5_path is None:
            output_folder,dvs_h5 = None,None
        else:
            output_folder = os.path.dirname(self.h5_path) or "."
            os.makedirs(output_folder, exist_ok=True)
            dvs_h5 = os.path.basename(self.h5_path)

        if self.emulator_factory is None:
            import torch
            from v2ecore.emulator import EventEmulator

            self.emulator_factory = EventEmulator
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device or "cpu"

        self.emulator = self.emulator_factory(
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz=300.0,
            leak_rate_hz=0.01,
            shot_noise_rate_hz=0.001,
            refractory_period_s=0.0005,
            seed=0,
            output_folder=output_folder,
            dvs_h5=dvs_h5,
            output_width=self.width,
            output_height=self.height,
            device=device,
        )

    def process_frame(self, rgb: np.ndarray, timestamp: float,
                      close_window: bool) -> np.ndarray | None:
        """Process a frame; close the current window after its returned events."""
        return self.process_gray_frame(_rgb_to_gray(rgb),timestamp,close_window)

    def process_gray_frame(self, gray: np.ndarray, timestamp: float,
                           close_window: bool) -> np.ndarray | None:
        """Process a grayscale frame and optionally close its event window."""
        if self.closed:
            raise RuntimeError("Cannot process frames after the recorder has been closed.")
        if self.emulator is None:
            self._start(gray)

        events = self.emulator.generate_events(gray, float(timestamp))
        if events is not None and len(events):
            self.window_events.append(np.asarray(events))

        # Closing after accumulation includes events exactly on the upper bound.
        image = None
        if close_window:
            combined = (
                np.concatenate(self.window_events, axis=0)
                if self.window_events
                else None
            )
            image = events_to_kronecker(combined, self.height, self.width)
            if self.retain_images:
                self.images.append(image)
            self.window_count += 1
            self.window_events.clear()
        return image

    def close(self) -> np.ndarray | None:
        if not self.closed:
            if self.emulator is not None:
                self.emulator.cleanup()
            self.closed = True

        if self.height is None or self.width is None:
            raise RuntimeError("No frames were supplied to the v2e recorder.")
        if self.window_count != self.expected_windows:
            raise RuntimeError(
                f"Expected {self.expected_windows} event windows, generated {self.window_count}."
            )
        if not self.retain_images:
            return None
        return np.stack(self.images, axis=0).astype(np.uint8, copy=False)

    def abort(self) -> None:
        """Close v2e and remove a rollout that was rejected or failed."""
        if not self.closed and self.emulator is not None:
            self.emulator.cleanup()
        self.closed = True
        if self.h5_path is not None and os.path.isfile(self.h5_path):
            os.unlink(self.h5_path)


class OnlineEventImageGenerator(V2ERolloutRecorder):
    """Generate aligned Kronecker images in memory without event persistence."""

    def __init__(self,expected_windows:int,device:str|None=None) -> None:
        super().__init__(
            h5_path=None,expected_windows=expected_windows,
            device=device,retain_images=False)
