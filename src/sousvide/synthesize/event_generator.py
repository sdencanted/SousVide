"""Utilities for producing aligned v2e and Kronecker-delta rollout data."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from sousvide.synthesize.event_surfaces import (
    EventModality,create_event_surface,resolve_event_surface_options,
    validate_event_modalities,
)


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
    scale = 255.0 / max(1.0, float(np.percentile(nonzero, 98)) - event_min)
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
    """Stream rendered frames through v2e and collect aligned event images."""

    def __init__(
        self,
        h5_path: str | None,
        expected_windows: int,
        emulator_factory: Callable[..., object] | None = None,
        device: str | None = None,
        retain_images: bool = True,
        event_modalities: tuple[EventModality, ...] | list[EventModality] | None = None,
        event_surface_options: dict[str,dict] | None = None,
    ) -> None:
        self.h5_path = h5_path
        self.expected_windows = expected_windows
        self.emulator_factory = emulator_factory
        self.device = device
        self.retain_images = retain_images
        self.event_modalities = validate_event_modalities(
            ("kronecker_delta",)
            if event_modalities is None else event_modalities)
        self.event_surface_options = resolve_event_surface_options(
            self.event_modalities,event_surface_options)
        self.output_modality = self.event_modalities[0]
        self.emulator = None
        self.height = None
        self.width = None
        self.window_events: list[np.ndarray] = []
        self.surfaces = {}
        self.images_by_modality = {
            modality:[] for modality in self.event_modalities}
        # Preserve the legacy attribute for callers that inspect it directly.
        self.images = self.images_by_modality[self.output_modality]
        self._closed_images = None
        self.window_count = 0
        self.closed = False

    def _start(self, rgb: np.ndarray) -> None:
        self.height, self.width = rgb.shape[:2]
        self.surfaces = {
            modality:create_event_surface(
                modality,self.height,self.width,
                self.event_surface_options[modality])
            for modality in self.event_modalities
            if modality != "kronecker_delta"
        }
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
            events = np.asarray(events)
            if "kronecker_delta" in self.event_modalities:
                self.window_events.append(events)
            for surface in self.surfaces.values():
                surface.update(events)

        # Closing after accumulation includes events exactly on the upper bound.
        image = None
        if close_window:
            boundary_images = {}
            if "kronecker_delta" in self.event_modalities:
                combined = (
                    np.concatenate(self.window_events, axis=0)
                    if self.window_events
                    else None
                )
                boundary_images["kronecker_delta"] = events_to_kronecker(
                    combined,self.height,self.width)
            for modality,surface in self.surfaces.items():
                boundary_images[modality] = surface.snapshot()
            image = boundary_images[self.output_modality]
            if self.retain_images:
                for modality,boundary_image in boundary_images.items():
                    self.images_by_modality[modality].append(boundary_image)
            self.window_count += 1
            self.window_events.clear()
        return image

    def close(self) -> np.ndarray | None:
        images = self.close_all()
        if images is None:
            return None
        return images[self.output_modality]

    def close_all(self) -> dict[EventModality,np.ndarray] | None:
        """Close v2e and return every retained, aligned event representation."""
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
        if self._closed_images is None:
            self._closed_images = {
                modality:np.stack(images,axis=0).astype(np.uint8,copy=False)
                for modality,images in self.images_by_modality.items()
            }
        return self._closed_images

    def abort(self) -> None:
        """Close v2e and remove a rollout that was rejected or failed."""
        if not self.closed and self.emulator is not None:
            self.emulator.cleanup()
        self.closed = True
        if self.h5_path is not None and os.path.isfile(self.h5_path):
            os.unlink(self.h5_path)


class OnlineEventImageGenerator(V2ERolloutRecorder):
    """Generate one selected event-image modality without file persistence."""

    def __init__(self,expected_windows:int,device:str|None=None,
                 image_modality:EventModality="kronecker_delta",
                 event_surface_options:dict[str,dict]|None=None) -> None:
        super().__init__(
            h5_path=None,expected_windows=expected_windows,
            device=device,retain_images=False,
            event_modalities=(image_modality,),
            event_surface_options=event_surface_options)
