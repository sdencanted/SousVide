"""Utilities for producing aligned v2e event representations.

The voxel-grid voting rule is derived from ``rpg_e2vid``'s GPL-3.0
``events_to_voxel_grid_pytorch`` implementation:
https://github.com/cedric-scheerlinck/rpg_e2vid/blob/master/utils/inference_utils.py#L480
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from sousvide.synthesize.event_surfaces import (
    EVENT_SURFACE_MODALITIES,EventModality,create_event_surface,
    resolve_event_surface_options,validate_event_modalities,
)


MIN_EVENTS_PER_IMAGE = 10
VOXEL_GRID_BINS = 5
VOXEL_GRID_MODALITIES = (
    "event_voxel_grid", "event_voxel_grid_polarity",
)


def _voxel_event_components(
        events: np.ndarray | None, height: int, width: int):
    """Return finite, in-bounds event components without modifying the input."""
    if height <= 0 or width <= 0:
        raise ValueError("Voxel-grid dimensions must be positive.")
    if events is None or len(events) == 0:
        return tuple(np.empty(0,dtype=dtype) for dtype in (
            np.float64,np.int64,np.int64,np.float32))

    event_array = np.asarray(events)
    if event_array.ndim != 2 or event_array.shape[1] < 4:
        raise ValueError(
            "Events must have shape (N, >=4) with columns [t, x, y, polarity].")

    values = event_array[:,:4]
    finite = np.all(np.isfinite(values),axis=1)
    values = values[finite]
    if not len(values):
        return tuple(np.empty(0,dtype=dtype) for dtype in (
            np.float64,np.int64,np.int64,np.float32))

    timestamps = values[:,0].astype(np.float64,copy=True)
    xs = values[:,1].astype(np.int64,copy=False)
    ys = values[:,2].astype(np.int64,copy=False)
    polarities = values[:,3].astype(np.float32,copy=True)
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return (
        timestamps[in_bounds],xs[in_bounds],ys[in_bounds],
        polarities[in_bounds])


def _accumulate_voxel_grid(
        timestamps:np.ndarray,xs:np.ndarray,ys:np.ndarray,
        weights:np.ndarray,height:int,width:int,
        num_bins:int=VOXEL_GRID_BINS) -> np.ndarray:
    grid = np.zeros((num_bins,height,width),dtype=np.float32)
    if not len(timestamps):
        return grid

    delta_t = timestamps[-1]-timestamps[0]
    if delta_t == 0:
        normalized_timestamps = np.zeros_like(timestamps)
    else:
        normalized_timestamps = (
            (num_bins-1)*(timestamps-timestamps[0])/delta_t)

    left_bins = np.floor(normalized_timestamps).astype(np.int64)
    fractions = (normalized_timestamps-left_bins).astype(np.float32)
    left_weights = weights*(1.0-fractions)
    right_weights = weights*fractions

    valid = (left_bins >= 0) & (left_bins < num_bins)
    np.add.at(
        grid,(left_bins[valid],ys[valid],xs[valid]),left_weights[valid])
    right_bins = left_bins+1
    valid = (right_bins >= 0) & (right_bins < num_bins)
    np.add.at(
        grid,(right_bins[valid],ys[valid],xs[valid]),right_weights[valid])
    return grid


def events_to_voxel_grid(
        events:np.ndarray|None,height:int,width:int,
        num_bins:int=VOXEL_GRID_BINS) -> np.ndarray:
    """Build a signed ``(num_bins,H,W)`` temporal voxel grid."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    timestamps,xs,ys,polarities = _voxel_event_components(
        events,height,width)
    weights = np.where(polarities > 0,1.0,-1.0).astype(np.float32)
    return _accumulate_voxel_grid(
        timestamps,xs,ys,weights,height,width,num_bins)


def events_to_polarity_voxel_grid(
        events:np.ndarray|None,height:int,width:int,
        num_bins:int=VOXEL_GRID_BINS) -> np.ndarray:
    """Build positive then negative temporal grids with ``2*num_bins`` channels."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    timestamps,xs,ys,polarities = _voxel_event_components(
        events,height,width)
    positive = _accumulate_voxel_grid(
        timestamps,xs,ys,(polarities > 0).astype(np.float32),
        height,width,num_bins)
    negative = _accumulate_voxel_grid(
        timestamps,xs,ys,(polarities <= 0).astype(np.float32),
        height,width,num_bins)
    return np.concatenate((positive,negative),axis=0)


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
        image_output_paths: dict[EventModality,str] | None = None,
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
        if image_output_paths is not None and not retain_images:
            raise ValueError(
                "image_output_paths cannot be used when retain_images=False.")
        if (image_output_paths is not None
                and set(image_output_paths) != set(self.event_modalities)):
            raise ValueError(
                "Image output paths must match the selected event modalities.")
        self.image_output_paths = (
            {} if image_output_paths is None else dict(image_output_paths))
        self.output_modality = self.event_modalities[0]
        self.emulator = None
        self.height = None
        self.width = None
        self.window_events: list[np.ndarray] = []
        self.surfaces = {}
        self.images_by_modality = {
            modality:[] for modality in self.event_modalities}
        self._image_memmaps: dict[EventModality,np.memmap] = {}
        # Preserve the legacy attribute for callers that inspect it directly.
        self.images = self.images_by_modality[self.output_modality]
        self._closed_images = None
        self._retain_window_events = any(
            modality == "kronecker_delta" or modality in VOXEL_GRID_MODALITIES
            for modality in self.event_modalities)
        self.window_count = 0
        self.closed = False

    def _start(self, rgb: np.ndarray) -> None:
        self.height, self.width = rgb.shape[:2]
        for modality,path in self.image_output_paths.items():
            output_folder = os.path.dirname(path) or "."
            os.makedirs(output_folder,exist_ok=True)
            if modality == "event_voxel_grid":
                shape = (self.expected_windows,VOXEL_GRID_BINS,
                         self.height,self.width)
                dtype = np.float32
            elif modality == "event_voxel_grid_polarity":
                shape = (self.expected_windows,2*VOXEL_GRID_BINS,
                         self.height,self.width)
                dtype = np.float32
            else:
                shape = (self.expected_windows,self.height,self.width)
                dtype = np.uint8
            self._image_memmaps[modality] = np.lib.format.open_memmap(
                path,mode="w+",dtype=dtype,shape=shape)
        if self._image_memmaps:
            self.images = self._image_memmaps[self.output_modality]
        self.surfaces = {
            modality:create_event_surface(
                modality,self.height,self.width,
                self.event_surface_options[modality])
            for modality in self.event_modalities
            if modality in EVENT_SURFACE_MODALITIES
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
            if self._retain_window_events:
                self.window_events.append(events)
            for surface in self.surfaces.values():
                surface.update(events)

        # Closing after accumulation includes events exactly on the upper bound.
        image = None
        if close_window:
            boundary_images = {}
            combined = (
                np.concatenate(self.window_events,axis=0)
                if self.window_events else None)
            if "kronecker_delta" in self.event_modalities:
                boundary_images["kronecker_delta"] = events_to_kronecker(
                    combined,self.height,self.width)
            if "event_voxel_grid" in self.event_modalities:
                boundary_images["event_voxel_grid"] = events_to_voxel_grid(
                    combined,self.height,self.width)
            if "event_voxel_grid_polarity" in self.event_modalities:
                boundary_images["event_voxel_grid_polarity"] = (
                    events_to_polarity_voxel_grid(
                        combined,self.height,self.width))
            for modality,surface in self.surfaces.items():
                boundary_images[modality] = surface.snapshot()
            image = boundary_images[self.output_modality]
            if self.retain_images:
                if self.window_count >= self.expected_windows:
                    raise RuntimeError(
                        "Generated more event windows than expected.")
                for modality,boundary_image in boundary_images.items():
                    if self._image_memmaps:
                        self._image_memmaps[modality][self.window_count] = (
                            boundary_image)
                    else:
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
            if self._image_memmaps:
                for images in self._image_memmaps.values():
                    images.flush()
                self._closed_images = dict(self._image_memmaps)
            else:
                self._closed_images = {
                    modality:np.stack(images,axis=0).astype(
                        np.float32
                        if modality in VOXEL_GRID_MODALITIES else np.uint8,
                        copy=False)
                    for modality,images in self.images_by_modality.items()
                }
        return self._closed_images

    def abort(self) -> None:
        """Close v2e and remove a rollout that was rejected or failed."""
        if not self.closed and self.emulator is not None:
            self.emulator.cleanup()
        self.closed = True
        for images in self._image_memmaps.values():
            images.flush()
            images._mmap.close()
        self._image_memmaps.clear()
        self._closed_images = None
        for path in self.image_output_paths.values():
            if os.path.isfile(path):
                os.unlink(path)
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
