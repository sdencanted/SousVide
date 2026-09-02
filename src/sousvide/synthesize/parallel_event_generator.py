"""Disk-backed CPU workers for parallel per-rollout v2e processing."""

from __future__ import annotations

import os

import h5py
import numpy as np

from sousvide.synthesize.event_generator import (
    V2ERolloutRecorder,
    _rgb_to_gray,
    events_to_bilinear,
    events_to_kronecker,
    events_to_polarity_voxel_grid,
    events_to_pseudo_gaussian,
    events_to_voxel_grid,
)
from sousvide.synthesize.event_surfaces import (
    EVENT_SURFACE_MODALITIES,
    create_event_surface,
    resolve_event_surface_options,
    validate_event_modalities,
)
from sousvide.synthesize.event_cloud import (
    events_to_event_cloud,resolve_event_cloud_options)


class EventFrameBuffer:
    """Collect simulation-rate frames without running v2e in the simulator process."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.close_windows: list[bool] = []

    def process_frame(self, rgb: np.ndarray, timestamp: float,
                      close_window: bool) -> None:
        self.frames.append(_rgb_to_gray(rgb))
        self.timestamps.append(float(timestamp))
        self.close_windows.append(bool(close_window))

    def save(self, frame_path: str) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Write frames to a mmap-compatible file and release their RAM."""
        if not self.frames:
            raise RuntimeError("No event-source frames were captured.")
        np.save(frame_path,np.stack(self.frames,axis=0),allow_pickle=False)
        timestamps = tuple(self.timestamps)
        close_windows = tuple(self.close_windows)
        self.frames.clear()
        self.timestamps.clear()
        self.close_windows.clear()
        return timestamps,close_windows


def process_buffered_rollout(
    frame_path: str,
    timestamps: tuple[float, ...],
    close_windows: tuple[bool, ...],
    h5_path: str,
    kronecker_path: str,
    expected_windows: int,
    event_modalities=None,
    event_surface_options=None,
    event_output_paths:dict[str,str]|None=None,
    retain_images:bool=True,
):
    """Run v2e and write event stacks directly to disk in a CPU worker."""
    import torch

    # A worker represents one CPU lane; prevent each torch instance from
    # internally claiming every core and oversubscribing the host.
    torch.set_num_threads(1)
    output_paths = (
        {"kronecker_delta":kronecker_path}
        if event_output_paths is None else event_output_paths)
    recorder = V2ERolloutRecorder(
        h5_path,expected_windows,device="cpu",
        retain_images=retain_images,
        event_modalities=event_modalities,
        event_surface_options=event_surface_options,
        image_output_paths=output_paths if retain_images else None)
    try:
        frames = np.load(frame_path,mmap_mode="r",allow_pickle=False)
        if not (len(frames) == len(timestamps) == len(close_windows)):
            raise ValueError("Buffered frame timestamps and window flags are misaligned.")
        for frame,timestamp,close_window in zip(frames,timestamps,close_windows):
            recorder.process_gray_frame(frame,timestamp,close_window)
        event_images = recorder.close_all()
        if retain_images and set(output_paths) != set(event_images):
            raise ValueError(
                "Event output paths do not match generated modalities.")
    except Exception:
        recorder.abort()
        for output_path in output_paths.values():
            if os.path.isfile(output_path):
                os.unlink(output_path)
        raise
    finally:
        if os.path.isfile(frame_path):
            os.unlink(frame_path)
    if event_output_paths is None:
        return kronecker_path,h5_path
    return event_output_paths,h5_path


def process_event_stream_rollout(
        h5_path:str,window_end_times:tuple[float,...],height:int,width:int,
        event_modalities,event_surface_options,event_output_paths:dict[str,str],
        event_cloud_options=None,stream_id:str|None=None):
    """Generate aligned event representations from one persisted v2e stream."""
    requested = tuple(event_modalities)
    has_event_cloud = "event_cloud" in requested
    modalities = validate_event_modalities(
        tuple(item for item in requested if item != "event_cloud")) \
        if any(item != "event_cloud" for item in requested) else ()
    options = resolve_event_surface_options(
        modalities,event_surface_options) if modalities else {}
    cloud_options = (
        resolve_event_cloud_options(event_cloud_options)
        if has_event_cloud else None)
    if set(event_output_paths) != set(requested):
        raise ValueError(
            "Event output paths must match the selected event modalities.")
    if height <= 0 or width <= 0:
        raise ValueError("Event-image dimensions must be positive.")
    if not window_end_times:
        raise ValueError("At least one event window is required.")
    boundaries_us = np.rint(
        np.asarray(window_end_times,dtype=np.float64)*1e6).astype(np.int64)
    if np.any(np.diff(boundaries_us) <= 0):
        raise ValueError("Event window end times must be strictly increasing.")

    with h5py.File(h5_path,"r") as event_file:
        if "events" not in event_file:
            raise ValueError(f"Event stream has no 'events' dataset: {h5_path}")
        stored_events = np.asarray(event_file["events"])
    if stored_events.ndim != 2 or stored_events.shape[1] < 4:
        raise ValueError(
            f"Event stream must have shape (N, >=4): {h5_path}")
    if len(stored_events) and np.any(np.diff(stored_events[:,0]) < 0):
        raise ValueError(f"Event timestamps are not monotonic: {h5_path}")

    output_arrays = {}
    for modality,path in event_output_paths.items():
        output_folder = os.path.dirname(path) or "."
        os.makedirs(output_folder,exist_ok=True)
        if modality == "event_cloud":
            shape = (
                len(boundaries_us),cloud_options["num_points"],4)
            dtype = np.float32
        elif modality == "event_voxel_grid":
            shape = (len(boundaries_us),5,height,width)
            dtype = np.float32
        elif modality == "event_voxel_grid_polarity":
            shape = (len(boundaries_us),10,height,width)
            dtype = np.float32
        else:
            shape = (len(boundaries_us),height,width)
            dtype = np.uint8
        output_arrays[modality] = np.lib.format.open_memmap(
            path,mode="w+",dtype=dtype,shape=shape)

    surfaces = {
        modality:create_event_surface(
            modality,height,width,options[modality])
        for modality in modalities if modality in EVENT_SURFACE_MODALITIES}
    start_index = 0
    raw_event_counts = np.zeros(len(boundaries_us),dtype=np.int64)
    resolved_stream_id = (
        os.path.splitext(os.path.basename(h5_path))[0]
        if stream_id is None else str(stream_id))
    try:
        timestamps = stored_events[:,0]
        for window_index,boundary_us in enumerate(boundaries_us):
            end_index = int(np.searchsorted(
                timestamps,boundary_us,side="right"))
            events = stored_events[start_index:end_index].astype(
                np.float64,copy=True)
            if len(events):
                events[:,0] /= 1e6
                events[:,3] = np.where(events[:,3] > 0,1.0,-1.0)
            for surface in surfaces.values():
                surface.update(events)
            raw_event_counts[window_index] = len(events)
            for modality in requested:
                if modality == "event_cloud":
                    image,_ = events_to_event_cloud(
                        events,width,height,stream_id=resolved_stream_id,
                        window_index=window_index,options=cloud_options,
                        polarity_signed=True)
                elif modality == "kronecker_delta":
                    image = events_to_kronecker(events,height,width)
                elif modality == "event_pseudo_gaussian":
                    image = events_to_pseudo_gaussian(
                        events,height,width,**options[modality])
                elif modality == "event_bilinear":
                    image = events_to_bilinear(
                        events,height,width,**options[modality])
                elif modality == "event_voxel_grid":
                    image = events_to_voxel_grid(events,height,width)
                elif modality == "event_voxel_grid_polarity":
                    image = events_to_polarity_voxel_grid(
                        events,height,width)
                else:
                    image = surfaces[modality].snapshot()
                output_arrays[modality][window_index] = image
            start_index = end_index
        if start_index != len(stored_events):
            raise ValueError(
                f"Event stream contains events after the final window: {h5_path}")
        for images in output_arrays.values():
            images.flush()
    except Exception:
        for images in output_arrays.values():
            images.flush()
            images._mmap.close()
        for path in event_output_paths.values():
            if os.path.isfile(path):
                os.unlink(path)
        raise
    return event_output_paths,raw_event_counts
