"""Disk-backed CPU workers for parallel per-rollout v2e processing."""

from __future__ import annotations

import os

import numpy as np

from sousvide.synthesize.event_generator import V2ERolloutRecorder, _rgb_to_gray


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
        event_modalities=event_modalities,
        event_surface_options=event_surface_options,
        image_output_paths=output_paths)
    try:
        frames = np.load(frame_path,mmap_mode="r",allow_pickle=False)
        if not (len(frames) == len(timestamps) == len(close_windows)):
            raise ValueError("Buffered frame timestamps and window flags are misaligned.")
        for frame,timestamp,close_window in zip(frames,timestamps,close_windows):
            recorder.process_gray_frame(frame,timestamp,close_window)
        event_images = recorder.close_all()
        if set(output_paths) != set(event_images):
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
