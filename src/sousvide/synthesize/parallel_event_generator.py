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
    """Run v2e for one rollout in an isolated CPU worker process."""
    import torch

    # A worker represents one CPU lane; prevent each torch instance from
    # internally claiming every core and oversubscribing the host.
    torch.set_num_threads(1)
    recorder = V2ERolloutRecorder(
        h5_path,expected_windows,device="cpu",
        event_modalities=event_modalities,
        event_surface_options=event_surface_options)
    try:
        frames = np.load(frame_path,mmap_mode="r",allow_pickle=False)
        if not (len(frames) == len(timestamps) == len(close_windows)):
            raise ValueError("Buffered frame timestamps and window flags are misaligned.")
        for frame,timestamp,close_window in zip(frames,timestamps,close_windows):
            recorder.process_gray_frame(frame,timestamp,close_window)
        event_images = recorder.close_all()
        if event_output_paths is None:
            # Preserve the legacy single-Kronecker worker contract.
            np.save(
                kronecker_path,event_images["kronecker_delta"],
                allow_pickle=False)
        else:
            if set(event_output_paths) != set(event_images):
                raise ValueError(
                    "Event output paths do not match generated modalities.")
            for modality,output_path in event_output_paths.items():
                np.save(output_path,event_images[modality],allow_pickle=False)
    except Exception:
        recorder.abort()
        cleanup_paths = (
            [kronecker_path] if event_output_paths is None
            else list(event_output_paths.values()))
        for output_path in cleanup_paths:
            if os.path.isfile(output_path):
                os.unlink(output_path)
        raise
    finally:
        if os.path.isfile(frame_path):
            os.unlink(frame_path)
    if event_output_paths is None:
        return kronecker_path,h5_path
    return event_output_paths,h5_path
