"""Validation helpers for stack-aligned rollout modalities."""

import numpy as np


def validate_aligned_rollouts(
        trajectories,images,event_images,image_modality="kronecker_delta") -> None:
    if not (len(trajectories) == len(images) == len(event_images)):
        raise ValueError(
            f"Trajectory, RGB, and {image_modality} rollout counts do not match.")

    for trajectory, image_data, event_data in zip(
        trajectories, images, event_images
    ):
        rollout_id = trajectory["rollout_id"]
        if (
            image_data["rollout_id"] != rollout_id
            or event_data["rollout_id"] != rollout_id
        ):
            raise ValueError(
                f"Trajectory, RGB, and {image_modality} rollout IDs do not match.")

        rgb = image_data["rgb"]
        event_frames = event_data[image_modality]
        ndata = trajectory["Ndata"]
        if rgb.shape[0] != ndata or event_frames.shape[0] != ndata:
            raise ValueError(f"Frame count mismatch for rollout {rollout_id}.")
        if event_frames.dtype != np.uint8 or event_frames.ndim != 3:
            raise ValueError("Event arrays must have uint8 (N,H,W) format.")
        if rgb.shape[1:3] != event_frames.shape[1:3]:
            raise ValueError(f"Image dimensions do not match for rollout {rollout_id}.")
