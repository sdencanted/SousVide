"""Validation helpers for stack-aligned rollout modalities."""

import numpy as np


def validate_aligned_rollouts(trajectories, images, kronecker_images) -> None:
    if not (len(trajectories) == len(images) == len(kronecker_images)):
        raise ValueError("Trajectory, RGB, and Kronecker rollout counts do not match.")

    for trajectory, image_data, kronecker_data in zip(
        trajectories, images, kronecker_images
    ):
        rollout_id = trajectory["rollout_id"]
        if (
            image_data["rollout_id"] != rollout_id
            or kronecker_data["rollout_id"] != rollout_id
        ):
            raise ValueError("Trajectory, RGB, and Kronecker rollout IDs do not match.")

        rgb = image_data["rgb"]
        delta = kronecker_data["kronecker_delta"]
        ndata = trajectory["Ndata"]
        if rgb.shape[0] != ndata or delta.shape[0] != ndata:
            raise ValueError(f"Frame count mismatch for rollout {rollout_id}.")
        if delta.dtype != np.uint8 or delta.ndim != 3:
            raise ValueError("Kronecker arrays must have uint8 (N,H,W) format.")
        if rgb.shape[1:3] != delta.shape[1:3]:
            raise ValueError(f"Image dimensions do not match for rollout {rollout_id}.")
