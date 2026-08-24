"""Shared helpers for RGB and Kronecker-delta observation inputs."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

import sousvide.synthesize.data_compress_helper as dch


ImageModality = Literal[
    "rgb", "kronecker_delta", "event_bin", "event_eros", "event_tos"
]

IMAGE_MODALITIES: tuple[ImageModality, ...] = (
    "rgb", "kronecker_delta", "event_bin", "event_eros", "event_tos"
)
EVENT_IMAGE_MODALITIES = IMAGE_MODALITIES[1:]

_MODALITY_STORAGE = {
    "rgb": ("images", "images"),
    "kronecker_delta": ("kronecker", "kronecker"),
    "event_bin": ("event_bin", "event_bin"),
    "event_eros": ("event_eros", "event_eros"),
    "event_tos": ("event_tos", "event_tos"),
}


def kronecker_to_three_channels(image: np.ndarray) -> np.ndarray:
    """Repeat one grayscale Kronecker image for existing RGB backbones."""
    if image.ndim != 2:
        raise ValueError("A Kronecker image must have shape (H,W).")
    return np.repeat(image[...,None],3,axis=-1)


def event_image_to_three_channels(image: np.ndarray) -> np.ndarray:
    """Repeat one grayscale event image for the existing RGB backbones."""
    if image.ndim != 2:
        raise ValueError("An event image must have shape (H,W).")
    return np.repeat(image[...,None],3,axis=-1)


def is_event_modality(image_modality: str) -> bool:
    return image_modality in EVENT_IMAGE_MODALITIES


def modality_storage(image_modality: ImageModality) -> tuple[str, str]:
    image_modality = validate_image_modality(image_modality)
    return _MODALITY_STORAGE[image_modality]


def validate_image_modality(image_modality: str) -> ImageModality:
    if image_modality not in IMAGE_MODALITIES:
        raise ValueError(
            f"image_modality must be one of {IMAGE_MODALITIES}."
        )
    return image_modality


def get_aligned_stack_files(course_path: str, image_modality: ImageModality):
    """Return trajectory/modality stack paths keyed by their numeric suffix."""
    image_modality = validate_image_modality(image_modality)
    modality_folder,modality_prefix = modality_storage(image_modality)

    def indexed(folder: str, prefix: str) -> dict[str, str]:
        folder_path = os.path.join(course_path, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing rollout modality folder: {folder_path}")
        files = {}
        for filename in os.listdir(folder_path):
            if filename.startswith(prefix) and filename.endswith(".pt"):
                suffix = filename[len(prefix):-3]
                files[suffix] = os.path.join(folder_path, filename)
        return files

    trajectories = indexed("trajectories", "trajectories")
    modalities = indexed(modality_folder, modality_prefix)
    if trajectories.keys() != modalities.keys():
        raise ValueError(
            f"Trajectory and {image_modality} stack IDs do not match in {course_path}."
        )
    return [(trajectories[key], modalities[key]) for key in sorted(trajectories)]


def prepare_rollout_images(traj_data: dict, image_data: dict,
                           image_modality: ImageModality) -> np.ndarray:
    """Validate one aligned rollout and return three-channel HWC images."""
    image_modality = validate_image_modality(image_modality)
    if traj_data["rollout_id"] != image_data["rollout_id"]:
        raise ValueError("Trajectory and image rollout IDs do not match.")

    image_data = dch.decompress_data(image_data, key=image_modality)
    images = image_data[image_modality]
    if not isinstance(images, np.ndarray):
        raise TypeError(f"{image_modality} frames must decompress to an ndarray.")
    if images.shape[0] != traj_data["Ndata"]:
        raise ValueError(
            f"Frame count mismatch for rollout {traj_data['rollout_id']}."
        )

    if is_event_modality(image_modality):
        if images.ndim != 3:
            raise ValueError("Event frames must have shape (N,H,W).")
        images = np.repeat(images[..., None], 3, axis=-1)
    elif images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("RGB frames must have shape (N,H,W,3).")

    return images
