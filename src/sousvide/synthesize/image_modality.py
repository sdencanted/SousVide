"""Shared helpers for RGB, grayscale, and event-based observation inputs."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

import sousvide.synthesize.data_compress_helper as dch


ImageModality = Literal[
    "rgb", "grayscale", "kronecker_delta", "event_pseudo_gaussian",
    "event_bilinear", "event_bin", "event_eros", "event_tos",
    "event_voxel_grid", "event_voxel_grid_polarity",
]

IMAGE_MODALITIES: tuple[ImageModality, ...] = (
    "rgb", "grayscale", "kronecker_delta", "event_pseudo_gaussian",
    "event_bilinear", "event_bin", "event_eros", "event_tos",
    "event_voxel_grid", "event_voxel_grid_polarity",
)
EVENT_IMAGE_MODALITIES = IMAGE_MODALITIES[2:]
VOXEL_GRID_MODALITIES: tuple[ImageModality, ...] = (
    "event_voxel_grid", "event_voxel_grid_polarity",
)

_MODALITY_STORAGE = {
    "rgb": ("images", "images"),
    # Grayscale observations are derived from the original RGB rollout stack;
    # no duplicate rollout artifact is needed.
    "grayscale": ("images", "images"),
    "kronecker_delta": ("kronecker", "kronecker"),
    "event_pseudo_gaussian": (
        "event_pseudo_gaussian", "event_pseudo_gaussian"),
    "event_bilinear": ("event_bilinear", "event_bilinear"),
    "event_bin": ("event_bin", "event_bin"),
    "event_eros": ("event_eros", "event_eros"),
    "event_tos": ("event_tos", "event_tos"),
    "event_voxel_grid": ("event_voxel_grid", "event_voxel_grid"),
    "event_voxel_grid_polarity": (
        "event_voxel_grid_polarity", "event_voxel_grid_polarity"),
}


def rgb_to_grayscale(image: np.ndarray, repeat_channels: bool = True) -> np.ndarray:
    """Convert RGB image content to BT.601 luminance.

    By default the luminance plane is repeated across three channels. This
    keeps grayscale usable by all existing RGB backbones, including pretrained
    feature extractors, while ensuring that color information is removed.
    """
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError("An image must have at least two dimensions.")

    if image.ndim == 2:
        grayscale = image[...,None]
    elif image.shape[-1] == 1:
        grayscale = image
    elif image.shape[-1] == 3:
        luminance = np.sum(
            image.astype(np.float32)*np.array(
                [0.299,0.587,0.114],dtype=np.float32),
            axis=-1,
        )
        if np.issubdtype(image.dtype,np.integer):
            info = np.iinfo(image.dtype)
            luminance = np.clip(np.rint(luminance),info.min,info.max)
        grayscale = luminance.astype(image.dtype,copy=False)[...,None]
    else:
        raise ValueError("An RGB image must have shape (...,H,W,3).")

    return np.repeat(grayscale,3,axis=-1) if repeat_channels else grayscale


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


def event_image_to_model_channels(
        image: np.ndarray, image_modality: ImageModality) -> np.ndarray:
    """Convert one stored event representation to the policy's HWC layout."""
    image_modality = validate_image_modality(image_modality)
    image = np.asarray(image)
    if is_voxel_grid_modality(image_modality):
        channels = image_modality_channels(image_modality)
        if image.ndim != 3 or image.shape[0] != channels:
            raise ValueError(
                f"{image_modality} must have shape ({channels},H,W).")
        return np.moveaxis(image,0,-1)
    return event_image_to_three_channels(image)


def is_event_modality(image_modality: str) -> bool:
    return image_modality in EVENT_IMAGE_MODALITIES


def is_grayscale_modality(image_modality: str) -> bool:
    return image_modality == "grayscale"


def uses_modality_observation_folder(image_modality: ImageModality) -> bool:
    """Return whether observations must be isolated from legacy RGB data."""
    return validate_image_modality(image_modality) != "rgb"


def is_voxel_grid_modality(image_modality: str) -> bool:
    return image_modality in VOXEL_GRID_MODALITIES


def image_modality_channels(image_modality: ImageModality) -> int:
    """Return the number of channels presented to an image-consuming model."""
    image_modality = validate_image_modality(image_modality)
    if image_modality == "event_voxel_grid":
        return 5
    if image_modality == "event_voxel_grid_polarity":
        return 10
    return 3


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
    """Validate one aligned rollout and return model-ready HWC images."""
    image_modality = validate_image_modality(image_modality)
    if traj_data["rollout_id"] != image_data["rollout_id"]:
        raise ValueError("Trajectory and image rollout IDs do not match.")

    # Grayscale is generated from the lossless RGB rollout on demand.
    storage_key = "rgb" if is_grayscale_modality(image_modality) else image_modality
    image_data = dch.decompress_data(image_data, key=storage_key)
    images = image_data[storage_key]
    if not isinstance(images, np.ndarray):
        raise TypeError(f"{image_modality} frames must decompress to an ndarray.")
    if images.shape[0] != traj_data["Ndata"]:
        raise ValueError(
            f"Frame count mismatch for rollout {traj_data['rollout_id']}."
        )

    if is_grayscale_modality(image_modality):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("RGB source frames must have shape (N,H,W,3).")
        images = rgb_to_grayscale(images)
    elif is_voxel_grid_modality(image_modality):
        channels = image_modality_channels(image_modality)
        if (images.ndim != 4 or images.shape[1] != channels
                or images.dtype != np.float32):
            raise ValueError(
                f"{image_modality} frames must have float32 "
                f"(N,{channels},H,W) format.")
        images = np.moveaxis(images,1,-1)
    elif is_event_modality(image_modality):
        if images.ndim != 3:
            raise ValueError("Event frames must have shape (N,H,W).")
        images = np.repeat(images[..., None], 3, axis=-1)
    elif images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("RGB frames must have shape (N,H,W,3).")

    return images
