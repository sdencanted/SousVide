"""
Helper functions for data synthesis.
"""

import numpy as np

from PIL import Image
from io import BytesIO


def decompress_data(
    image_dict:dict[str,str|np.ndarray],key:str="rgb"
) -> dict[str,str|np.ndarray]:
    """
    Decompress PNG frames back to their original RGB or grayscale arrays.

    Args:
        image_dict:    Dictionary containing image data.
        key:           Image modality key to decompress.

    Returns:
        image_dict:    Dictionary with decompressed images.
    """
    assert key in image_dict, f"No {key} images found in data dictionary"
    frames = image_dict[key]
    # Voxel stacks are serialized as tensors so torch.save streams their
    # disk-backed storage instead of constructing one enormous pickle buffer.
    try:
        import torch
        if isinstance(frames,torch.Tensor):
            frames = frames.detach().cpu().numpy()
            image_dict[key] = frames
    except ImportError:
        pass
    if len(frames) == 0:
        return image_dict

    if isinstance(frames[0],bytes):
        raw_imgs = []
        for frame in frames:
            with Image.open(BytesIO(frame)) as image:
                raw_imgs.append(np.array(image))
        image_dict[key] = np.stack(raw_imgs,axis=0)

    return image_dict


def compress_data(Images,key:str="rgb"):
    """
    Compress RGB ``(N,H,W,3)`` or grayscale ``(N,H,W)`` uint8 frames as PNG.

    Args:
        Images:    List of dictionaries containing image data.
        key:       Image modality key to compress.

    Returns:
        Images:    List of dictionaries with compressed images.
    """
    if not Images:
        return Images
    assert all(key in item for item in Images), f"No {key} images found in data dictionary"

    for image_dict in Images:
        raw_imgs = image_dict[key]
        if isinstance(raw_imgs,np.ndarray):
            compressed = []
            for image_array in raw_imgs:
                buffer = BytesIO()
                Image.fromarray(image_array).save(buffer,format="PNG")
                compressed.append(buffer.getvalue())
            image_dict[key] = compressed

    return Images
