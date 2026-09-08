import os

from sousvide.synthesize.image_modality import (
    VisualModality,validate_visual_modality)


def get_losses_path(
        pilot_path:str,net_name:str,
        image_modality:VisualModality="rgb") -> str:
    """Return the loss-log path for a trained policy network."""
    image_modality = validate_visual_modality(image_modality)
    filename = (
        f"losses_{net_name}_{image_modality}.pt"
        if net_name == "commNet"
        else f"losses_{net_name}.pt"
    )
    return os.path.join(pilot_path,filename)
