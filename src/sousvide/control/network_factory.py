import os
import json
import torch
import sousvide.control.network_helper as nh

from typing import Dict,Union,List
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.sifu import SIFU
from sousvide.control.networks.svnet import SVNet
from sousvide.control.networks.dnnet import DNNet
from sousvide.control.networks.feature_extractors import (
    DINO
)
from sousvide.control.networks.pave import Pave
from sousvide.synthesize.image_modality import (
    IMAGE_MODALITIES,ImageModality,image_modality_channels,
    is_voxel_grid_modality,validate_image_modality)


def get_network_path(
        pilot_path:str,net_name:str,
        image_modality:ImageModality="rgb") -> str:
    """Return the canonical save path for a policy network."""
    image_modality = validate_image_modality(image_modality)
    filename = (
        f"{net_name}_{image_modality}.pt"
        if net_name == "commNet"
        else f"{net_name}.pt"
    )
    return os.path.join(pilot_path,filename)


def get_network_load_path(
        pilot_path:str,net_name:str,
        image_modality:ImageModality="rgb") -> str:
    """Resolve a network path, including a modality-matched legacy commNet."""
    network_path = get_network_path(pilot_path,net_name,image_modality)
    if os.path.isfile(network_path):
        return network_path

    if net_name == "commNet":
        legacy_path = os.path.join(pilot_path,"commNet.pt")
        if (os.path.isfile(legacy_path) and
            _get_legacy_commnet_modality(pilot_path) == image_modality):
            return legacy_path

    return network_path


def _get_legacy_commnet_modality(pilot_path:str) -> ImageModality:
    """Infer which modality produced ``commNet.pt`` from its latest loss log."""
    losses_path = os.path.join(pilot_path,"losses_commNet.pt")
    if os.path.isfile(losses_path):
        try:
            losses = torch.load(losses_path,weights_only=False)
            for loss_entry in reversed(list(losses.values())):
                modality = loss_entry.get("image_modality")
                if modality in IMAGE_MODALITIES:
                    return modality
        except (AttributeError,EOFError,RuntimeError,TypeError,ValueError):
            pass

    # Before modalities were introduced, commNet was trained on RGB.
    return "rgb"

def generate_network(
        net_config:Dict[str,Union[str,Dict[str,List[List[Union[str,int]]]]]],
        net_name:str,
        pilot_path:str,
        image_modality:ImageModality="rgb",
        require_commnet_weights:bool=False) -> BaseNet:
    """
    Generate a network based on the configuration dictionary. If the network does
    not already exist as a .pth file, it will be created and saved to the specified
    path. Only one network of each type can exist per pilot.

    Args:
        config:     Configuration dictionary for the network.
        net_name:   Name of the network.
        pilot_path: Pilot path.
        image_modality: Image modality selecting the commNet weights.
        require_commnet_weights: Raise if the selected commNet weights do not exist.

    Returns:
        network:    The generated network.
        nhy:        The maximum sequence length (if any).
    """

    # Some useful intermediate variables
    network_type = net_config["network_type"]
    if network_type == "dino" and is_voxel_grid_modality(image_modality):
        raise ValueError(
            "Voxel-grid image modalities are supported only by SVNet; "
            "DINO image inputs require three channels.")
    network_path = get_network_path(pilot_path,net_name,image_modality)
    load_path = get_network_load_path(pilot_path,net_name,image_modality)
    print(f"Generating network '{net_name}' of type '{network_type}' at path '{network_path}'...")

    if (require_commnet_weights and net_name == "commNet" and
        not os.path.isfile(load_path)):
        raise FileNotFoundError(
            f"No {image_modality} commNet weights found at '{network_path}'. "
            "Train commNet with the same image_modality before deployment.")

    # If the network already exists, load it. Otherwise, create it.
    if os.path.isfile(load_path):
        network = torch.load(load_path,weights_only=False)
    else:
        # Simple Networks
        if network_type == "simple":
            network = BaseNet(**net_config)
        # Feature Extractors
        elif network_type == "dino":
            network = DINO(**net_config)
        # History Networks
        elif network_type == "sifu":
            network = SIFU(**net_config)
        # Command Networks
        elif network_type == "svnet":
            network = SVNet(
                **net_config,
                image_channels=image_modality_channels(image_modality))
        elif network_type == "dnnet":
            network = DNNet(**net_config)
        elif network_type == "pave":
            network = Pave(**net_config)
        # Mixture of Experts Networks
        # Invalid Network Type
        else:
            raise ValueError(f"Invalid network type: {net_config['network_type']}")

    return network
