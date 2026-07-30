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

def generate_network(
        net_config:Dict[str,Union[str,Dict[str,List[List[Union[str,int]]]]]],
        net_name:str,
        pilot_path:str) -> BaseNet:
    """
    Generate a network based on the configuration dictionary. If the network does
    not already exist as a .pth file, it will be created and saved to the specified
    path. Only one network of each type can exist per pilot.

    Args:
        config:     Configuration dictionary for the network.
        net_name:   Name of the network.
        pilot_path: Pilot path.

    Returns:
        network:    The generated network.
        nhy:        The maximum sequence length (if any).
    """

    # Some useful intermediate variables
    network_type = net_config["network_type"]
    network_path = os.path.join(pilot_path,net_name+".pt")

    # If the network already exists, load it. Otherwise, create it.
    if os.path.isfile(network_path):
        network = torch.load(network_path,weights_only=False)
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
            network = SVNet(**net_config)
        elif network_type == "dnnet":
            network = DNNet(**net_config)
        elif network_type == "pave":
            network = Pave(**net_config)
        # Mixture of Experts Networks
        # Invalid Network Type
        else:
            raise ValueError(f"Invalid network type: {net_config['network_type']}")
    
    return network
    