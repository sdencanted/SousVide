import copy
import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet
import sousvide.visualize.rich_utilities as ru
from sousvide.synthesize.event_cloud import resolve_event_cloud_options
from sousvide.synthesize.image_modality import (
    VisualModality,is_event_cloud_modality,validate_visual_modality)

class Policy(nn.Module):
    def __init__(self,
                 policy_config:dict[str,dict],
                 policy_name:str,
                 policy_path:str,
                 image_modality:VisualModality="rgb",
                 event_cloud_options:dict|None=None,
                 require_commnet_weights:bool=False):
        """
        Initialize a Learned Control Policy.

        Args:
            policy_config:  Policy configuration dictionary.
            policy_path:    Policy path.
            image_modality: Image modality selecting the commNet weights.
            require_commnet_weights: Require existing weights for the selected commNet.
            
        Variables:
            network_type:   Type of network.
            pd_idxs:        Indices of the prediction output.
            dp_idxs:        Indices of the deployment output.
            networks:       Network layers.
            use_deploy:     Flag to use forward-pass.
            Nhy:            Maximum sequence length.
        """
        
        # Initial Parent Call
        super().__init__()

        # Populate the network
        image_modality = validate_visual_modality(image_modality)
        if is_event_cloud_modality(image_modality):
            resolved_cloud_options = resolve_event_cloud_options(
                event_cloud_options)
        else:
            if event_cloud_options:
                raise ValueError(
                    "event_cloud_options requires image_modality='event_cloud'.")
            resolved_cloud_options = None
        networks:dict[str,BaseNet] = nn.ModuleDict()
        network_paths = {}
        Nhy = 1
        for name,config in policy_config["networks"].items():
            config = copy.deepcopy(config)
            if (name == "commNet"
                    and is_event_cloud_modality(image_modality)):
                config.setdefault("layers",{}).setdefault("secnet",{})[
                    "num_points"] = resolved_cloud_options["num_points"]
            networks[name] = nf.generate_network(
                config,name,policy_path,image_modality=image_modality,
                require_commnet_weights=require_commnet_weights)
            if (name == "commNet"
                    and is_event_cloud_modality(image_modality)):
                if getattr(networks[name],"visual_backbone",None) != "secnet":
                    raise ValueError(
                        "event_cloud CommNet checkpoint is not a SECNet model; "
                        "train a fresh modality-specific checkpoint.")
                feature_network = networks[name].networks["feat"]
                if feature_network.num_points != resolved_cloud_options["num_points"]:
                    raise ValueError(
                        "Existing SECNet CommNet point count does not match "
                        "event_cloud_options; train a fresh matching checkpoint.")
            elif (name == "commNet"
                  and getattr(networks[name],"visual_backbone",None) == "secnet"):
                raise ValueError(
                    "SECNet CommNet checkpoints cannot be used with dense images "
                    "or voxel grids.")
            network_paths[name] = nf.get_network_path(
                policy_path,name,image_modality)

            # Update the max sequence length variable
            Nhy = max(Nhy,networks[name].Nhy)

            # Ensure all deploy flags are true
            for network in networks.values():
                network.use_deploy = True

        # Class Variables (last network outputs command)
        self.io_idxs = network.io_idxs
        self.network_type = policy_name
        self.Nhy = int(Nhy)
        self.image_modality = image_modality

        self.networks = networks
        self.network_paths = network_paths

    def collect_prediction_inputs(
            self,Xnn:dict[str,torch.Tensor],
            network_names:list[str]|None=None
            ) -> dict[str,dict[str,torch.Tensor]]:
        """Collect training inputs without executing the final target network.

        Networks preceding the last requested network still execute because their
        outputs may be inputs to a downstream target. The last target itself is not
        executed, avoiding an unnecessary CommNet forward pass during synthesis.
        """
        ordered_names = list(self.networks.keys())
        requested = ordered_names if network_names is None else network_names
        unknown = set(requested)-set(ordered_names)
        if unknown:
            raise ValueError(f"Unknown policy networks requested: {sorted(unknown)}")
        if not requested:
            return {}

        requested_set = set(requested)
        last_target_index = max(ordered_names.index(name) for name in requested)
        prediction_inputs = {}
        with torch.inference_mode():
            for index,net_name in enumerate(ordered_names[:last_target_index+1]):
                network = self.networks[net_name]
                network_inputs = nh.extract_io(Xnn,network.io_idxs["xdp"])
                if net_name in requested_set:
                    prediction_inputs[net_name] = network_inputs

                if index < last_target_index:
                    Xnn = Xnn|network(network_inputs)

        return prediction_inputs

    def forward(self,Xnn:dict[str,torch.Tensor]) -> tuple[
                    torch.Tensor,dict[str,torch.Tensor],dict[str,torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            Xnn:    Dictionary of input tensors.

        Returns:
            ynn:    Policy output.
            znn:    Feature output.
            Xpd:    Dictionary for prediction inputs.
        """

        # Initialize output variables
        ynn,pch,cls,Xpd = None,None,None,{}

        # Forward Pass through the networks
        for net_name,network in self.networks.items():
            # Extract the Network Inputs and Input Key
            xnn_idxs = network.io_idxs["xdp"]
            Xnn_net = nh.extract_io(Xnn,xnn_idxs)
        
            # Update dictionary with forward pass through the network
            Ynn_net:dict = network(Xnn_net)

            # Extract policy outputs (first value of featNet/commNet)
            if net_name == "featNet":
                pch = Ynn_net["patches"]
                cls = Ynn_net["class_token"]
            elif net_name == "commNet":
                ynn = next(iter(Ynn_net.values()))

            Xpd[net_name] = Xnn_net

            # Update Xnn
            Xnn = Xnn|Ynn_net
            
        return ynn,pch,cls,Xpd
