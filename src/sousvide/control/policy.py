import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet
import sousvide.visualize.rich_utilities as ru
from sousvide.synthesize.image_modality import ImageModality,validate_image_modality

class Policy(nn.Module):
    def __init__(self,
                 policy_config:dict[str,dict],
                 policy_name:str,
                 policy_path:str,
                 image_modality:ImageModality="rgb",
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
        image_modality = validate_image_modality(image_modality)
        networks:dict[str,BaseNet] = nn.ModuleDict()
        network_paths = {}
        Nhy = 1
        for name,config in policy_config["networks"].items():
            networks[name] = nf.generate_network(
                config,name,policy_path,image_modality=image_modality,
                require_commnet_weights=require_commnet_weights)
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
