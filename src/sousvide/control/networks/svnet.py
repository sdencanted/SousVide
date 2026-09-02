import torch
import copy

from torch import nn
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.secnet import SECNetEncoder
from sousvide.synthesize.image_modality import (
    VisualModality,is_event_cloud_modality,image_modality_channels)

class SVNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 image_modality:VisualModality="rgb",
                 network_type="svnet"):
        """
        Initialize a SousVide Command Network.

        The network takes images, current states and histrory feature vector to
        output a command vector. The network uses a SqueezeNet backbone to extract
        image features.

        """

        inputs = copy.deepcopy(inputs)
        input_groups = (
            inputs.values()
            if "prediction" in inputs and "deployment" in inputs
            else (inputs,))
        backbones = layers.get("visual_backbones",{})
        backbone_name = backbones.get(
            image_modality,backbones.get("default","squeezenet1_1"))
        secnet_config = layers.get("secnet",{})
        if is_event_cloud_modality(image_modality):
            if backbone_name != "secnet":
                raise ValueError("event_cloud inputs require the SECNet backbone.")
            num_points = secnet_config.get("num_points",4096)
            for input_group in input_groups:
                if "rgb_image" not in input_group:
                    raise ValueError(
                        "SVNet event_cloud adaptation requires an rgb_image input template.")
                del input_group["rgb_image"]
                input_group["event_cloud"] = [
                    num_points,["t","x","y","p"]]
        else:
            if backbone_name != "squeezenet1_1":
                raise ValueError(
                    "Dense image and voxel inputs require the SqueezeNet backbone.")
            image_channels = image_modality_channels(image_modality)
            for input_group in input_groups:
                if "rgb_image" in input_group:
                    input_group["rgb_image"][-1] = image_channels

        # Initialize the parent class
        # Zero-argument super remains valid when a notebook reloads this
        # module while network_factory still references the previous class.
        super().__init__(inputs,outputs,network_type)

        # Unpack network configs
        io_dims = self.get_io_dims()
        io_sizes = self.get_io_sizes()
        dropout = layers["dropout"]
        hidden_sizes = layers["hidden_sizes"]
        idx_cmd = layers["cmd_aug_layer"]

        # Populate the network
        if backbone_name == "secnet":
            feature_network = SECNetEncoder(**secnet_config)
            feature_size = feature_network.out_features
            visual_input_key = "event_cloud"
        else:
            feature_network = squeezenet1_1()
            resolved_image_channels = io_dims["xdp"]["rgb_image"][0]
            if resolved_image_channels != 3:
                first_conv = feature_network.features[0]
                feature_network.features[0] = nn.Conv2d(
                    resolved_image_channels,first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,stride=first_conv.stride,
                    padding=first_conv.padding,dilation=first_conv.dilation,
                    groups=first_conv.groups,bias=first_conv.bias is not None,
                    padding_mode=first_conv.padding_mode)
            feature_size = layers["sqnet_size"]
            visual_input_key = "rgb_image"

        prev_size = io_sizes["xdp"]["current1"]+feature_size

        mlp0 = []
        for layer_size in hidden_sizes[:idx_cmd]:
            mlp0.append(nn.Linear(prev_size, layer_size))
            mlp0.append(nn.ReLU())
            mlp0.append(nn.Dropout(dropout))

            prev_size = layer_size

        prev_size += io_sizes["xdp"]["current2"] + io_sizes["xdp"]["feature_vector"]

        mlp1 = []
        for layer_size in hidden_sizes[idx_cmd:]:
            mlp1.append(nn.Linear(prev_size, layer_size))
            mlp1.append(nn.ReLU())
            mlp1.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        mlp1.append(nn.Linear(prev_size, io_sizes["ypd"]["command"]))

        networks = nn.ModuleDict({
            "feat": feature_network,
            "mlp0": nn.Sequential(*mlp0),
            "mlp1": nn.Sequential(*mlp1)
        })

        # Class Variables
        self.networks = networks
        self.visual_input_key = visual_input_key
        self.visual_backbone = backbone_name
    
    def forward(self,Xnn:dict[str,torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            Xnn:    Input dictionary.

        Returns:
            Ynn:    Output dictionary.
        """

        # Unpack inputs
        # Full-object SqueezeNet checkpoints created before event-cloud
        # support do not contain ``visual_input_key``. Their visual input was
        # always ``rgb_image``, so retain that legacy schema when loading them.
        visual_input_key = getattr(self,"visual_input_key","rgb_image")
        xnn_img,xnn_hL = Xnn[visual_input_key],Xnn["feature_vector"]
        xnn_cr1,xnn_cr2 = Xnn["current1"],Xnn["current2"]

        # Feature extraction
        ynn_ft = self.networks["feat"](xnn_img)

        # Image MLP
        xnn_im = torch.cat([ynn_ft,xnn_cr1],dim=1)
        ynn_im = self.networks["mlp0"](xnn_im)

        # Command MLP
        xnn_cm = torch.cat([ynn_im,xnn_cr2,xnn_hL],dim=1)
        ynn_cm = self.networks["mlp1"](xnn_cm)     

        # Create the output dictionary
        Ynn = {"command": ynn_cm}

        return Ynn
