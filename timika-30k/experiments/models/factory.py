"""
Model Factory for Multi-Model Segmentation Architecture.
Supports:
1. U-Net++ (Nested U-Net with dense skip pathways)
2. FPN (Feature Pyramid Network)
3. SegFormer (Transformer encoder with multi-scale attention via MiT backbones)
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    model_name: str = "unetplusplus",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 1,
    num_classes: int = 7,
) -> nn.Module:
    """
    Constructs the specified 2D segmentation model.
    
    Args:
        model_name: 'unetplusplus', 'fpn', 'segformer'
        encoder_name: Backbone name (e.g. 'convnext_small', 'resnet34', 'mit_b3')
        encoder_weights: Pretrained weights (e.g. 'imagenet' or None)
        in_channels: 1 for grayscale radiographs
        num_classes: Number of target pathology masks (7)
    """
    model_name_lower = model_name.lower().replace("-", "").replace("_", "")

    if model_name_lower in ["unetplusplus", "unet++"]:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name_lower == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name_lower == "segformer":
        # Ensure encoder is an MiT backbone if generic name was passed
        if not encoder_name.startswith("mit_"):
            encoder_name = "mit_b3"
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from ['unetplusplus', 'fpn', 'segformer']"
        )

    return model
