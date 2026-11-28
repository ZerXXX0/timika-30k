"""
Data Transforms and Augmentations for Chest Radiographs using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str = "train", image_size: tuple = (512, 512)) -> A.Compose:
    """
    Build albumentations transform pipeline for medical chest radiographs.
    
    In CXR imaging:
    - Horizontal flip is safe (though situs inversus is rare, thoracic symmetry remains consistent).
    - Modest rotation (±10 deg) and small translation simulate slight patient positioning variations.
    - Extreme elastic distortions or vertical flips are strictly avoided as they violate anatomical priors.
    """
    height, width = image_size

    if split == "train":
        transform_list = [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.04, 0.04),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4,
            ),
            A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    else:
        # Validation / Test
        transform_list = [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ]

    return A.Compose(transform_list)
