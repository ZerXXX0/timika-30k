"""
Experiment Configuration & Hyperparameters
Multi-Model CXR Segmentation Framework (SMP U-Net++/FPN, SegFormer MiT, nnU-Net v2)
Across 3 Preprocessing Scenarios (CLAHE, Bone-Suppressed, Bone-Suppression + CLAHE)
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

# Base directories (all derived from this file's location, so commands work from any working directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_BASE = PROJECT_ROOT
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # <model>_<encoder>_<scenario>/best_model.pth
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")          # inference samples, schema runs, nnU-Net raw data
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")            # image-id lists (e.g. the 553 ChestX-Det test images)
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Disease classes in timika dataset (7 multi-label classes)
DISEASE_CLASSES = [
    "atelectasis",
    "cavitation",
    "infiltrate",
    "lymphadenopathy",
    "pleural_effusion",
    "pneumothorax",
    "tb_lesion",
]

# Color map for visual overlays during inference sampling (RGB 0-255)
CLASS_COLORS = {
    "atelectasis": (255, 99, 71),        # Tomato Red
    "cavitation": (255, 215, 0),         # Gold
    "infiltrate": (50, 205, 50),         # Lime Green
    "lymphadenopathy": (138, 43, 226),   # Blue Violet
    "pleural_effusion": (30, 144, 255),  # Deep Sky Blue
    "pneumothorax": (255, 20, 147),      # Deep Pink
    "tb_lesion": (255, 140, 0),          # Dark Orange
}

# The 3 Preprocessing Scenarios
SCENARIOS = {
    "clahe": {
        "name": "CLAHE Enhanced",
        "description": "Contrast-limited adaptive histogram equalization (without bone suppression)",
        "image_dir": os.path.join(DATASET_BASE, "preprocessed_clahe", "images"),
        "manifest": os.path.join(DATASET_BASE, "preprocessed_clahe", "manifest.csv"),
    },
    "bone_suppressed": {
        "name": "Bone Suppressed",
        "description": "Deep-learning bone suppression removing rib/clavicle shadows (without CLAHE)",
        "image_dir": os.path.join(DATASET_BASE, "preprocessed_bone_suppressed", "images"),
        "manifest": os.path.join(DATASET_BASE, "preprocessed_bone_suppressed", "manifest.csv"),
    },
    "bone_suppression_clahe": {
        "name": "Bone Suppressed + CLAHE",
        "description": "Dual preprocessed: Ribs suppressed + local soft-tissue contrast enhancement",
        "image_dir": os.path.join(DATASET_BASE, "preprocessed_bone_suppression_clahe", "images"),
        "manifest": os.path.join(DATASET_BASE, "preprocessed_bone_suppression_clahe", "manifest.csv"),
    },
}

# Label paths
LABELS_DISEASE_DIR = os.path.join(DATASET_BASE, "preprocessed_labels", "disease")
LABELS_MANIFEST = os.path.join(LABELS_DISEASE_DIR, "manifest.csv")


def experiment_name(model: str, encoder: str, scenario: str) -> str:
    """Folder name used for an experiment under checkpoints/ and outputs/."""
    return f"{model}_{encoder}_{scenario}"


@dataclass
class Hyperparameters:
    """Explicit Hyperparameter Specifications for Training and Inference."""

    # 1. Architecture & Model
    model_name: str = "unetplusplus"  # 'unetplusplus', 'fpn', 'segformer', 'nnunet'
    encoder_name: str = "resnet34"    # 'resnet34', 'resnet50', 'densenet121', 'efficientnet-b4', 'mit_b3' (for SegFormer)
    encoder_weights: str = "imagenet"
    in_channels: int = 1
    num_classes: int = len(DISEASE_CLASSES)

    # 2. Preprocessing & Input Dimensions
    scenario: str = "bone_suppression_clahe"  # 'clahe', 'bone_suppressed', 'bone_suppression_clahe'
    image_size: Tuple[int, int] = (512, 512)

    # 3. Optimization
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    lr_scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 2

    # 4. Training Schedule & Batch Size
    batch_size: int = 16  # Optimized for RTX 5070 (12GB/16GB VRAM)
    val_batch_size: int = 16
    num_workers: int = 4
    epochs: int = 40
    early_stopping_patience: int = 8
    amp_enabled: bool = True  # Mixed precision (fp16) via torch.cuda.amp

    # 5. Loss Function Hyperparameters
    loss_type: str = "masked_bce_dice"  # 'masked_bce_dice', 'masked_focal_tversky'
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    tversky_alpha: float = 0.7  # Penalizes False Negatives more (critical for rare medical lesions)
    tversky_beta: float = 0.3   # Penalizes False Positives
    tversky_gamma: float = 1.33

    # 6. Multi-Label Inference & Thresholding
    classification_threshold: float = 0.5  # Sigmoid decision threshold
    min_component_area_pixels: int = 30     # Remove isolated noise specks (<30 pixels)

    # 7. Inference Sampling Output
    num_inference_samples: int = 75  # Target 50-100 images from test set
    output_dir: str = OUTPUTS_DIR
    checkpoints_dir: str = CHECKPOINTS_DIR

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
