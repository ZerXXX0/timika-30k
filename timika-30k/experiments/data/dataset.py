"""
PyTorch Dataset for Timika Multi-Label CXR Disease Segmentation.
Supports the 3 Preprocessing Scenarios with Loss Masking for Partial Annotations.
"""

import os
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from experiments.config import (
    SCENARIOS,
    DISEASE_CLASSES,
    LABELS_DISEASE_DIR,
    LABELS_MANIFEST,
)
from experiments.data.transforms import get_transforms


class TimikaDiseaseDataset(Dataset):
    """
    Multi-label dataset loader for Timika chest radiographs.
    
    Key Features:
    1. Dynamic Preprocessing Scenario selection ('clahe', 'bone_suppressed', 'bone_suppression_clahe').
    2. Patient-leakage-free split handling (Train / Val / Test).
    3. Multi-label binary mask loading for all 7 disease classes.
    4. Annotated-mask tensor generation: signals which classes are officially labeled for each image
       to prevent penalizing unannotated findings as false negatives.
    """

    def __init__(
        self,
        scenario: str = "bone_suppression_clahe",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS.keys())}")

        self.scenario = scenario
        self.split = split
        self.image_size = image_size
        self.transforms = get_transforms(split=split, image_size=image_size)

        # Base directories
        self.image_dir = SCENARIOS[scenario]["image_dir"]
        self.manifest_path = SCENARIOS[scenario]["manifest"]
        self.labels_dir = LABELS_DISEASE_DIR

        # Load image manifest
        df_img = pd.read_csv(self.manifest_path)

        # Load label annotations manifest to construct fast lookup table
        df_lbl = pd.read_csv(LABELS_MANIFEST)
        # Set of (image_id, class_name) that are actively annotated
        self.annotated_set = set(zip(df_lbl["id"], df_lbl["class"]))

        # Build split filtering
        self.samples_df = self._filter_split(df_img, split=split, val_ratio=val_ratio, seed=seed)

        if max_samples is not None and len(self.samples_df) > max_samples:
            self.samples_df = self.samples_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        self.samples = self.samples_df.to_dict("records")
        print(f"[{scenario.upper()} - {split.upper()}] Loaded {len(self.samples)} images.")

    def _filter_split(
        self, df: pd.DataFrame, split: str, val_ratio: float = 0.1, seed: int = 42
    ) -> pd.DataFrame:
        """Partition images into train, val, and test while ensuring zero patient overlap."""
        if split == "test":
            # Exact official test split (5,145 images)
            return df[df["split"] == "test"].copy().reset_index(drop=True)

        # For train and val: take images from official 'train' and unassigned '-'
        train_pool = df[df["split"] != "test"].copy()

        # Group by patient_id to prevent patient leakage
        patient_ids = sorted(train_pool["patient_id"].astype(str).unique())
        
        # Deterministic patient-level hash split
        val_patients = set()
        for pid in patient_ids:
            h = int(hashlib.md5(f"{pid}_{seed}".encode("utf-8")).hexdigest(), 16)
            if (h % 1000) / 1000.0 < val_ratio:
                val_patients.add(pid)

        if split == "val":
            sub_df = train_pool[train_pool["patient_id"].astype(str).isin(val_patients)]
        elif split == "train":
            sub_df = train_pool[~train_pool["patient_id"].astype(str).isin(val_patients)]
        else:
            raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test']")

        return sub_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples[idx]
        image_id = row["id"]

        # 1. Load Preprocessed Image
        image_path = os.path.join(self.image_dir, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        image_np = np.array(image, dtype=np.uint8)  # (H, W)

        # 2. Load 7 Disease Masks
        mask_channels = []
        annotated_flags = []

        for cls_name in DISEASE_CLASSES:
            # Check if this image has an official annotation for this class
            is_annotated = (image_id, cls_name) in self.annotated_set
            annotated_flags.append(1.0 if is_annotated else 0.0)

            mask_path = os.path.join(self.labels_dir, cls_name, image_id)
            if os.path.exists(mask_path):
                mask_im = Image.open(mask_path).convert("L")
                mask_arr = (np.array(mask_im) > 127).astype(np.float32)
            else:
                mask_arr = np.zeros(image_np.shape, dtype=np.float32)
            mask_channels.append(mask_arr)

        # Stack masks to shape (H, W, 7) for Albumentations
        mask_np = np.stack(mask_channels, axis=-1)

        # 3. Apply Albumentations Transforms
        transformed = self.transforms(image=image_np, mask=mask_np)
        image_tensor = transformed["image"]  # (1, H, W)
        mask_trans = transformed["mask"]    # (H, W, 7)

        # Permute mask to (7, H, W)
        target_tensor = mask_trans.permute(2, 0, 1).float()
        annotated_tensor = torch.tensor(annotated_flags, dtype=torch.float32)  # (7,)

        return {
            "image": image_tensor,            # (1, 512, 512)
            "target": target_tensor,          # (7, 512, 512)
            "annotated_mask": annotated_tensor,  # (7,) - binary flag per class
            "id": image_id,
            "source": row.get("source", ""),
            "patient_id": str(row.get("patient_id", "")),
        }
