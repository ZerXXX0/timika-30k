"""
Compound Loss Functions for Partially-Labeled Multi-Label CXR Segmentation.
Combines Masked BCE + Soft Dice Loss or Focal Tversky Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiLabelBCEDiceLoss(nn.Module):
    """
    Computes BCE + Soft Dice Loss strictly on annotated pathology channels.
    
    Why this is critical:
    Source datasets (SIIM-ACR, TBX11K, etc.) only annotate specific findings.
    Treating unannotated channels as negative heavily penalizes the model when it
    correctly recognizes unannotated lesions. This loss zeroes out gradients for
    unannotated classes on each image.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        annotated_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw unnormalized model predictions.
            targets: (B, C, H, W) ground truth binary masks (0 or 1).
            annotated_mask: (B, C) binary flags (1 = class is labeled, 0 = unannotated).
        """
        # 1. Masked Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, C, H, W)
        bce_per_class = bce.mean(dim=(2, 3))  # (B, C)
        
        # Apply annotation mask
        masked_bce = (bce_per_class * annotated_mask).sum() / (annotated_mask.sum() + 1e-8)

        # 2. Masked Soft Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))  # (B, C)
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # (B, C)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        masked_dice = (dice_loss * annotated_mask).sum() / (annotated_mask.sum() + 1e-8)

        # Compound weighted loss
        total_loss = self.bce_weight * masked_bce + self.dice_weight * masked_dice
        return total_loss


class MaskedFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss with channel masking for highly imbalanced, small lesion segmentation.
    alpha=0.7, beta=0.3 prioritizes Recall (penalizes False Negatives more heavily).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        annotated_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # True Positives, False Positives, False Negatives per (B, C)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, 1.0 / self.gamma)

        masked_loss = (focal_tversky * annotated_mask).sum() / (annotated_mask.sum() + 1e-8)
        return masked_loss
