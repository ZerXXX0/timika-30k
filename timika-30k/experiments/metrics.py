"""
Evaluation Metrics for Partially Annotated Multi-Label Segmentation.
Computes Dice, IoU, Sensitivity, Precision per pathology and overall macro score.
"""

from typing import Dict, List
import numpy as np
import torch

from experiments.config import DISEASE_CLASSES


class SegmentationMetricTracker:
    """Accumulates and computes per-class and macro segmentation metrics."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.tp = np.zeros(len(DISEASE_CLASSES), dtype=np.float64)
        self.fp = np.zeros(len(DISEASE_CLASSES), dtype=np.float64)
        self.fn = np.zeros(len(DISEASE_CLASSES), dtype=np.float64)
        self.evaluated_counts = np.zeros(len(DISEASE_CLASSES), dtype=np.int64)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        annotated_mask: torch.Tensor,
    ):
        """
        Updates running confusion matrix values only for annotated classes.
        """
        preds = (torch.sigmoid(logits) > self.threshold).float()
        targets = targets.float()

        # (B, C)
        tp_batch = (preds * targets).sum(dim=(2, 3)).cpu().numpy()
        fp_batch = (preds * (1.0 - targets)).sum(dim=(2, 3)).cpu().numpy()
        fn_batch = ((1.0 - preds) * targets).sum(dim=(2, 3)).cpu().numpy()
        mask_batch = annotated_mask.cpu().numpy()  # (B, C)

        for c in range(len(DISEASE_CLASSES)):
            valid_idx = mask_batch[:, c] > 0.5
            if np.any(valid_idx):
                self.tp[c] += tp_batch[valid_idx, c].sum()
                self.fp[c] += fp_batch[valid_idx, c].sum()
                self.fn[c] += fn_batch[valid_idx, c].sum()
                self.evaluated_counts[c] += valid_idx.sum()

    def compute(self) -> Dict[str, float]:
        metrics = {}
        dice_list = []
        iou_list = []

        for c, cls_name in enumerate(DISEASE_CLASSES):
            tp = self.tp[c]
            fp = self.fp[c]
            fn = self.fn[c]

            dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
            iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
            precision = (tp + self.smooth) / (tp + fp + self.smooth)
            recall = (tp + self.smooth) / (tp + fn + self.smooth)

            metrics[f"{cls_name}_dice"] = float(dice)
            metrics[f"{cls_name}_iou"] = float(iou)
            metrics[f"{cls_name}_precision"] = float(precision)
            metrics[f"{cls_name}_recall"] = float(recall)

            if self.evaluated_counts[c] > 0:
                dice_list.append(dice)
                iou_list.append(iou)

        metrics["macro_dice"] = float(np.mean(dice_list)) if dice_list else 0.0
        metrics["macro_iou"] = float(np.mean(iou_list)) if iou_list else 0.0
        return metrics
