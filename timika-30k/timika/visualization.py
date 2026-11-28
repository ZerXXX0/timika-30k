"""
Visualization helpers shared by the inference and overlay commands.

Both functions work in RGB; ``cv2.imwrite`` expects BGR, so callers must convert before saving.
"""

from typing import Dict, List

import cv2
import numpy as np

from timika.config import CLASS_COLORS, DISEASE_CLASSES


def create_color_overlay(
    base_gray: np.ndarray,
    masks: np.ndarray,
    class_names: List[str] = DISEASE_CLASSES,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Renders multi-label color overlays with contours on top of grayscale CXR.
    Args:
        base_gray: (H, W) uint8 grayscale image [0..255]
        masks: (C, H, W) binary float array [0 or 1]
    """
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()

    for c, cls_name in enumerate(class_names):
        mask_c = masks[c]
        if mask_c.sum() > 0:
            color = CLASS_COLORS.get(cls_name, (255, 255, 0))
            # Colored fill
            overlay[mask_c > 0.5] = color

    blended = cv2.addWeighted(overlay, alpha, rgb, 1.0 - alpha, 0)

    # Draw solid contour boundary lines for high contrast
    for c, cls_name in enumerate(class_names):
        mask_c = masks[c].astype(np.uint8)
        if mask_c.sum() > 0:
            color = CLASS_COLORS.get(cls_name, (255, 255, 0))
            contours, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, color, 2)

    return blended


def render_composite_card(
    base_gray: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    image_id: str,
    patient_id: str,
    source: str,
    per_class_metrics: Dict[str, float],
) -> np.ndarray:
    """
    Assembles a 3-panel comparison: [Original CXR] | [Ground Truth] | [Predicted Masks]
    with an informative header and color-coded pathology legend.
    """
    h, w = base_gray.shape

    # 1. Base RGB
    panel_orig = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)

    # 2. Ground Truth Overlay
    panel_gt = create_color_overlay(base_gray, gt_masks)

    # 3. Prediction Overlay
    panel_pred = create_color_overlay(base_gray, pred_masks)

    # Header Bar & Canvas Dimensions
    header_h = 70
    pad = 16
    canvas_w = pad * 4 + w * 3
    canvas_h = h + header_h + 30
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 35  # Dark slate background

    # Title & Metadata
    title_text = f"Sample: {image_id}  |  Patient: {patient_id}  |  Source: {source.upper()}"
    cv2.putText(canvas, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    # Panel Sub-headers
    cv2.putText(canvas, "Input CXR", (pad, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ground Truth", (pad * 2 + w, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Model Prediction", (pad * 3 + w * 2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)

    # Place Panels
    canvas[header_h : header_h + h, pad : pad + w] = panel_orig
    canvas[header_h : header_h + h, pad * 2 + w : pad * 2 + w * 2] = panel_gt
    canvas[header_h : header_h + h, pad * 3 + w * 2 : pad * 3 + w * 3] = panel_pred

    # Footer legend for active findings
    footer_y = header_h + h + 22
    x_offset = 20
    for cls_name in DISEASE_CLASSES:
        c_idx = DISEASE_CLASSES.index(cls_name)
        gt_present = gt_masks[c_idx].sum() > 0
        pred_present = pred_masks[c_idx].sum() > 0
        if gt_present or pred_present:
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            dice_val = per_class_metrics.get(f"{cls_name}_dice", 0.0)
            tag = f"{cls_name.upper()} (Dice: {dice_val:.2f})"
            # Draw color pill
            cv2.circle(canvas, (x_offset + 8, footer_y - 5), 7, color, -1)
            cv2.putText(canvas, tag, (x_offset + 22, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            x_offset += len(tag) * 11 + 35

    return canvas
