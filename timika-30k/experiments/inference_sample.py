"""
Inference & Visualization Pipeline for 50-100 Test Samples.
Outputs side-by-side composite images: [Original CXR] | [Ground Truth] | [Predicted Masks]
Generates per-image metrics and an interactive HTML visual inspection gallery.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from experiments.config import (
    DISEASE_CLASSES,
    CLASS_COLORS,
    SCENARIOS,
    PROJECT_ROOT,
)
from experiments.data.dataset import TimikaDiseaseDataset
from experiments.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 50-100 Inference Samples from Test Split")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth checkpoint")
    parser.add_argument("--model", type=str, default="unetplusplus", choices=["unetplusplus", "fpn", "segformer"])
    parser.add_argument("--encoder", type=str, default=None, help="Backbone encoder name (e.g. convnext_small, mit_b3)")
    parser.add_argument("--scenario", type=str, default="bone_suppression_clahe", choices=["clahe", "bone_suppressed", "bone_suppression_clahe"])
    parser.add_argument("--num-samples", type=int, default=75, help="Number of test images to output (50-100)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid classification threshold")
    parser.add_argument("--min-pixel-size", type=int, default=30, help="Minimum connected component area in pixels")
    parser.add_argument("--output-dir", type=str, default=None, help="Target directory for output samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def filter_tiny_components(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Removes small isolated noise speckles from binary mask."""
    if mask.sum() == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1.0
    return cleaned


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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.encoder is None:
        args.encoder = "mit_b3" if args.model == "segformer" else "resnet34"

    # Output directory
    exp_name = f"{args.model}_{args.encoder}_{args.scenario}"
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "experiments", "outputs", exp_name)

    vis_dir = os.path.join(args.output_dir, "visualizations")
    masks_dir = os.path.join(args.output_dir, "predicted_masks")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"INFERENCE SAMPLING PIPELINE")
    print(f"Model: {args.model.upper()} ({args.encoder}) | Scenario: {args.scenario}")
    print(f"Test Samples to generate: {args.num_samples} (target 50-100)")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*70}\n")

    # 1. Load Test Dataset
    test_dataset = TimikaDiseaseDataset(
        scenario=args.scenario,
        split="test",
        seed=args.seed,
    )
    print(f"Test pool size: {len(test_dataset)} total images.")

    # Select representative 50-100 test samples (prioritizing diverse positive cases and negatives)
    # Use pre-loaded manifest to do instantaneous stratification
    print("Stratifying test set samples via manifest metadata...")
    df_lbl = pd.read_csv(os.path.join(PROJECT_ROOT, "preprocessed_labels", "disease", "manifest.csv"))
    known_pos_ids = set(df_lbl[df_lbl["positive"] == True]["id"])

    positive_indices = []
    negative_indices = []

    for idx, sample_info in enumerate(test_dataset.samples):
        if sample_info["id"] in known_pos_ids:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)

    print(f"Found {len(positive_indices)} positive test images and {len(negative_indices)} negative test images.")

    # Target ratio: 70% positive cases, 30% confirmed negative cases
    target_pos = min(int(args.num_samples * 0.7), len(positive_indices))
    target_neg = min(args.num_samples - target_pos, len(negative_indices))

    selected_indices = random.sample(positive_indices, target_pos) + random.sample(negative_indices, target_neg)
    random.shuffle(selected_indices)
    print(f"Selected {len(selected_indices)} balanced samples for inference inspection.")

    # 2. Build and Load Model
    model = build_model(
        model_name=args.model,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=1,
        num_classes=len(DISEASE_CLASSES),
    ).to(args.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Note: Running inference with initialized model weights (demo mode).")

    model.eval()

    # 3. Run Inference and Generate Visualizations
    records = []
    html_cards = []

    print("\nGenerating sample visualizations...")
    with torch.no_grad():
        for i, sample_idx in enumerate(selected_indices, 1):
            sample = test_dataset[sample_idx]
            image_tensor = sample["image"].unsqueeze(0).to(args.device)  # (1, 1, 512, 512)
            gt_masks = sample["target"].numpy()                          # (7, 512, 512)
            annotated_mask = sample["annotated_mask"].numpy()            # (7,)
            image_id = sample["id"]
            patient_id = sample["patient_id"]
            source = sample["source"]

            # Model Forward Pass
            logits = model(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (7, 512, 512)

            # Apply threshold & component filtering
            pred_masks = np.zeros_like(probs)
            for c in range(len(DISEASE_CLASSES)):
                binary_c = (probs[c] > args.threshold).astype(np.float32)
                pred_masks[c] = filter_tiny_components(binary_c, min_size=args.min_pixel_size)

            # Calculate metrics for active/annotated classes
            per_class_metrics = {}
            for c, cls_name in enumerate(DISEASE_CLASSES):
                if annotated_mask[c] > 0.5:
                    intersection = (pred_masks[c] * gt_masks[c]).sum()
                    union = pred_masks[c].sum() + gt_masks[c].sum()
                    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                    iou = (intersection + 1e-5) / (union - intersection + 1e-5)
                    per_class_metrics[f"{cls_name}_dice"] = float(dice)
                    per_class_metrics[f"{cls_name}_iou"] = float(iou)

            # Base raw grayscale image [0..255]
            # Reverse normalization (mean=0.5, std=0.5 -> [0..255])
            raw_img_norm = sample["image"].squeeze(0).numpy()
            raw_img_uint8 = np.clip((raw_img_norm * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

            # Render 3-panel card
            card = render_composite_card(
                base_gray=raw_img_uint8,
                gt_masks=gt_masks,
                pred_masks=pred_masks,
                image_id=image_id,
                patient_id=patient_id,
                source=source,
                per_class_metrics=per_class_metrics,
            )

            # Save Composite Visualization
            vis_filename = f"sample_{i:03d}_{image_id}"
            vis_path = os.path.join(vis_dir, vis_filename)
            cv2.imwrite(vis_path, card)

            # Save Predicted Masks
            sample_masks_dir = os.path.join(masks_dir, os.path.splitext(image_id)[0])
            os.makedirs(sample_masks_dir, exist_ok=True)
            for c, cls_name in enumerate(DISEASE_CLASSES):
                if pred_masks[c].sum() > 0:
                    mask_uint8 = (pred_masks[c] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(sample_masks_dir, f"{cls_name}.png"), mask_uint8)

            # Record for CSV summary
            rec = {
                "sample_idx": i,
                "image_id": image_id,
                "patient_id": patient_id,
                "source": source,
                "has_ground_truth": bool(gt_masks.sum() > 0),
                "has_prediction": bool(pred_masks.sum() > 0),
                **per_class_metrics,
            }
            records.append(rec)

            # HTML card snippet
            rel_vis_path = os.path.join("visualizations", vis_filename).replace("\\", "/")
            html_cards.append(f"""
            <div class="sample-card">
                <h4>#{i:03d}: {image_id} (Patient: {patient_id} | {source.upper()})</h4>
                <img src="{rel_vis_path}" alt="{image_id}" loading="lazy"/>
            </div>
            """)

            if i % 15 == 0 or i == len(selected_indices):
                print(f"Processed [{i}/{len(selected_indices)}] samples...")

    # 4. Save Summary CSV
    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(args.output_dir, "test_inference_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    # 5. Generate Interactive HTML Gallery
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Inference Samples - {exp_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-bottom: 8px; }}
        .meta {{ background: #1e293b; padding: 16px; border-radius: 8px; margin-bottom: 24px; }}
        .grid {{ display: flex; flex-direction: column; gap: 20px; }}
        .sample-card {{ background: #1e293b; padding: 16px; border-radius: 10px; border: 1px solid #334155; }}
        .sample-card h4 {{ margin: 0 0 12px 0; color: #94a3b8; font-weight: 500; }}
        .sample-card img {{ width: 100%; max-width: 1560px; height: auto; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>CXR Segmentation Inference Gallery</h1>
    <div class="meta">
        <p><strong>Model:</strong> {args.model.upper()} ({args.encoder}) &nbsp;|&nbsp; <strong>Scenario:</strong> {SCENARIOS[args.scenario]['name']}</p>
        <p><strong>Total Samples:</strong> {len(selected_indices)} &nbsp;|&nbsp; <strong>Decision Threshold:</strong> {args.threshold} &nbsp;|&nbsp; <strong>Min Component Pixels:</strong> {args.min_pixel_size}</p>
        <p><strong>Summary Table:</strong> <a href="test_inference_summary.csv" style="color: #38bdf8;">Download test_inference_summary.csv</a></p>
    </div>
    <div class="grid">
        {"".join(html_cards)}
    </div>
</body>
</html>"""
    html_path = os.path.join(args.output_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{'='*70}")
    print(f"COMPLETE! Successfully generated {len(selected_indices)} test inference samples.")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Interactive Gallery: {html_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
