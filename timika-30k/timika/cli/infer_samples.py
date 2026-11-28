"""
Inference & Visualization Pipeline for 50-100 Test Samples.
Outputs side-by-side composite images: [Original CXR] | [Ground Truth] | [Predicted Masks]
Generates per-image metrics and an interactive HTML visual inspection gallery.

Usage (from the project root):
    python -m timika.cli.infer_samples --model unetplusplus --scenario bone_suppressed \
        --checkpoint checkpoints/unetplusplus_resnet34_bone_suppressed/best_model.pth
Output: outputs/<model>_<encoder>_<scenario>/{visualizations/, predicted_masks/, test_inference_summary.csv, gallery.html}
"""

import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from timika.config import (
    DISEASE_CLASSES,
    LABELS_MANIFEST,
    OUTPUTS_DIR,
    SCENARIOS,
    experiment_name,
)
from timika.datasets.dataset import TimikaDiseaseDataset
from timika.models.factory import build_model
from timika.visualization import render_composite_card


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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.encoder is None:
        args.encoder = "mit_b3" if args.model == "segformer" else "resnet34"

    # Output directory
    exp_name = experiment_name(args.model, args.encoder, args.scenario)
    if args.output_dir is None:
        args.output_dir = os.path.join(OUTPUTS_DIR, exp_name)

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
    df_lbl = pd.read_csv(LABELS_MANIFEST)
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
