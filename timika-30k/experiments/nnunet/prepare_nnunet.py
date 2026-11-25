"""
nnU-Net v2 Dataset Preparation Script for Timika CXR Segmentation.
Converts any of the 3 scenarios into official nnU-Net v2 directory structure and dataset.json.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import shutil
import argparse
import pandas as pd
from PIL import Image
import numpy as np

from experiments.config import SCENARIOS, LABELS_DISEASE_DIR, LABELS_MANIFEST, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Dataset for nnU-Net v2 (2D)")
    parser.add_argument("--scenario", type=str, default="bone_suppression_clahe", choices=["clahe", "bone_suppressed", "bone_suppression_clahe"])
    parser.add_argument("--dataset-id", type=int, default=501, help="nnU-Net Dataset ID (e.g. 501 -> Dataset501_Timika)")
    parser.add_argument("--task-type", type=str, default="any_disease", choices=["any_disease", "pneumothorax", "tb_lesion", "infiltrate"])
    parser.add_argument("--max-train-cases", type=int, default=1500, help="Subset size for efficient 2D training")
    parser.add_argument("--max-test-cases", type=int, default=100, help="Test cases for inference sampling")
    parser.add_argument("--output-base", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_cfg = SCENARIOS[args.scenario]
    dataset_name = f"Dataset{args.dataset_id:03d}_Timika_{args.scenario}_{args.task_type}"

    base_dir = args.output_base or os.path.join(PROJECT_ROOT, "experiments", "nnunet_raw")
    dataset_dir = os.path.join(base_dir, dataset_name)

    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")

    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PREPARING nnU-Net v2 DATASET: {dataset_name}")
    print(f"Scenario: {args.scenario} | Task: {args.task_type}")
    print(f"Directory: {dataset_dir}")
    print(f"{'='*70}\n")

    # Load manifests
    df_img = pd.read_csv(scenario_cfg["manifest"])
    train_pool = df_img[df_img["split"] == "train"]
    test_pool = df_img[df_img["split"] == "test"]

    if args.max_train_cases:
        train_pool = train_pool.sample(n=min(args.max_train_cases, len(train_pool)), random_state=42)
    if args.max_test_cases:
        test_pool = test_pool.sample(n=min(args.max_test_cases, len(test_pool)), random_state=42)

    def process_subset(pool, img_dest, lbl_dest, split_name):
        case_ids = []
        for idx, row in pool.iterrows():
            img_id = row["id"]
            case_id = f"case_{idx:05d}"
            case_ids.append(case_id)

            # Copy image with nnU-Net channel suffix _0000.png
            src_img = os.path.join(scenario_cfg["image_dir"], img_id)
            dst_img = os.path.join(img_dest, f"{case_id}_0000.png")
            shutil.copyfile(src_img, dst_img)

            # Build label mask (0: Background, 1: Lesion)
            if args.task_type == "any_disease":
                # Union of all 7 diseases
                combined_mask = None
                for cls_name in ["atelectasis", "cavitation", "infiltrate", "lymphadenopathy", "pleural_effusion", "pneumothorax", "tb_lesion"]:
                    m_path = os.path.join(LABELS_DISEASE_DIR, cls_name, img_id)
                    if os.path.exists(m_path):
                        m_arr = (np.array(Image.open(m_path).convert("L")) > 127).astype(np.uint8)
                        if combined_mask is None:
                            combined_mask = m_arr
                        else:
                            combined_mask = np.maximum(combined_mask, m_arr)
                if combined_mask is None:
                    combined_mask = np.zeros((512, 512), dtype=np.uint8)
            else:
                m_path = os.path.join(LABELS_DISEASE_DIR, args.task_type, img_id)
                if os.path.exists(m_path):
                    combined_mask = (np.array(Image.open(m_path).convert("L")) > 127).astype(np.uint8)
                else:
                    combined_mask = np.zeros((512, 512), dtype=np.uint8)

            dst_lbl = os.path.join(lbl_dest, f"{case_id}.png")
            Image.fromarray(combined_mask).save(dst_lbl)

        return case_ids

    print("Formatting Train split...")
    train_cases = process_subset(train_pool, images_tr, labels_tr, "train")
    print(f"Written {len(train_cases)} training cases.")

    print("Formatting Test split...")
    test_cases = process_subset(test_pool, images_ts, labels_ts, "test")
    print(f"Written {len(test_cases)} test cases.")

    # Generate dataset.json for nnU-Net v2
    dataset_json = {
        "channel_names": {
            "0": "CXR"
        },
        "labels": {
            "background": 0,
            args.task_type: 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".png"
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n[DONE] Created official nnU-Net v2 dataset.json at: {json_path}")
    print("\nTo train nnU-Net v2:")
    print(f"1. set nnUNet_raw={base_dir}")
    print(f"2. nnUNetv2_plan_and_preprocess -d {args.dataset_id} -c 2d")
    print(f"3. nnUNetv2_train {args.dataset_id} 2d 0")


if __name__ == "__main__":
    main()
