"""
Automated Batch Runner: Trains a Model across All 3 Preprocessing Scenarios Sequentially
and Automatically Generates Test Inference Samples & Gallery for Each.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config import SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Model across All 3 Scenarios Sequentially")
    parser.add_argument("--model", type=str, default="segformer", choices=["segformer", "unetplusplus", "fpn"])
    parser.add_argument("--encoder", type=str, default=None, help="Backbone encoder (e.g. mit_b3, resnet34)")
    parser.add_argument("--scenarios", nargs="+", default=["bone_suppressed", "clahe", "bone_suppression_clahe"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=75, help="Test inference samples per scenario")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.encoder is None:
        args.encoder = "mit_b3" if args.model == "segformer" else "resnet34"

    print("\n" + "=" * 80)
    print(f"BATCH EXPERIMENT RUNNER: {args.model.upper()} ({args.encoder})")
    print(f"Scenarios to run sequentially: {args.scenarios}")
    print(f"Epochs per scenario: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 80 + "\n")

    summary_records = []

    for i, scenario in enumerate(args.scenarios, 1):
        print("\n" + "#" * 80)
        print(f"[{i}/{len(args.scenarios)}] STARTING SCENARIO: {scenario.upper()} ({SCENARIOS[scenario]['name']})")
        print("#" * 80 + "\n")

        # 1. Run Training
        train_cmd = [
            args.python_bin,
            "experiments/train.py",
            "--model", args.model,
            "--encoder", args.encoder,
            "--scenario", scenario,
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ]
        print(f"Running command: {' '.join(train_cmd)}")
        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            print(f"[ERROR] Training failed for scenario {scenario} with exit code {ret.returncode}")
            continue

        # 2. Check for checkpoint
        exp_name = f"{args.model}_{args.encoder}_{scenario}"
        ckpt_path = os.path.join("experiments", "checkpoints", exp_name, "best_model.pth")
        history_path = os.path.join("experiments", "checkpoints", exp_name, "training_history.csv")

        best_dice = 0.0
        best_epoch = 0
        if os.path.exists(history_path):
            df_hist = pd.read_csv(history_path)
            best_idx = df_hist["val_macro_dice"].idxmax()
            best_dice = float(df_hist.loc[best_idx, "val_macro_dice"])
            best_epoch = int(df_hist.loc[best_idx, "epoch"])

        summary_records.append({
            "scenario": scenario,
            "best_epoch": best_epoch,
            "val_macro_dice": best_dice,
            "checkpoint": ckpt_path,
        })

        # 3. Automatically Generate Test Set Inference Samples & Gallery
        if os.path.exists(ckpt_path):
            print(f"\n[INFERENCE] Generating {args.num_samples} test samples for {exp_name}...")
            infer_cmd = [
                args.python_bin,
                "experiments/inference_sample.py",
                "--checkpoint", ckpt_path,
                "--model", args.model,
                "--encoder", args.encoder,
                "--scenario", scenario,
                "--num-samples", str(args.num_samples),
            ]
            subprocess.run(infer_cmd)

    # Final summary table
    print("\n" + "=" * 80)
    print("ALL SCENARIOS COMPLETED! SUMMARY TABLE:")
    print("=" * 80)
    df_summary = pd.DataFrame(summary_records)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
