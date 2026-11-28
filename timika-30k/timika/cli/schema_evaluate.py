"""
Evaluasi hasil inferensi skema_inferensi_ghozy-261125 terhadap ground truth.

Membaca keluaran dari timika/cli/schema_infer.py:
    <output-dir>/<nama-inferensi>/predicted_masks/<id_citra>/<kelas>.png
    <output-dir>/<nama-inferensi>/probabilitas_maks_sigmoid.csv
dan membandingkannya dengan mask ground truth di preprocessed_labels/disease/<kelas>/<id_citra>.

Metrik per kelas (empat kelas yang disimpan) dan rata-rata makro:
    dice, iou, precision, recall   dihitung dari TP/FP/FN yang dijumlahkan pada seluruh citra
                                   (definisi yang sama dengan validasi pada timika/metrics.py)
    dice_per_citra, iou_per_citra  rata-rata Dice / IoU per citra pada seluruh citra
                                   (citra tanpa temuan dan tanpa prediksi bernilai 1.0)
    dice_per_citra_positif         rata-rata Dice per citra hanya pada citra yang memiliki temuan pada GT
    sens_citra, spes_citra         tingkat citra: prediksi tidak kosong vs. GT tidak kosong
    auroc_citra                    tingkat citra: nilai sigmoid tertinggi vs. GT tidak kosong

Hasil ditulis ke:
    <output-dir>/<nama-inferensi>/info_inferensi.json   (kunci "metrik_evaluasi")
    <output-dir>/info_inferensi.csv                     (satu baris per inferensi dan kelas, plus baris MAKRO)

Contoh (dijalankan dari root proyek):
    python -m timika.cli.schema_evaluate
    python -m timika.cli.schema_evaluate --runs 1 2
"""

import argparse
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from timika.config import LABELS_DISEASE_DIR
from timika.inference_schema import DEFAULT_OUTPUT_DIR, RUNS, SAVED_CLASSES

SMOOTH = 1e-5  # sama dengan timika/metrics.py
METRIC_KEYS = [
    "dice", "iou", "precision", "recall",
    "dice_per_citra", "iou_per_citra", "dice_per_citra_positif",
    "sens_citra", "spes_citra", "auroc_citra",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi hasil inferensi skema_inferensi_ghozy-261125")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--runs", type=int, nargs="+", default=[r.number for r in RUNS], choices=[r.number for r in RUNS])
    return p.parse_args()


def read_binary(path: str) -> np.ndarray:
    """Baca PNG mask -> (512, 512) bool. Mask ground truth diubah ke 512x512 (nearest) bila ukurannya berbeda."""
    arr = np.array(Image.open(path).convert("L")) > 127
    if arr.shape != (512, 512):
        arr = cv2.resize(arr.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    return arr


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC (Mann-Whitney U) dengan peringkat rata-rata untuk nilai kembar. NaN bila hanya ada satu kelas."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_run(run_dir: str) -> Dict:
    prob_df = pd.read_csv(os.path.join(run_dir, "probabilitas_maks_sigmoid.csv"))
    ids = prob_df["id_citra"].astype(str).tolist()
    masks_root = os.path.join(run_dir, "predicted_masks")

    per_class = {}
    for cls in SAVED_CLASSES:
        tp = fp = fn = 0.0
        dice_img, iou_img, gt_pos, pred_pos = [], [], [], []
        for image_id in ids:
            pred = read_binary(os.path.join(masks_root, os.path.splitext(image_id)[0], f"{cls}.png"))
            gt = read_binary(os.path.join(LABELS_DISEASE_DIR, cls, image_id))
            inter = float(np.logical_and(pred, gt).sum())
            n_pred, n_gt = float(pred.sum()), float(gt.sum())
            tp += inter
            fp += n_pred - inter
            fn += n_gt - inter
            dice_img.append((2.0 * inter + SMOOTH) / (n_pred + n_gt + SMOOTH))
            iou_img.append((inter + SMOOTH) / (n_pred + n_gt - inter + SMOOTH))
            gt_pos.append(n_gt > 0)
            pred_pos.append(n_pred > 0)

        gt_pos, pred_pos = np.array(gt_pos), np.array(pred_pos)
        dice_img, iou_img = np.array(dice_img), np.array(iou_img)
        n_pos = int(gt_pos.sum())
        n_neg = len(ids) - n_pos
        per_class[cls] = {
            "n_citra": len(ids),
            "n_citra_positif": n_pos,
            "dice": (2.0 * tp + SMOOTH) / (2.0 * tp + fp + fn + SMOOTH),
            "iou": (tp + SMOOTH) / (tp + fp + fn + SMOOTH),
            "precision": (tp + SMOOTH) / (tp + fp + SMOOTH),
            "recall": (tp + SMOOTH) / (tp + fn + SMOOTH),
            "dice_per_citra": float(dice_img.mean()),
            "iou_per_citra": float(iou_img.mean()),
            "dice_per_citra_positif": float(dice_img[gt_pos].mean()) if n_pos else float("nan"),
            "sens_citra": float((pred_pos & gt_pos).sum() / n_pos) if n_pos else float("nan"),
            "spes_citra": float((~pred_pos & ~gt_pos).sum() / n_neg) if n_neg else float("nan"),
            "auroc_citra": auroc(prob_df[cls].to_numpy(dtype=np.float64), gt_pos),
        }

    macro = {k: float(np.nanmean([per_class[c][k] for c in SAVED_CLASSES])) for k in METRIC_KEYS}
    return {"jumlah_citra": len(ids), "per_kelas": per_class, "makro": macro}


def main():
    args = parse_args()
    specs = [r for r in RUNS if r.number in args.runs]

    csv_rows: List[Dict] = []
    for spec in specs:
        run_dir = os.path.join(args.output_dir, spec.name)
        if not os.path.isfile(os.path.join(run_dir, "probabilitas_maks_sigmoid.csv")):
            raise SystemExit(f"Keluaran inferensi {spec.number} tidak ditemukan di {run_dir}. Jalankan timika.cli.schema_infer dahulu.")

        print(f"Mengevaluasi inferensi {spec.number}: {spec.name} ...")
        result = evaluate_run(run_dir)

        # Tulis ke info_inferensi.json milik inferensi tersebut
        info_path = os.path.join(run_dir, "info_inferensi.json")
        info = {}
        if os.path.isfile(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        info["metrik_evaluasi"] = result
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)  # NaN ditulis sebagai NaN

        meta = {
            "nomor": spec.number,
            "inferensi": spec.name,
            "model": spec.model,
            "encoder": spec.encoder,
            "bentuk_citra_masukan": spec.input_form,
            "checkpoint_epoch": info.get("checkpoint_epoch"),
        }
        for cls in SAVED_CLASSES:
            csv_rows.append({**meta, "kelas": cls, **result["per_kelas"][cls]})
        csv_rows.append({
            **meta, "kelas": "MAKRO", "n_citra": result["jumlah_citra"], "n_citra_positif": np.nan, **result["makro"],
        })

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.output_dir, "info_inferensi.csv")
    df.to_csv(csv_path, index=False)

    # Ringkasan di layar
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = df[["nomor", "inferensi", "kelas", "n_citra_positif", "dice", "iou", "precision", "recall",
               "dice_per_citra", "sens_citra", "spes_citra", "auroc_citra"]].copy()
    print("\n" + show.round(4).to_string(index=False))
    print(f"\nTersimpan: {csv_path}")
    print("Metrik per inferensi juga ditambahkan ke info_inferensi.json (kunci 'metrik_evaluasi').")


if __name__ == "__main__":
    main()
