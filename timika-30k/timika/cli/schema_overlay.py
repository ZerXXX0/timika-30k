"""
Overlay prediksi vs. ground truth untuk hasil inferensi skema_inferensi_ghozy-261125.

Untuk setiap citra pada setiap inferensi dibuat satu gambar tiga panel:
    [Citra masukan] | [Ground truth] | [Prediksi model]
dengan warna per kelas (atelectasis, infiltrate, pleural_effusion, pneumothorax), kontur, dan
nilai Dice per kelas pada legenda bagian bawah.

Membaca keluaran timika/cli/schema_infer.py dan menulis ke:
    <output-dir>/<nama-inferensi>/overlay/<id_citra>.png

Contoh (dijalankan dari root proyek):
    python -m timika.cli.schema_overlay
    python -m timika.cli.schema_overlay --runs 1 --limit 10
    python -m timika.cli.schema_overlay --format jpg        # berkas jauh lebih kecil
"""

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from timika.config import DISEASE_CLASSES, LABELS_DISEASE_DIR, SCENARIOS
from timika.inference_schema import DEFAULT_OUTPUT_DIR, RUNS, SAVED_CLASSES
from timika.visualization import render_composite_card

SMOOTH = 1e-5


def parse_args():
    p = argparse.ArgumentParser(description="Overlay prediksi vs ground truth")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--runs", type=int, nargs="+", default=[r.number for r in RUNS], choices=[r.number for r in RUNS])
    p.add_argument("--limit", type=int, default=None, help="Hanya N citra pertama per inferensi (uji cepat)")
    p.add_argument("--format", type=str, default="png", choices=["png", "jpg"],
                   help="png: tanpa kehilangan kualitas tetapi besar; jpg: jauh lebih kecil")
    return p.parse_args()


def read_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.shape != (512, 512):
        arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    return arr


def read_mask(path: str) -> np.ndarray:
    arr = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    if arr.shape != (512, 512):
        arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.float32)


def main():
    args = parse_args()
    manifest = pd.read_csv(SCENARIOS["bone_suppressed"]["manifest"], usecols=["id", "source", "patient_id"])
    meta = manifest.set_index("id").to_dict("index")

    for spec in [r for r in RUNS if r.number in args.runs]:
        run_dir = os.path.join(args.output_dir, spec.name)
        prob_csv = os.path.join(run_dir, "probabilitas_maks_sigmoid.csv")
        if not os.path.isfile(prob_csv):
            raise SystemExit(f"Keluaran inferensi {spec.number} tidak ditemukan di {run_dir}. Jalankan timika.cli.schema_infer dahulu.")

        ids = pd.read_csv(prob_csv)["id_citra"].astype(str).tolist()
        if args.limit is not None:
            ids = ids[: args.limit]
        image_dir = SCENARIOS[spec.scenario]["image_dir"]
        overlay_dir = os.path.join(run_dir, "overlay")
        os.makedirs(overlay_dir, exist_ok=True)

        print(f"Inferensi {spec.number}: {spec.name} -> {overlay_dir}")
        t0 = time.time()
        for n, image_id in enumerate(ids, 1):
            stem = os.path.splitext(image_id)[0]
            base = read_gray(os.path.join(image_dir, image_id))

            # Susun ke (7, 512, 512) agar cocok dengan helper render; kelas selain empat kelas tersimpan dibiarkan kosong
            gt = np.zeros((len(DISEASE_CLASSES), 512, 512), dtype=np.float32)
            pred = np.zeros_like(gt)
            metrics = {}
            for cls in SAVED_CLASSES:
                c = DISEASE_CLASSES.index(cls)
                gt[c] = read_mask(os.path.join(LABELS_DISEASE_DIR, cls, image_id))
                pred[c] = read_mask(os.path.join(run_dir, "predicted_masks", stem, f"{cls}.png"))
                inter = float((gt[c] * pred[c]).sum())
                metrics[f"{cls}_dice"] = (2.0 * inter + SMOOTH) / (float(gt[c].sum()) + float(pred[c].sum()) + SMOOTH)

            info = meta.get(image_id, {})
            card = render_composite_card(
                base_gray=base,
                gt_masks=gt,
                pred_masks=pred,
                image_id=image_id,
                patient_id=str(info.get("patient_id", "-")),
                source=str(info.get("source", "chestxdet")),
                per_class_metrics=metrics,
            )
            # Helper bekerja dalam RGB; cv2.imwrite membutuhkan BGR agar warna sesuai CLASS_COLORS
            card = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(overlay_dir, f"{stem}.{args.format}")
            if args.format == "jpg":
                cv2.imwrite(out_path, card, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                cv2.imwrite(out_path, card)

            if n % 100 == 0 or n == len(ids):
                print(f"  [{n}/{len(ids)}] ({time.time() - t0:.0f} dtk)")

    print(f"\nSelesai. Overlay ada di <output-dir>/<nama-inferensi>/overlay/ ({args.output_dir})")


if __name__ == "__main__":
    main()
