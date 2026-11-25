"""
Inferensi sesuai "Skema Inferensi Model Timika-30K" (for prompt/skema_inferensi_ghozy-261125.md).

Empat inferensi pada 553 citra uji ChestX-Det (daftar id pada
for prompt/daftar_citra_uji_chestxdet-261125.csv):

    1  SegFormer (mit_b3)       citra bone              -> preprocessed_bone_suppressed/images
    2  U-Net++ (resnet34)       citra bone              -> preprocessed_bone_suppressed/images
    3  U-Net++ (resnet34)       citra contrast (CLAHE)  -> preprocessed_clahe/images
    4  U-Net++ (resnet34)       citra bone and contrast -> preprocessed_bone_suppression_clahe/images

Pengaturan: checkpoint epoch terbaik (best_model.pth), masukan 512x512 satu kanal abu-abu,
ambang sigmoid 0.5, komponen terhubung < 30 piksel dihapus, fp16, dan hanya empat kelas
yang disimpan: atelectasis, infiltrate, pleural_effusion, pneumothorax.

Keluaran per inferensi (di dalam <output-dir>/<nama-inferensi>/):
    predicted_masks/<id_citra>/<kelas>.png   PNG biner 512x512 (0 / 255), satu per citra dan per kelas
    probabilitas_maks_sigmoid.csv            satu baris per citra, nilai sigmoid tertinggi per kelas
    info_inferensi.json                      pengaturan dan checkpoint yang dipakai

Contoh:
    python experiments/inference_skema_ghozy.py                    # keempat inferensi
    python experiments/inference_skema_ghozy.py --runs 1 3         # hanya inferensi 1 dan 3
    python experiments/inference_skema_ghozy.py --limit 8          # uji cepat pada 8 citra pertama
"""

import os
import sys

# Pastikan root proyek ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.config import DISEASE_CLASSES, PROJECT_ROOT, SCENARIOS
from experiments.data.transforms import get_transforms
from experiments.models.factory import build_model

SAVED_CLASSES = ["atelectasis", "infiltrate", "pleural_effusion", "pneumothorax"]
SAVED_IDX = [DISEASE_CLASSES.index(c) for c in SAVED_CLASSES]

DEFAULT_ID_LIST = os.path.join(PROJECT_ROOT, "for prompt", "daftar_citra_uji_chestxdet-261125.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "outputs", "inferensi_ghozy-261125")


@dataclass(frozen=True)
class RunSpec:
    number: int
    name: str
    model: str
    encoder: str
    scenario: str  # kunci pada config.SCENARIOS
    input_form: str


RUNS = [
    RunSpec(1, "1_segformer_bone", "segformer", "mit_b3", "bone_suppressed", "Citra bone"),
    RunSpec(2, "2_unetpp_bone", "unetplusplus", "resnet34", "bone_suppressed", "Citra bone"),
    RunSpec(3, "3_unetpp_contrast", "unetplusplus", "resnet34", "clahe", "Citra contrast"),
    RunSpec(4, "4_unetpp_bone_contrast", "unetplusplus", "resnet34", "bone_suppression_clahe", "Citra bone and contrast"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Inferensi sesuai skema_inferensi_ghozy-261125")
    p.add_argument("--id-list", type=str, default=DEFAULT_ID_LIST, help="CSV daftar id citra uji (kolom id_citra)")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--runs", type=int, nargs="+", default=[r.number for r in RUNS], choices=[r.number for r in RUNS],
                   help="Nomor inferensi yang dijalankan (default: semua)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 disarankan di Windows)")
    p.add_argument("--threshold", type=float, default=0.5, help="Ambang keluaran sigmoid")
    p.add_argument("--min-pixel-size", type=int, default=30, help="Komponen terhubung yang lebih kecil dihapus")
    p.add_argument("--limit", type=int, default=None, help="Hanya proses N citra pertama (uji cepat)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def checkpoint_path(spec: RunSpec) -> str:
    exp_name = f"{spec.model}_{spec.encoder}_{spec.scenario}"
    return os.path.join(PROJECT_ROOT, "experiments", "checkpoints", exp_name, "best_model.pth")


class IdListDataset(Dataset):
    """Memuat citra berdasarkan id, dengan praproses yang sama seperti split uji pada pelatihan."""

    def __init__(self, ids: List[str], image_dir: str):
        self.ids = ids
        self.image_dir = image_dir
        # Resize 512x512 (bilinear) lalu normalisasi mean=0.5, std=0.5 -> tensor (1, 512, 512)
        self.transforms = get_transforms(split="test", image_size=(512, 512))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        image = Image.open(os.path.join(self.image_dir, image_id)).convert("L")
        image_np = np.array(image, dtype=np.uint8)
        return self.transforms(image=image_np)["image"], image_id


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """mask: (H, W) uint8 {0,1}. Hapus komponen terhubung (8-konektivitas) dengan luas < min_size piksel."""
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False  # label 0 = latar belakang
    return keep[labels].astype(np.uint8)


def load_model(spec: RunSpec, ckpt_path: str, device: str):
    # encoder_weights=None: bobot dimuat dari checkpoint, tidak perlu mengunduh bobot ImageNet.
    model = build_model(
        model_name=spec.model,
        encoder_name=spec.encoder,
        encoder_weights=None,
        in_channels=1,
        num_classes=len(DISEASE_CLASSES),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    cfg = ckpt.get("config", {}) or {}
    for key, expected in (("model", spec.model), ("scenario", spec.scenario)):
        if key in cfg and cfg[key] != expected:
            print(f"  [PERINGATAN] checkpoint dilatih dengan {key}={cfg[key]!r}, inferensi memakai {expected!r}")
    return model, {"epoch": ckpt.get("epoch"), "best_macro_dice": ckpt.get("best_macro_dice")}


def run_one(spec: RunSpec, ids: List[str], args) -> None:
    ckpt_path = checkpoint_path(spec)
    image_dir = SCENARIOS[spec.scenario]["image_dir"]
    out_dir = os.path.join(args.output_dir, spec.name)
    masks_root = os.path.join(out_dir, "predicted_masks")
    os.makedirs(masks_root, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Inferensi {spec.number}: {spec.model} ({spec.encoder}) | {spec.input_form}")
    print(f"Folder citra : {image_dir}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Keluaran     : {out_dir}")
    print(f"{'=' * 70}")

    model, ckpt_info = load_model(spec, ckpt_path, args.device)
    print(f"Checkpoint dimuat: epoch {ckpt_info['epoch']}, val Macro Dice {ckpt_info['best_macro_dice']}")

    loader = DataLoader(
        IdListDataset(ids, image_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    use_amp = args.device.startswith("cuda")
    rows = []
    t0 = time.time()

    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(args.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits[:, SAVED_IDX].float())  # (B, 4, 512, 512)

            max_prob = probs.amax(dim=(2, 3)).cpu().numpy()                  # (B, 4)
            binary = (probs > args.threshold).to(torch.uint8).cpu().numpy()  # (B, 4, 512, 512)

            for b, image_id in enumerate(batch_ids):
                stem = os.path.splitext(image_id)[0]
                sample_dir = os.path.join(masks_root, stem)
                os.makedirs(sample_dir, exist_ok=True)
                for k, cls_name in enumerate(SAVED_CLASSES):
                    mask = remove_small_components(binary[b, k], args.min_pixel_size)
                    cv2.imwrite(os.path.join(sample_dir, f"{cls_name}.png"), mask * 255)
                rows.append({"id_citra": image_id, **{c: float(max_prob[b, k]) for k, c in enumerate(SAVED_CLASSES)}})

            if len(rows) % 80 < len(batch_ids) or len(rows) == len(ids):
                print(f"  [{len(rows)}/{len(ids)}] citra diproses ({time.time() - t0:.0f} dtk)")

    csv_path = os.path.join(out_dir, "probabilitas_maks_sigmoid.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(os.path.join(out_dir, "info_inferensi.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "nomor": spec.number,
                "model": spec.model,
                "encoder": spec.encoder,
                "bentuk_citra_masukan": spec.input_form,
                "folder_citra": image_dir,
                "checkpoint": ckpt_path,
                "checkpoint_epoch": ckpt_info["epoch"],
                "checkpoint_val_macro_dice": ckpt_info["best_macro_dice"],
                "jumlah_citra": len(rows),
                "ukuran_masukan": [512, 512],
                "ambang_sigmoid": args.threshold,
                "komponen_dihapus_kurang_dari_piksel": args.min_pixel_size,
                "presisi": "fp16" if use_amp else "fp32",
                "kelas_disimpan": SAVED_CLASSES,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Selesai inferensi {spec.number}: {len(rows)} citra dalam {time.time() - t0:.0f} dtk")
    print(f"  Mask : {masks_root}")
    print(f"  CSV  : {csv_path}")

    del model
    if use_amp:
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    specs = [r for r in RUNS if r.number in args.runs]

    ids = pd.read_csv(args.id_list)["id_citra"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Daftar id citra mengandung duplikat.")
    if args.limit is not None:
        ids = ids[: args.limit]
    print(f"Daftar id: {len(ids)} citra dari {args.id_list}")

    # Periksa semua prasyarat sebelum memulai inferensi pertama
    problems = []
    for spec in specs:
        if not os.path.isfile(checkpoint_path(spec)):
            problems.append(f"Inferensi {spec.number}: checkpoint tidak ada: {checkpoint_path(spec)}")
        image_dir = SCENARIOS[spec.scenario]["image_dir"]
        missing = [i for i in ids if not os.path.isfile(os.path.join(image_dir, i))]
        if missing:
            problems.append(f"Inferensi {spec.number}: {len(missing)} citra tidak ada di {image_dir} (contoh: {missing[:3]})")
    if problems:
        raise SystemExit("Prasyarat tidak terpenuhi:\n  " + "\n  ".join(problems))

    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name(0)}: {free / 2**30:.1f} GiB bebas dari {total / 2**30:.1f} GiB")

    os.makedirs(args.output_dir, exist_ok=True)
    for spec in specs:
        run_one(spec, ids, args)

    print(f"\nSEMUA SELESAI. Keluaran di: {args.output_dir}")


if __name__ == "__main__":
    main()
