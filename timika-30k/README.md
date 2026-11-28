# Timika-30k

Multi-label chest X-ray disease segmentation (7 findings: atelectasis, cavitation, infiltrate,
lymphadenopathy, pleural effusion, pneumothorax, TB lesion) with U-Net++, FPN and SegFormer, trained on
three preprocessing scenarios (`clahe`, `bone_suppressed`, `bone_suppression_clahe`).

The images and masks are a separate, large dataset that is **not** stored in git; see
[docs/dataset/README.md](docs/dataset/README.md) for how it was assembled. Results are in
[docs/REPORT.md](docs/REPORT.md) and [docs/TRAINING_RESULTS.md](docs/TRAINING_RESULTS.md).

## Layout

```
.
├── timika/                   Python package
│   ├── config.py             paths, class list, scenarios, hyperparameter defaults
│   ├── datasets/             TimikaDiseaseDataset (patient-level split, partial-annotation masks), augmentations
│   ├── models/               build_model(): U-Net++, FPN, SegFormer via segmentation_models_pytorch
│   ├── losses/               masked BCE + Dice, masked focal Tversky
│   ├── metrics.py            validation metric tracker
│   ├── visualization.py      colour overlays and 3-panel (input | ground truth | prediction) cards
│   ├── inference_schema.py   the four runs defined by docs/inference_schema_ghozy-261125.md
│   └── cli/                  command-line entry points (see Usage)
├── tests/                    smoke tests (no data or checkpoints needed)
├── docs/                     reports, inference schema, dataset card, paper
├── splits/                   image-id lists (e.g. the 553 ChestX-Det test images)
├── checkpoints/              <model>_<encoder>_<scenario>/  best_model.pth (git-ignored) + training_history.csv
├── outputs/                  inference samples, schema runs, nnU-Net raw data (bulk files git-ignored)
├── data/  labels/  preprocessed_*/  preprocessed_labels/     dataset (git-ignored)
├── pyproject.toml  requirements.txt  .gitignore
```

## Setup

```
pip install -r requirements.txt     # install the CUDA build of PyTorch first, see the note in the file
pip install -e .                    # optional: adds the timika-* commands and makes `timika` importable anywhere
python -m pytest                    # optional smoke tests (needs: pip install pytest)
```

The dataset folders (`data/`, `labels/`, `preprocessed_*/`, `preprocessed_labels/`) must sit in this folder.

## Usage

Run everything **from this folder** (the project root) as `python -m timika.cli.<name>`. Paths are resolved from
the package location, so checkpoints and outputs always go to `checkpoints/` and `outputs/`.

| Task | Command |
|---|---|
| Train one model on one scenario | `python -m timika.cli.train --model segformer --scenario bone_suppressed` |
| Resume training | `python -m timika.cli.train --model segformer --scenario bone_suppressed --resume checkpoints/segformer_mit_b3_bone_suppressed/best_model.pth` |
| Train a model on all scenarios, then sample test inference | `python -m timika.cli.train_all --model unetplusplus` |
| Sample test images with cards and an HTML gallery | `python -m timika.cli.infer_samples --model unetplusplus --scenario bone_suppressed --checkpoint checkpoints/unetplusplus_resnet34_bone_suppressed/best_model.pth` |
| Schema inference: 4 runs on the 553 ChestX-Det test images | `python -m timika.cli.schema_infer` |
| Evaluate the schema runs against ground truth | `python -m timika.cli.schema_evaluate` |
| Overlays (input / ground truth / prediction) for the schema runs | `python -m timika.cli.schema_overlay` |
| Export a scenario as an nnU-Net v2 raw dataset | `python -m timika.cli.prepare_nnunet --scenario bone_suppression_clahe` |

Add `--help` to any command for its options. With `pip install -e .` the same commands are available as
`timika-train`, `timika-train-all`, `timika-infer-samples`, `timika-schema-infer`, `timika-schema-evaluate`,
`timika-schema-overlay` and `timika-prepare-nnunet`.

Schema commands write to `outputs/inferensi_ghozy-261125/<run>/` (`predicted_masks/`, `probabilitas_maks_sigmoid.csv`,
`info_inferensi.json`, and with the commands above `overlay/`), plus a combined `info_inferensi.csv`.

## What is tracked in git

Source, docs, `splits/`, `training_history.csv` logs and the small summaries in `outputs/` (CSV, JSON, HTML) are
tracked. Model weights, the dataset, per-image masks, overlays, and archives are ignored (see `.gitignore`).
Do not name a source folder after a rule in the repository-root `.gitignore` (`data/`, `labels/`, `nnunet/`, `lib/`,
`build/`, ...): its code would be silently ignored. This is why the package is `timika/datasets/`, not `timika/data/`.
