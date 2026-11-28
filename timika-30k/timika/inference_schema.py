"""
Definition of the "Skema Inferensi Model Timika-30K" (docs/inference_schema_ghozy-261125.md).

Four inference runs on the 553 ChestX-Det test images listed in
splits/daftar_citra_uji_chestxdet-261125.csv. Shared by ``timika.cli.schema_infer``,
``timika.cli.schema_evaluate`` and ``timika.cli.schema_overlay`` so the three commands agree on run
names, checkpoints, saved classes and default locations.
"""

import os
from dataclasses import dataclass
from typing import List

from timika.config import CHECKPOINTS_DIR, DISEASE_CLASSES, OUTPUTS_DIR, SPLITS_DIR, experiment_name

# Only these four classes are written to disk (the schema's requirement).
SAVED_CLASSES: List[str] = ["atelectasis", "infiltrate", "pleural_effusion", "pneumothorax"]
SAVED_IDX: List[int] = [DISEASE_CLASSES.index(c) for c in SAVED_CLASSES]

DEFAULT_ID_LIST = os.path.join(SPLITS_DIR, "daftar_citra_uji_chestxdet-261125.csv")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "inferensi_ghozy-261125")


@dataclass(frozen=True)
class RunSpec:
    number: int
    name: str
    model: str
    encoder: str
    scenario: str  # key in timika.config.SCENARIOS
    input_form: str


RUNS: List[RunSpec] = [
    RunSpec(1, "1_segformer_bone", "segformer", "mit_b3", "bone_suppressed", "Citra bone"),
    RunSpec(2, "2_unetpp_bone", "unetplusplus", "resnet34", "bone_suppressed", "Citra bone"),
    RunSpec(3, "3_unetpp_contrast", "unetplusplus", "resnet34", "clahe", "Citra contrast"),
    RunSpec(4, "4_unetpp_bone_contrast", "unetplusplus", "resnet34", "bone_suppression_clahe", "Citra bone and contrast"),
]


def checkpoint_path(spec: RunSpec) -> str:
    """Best-epoch checkpoint of the model trained for this run."""
    return os.path.join(CHECKPOINTS_DIR, experiment_name(spec.model, spec.encoder, spec.scenario), "best_model.pth")
