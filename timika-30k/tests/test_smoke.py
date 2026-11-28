"""
Smoke tests for the restructured project. They need neither the image data nor the checkpoints,
so they run on any machine:  python -m pytest
"""

import importlib
import os
import pkgutil

import numpy as np
import pandas as pd
import pytest
import torch

import timika
from timika import config, inference_schema
from timika.cli.schema_evaluate import auroc
from timika.cli.schema_infer import remove_small_components
from timika.datasets.transforms import get_transforms
from timika.losses.compound_loss import MaskedMultiLabelBCEDiceLoss
from timika.models.factory import build_model
from timika.visualization import render_composite_card


def _all_modules():
    return [m.name for m in pkgutil.walk_packages(timika.__path__, prefix="timika.")]


@pytest.mark.parametrize("name", _all_modules())
def test_every_module_imports(name):
    importlib.import_module(name)


def test_every_cli_has_main():
    for name in _all_modules():
        if name.startswith("timika.cli.") and not name.endswith("__init__"):
            assert callable(getattr(importlib.import_module(name), "main", None)), name


def test_paths_are_under_project_root():
    root = config.PROJECT_ROOT
    assert os.path.isdir(os.path.join(root, "timika"))
    for path in (config.CHECKPOINTS_DIR, config.OUTPUTS_DIR, config.SPLITS_DIR, config.DOCS_DIR):
        assert os.path.commonpath([root, path]) == root
    assert config.experiment_name("segformer", "mit_b3", "bone_suppressed") == "segformer_mit_b3_bone_suppressed"


def test_inference_schema_is_consistent():
    runs = inference_schema.RUNS
    assert [r.number for r in runs] == [1, 2, 3, 4]
    assert len({r.name for r in runs}) == len(runs)
    assert all(r.scenario in config.SCENARIOS for r in runs)
    assert set(inference_schema.SAVED_CLASSES) <= set(config.DISEASE_CLASSES)
    for r in runs:
        assert os.path.commonpath([config.CHECKPOINTS_DIR, inference_schema.checkpoint_path(r)]) == config.CHECKPOINTS_DIR


def test_test_image_list_is_the_553_chestxdet_images():
    ids = pd.read_csv(inference_schema.DEFAULT_ID_LIST)["id_citra"].astype(str)
    assert len(ids) == 553
    assert ids.is_unique


def test_remove_small_components():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[2:4, 2:4] = 1      # 4 px  -> removed
    mask[20:40, 20:40] = 1  # 400 px -> kept
    out = remove_small_components(mask, min_size=30)
    assert out[3, 3] == 0 and out[30, 30] == 1 and out.sum() == 400


def test_auroc():
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_transforms_resize_and_normalise():
    img = np.random.randint(0, 255, (300, 200), dtype=np.uint8)
    mask = np.zeros((300, 200, 7), dtype=np.float32)
    out = get_transforms("test", (512, 512))(image=img, mask=mask)
    assert tuple(out["image"].shape) == (1, 512, 512)
    assert tuple(out["mask"].shape) == (512, 512, 7)
    assert -1.0 <= float(out["image"].min()) and float(out["image"].max()) <= 1.0


def test_masked_loss_ignores_unannotated_classes():
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 16, 16, requires_grad=True)
    targets = (torch.rand(2, 7, 16, 16) > 0.5).float()
    annotated = torch.zeros(2, 7)
    annotated[:, 2] = 1.0
    loss = MaskedMultiLabelBCEDiceLoss()(logits, targets, annotated)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[:, 2].abs().sum() > 0
    assert logits.grad[:, [0, 1, 3, 4, 5, 6]].abs().sum() == 0


@pytest.mark.parametrize("model,encoder", [("unetplusplus", "resnet34"), ("segformer", "mit_b3")])
def test_models_forward_shape(model, encoder):
    net = build_model(model_name=model, encoder_name=encoder, encoder_weights=None, in_channels=1,
                      num_classes=len(config.DISEASE_CLASSES)).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 1, 64, 64))
    assert tuple(out.shape) == (1, len(config.DISEASE_CLASSES), 64, 64)


def test_render_composite_card_shape():
    base = np.full((512, 512), 128, dtype=np.uint8)
    gt = np.zeros((len(config.DISEASE_CLASSES), 512, 512), dtype=np.float32)
    pred = gt.copy()
    gt[0, 100:200, 100:200] = 1
    card = render_composite_card(base, gt, pred, "x.png", "p", "chestxdet", {"atelectasis_dice": 0.0})
    assert card.ndim == 3 and card.shape[2] == 3 and card.dtype == np.uint8
