# timika-50k

Combined chest X-ray dataset for the Timika score pipeline, assembled from
seven already-canonical datasets in this store, pseudolabeled with two
segmentation label types: organ region (6 lung zones) and disease. Not a
downloaded dataset, no single paper or source, see "Provenance" below.

## Made from

51,659 images pulled from seven of this store's own datasets, each kept in
its own subfolder under `data/` with original filenames intact so every
image stays traceable to its source.

| Source | Images in timika-50k | Images in its own canonical copy |
|---|---|---|
| Shenzhen chest X-ray set | 662 | 662 |
| Montgomery County chest X-ray set | 138 | 138 |
| TBX11K | 11,490 | 12,278 |
| ChestX-Det | 3,577 | 3,578 |
| COVID-19 Radiography Database | 21,165 | 21,165 |
| BIMCV-CAAXR | 2,539 | 2,591 |
| SIIM-ACR Pneumothorax | 12,088 | 12,089 |

TBX11K, ChestX-Det, BIMCV-CAAXR, and SIIM-ACR Pneumothorax are the only
sources that aren't a full copy of their own dataset, see "Invalid and
duplicate handling" below for why.

Three more of this store's datasets were considered and excluded entirely:
`chestxray_pneumonia_covid_tb` (7,135 images), `chestxray_covid19_pneumonia`
(6,432), and `tb4200` (4,200). All three are classification-only, folder-
encoded labels with no bounding box, mask, or polygon anywhere, nothing to
seed a pseudolabel from.

## Invalid and duplicate handling

This project's standing policy: data-quality problems get marked, never
deleted. Nothing described here was removed from any source dataset, files
just weren't copied into timika-50k's own `data/`.

- TBX11K's `data/extra/` folder (576 images) was dropped entirely before
  copying. It repackages Montgomery and Shenzhen images already copied
  separately here under their own names, plus an unvetted external source
  ("DA+DB") that was never independently registered or licensed-checked.
  TBX11K's own manifest already documents this folder as bonus augmentation.
- 216 of TBX11K's remaining images were copied, then removed from this
  copy specifically, after confirming each one against
  `C:\dataset\tbx11k\labels\quality_flags.csv.bz2`: internal pixel-duplicates
  within TBX11K itself, 73 of the 216 pairs straddle TBX11K's own train/test
  split. TBX11K's canonical copy still has all of them; only timika-50k's
  copy is missing the 216.
- Shenzhen, Montgomery, and COVID-19 Radiography Database had no confirmed
  duplicates as of the last project-wide dedup sweep, so every image from
  those three was copied.
- 1 of ChestX-Det's images was excluded, a confirmed train/test leakage
  duplicate (`test/41518.png` pixel-identical to `train/41520.png`) per
  `E:\dataset\chestxdet\labels\quality_flags.csv.bz2`. The two copies
  carried different annotations, test's 8 findings versus train's 5;
  test/41518.png (the more complete label) was kept, train's copy excluded.
- 52 of BIMCV-CAAXR's images were excluded, confirmed internal
  near-duplicates (repeat acquisitions, same subject/session) per
  `E:\dataset\bimcv_caaxr\labels\quality_flags.csv.bz2`.
- 1 of SIIM-ACR Pneumothorax's images was excluded, a confirmed train/test
  leakage duplicate (a test-split image pixel-identical to a train-split
  image) per `E:\dataset\siim_acr_pneumothorax\labels\quality_flags.csv.bz2`.
  A second internal duplicate pair exists (two independently-drawn
  pneumothorax masks on the same underlying image, IoU 0.068) and was kept,
  not excluded, both entries carry real, disagreeing annotations, not a
  redundant copy.
- BIMCV-CAAXR and SIIM-ACR Pneumothorax were each also checked against the
  5 original sources and against each other: 0 further cross-source matches
  beyond the exclusions above.

Full per-source counts and the dedup methodology: dataset/manifest.md in
research-cxr-timika, and the "CXR Dataset Roles" figure,
`https://claude.ai/code/artifact/a386363f-d55d-404a-8ba9-399ad4351551`.

## Eligibility for inclusion

Beyond the duplicate handling above, a source image only qualifies for
timika-50k at all if it already carries some form of localization
(bounding box, mask, or polygon, not a bare classification label). This is
why `chestxray_pneumonia_covid_tb`, `chestxray_covid19_pneumonia`, and
`tb4200` are excluded outright, see "Made from" above. Full breakdown by
source: the "Timika-37K Eligibility" figure,
`https://claude.ai/code/artifact/433b3f05-68d8-433f-aa86-2ea86847ae4d`.

## Labels

Two label types, meant to end up covering every image in `data/` uniformly.
Neither reuses a source dataset's own pre-existing annotations directly,
see "What each source's original labels contribute" below for why.

`labels/organ_region/`: PSPNet lung segmentation (torchxrayvision
`chestx_det.PSPNet`, combined lung Dice 0.870 on all 138 Montgomery
ground-truth masks, beating the CXAS alternative's 0.758) plus the Timika
paper's own 6-zone geometry: two horizontal lines splitting the lung field
into three equal-height bands, each band split left/right. One label PNG
per image, same relative path as its source under `data/`, single channel,
pixel values 0 (background) through 6 (zone id: 1 upper-left, 2
upper-right, 3 mid-left, 4 mid-right, 5 lower-left, 6 lower-right). Build
script: `analyses/timika50k_organ_labels/` in research-cxr-timika. Complete
as of 2026-08-30: all 51,660 images then in `data/` labeled, 0 errors (now
51,659, ChestX-Det's 1 train/test-duplicate exclusion landed after this
labeling pass, see "Invalid and duplicate handling" above; its label file
was removed along with the image, not left orphaned). The original
37,033 finished 2026-08-28; the 14,627 images added from BIMCV-CAAXR and
SIIM-ACR Pneumothorax (2026-08-29 fold-in) finished 2026-08-30, after a
real preprocessing bug fix along the way, `load_grayscale` originally
only rescaled a source image when its native pixel range exceeded 255
(fixing CAAXR's higher-bit-depth overflow), missing the converse case of
a native range far *below* 255 (observed as low as 0-64 on one CAAXR
image), which starves PSPNet of contrast just as badly. Fixed to rescale
via each image's own min/max unconditionally, a no-op for already
full-range images, a real fix for under-windowed ones; recovered 1 of the
7 images that first came back with an empty label.

Followed up with a full-dataset scan of every image's native pixel range
(`analyses/timika50k_organ_labels/scan_contrast_range.py`) to check for
more of the same under-windowing, not just trust the one case found by
accident. Found exactly 1 more: `covidrad/COVID/COVID-1876.png` (native
range 0-77), already flagged `label_failed` from the original 2026-08-28
build for this exact reason. Reprocessed the same way, recovered.
6 milder cases (native max 146-149, all `covidrad/`) exist but already
had a working non-empty mask, not touched.

`labels/organ_region/manifest.csv` (2026-09-03): one row per file, same
template as `labels/disease/manifest.csv`, method/confidence/valid/empty/
zones_present/evidence per image, matching the disease layer's own
provenance discipline rather than leaving organ_region's provenance
implicit in folder structure. 51,650 of 51,659 rows `confidence: model`
(the only method used anywhere in this layer, by design, see "What each
source's original labels contribute" below), 8 `failed`, 1 `invalid`.
The file's own `empty` column is computed from each mask's real pixel
content, not copied from `quality_flags.csv.bz2`, and cross-checked
against it at build time: 0 mismatches, all 9 currently-flagged cases
confirmed, no new ones found.

11 images total came back with an empty (all-background) label across
the full build, checked individually by hand,
`labels/organ_region/quality_flags.csv.bz2`: COVID/COVID-698.png isn't a
real diagnostic CXR (marked invalid at the source, see
`C:\dataset\covid19_radiography\manifest.yaml`); 2 (sub-S03240, COVID-1876)
were the under-windowing bug, since fixed and recovered; the remaining 8
are valid CXRs (off-center framing, portable-scan artifacts, low
partial-confidence signal below PSPNet's 0.5 threshold, or a lateral
projection PSPNet isn't validated on) that PSPNet simply failed to
segment, marked `label_failed`. None were removed, per this project's
mark-not-delete policy.

`labels/disease/`: 120,634 label files, `manifest.csv` documents every
one's provenance (image, class, method, confidence, evidence). Built in
two rounds:

**DS group, 2026-09-01**: Shenzhen, ChestX-Det, SIIM-ACR Pneumothorax,
27,631 files, real masks kept as-is, no pseudolabeler needed: cavitation
and lymphadenopathy 662 (Shenzhen only), pleural_effusion and infiltrate
4,239 (Shenzhen + ChestX-Det), atelectasis 3,577 (ChestX-Det only),
pneumothorax 14,252 (ChestX-Det + SIIM-ACR). No file where no ground
truth exists (SIIM-ACR's 1,413 unlabeled images), not a fabricated zero.

**Confirmed negatives and box masks, 2026-09-02**: closes most of the
remaining gap without Model X, the disease pseudolabeler, which still
doesn't work (see `repo/timika50k_pseudolabels/README.md`). 16,749 more
images now carry at least one disease label: 14,277 confirmed negatives
(all 6 classes, all-zero, a real label, not a placeholder) from TBX11K's
`health/` (3,800), COVID-19 Radiography Database's `Normal/` (10,192),
Montgomery's `_0` images (80), and BIMCV-CAAXR's Study-Level-Normal
images with no disease box (205 present on disk of 600 computed, the
rest are in BIMCV-CAAXR's own annotation set but were never in this
store's filtered download). Plus real boxes turned into masks via SAM
ViT-B, box-prompted, hard-intersected with the box (0.779 IoU against
ground truth on this project's own measurement, beating MedSAM's 0.635):
1,823 BIMCV-CAAXR images get a class-specific mask (infiltrate 1,530,
pleural_effusion 118, atelectasis 108, pneumothorax 7, from the 5 of 22
raw `finding_name` strings that map onto an RSHS class), 854 TBX11K
`tb/` + Montgomery images get a `tb_lesion` mask with no RSHS class
assigned (a generic TB box names no specific finding, never guessed),
and 15 of Montgomery's also get a class-specific copy where its own
free-text reading names exactly one finding.

Montgomery's box coordinate frame is undocumented in the source release;
calibrated against Shenzhen's real polygon masks (the only source with
both a same-format box file and independent pixel ground truth) and
against Montgomery's own lung masks, square-1024 non-uniform resize won
on both (91.5% of box area inside the lung field vs. 88.5% for the next
best hypothesis), used with `confidence: low` recorded on every affected
row, not silently treated as certain.

Verified independently 2026-09-02 by 3 agents against raw source data,
not this build's own log: 0 SAM masks found outside their prompting
box (one agent shrank a real box and emptied another to confirm its own
checker actually catches a violation, 71,772 and 106,762 pixels flagged);
every confirmed-negative source rule re-derived from the raw CSVs
independently, 0 mismatches; 0 rows where a generic box got assigned an
invented class. Still fully blocked on Model X: TBX11K `sick`, COVID-19
Radiography Database's 3 abnormal classes, BIMCV-CAAXR's 36
Abnormal-no-box images, both withheld test splits (TBX11K 3,216,
SIIM-ACR 1,376), and tuberculoma/bronchiectasis, no source's taxonomy
covers either.

## Preprocessing

`preprocessed/`: materializes `DiseaseSegDataset.__getitem__`'s
(`repo/timika_score/src/timika_score/training/dataset.py`) own
training-time image preprocessing to disk, flat and manifest-driven. Built
2026-09-02 at `C:\research\research-cxr-timika\dataset\timika-50k\preprocessed\`,
the project-local C: copy specifically, not mirrored here to the E:
canonical tree (the two copies are otherwise kept identical, see "Made
from" above); full method and script:
`analyses/timika50k_preprocessing/build_preprocessed.py` and its own
README in research-cxr-timika.

Grayscale load, crop, and resize match `DiseaseSegDataset` exactly for
Shenzhen and ChestX-Det, the only two sources it currently trains on:
plain `Image.open(path).convert("L")`, center-crop to square, resize to
512x512 via `torch.nn.functional.interpolate(mode="bilinear",
align_corners=False)`, the same call, not a PIL-resize approximation of
it. Every other source gets an own-image min/max rescale before that same
crop/resize, the fix already validated in this dataset's own
`labels/organ_region/` build (see "Labels" above), extended here since it
also hits an 8-bit source (COVID-19 Radiography Database), not just
CAAXR's 16-bit case: naive `.convert("L")` on a real CAAXR file gives mean
246.9/255 (washed out, near-white); the fix gives mean 140.5/255, full
0-255 range. `xrv.datasets.normalize` and `preprocess_image` (ImageNet 3ch
conversion) are deliberately left out of the static files, both stay live
at train time: normalize is a fixed linear rescale, so resizing before or
after it gives the same result, and augmentation has to run on the
not-yet-3-channel tensor before preprocess_image anyway, on the train
split only.

51,659/51,659 images written, 0 errors. Spot-checked visually and
numerically after the run: 28 images sampled across all 7 sources, real
contrast throughout (pixel std 44-97, none flat/blank), correct 512x512
shape. Montgomery's own source images carry a real blank/black scan
region in some captures (confirmed directly against the raw file, e.g.
`MCUCXR_0048_0.png`'s own pixels are exactly 0 from row 3424/4892
onward), inconsistent per image (14-21% of rows in 3 of 4 samples, 0% in
the 4th); the center-crop, matching `DiseaseSegDataset`'s own convention,
faithfully includes whatever sits in an image's vertical middle,
including this. Not fixed here, since it is a real source characteristic
and "match those params" was the mandate, not a preprocessing bug; worth
knowing before training on Montgomery specifically.

## Splits

Two manifests, both id -> split (train/val/test), covering all 51,659
`preprocessed/` rows: `splits_official.csv` and `splits_fair.csv`. Built
2026-09-02, script `analyses/timika50k_preprocessing/build_splits.py` in
research-cxr-timika, full method in that script's own module docstring
and its README.

**splits_official.csv** respects each source's own real official split
where one exists. ChestX-Det and SIIM-ACR Pneumothorax: real train/test
folder membership, their official test set kept exactly as-is, their
official train pool carved train:val at 8:1 (neither has an official
val). TBX11K: real official train/val list files
(`E:\dataset\tbx11k\labels\lists\TBX11K_{train,val}.txt.bz2`, the paper's
own 6,600/1,800 core-release split) plus its own withheld `test/` folder
kept as test unchanged, so TBX11K is the one source needing no carve at
all, its official split is already 3-way. Shenzhen, Montgomery, COVID-19
Radiography Database, and BIMCV-CAAXR carry no official split of any
kind, full 8:1:1 self-split.

**splits_fair.csv** ignores every official assignment, full 8:1:1
self-split for all 7 sources uniformly, a controlled comparison split
with no per-source procedural differences.

Self-splitting is patient-grouped everywhere it's used: every image
sharing a real patient_id lands in the same split, so BIMCV-CAAXR (~2.65
images/subject) can't leak one subject across train/val/test. A source
with no real per-patient identity in its filenames (SIIM-ACR's are DICOM
instance UIDs, not patient IDs) falls back to one group per image. Groups
are sorted, shuffled with a seeded RNG (seed 42), then greedily assigned
to whichever split is currently furthest below its target share.

Verified independently 2026-09-02 by 3 separate agents, checking the raw
source data directly rather than trusting this script's own log: 0
patient-leakage violations across 37,991 real patient groups in either
file; every official-split row in ChestX-Det, SIIM-ACR, and TBX11K
confirmed against real folder counts and the decompressed TBX11K list
files (1,000 TBX11K rows sampled by hand, 0 mismatches); row counts and
id sets match `preprocessed/manifest.csv` exactly in both files, 0
missing/extra/duplicated; splits_fair.csv's per-source ratios land within
0.3 percentage points of 80/10/10 everywhere. TBX11K's official test
share is 28.0% in splits_official.csv, far from 10%, a real, expected
property of its own withheld test set being that large relative to its
core release, not a bug.

## What each source's original labels contribute

Every image gets organ-region (and eventually disease) labels the same
way, fresh model inference plus the same geometric heuristic, regardless
of what that image's source dataset originally shipped. A source's own
prior localization is the eligibility gate (see above), not a shortcut
into the output label:

| Source | Its own original localization | Reused directly in timika-50k's labels? |
|---|---|---|
| Shenzhen | Lesion masks (337 of 662 images) | No, PSPNet runs on all 662 uniformly |
| Montgomery | Real, human-drawn left/right lung masks (all 138) | No, PSPNet runs on all 138 uniformly, even though this is arguably better than a model prediction |
| TBX11K | TB bounding boxes | No, boxes locate disease, not organ region; unused for the organ_region label, would matter once `labels/disease/` starts |
| ChestX-Det | 14-class organ pseudo-label layer (lungs, heart, clavicles, and others), plus 13-class disease polygons on 2,967 images | No, its 14 classes are a different anatomical taxonomy, not the Timika paper's 6-zone lung split, and would need re-deriving either way |
| COVID-19 Radiography Database | Lung region masks (all 21,165) | No, PSPNet runs on all 21,165 uniformly |
| BIMCV-CAAXR | Bounding boxes (`BoundingBoxes.csv`, 22 raw finding-name strings, not a clean taxonomy) plus study-level annotations | No, disease-relevant, not organ-region; unused for organ_region, would matter once `labels/disease/` starts. organ_region coverage itself is pending, see "Labels" above |
| SIIM-ACR Pneumothorax | Real radiologist pixel masks, pneumothorax only (2,379 of 12,088 positive) | No, disease-specific, not organ-region; unused for organ_region, would matter once `labels/disease/` starts. organ_region coverage itself is pending, see "Labels" above |

This is a deliberate choice, one method across the whole dataset keeps the
organ_region label consistent in style and quality regardless of source,
rather than a patchwork of real ground truth for some images and model
predictions for others. Montgomery's real masks and ChestX-Det's own organ
layer both still exist in their own canonical copies if a future revision
wants to compare against them or swap them in.

## Provenance

Derived, not downloaded: no single `{name}.zip` archival copy and no single
`paper`/`source` for `manifest.yaml` in the usual sense, this store's own
naming convention doesn't yet have a resolved answer for generated/derived
content (flagged under "Open / not decided" in
`C:\dataset\context\naming-convention.md`). `manifest.yaml` and the
`dataset/manifest.md` registration row are both still open, deferred until
composition is final.

## Known issues fixed

ChestX-Det's own `data/train/`+`data/test/` files carried a doubled
`.png.png` extension in the canonical store, mismatching its own JSON
labels (which always used the correct single extension). Fixed 2026-08-28,
renamed at the source and in this dataset's copy, full detail in
`C:\dataset\chestxdet\manifest.yaml`.

BIMCV-CAAXR and SIIM-ACR Pneumothorax hit the same doubled-extension bug in
`general-compression`, root-caused and fixed in the tool itself 2026-08-29.
Fixed at the source in both cases before copying into this dataset, full
detail in each one's own `manifest.yaml`.
