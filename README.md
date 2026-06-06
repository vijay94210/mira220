# Mira220 RGB-IR Reflectance Pipeline

This repository has one supported workflow:

```powershell
python -m mira220 fit
python -m mira220 process
python -m mira220 process-session
python -m mira220 inspect
```

The default model is `config/models/flat_patch_v1.yaml`. It processes Mira220
RAW files with openRGB-IR using IRC cut `0.0`, AWB disabled, and CCM disabled,
then applies the fitted post-demosaic Red, Green, and IR correction.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The default openRGB-IR checkout is:

```text
C:\Users\jayal\git\openRGB-IR
```

Override it with `--openrgbir-repo` when necessary.

## Process RAW Images

Put unpacked 1600x1400 uint16 Mira220 RAWs in `data/raw/`, then run:

```powershell
python -m mira220 process
```

Process another directory:

```powershell
python -m mira220 process "D:\captures" --recursive --run-id survey-001
```

Results are written to `results/<run-id>/<raw-name>/`. Incompatible RAW sizes
are skipped and recorded in `run_summary.json`.

Every processed capture writes calibrated reflectance TIFFs, `ndvi.tiff`,
`ndvi_gray.png`, `ndvi_false_color.png`, `rgn_preview.png`, and
`rgb_preview.png`. ArUco target detection is not required for this base
processing path.

## Process A Session With Optional Target Adjustment

Use `process-session` when a run may contain the ArUco marker calibration
target and you want both the validated model output and an experimental
session-affine variant:

```powershell
python -m mira220 process-session "D:\captures" --recursive --run-id survey-001
```

Results are written to:

```text
results/<run-id>/validated/<raw-name>/
results/<run-id>/affine-adjusted/<raw-name>/
```

`validated/` is always produced from `flat_patch_v1` and does not require a
marker. `affine-adjusted/` is produced only when at least one processed capture
contains a detectable target. The affine fit uses all measured target patches;
Mapir images are not used for this adjustment. Markerless captures in the same
run inherit the session affine correction when the fit succeeds.

## Fit a Model

```powershell
python -m mira220 fit `
  --mira-raw "C:\path\to\mira.raw" `
  --gold-tiff "C:\path\to\gold_reflectance_rgn.tif"
```

This updates `config/models/flat_patch_v1.yaml` and writes the compact evidence
set to `calibration/evidence/`.

## Inspect a Processed Image

```powershell
python -m mira220 inspect "results\survey-001\capture-name"
```

The inspection report includes model provenance, patch Red/Green/IR/NDVI, and
channel clipping fractions.

See [docs/pipeline.md](docs/pipeline.md) for the correction equations and
[docs/experiments.md](docs/experiments.md) for the historical experiment record.

## Compare Mira NDVI With Gold-Camera TIFFs

Use this optional workflow when validating outputs or refining the model with
additional training data. Place calibrated three-channel Red/Green/NIR
gold-camera TIFFs containing ArUco marker `830` under the gitignored
`data/gold/` directory. Mira inputs are processed output directories containing
`ndvi.tiff` and `rgn_reflectance.tiff`.

Compare one pair:

```powershell
python -m mira220 compare "results\survey-001\capture-name" "data\gold\gold-001.tif"
```

Compare every discovered Mira output with every gold TIFF and rank the
candidates for each Mira capture:

```powershell
python -m mira220 compare-all --mira-root results --gold-dir data/gold
```

Results are written under `results/comparisons/`. The top-ranked pair is only a
candidate; visually confirm its `alignment_overlay.png` before treating its
regression metrics as authoritative. Regression plots and primary metrics use
valid-pixel-weighted 16x16 NDVI block averages; reports also retain pixel-level
metrics under `pixel_metrics`.

## Apply The Scene-Trained Channel Correction

`config/models/scene_reference_v1.yaml` contains an optional second-stage
Red/NIR correction trained from the aligned June 4 dataset. It is scene-specific
and does not replace the general `flat_patch_v1` model.

Apply it to existing reflectance outputs without rerunning openRGB-IR:

```powershell
python -m mira220 recalibrate "results\survey-001" `
  --output-root "results\survey-001-scene-reference-v1"
```

For future RAW processing of the same scene:

```powershell
python -m mira220 process --model config/models/scene_reference_v1.yaml
```

## Validate A New Lighting Session Against Mapir

Place a session's Mira RAW and calibrated Mapir R/G/NIR TIFF files under:

```text
data/sessions/<session-id>/mira/raw/
data/sessions/<session-id>/mapir/tiff/
```

This evaluation command processes both models, fits Mira-only target lighting
adjustments, and compares all four variants against held-out Mapir images:

```powershell
python -m mira220 validate-session "data\sessions\2026-06-05"
```

Results are written to `results/sessions/<session-id>/`. Mapir images are never
used to fit the target session adjustment.

