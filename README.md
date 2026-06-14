# Mira220 RGB-IR Reflectance Pipeline

This repository has one supported workflow:

```powershell
python -m mira220 fit
python -m mira220 process
python -m mira220 process-session
python -m mira220 inspect
python -m mira220 detect-candidates
python -m mira220 collage
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

## Data Root

Large inputs and generated products live outside the Git worktree. By default
the CLI uses:

```text
C:\Users\jayal\droid-data
```

Override it for a shell session with:

```powershell
$env:MIRA220_DATA_ROOT = "D:\mira220-data"
```

The data root has three main roles:

- `raw/`: loose Mira RAW inputs for normal batch processing.
- `sessions/`: structured capture bundles that keep Mira RAWs and Mapir TIFFs
  together for a field session.
- `outputs/`: generated reflectance products, NDVI images, comparisons,
  summaries, smoke-test outputs, and presentations.

## Process RAW Images

Put unpacked 1600x1400 uint16 Mira220 RAWs in
`C:\Users\jayal\droid-data\raw\`, then run:

```powershell
python -m mira220 process
```

Process another directory:

```powershell
python -m mira220 process "C:\Users\jayal\droid-data\raw\survey-001" --recursive --run-id survey-001
```

Process RAWs and write candidate detection outputs into each processed capture
folder beside `ndvi.tiff`:

```powershell
python -m mira220 process "C:\Users\jayal\droid-data\raw\survey-001" --recursive --run-id survey-001 --detect-candidates
```

Use NDVI-only output mode to reduce disk usage for future review runs:

```powershell
python -m mira220 process "C:\Users\jayal\droid-data\raw\survey-001" --recursive --run-id survey-001 --product-set ndvi --detect-candidates
```

`--product-set ndvi` skips bulky per-channel reflectance TIFFs and keeps the
NDVI/candidate review products. Existing candidate outputs are reused; add
`--force-candidates` to regenerate them.

Results are written to
`C:\Users\jayal\droid-data\outputs\runs\<run-id>\<raw-name>\`.
Incompatible RAW sizes are skipped and recorded in `run_summary.json`. RAW
frames with more than 5% sensor-max pixels are also skipped as saturated and
recorded in the same summary.

Every processed capture writes calibrated reflectance TIFFs, `ndvi.tiff`,
`ndvi_gray.png`, `ndvi_false_color.png`, `ndvi_false_color_crop.png`,
`rgn_preview.png`, and `rgb_preview.png` in the default `full` product set.
ArUco target detection is not required for this base processing path.

## Process A Session With Optional Target Adjustment

Use `process-session` when a run may contain the ArUco marker calibration
target and you want both the validated model output and an experimental
session-affine variant:

```powershell
python -m mira220 process-session "C:\Users\jayal\droid-data\raw\survey-001" --recursive --run-id survey-001
```

Results are written to:

```text
C:\Users\jayal\droid-data\outputs\runs\<run-id>\validated\<raw-name>\
C:\Users\jayal\droid-data\outputs\runs\<run-id>\affine-adjusted\<raw-name>\
```

`validated/` is always produced from `flat_patch_v1` and does not require a
marker. `affine-adjusted/` is produced only when at least one processed capture
contains a detectable target. The affine fit uses all measured target patches;
Mapir images are not used for this adjustment. Markerless captures in the same
run inherit the session affine correction when the fit succeeds.

## Fit a Model

```powershell
python -m mira220 fit `
  --mira-raw "C:\Users\jayal\droid-data\calibration\mira\mira.raw" `
  --gold-tiff "C:\Users\jayal\droid-data\calibration\mapir\gold_reflectance_rgn.tif"
```

This updates `config/models/flat_patch_v1.yaml` and writes the compact evidence
set to `calibration/evidence/`.

## Inspect a Processed Image

```powershell
python -m mira220 inspect "C:\Users\jayal\droid-data\outputs\runs\survey-001\capture-name"
```

The inspection report includes model provenance, patch Red/Green/IR/NDVI, and
channel clipping fractions.

## Detect Cropped NDVI Candidate Regions

Use candidate detection on a processed output directory or a direct NDVI TIFF:

```powershell
python -m mira220 detect-candidates "C:\Users\jayal\droid-data\outputs\runs\survey-001\capture-name"
```

By default, results are written under
`C:\Users\jayal\droid-data\outputs\candidates\<input-name>\`. Detection uses
cropped physical `ndvi.tiff` values and overlays results on
`ndvi_false_color_crop.png` when available. The command writes
`candidate_overlay.png`, `candidate_mask.png`, `candidates.csv`, and
`candidate_summary.json`.

When candidate detection is run through `process --detect-candidates`, those
same files are written directly into each processed capture folder instead of
the shared `outputs\candidates` folder.

The detector builds broad NDVI regions first, then adds texture-heavy regions.
The overlay colors are red for broad NDVI regions and orange for texture-heavy
regions. Candidate regions are labeled with their candidate IDs, and the overlay
adds a right-side panel with an NDVI false-color scale plus the NDVI class
legend. The mask image is an ID mask where each candidate pixel value maps to a
candidate row.

Useful controls:

```powershell
python -m mira220 detect-candidates "<processed-output-dir>" `
  --threshold-mode percentile `
  --min-area 15000 `
  --morph-open 5 `
  --morph-close 31 `
  --min-contrast 0.03 `
  --texture-sensitivity 0.18 `
  --simplify-epsilon 8 `
  --geometry contour
```

## Create Dated Collages

Create one contact sheet per capture date from processed output images:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\survey-001"
```

By default, this uses `ndvi_false_color_crop.png` from each processed capture
folder and writes dated PNGs plus `collage_summary.json` under
`C:\Users\jayal\droid-data\outputs\collages\`.

Use candidate overlays instead:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\survey-001" `
  --image-name candidate_overlay.png `
  --output-dir "C:\Users\jayal\droid-data\outputs\collages\survey-001-candidates"
```

See [docs/pipeline.md](docs/pipeline.md) for the correction equations and
[docs/experiments.md](docs/experiments.md) for the historical experiment record.

## Compare Mira NDVI With Gold-Camera TIFFs

Use this optional workflow when validating outputs or refining the model with
additional training data. Place calibrated three-channel Red/Green/NIR
gold-camera TIFFs containing ArUco marker `830` under the gitignored
`C:\Users\jayal\droid-data\gold\` directory. Mira inputs are processed output
directories containing `ndvi.tiff` and `rgn_reflectance.tiff`.

Compare one pair:

```powershell
python -m mira220 compare `
  "C:\Users\jayal\droid-data\outputs\runs\survey-001\capture-name" `
  "C:\Users\jayal\droid-data\gold\gold-001.tif"
```

Compare every discovered Mira output with every gold TIFF and rank the
candidates for each Mira capture:

```powershell
python -m mira220 compare-all
```

Results are written under
`C:\Users\jayal\droid-data\outputs\comparisons\`. The top-ranked pair is only
a candidate; visually confirm its `alignment_overlay.png` before treating its
regression metrics as authoritative. Regression plots and primary metrics use
valid-pixel-weighted 16x16 NDVI block averages; reports also retain pixel-level
metrics under `pixel_metrics`.

## Apply The Scene-Trained Channel Correction

`config/models/scene_reference_v1.yaml` contains an optional second-stage
Red/NIR correction trained from the aligned June 4 dataset. It is scene-specific
and does not replace the general `flat_patch_v1` model.

Apply it to existing reflectance outputs without rerunning openRGB-IR:

```powershell
python -m mira220 recalibrate "C:\Users\jayal\droid-data\outputs\runs\survey-001" `
  --output-root "C:\Users\jayal\droid-data\outputs\runs\survey-001-scene-reference-v1"
```

For future RAW processing of the same scene:

```powershell
python -m mira220 process --model config/models/scene_reference_v1.yaml
```

## Validate A New Lighting Session Against Mapir

Place a session's Mira RAW and calibrated Mapir R/G/NIR TIFF files under:

```text
C:\Users\jayal\droid-data\sessions\<session-id>\mira\raw\
C:\Users\jayal\droid-data\sessions\<session-id>\mapir\tiff\
```

This evaluation command processes both models, fits Mira-only target lighting
adjustments, and compares all four variants against held-out Mapir images:

```powershell
python -m mira220 validate-session "C:\Users\jayal\droid-data\sessions\2026-06-05"
```

Results are written to
`C:\Users\jayal\droid-data\outputs\sessions\<session-id>\`. Mapir images are
never used to fit the target session adjustment.

