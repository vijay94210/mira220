# Mira220 RGB-IR Reflectance Pipeline

This repository has one supported workflow:

```powershell
python -m mira220 fit
python -m mira220 process
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

