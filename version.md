# Mira220 Codebase Evolution

This file summarizes how the Mira220 RGB-IR reflectance pipeline has evolved
over time. It is not a strict semantic-version changelog; it records the major
capability milestones visible in the repository history.

## Current State

The current pipeline processes Mira220 RAW files into calibrated reflectance and
NDVI products, supports optional session-level target adjustment, detects NDVI
candidate regions, and creates dated collages for review.

Main commands:

```powershell
python -m mira220 fit
python -m mira220 process
python -m mira220 process-session
python -m mira220 inspect
python -m mira220 detect-candidates
python -m mira220 collage
python -m mira220 compare
python -m mira220 compare-all
python -m mira220 recalibrate
python -m mira220 validate-session
```

Authoritative pipeline details live in [docs/pipeline.md](docs/pipeline.md).
Historical experiment notes live in [docs/experiments.md](docs/experiments.md).

## Evolution Timeline

### Foundation: Reflectance Pipeline

Commit: `63bd69f Refactor Mira220 reflectance pipeline`

The repository started around a calibrated Mira220 processing workflow. The core
pipeline runs openRGB-IR, normalizes demosaiced channels, applies a fitted
flat-patch correction, and writes reflectance and NDVI outputs.

Key ideas established:

- OpenRGB-IR is used for RAW conversion.
- AWB and CCM are disabled for the supported workflow.
- The calibrated model lives in `config/models/flat_patch_v1.yaml`.
- Outputs include reflectance TIFFs, NDVI TIFFs, and preview PNGs.

### Gold-Camera Comparison And Scene Correction

Commit: `3fb8bb7 Add ArUco-aligned NDVI comparison and scene correction`

The project added tooling to compare Mira outputs against calibrated gold-camera
TIFFs. ArUco marker geometry is used to align scenes and identify calibration
target patches.

Capabilities added:

- Pairwise Mira-to-gold comparison.
- Batch comparison across many outputs and gold TIFFs.
- Alignment overlays for visual confirmation.
- Optional scene-trained correction using `scene_reference_v1.yaml`.

### Production Session Processing

Commits:

- `80fdd24 Add production session processing workflow`
- `9f90dab Merge production session processing workflow`

The workflow expanded from processing single batches to handling field sessions.
The `process-session` command writes a validated output set and can also create
an affine-adjusted variant when target measurements are available.

Capabilities added:

- Session-oriented output structure.
- Validated and affine-adjusted output variants.
- Session-level target lighting adjustment.
- Session validation against held-out Mapir images.

### Safer Batch Processing

Commit: `96969de Skip already processed RAW captures`

Batch processing became safer for repeated runs. The processor now checks for
expected output files before rerunning expensive RAW conversion work.

Impact:

- Existing processed captures are skipped.
- Repeated runs can fill in missing outputs without starting over.
- Run summaries record processed and skipped captures.

### Improved NDVI Review Images

Commit: `ee947f3 Add cropped smoothed NDVI preview`

The NDVI review outputs became easier to inspect. The pipeline added a cropped
false-color NDVI preview that focuses on the valid image region.

Impact:

- Review images avoid invalid border regions.
- NDVI previews are smoother and easier to scan.
- Candidate detection later uses these cropped products for overlays.

### External Data Root

Commit: `618a47c Use external data root for inputs and outputs`

Large inputs and generated products were moved out of the Git worktree by
default. The CLI now uses an external data root, normally:

```text
C:\Users\jayal\droid-data
```

Impact:

- Large RAW, TIFF, and output files stay out of Git.
- The repository remains focused on source code, configs, tests, and docs.
- `MIRA220_DATA_ROOT` can override the default data location.

### NDVI Candidate Region Detection

Commit: `e1d0bd0 Add NDVI candidate region detection`

The project added the first candidate-region detector for NDVI review. The
detector identifies broad NDVI regions and texture-heavy regions, then writes
overlay, mask, CSV, and JSON summary outputs.

Outputs added:

```text
candidate_overlay.png
candidate_mask.png
candidates.csv
candidate_summary.json
```

### Batch Candidate Detection

Commit: `9e14f16 Add NDVI batch candidate detection`

Candidate detection was integrated into batch processing. The `process` command
can now write candidate outputs next to each processed capture.

Useful options:

```powershell
python -m mira220 process "<raw-dir>" --recursive --detect-candidates
python -m mira220 process "<raw-dir>" --recursive --product-set ndvi --detect-candidates
python -m mira220 process "<raw-dir>" --recursive --detect-candidates --force-candidates
```

Impact:

- Candidate review products can be generated during normal processing.
- NDVI-only processing can reduce storage use for review runs.
- Existing candidate outputs can be reused or force-regenerated.

### Target-Aware Candidate Filtering And Dated Collages

Commit: `557b942 Add target-aware candidate filtering and dated collages`

Candidate detection became more target-aware and review-friendly. The detector
can exclude the calibration target area so the board is less likely to be
reported as a plant candidate. The processor also skips RAW frames that are too
saturated to trust.

Capabilities added:

- Target exclusion masks from ArUco marker and patch geometry.
- Candidate summary metadata for target exclusion.
- Saturated RAW skip rule at more than 5% sensor-max pixels.
- Dated collages from existing processed outputs.
- Candidate-overlay collages for faster review.

Collage examples:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614"
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614" `
  --image-name candidate_overlay.png `
  --output-dir "C:\Users\jayal\droid-data\outputs\collages\reprocess-20260614-candidates"
```

### Student-Friendly Reporting

Commit: `6d31d46 Add student-friendly work summary report`

The repository added a plain-language report in Markdown and PDF form. The
report summarizes the recent pipeline work at an 8th-grade reading level.

Files added:

```text
docs/work_summary_8th_grade.md
docs/work_summary_8th_grade.pdf
```

## Design Direction

The project has moved through four broad phases:

1. Build a calibrated RAW-to-NDVI pipeline.
2. Validate the pipeline against target patches and gold-camera data.
3. Scale the workflow to sessions and repeated field runs.
4. Improve review tools with candidates, collages, summaries, and reports.

The current direction favors repeatable processing, clear audit outputs, and
fast visual review of large image batches.
