# Historical Experiments

The initial development phase generated several large output directories. They
were removed after the current `flat_patch_v1` model and compact evidence set
were promoted.

| Historical Directory | Purpose | Conclusion | Disposition |
|---|---|---|---|
| `validation`, `validation_calibrated`, `final_validation` | Early stage-one and ArUco validation | Pipeline and marker geometry worked; early reflectance model was unsuitable | Removed |
| `ndvi_revision_validation` | AWB-disabled NDVI and corrected display scale | NDVI must use linear Red, not colorimetric CCM output | Removed |
| `candidate_mira_zero_cut` | Compare June 4 Mira captures with IRC cut zero | `2026-06-04_19_48_23.raw` was selected for fitting | Removed |
| `irc_sweep_gold` | Test mosaic-domain IRC cuts against gold TIFF | Large cuts collapse visible detail; retain cut zero | Removed |
| `reference_fit_194823` | Scene-wide linear reference fit | Good scene fit but poor flat-patch NDVI | Removed |
| `reference_fit_constrained` | Weighted linear scene/patch fit | Improved patches but P4 remained high | Removed |
| `reference_fit_flat_patches` | Final quadratic flat-patch fit | All patch NDVI values approximately within +/-0.02 | Promoted to config and evidence |
| `batch_validation`, `batch_reflectance`, root capture outputs | Batch-run validation and duplicate products | Batch behavior validated | Removed |
| `gold_patch_identification` | Confirm patch ordering and gold values | P1/P2 top row, P3/P4 bottom row | Promoted to evidence |

The retained evidence is under `calibration/evidence/`.

