# Reconciled NDVI Candidate Detection Plan

## Summary
Rebuild candidate detection around raster-style region processing. The detector reads cropped NDVI, creates cleaned class masks, extracts broad regions first, and adds texture-heavy regions as a separate candidate type.

## Key Changes
- Load processed output directories, direct NDVI TIFFs, grayscale images, and false-color NDVI images.
- Prefer physical `ndvi.tiff` for measurements and crop with `NDVI_CROP_MARGINS` by default.
- Normalize with 1st-99th percentile clipping for mask generation and visualization.
- Reclassify into `high_vegetation`, `medium_vegetation`, `low_vegetation`, and `non_vegetation_artifact`.
- Clean masks with open/close morphology, small-region sieve removal, and small-hole filling.
- Detect:
  - red `broad_ndvi_region` candidates from cleaned class contours
  - orange `texture_region` candidates from local texture statistics
- Do not emit linear-artifact or geometric-box candidates by default; those created visual clutter on representative images.

## Outputs And CLI
- Keep `python -m mira220 detect-candidates <image_or_output_dir>`.
- Write `candidate_overlay.png`, `candidate_mask.png`, `candidates.csv`, and `candidate_summary.json`.
- Expose tuning for threshold mode, manual threshold, inversion, minimum area, contrast threshold, morphology, hole filling, texture sensitivity/window, geometry mode, simplification, and class thresholds.
- `candidate_mask.png` is a single-channel candidate ID mask whose pixel values map to CSV/JSON candidate IDs.

## Tests
- Cover processed-directory/direct-image loading, threshold modes, speckle removal, hole filling, broad region extraction, irregular contour simplification, texture regions, suppression of line/geometric artifacts, uniform-scene suppression, mask/CSV consistency, and CLI smoke output.
