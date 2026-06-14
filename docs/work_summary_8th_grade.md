# Mira220 Processing Update Summary

Date: June 14, 2026

## Big Idea

This project turns Mira220 RAW camera files into images that help us study
plants. The main image is called NDVI. NDVI helps show where plants look
healthier or less healthy.

This update made the pipeline better at three jobs:

1. Skipping RAW images that are too bright and washed out.
2. Finding candidate plant regions while ignoring the calibration target.
3. Making collage images grouped by date, so many processed images can be
   reviewed quickly.

## What Changed

### 1. Skipping Overly Bright RAW Images

Some RAW images have too many pixels at the highest sensor value. That usually
means the image is saturated, or too bright to trust.

The pipeline now checks each RAW file before processing it. If more than 5% of
the pixels are at the sensor maximum, that RAW file is skipped.

The skipped image is recorded in `run_summary.json` with:

- the file path
- the reason: `excessive RAW saturation`
- the saturated pixel fraction
- the threshold used
- the sensor maximum value

This helps prevent bad images from being mixed into the results.

### 2. Better Candidate Region Detection

Candidate regions are areas in the NDVI image that may be important for review.
For example, they may show strong vegetation, weaker vegetation, or texture that
could be worth checking.

This update made candidate detection more careful:

- It can find the ArUco calibration target in a processed capture.
- It builds a mask around the marker and target patches.
- It removes candidates that overlap the target area.
- It filters out skinny edge artifacts and board-like shapes that are probably
  not useful plant regions.

The goal is to keep the candidate list focused on real scene regions instead of
the calibration board or image-edge artifacts.

### 3. Candidate Summary Metadata

The candidate summary now includes target exclusion information. This means
`candidate_summary.json` can say whether the calibration target was found and
how many pixels were excluded.

This makes the output easier to audit later.

### 4. Dated Collages

A new command was added:

```powershell
python -m mira220 collage
```

This command looks through processed output folders, groups images by capture
date, and writes one collage per date.

By default, it uses:

```text
ndvi_false_color_crop.png
```

It can also make collages from candidate overlays:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614" `
  --image-name candidate_overlay.png `
  --output-dir "C:\Users\jayal\droid-data\outputs\collages\reprocess-20260614-candidates"
```

This is useful because it lets someone compare many images from the same day at
once instead of opening each file one by one.

## Work Completed

The changes were committed and merged into `master`.

Commit:

```text
557b942 Add target-aware candidate filtering and dated collages
```

The tests were run after the changes.

Result:

```text
54 tests passed
```

## Collages Created

NDVI crop collages were created from the already processed run:

```text
C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614
```

They were written to:

```text
C:\Users\jayal\droid-data\outputs\collages
```

Candidate overlay collages were also created from the same processed run.

They were written to:

```text
C:\Users\jayal\droid-data\outputs\collages\reprocess-20260614-candidates
```

The dates included were:

- 2026-03-29
- 2026-04-13
- 2026-04-20
- 2026-04-28
- 2026-04-29
- 2026-05-02
- 2026-06-04
- 2026-06-05
- 2026-06-06

## How To Use It

To create normal NDVI crop collages from an existing processed run:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614"
```

To create candidate-region collages from an existing processed run:

```powershell
python -m mira220 collage "C:\Users\jayal\droid-data\outputs\runs\reprocess-20260614" `
  --image-name candidate_overlay.png `
  --output-dir "C:\Users\jayal\droid-data\outputs\collages\reprocess-20260614-candidates"
```

These commands do not reprocess RAW files. They only use images that were
already created.

## Why This Matters

This update makes the workflow easier to trust and easier to review.

Bad saturated images are skipped. The calibration target is less likely to be
mistaken for a plant region. Collages make it faster to compare many processed
images from the same date.
