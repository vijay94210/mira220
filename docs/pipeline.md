# Current Flat-Patch Pipeline

## Processing

1. Run openRGB-IR with black-level correction, RGB-IR interpolation, and Malvar demosaic.
2. Keep IRC cut at `0.0`; disable AWB and CCM.
3. Normalize demosaiced channels by `4095`.
4. Apply the post-demosaic flat-patch correction.
5. Compute NDVI from corrected Red and IR.

The authoritative configuration is `config/models/flat_patch_v1.yaml`.

## Correction Values

```text
Red = clip(
    2.7299981 * raw_red
  - 5.5990033 * raw_ir
  + 3.9216752 * raw_ir^2
  + 0.3154759,
  0, 1
)

Green = clip(
    0.1680883 * raw_green
  + 0.2066993 * raw_ir
  + 0.3978048 * raw_ir^2
  + 0.1240077,
  0, 1
)

IR = clip(
    0.6501014 * raw_ir
  + 0.0275980 * raw_ir^2
  + 0.1559270,
  0, 1
)

NDVI = (IR - Red) / (IR + Red + 1e-6)
```

The PNG NDVI display clips to `[-0.2, 0.7]` and uses a red-yellow-green scale.
The float32 TIFF retains the full `[-1, 1]` range.

## Calibration Target

ArUco marker `830` identifies four flat-spectrum patches:

| Patch | Position | Shared Gold Target | Validated NDVI |
|---|---|---:|---:|
| P1 | Top-left | 0.7406068 | -0.0037 |
| P2 | Top-right | 0.6783373 | -0.0002 |
| P3 | Bottom-left | 0.3136492 | -0.0177 |
| P4 | Bottom-right | 0.2642964 | 0.0214 |

The fit uses each gold patch's mean R/G/IR reflectance as a shared target,
enforcing near-zero patch NDVI.

## Outputs

Each processed RAW produces:

```text
red_reflectance.tiff
green_reflectance.tiff
ir_reflectance.tiff
rgn_reflectance.tiff
ndvi.tiff
rgn_preview.png
ndvi_false_color.png
ndvi_gray.png
```

Temporary openRGB-IR outputs are removed after successful processing unless
`--keep-intermediates` is supplied.

