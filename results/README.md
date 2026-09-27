# results/

One directory per test run. Everything in here is **generated** and gitignored
except this file — the captures alone are hundreds of megabytes, and a result is
specific to one physical setup on one night.

## Layout

```
results/
  README.md                       <- tracked
  .gitkeep                        <- tracked
  YYYY-MM-DD_<slug>/              <- one run, ignored
    README.md                     what was set up, and what happened
    config.toml                   the exact config used, copy-pasteable
    scan/
      captures/                   one photograph per projected frame
      patterns/                   the frames that were projected
      decoded.npz                 camera -> projector map, masks, confidence
      detection.json              camera-space segments and regions
      white.png                   the all-white capture
      ground_truth.npz            simulated runs only
      export/
        mask.png                  projector resolution, white = project here
        regions.svg               labelled regions in projector pixels
        scan.json                 geometry + confidence as data
    report/
      result.mp4                  the run as a ~30 s video
      summary.json                numbers, for diffing between runs
      summary.md                  the same, readable
      stills/                     each video section as a PNG
```

## Naming

`YYYY-MM-DD_<slug>`, slug describing the *setup*, not the outcome:

- `2026-09-24_tabletop-sim-baseline`
- `2026-09-24_cardboard-box-usb-webcam`
- `2026-11-02_house-front-dslr`

Keep failed runs. A scan that decoded badly is the most useful thing to compare
the next one against, and the whole point of writing every stage to disk is that
you can re-run detection and export on it without going back outside.

## Making a run

```bash
facade-scan run    --scan results/<run>/scan --backend webcam
facade-scan report --scan results/<run>/scan --out results/<run>/report \
                   --title "<run>" --notes "what was set up"
```

## What to compare between runs

From `report/summary.json`:

| field | what it tells you |
|---|---|
| `decode.coverage` | fraction of the camera frame that decoded at all |
| `planarity_residual_px` | **measurement noise in projector pixels.** Needs no ground truth: the decoded map over any flat surface is exactly a homography, so the residual of a fitted one is the scan's own error. This is the number to watch on physical rigs. |
| `per_bit_confidence` | how much headroom there was. p05 near the threshold means you were nearly out of light. |
| `ground_truth.median_error_px` | simulated runs only |
| `projector_mask.lit_fraction` | how much of the panel ends up lit |
