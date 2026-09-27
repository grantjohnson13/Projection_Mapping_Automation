# facade_scan/assets/

## santa-head.png

Santa's head, hat and beard, used by the `santa` layer in `facade-scan animate`.

| | |
|---|---|
| source | [Openclipart 354120, "Santa Claus face drop shape"](https://openclipart.org/detail/354120/santa-claus-face-drop-shape) |
| author | MissKaLem, uploaded 2025-12-21 |
| licence | **Public domain (CC0)** — Openclipart releases all uploads as 100% public domain |
| retrieved | 2026-09-26 |
| file | 800 × 1367 RGBA PNG, alpha a true cutout (all four corners fully transparent) |

Tracked in git, unlike `assets/audio/`, because CC0 permits redistribution —
that is the whole difference. Supplied music is licensed to whoever bought it,
not to this repository, so it stays ignored.

## sleigh-reindeer.png

Santa's sleigh and four reindeer, flown across the subject by the `flyer` layer.

| | |
|---|---|
| source | [Openclipart 291102, "Santa's Sleigh And Reindeer Silhouette"](https://openclipart.org/detail/291102/santas-sleigh-and-reindeer-silhouette) |
| author | GDJ, uploaded 2017-11-29, derived from a Pixabay image |
| licence | **Public domain (CC0)** |
| retrieved | 2026-09-26 |
| file | 1970 × 525 RGBA PNG, transparent margin trimmed |

This one is a **silhouette**, and the `flyer` layer treats it as such: the
artwork is black, and black is the one colour a projector cannot produce, so
compositing it would put a sleigh-shaped hole in the wash. Its alpha is used as
a stencil and filled with `flyer_colour` instead, so it flies as light. Any
silhouette PNG works the same way; its own colours are ignored.

## Replacing it

Point `animate.santa_image` at any RGBA PNG. It is scaled to
`animate.santa_height_frac` of the subject's height and keeps its own aspect
ratio, so proportions are up to the file.

Two things matter for a projected image, both learned the hard way on the
candidates that were rejected for this one:

- **It must have a real alpha channel.** A PNG on a white background is
  composited as a white rectangle. `animate` refuses one without alpha rather
  than showing you the box.
- **Line art disappears.** Outlines with transparent interiors let the wash
  through where the face should be, so at throw distance there is nothing to
  see. Solid fills with strong internal contrast survive; drawings do not.
