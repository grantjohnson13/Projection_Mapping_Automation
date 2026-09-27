"""Gray-code structured-light pattern generation.

Why Gray code and not plain binary
----------------------------------
In plain binary, adjacent code words can differ in many bits at once: 0111 ->
1000 flips four. Every one of those bit boundaries falls at the *same* pixel
column, so a decode error there -- and there is always some error, because the
projector's stripe edge lands partway across a camera pixel -- can move the
decoded coordinate by half the image.

Gray code guarantees that adjacent code words differ in exactly one bit, so the
bit boundaries are spread across different columns and a misread at any one of
them costs exactly one projector pixel. That property is the whole reason this
technique works on a real facade at night.

Why every pattern also gets an inverse
--------------------------------------
A camera pixel looking at dark brick under a lit stripe can easily be darker
than a pixel looking at white trim under an unlit stripe. There is no global
intensity threshold that separates "lit" from "unlit" across a real facade.

So we never use one. Each pattern is projected, then its photographic inverse is
projected, and the bit is read as ``pattern > inverse`` *per pixel*. Surface
albedo, ambient light and lens vignetting all multiply both exposures equally
and cancel in the comparison. The magnitude ``|pattern - inverse|`` falls out as
a free per-pixel confidence measure. This doubles the frame count and is not
optional.

Frame set for a W x H projector
-------------------------------
- ``ceil(log2(W))`` vertical stripe planes, encoding the projector *x* coordinate
- ``ceil(log2(H))`` horizontal stripe planes, encoding the projector *y* coordinate
- an inverse of each of the above
- one all-white and one all-black frame

For 1920x1080 that is 11 + 11 planes, doubled to 44, plus 2 = **46 frames**.

A useful side effect of Gray code: its finest plane has *two*-pixel stripes,
where plain binary's least significant bit alternates every single pixel. The
hardest plane to resolve is therefore twice as wide as it would otherwise be,
which matters a great deal through a defocused projector lens at 15 metres.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .config import PatternConfig

Role = Literal["white", "black", "gray"]
Axis = Literal["x", "y"]


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Frame:
    """One projected frame, in capture order."""

    index: int
    filename: str
    #: "white", "black" or "gray"
    role: str
    #: "x" for vertical stripes, "y" for horizontal stripes, None otherwise
    axis: str | None = None
    #: Bit index within the axis, 0 = most significant. None for white/black.
    bit: int | None = None
    #: True if this frame is the photographic inverse of its pattern.
    inverted: bool = False

    @property
    def label(self) -> str:
        if self.role != "gray":
            return self.role
        assert self.axis is not None and self.bit is not None
        return f"{self.axis}{self.bit:02d}{'_inv' if self.inverted else ''}"


@dataclass
class Manifest:
    """Describes a generated pattern set, written as ``manifest.json``.

    The manifest is the contract between pattern generation, capture and decode:
    the capture backends replay ``frames`` in order, and the decoder uses the
    same list to know which captured image is which bit of which axis.
    """

    projector_width: int
    projector_height: int
    #: Number of Gray-code planes encoding x (= ceil(log2(width))).
    bits_x: int
    #: Number of Gray-code planes encoding y (= ceil(log2(height))).
    bits_y: int
    frames: list[Frame]
    version: int = 1

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def frames_for(self, axis: str, inverted: bool) -> list[Frame]:
        """Frames for one axis and polarity, ordered MSB first."""
        sel = [f for f in self.frames if f.role == "gray" and f.axis == axis
               and f.inverted == inverted]
        return sorted(sel, key=lambda f: f.bit or 0)

    def frame_by_role(self, role: str) -> Frame:
        for f in self.frames:
            if f.role == role:
                return f
        raise KeyError(f"no {role!r} frame in manifest")

    # ------------------------------------------------------------------ io --
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json())
        return path

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text())
        frames = [Frame(**f) for f in data.pop("frames")]
        return cls(frames=frames, **data)


# --------------------------------------------------------------------------- #
# Gray code
# --------------------------------------------------------------------------- #
def num_bits(size: int) -> int:
    """Number of Gray-code planes needed to address ``size`` columns/rows."""
    if size < 1:
        raise ValueError("size must be >= 1")
    if size == 1:
        return 1
    return int(np.ceil(np.log2(size)))


def binary_to_gray(values: np.ndarray) -> np.ndarray:
    """Standard reflected binary (Gray) encoding: ``g = v ^ (v >> 1)``."""
    v = np.asarray(values, dtype=np.uint32)
    return v ^ (v >> np.uint32(1))


def gray_to_binary(gray: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`binary_to_gray`.

    XOR-shift-fold: after folding by 1, 2, 4, ... < ``bits``, every bit has
    accumulated the XOR of all higher Gray bits, which is the binary value.
    """
    b = np.asarray(gray, dtype=np.uint32).copy()
    shift = 1
    while shift < bits:
        b ^= b >> np.uint32(shift)
        shift <<= 1
    return b


def gray_plane(size: int, bit: int, bits: int) -> np.ndarray:
    """One Gray-code bit plane as a 1-D uint8 array of 0/1 over ``size``.

    ``bit`` is indexed from the most significant end: bit 0 is the single
    widest stripe pair, bit ``bits - 1`` is the finest one-pixel stripe.
    """
    if not 0 <= bit < bits:
        raise ValueError(f"bit {bit} out of range for {bits} bits")
    coords = np.arange(size, dtype=np.uint32)
    gray = binary_to_gray(coords)
    shift = np.uint32(bits - 1 - bit)
    return ((gray >> shift) & np.uint32(1)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Frame generation
# --------------------------------------------------------------------------- #
def build_manifest(width: int, height: int, cfg: PatternConfig | None = None) -> Manifest:
    """Build the frame list without rendering any pixels."""
    cfg = cfg or PatternConfig()
    bits_x, bits_y = num_bits(width), num_bits(height)
    frames: list[Frame] = []

    def add(role: str, axis: str | None = None, bit: int | None = None,
            inverted: bool = False) -> None:
        idx = len(frames)
        stub = Frame(index=idx, filename="", role=role, axis=axis, bit=bit, inverted=inverted)
        filename = cfg.filename_template.format(index=idx, label=stub.label)
        frames.append(
            Frame(index=idx, filename=filename, role=role, axis=axis, bit=bit,
                  inverted=inverted)
        )

    # White and black come first: they double as the exposure check you look at
    # before committing to a 46-frame capture, and the decoder needs them for
    # the illumination mask and the house-mask shortcut.
    add("white")
    add("black")
    for bit in range(bits_x):
        add("gray", "x", bit, False)
        add("gray", "x", bit, True)
    for bit in range(bits_y):
        add("gray", "y", bit, False)
        add("gray", "y", bit, True)

    return Manifest(projector_width=width, projector_height=height,
                    bits_x=bits_x, bits_y=bits_y, frames=frames)


def render_frame(frame: Frame, width: int, height: int, bits_x: int, bits_y: int,
                 cfg: PatternConfig | None = None) -> np.ndarray:
    """Render one frame as a ``(height, width)`` uint8 image."""
    cfg = cfg or PatternConfig()
    lo, hi = np.uint8(cfg.black_level), np.uint8(cfg.white_level)

    if frame.role == "white":
        return np.full((height, width), hi, dtype=np.uint8)
    if frame.role == "black":
        return np.full((height, width), lo, dtype=np.uint8)

    assert frame.axis is not None and frame.bit is not None
    if frame.axis == "x":
        line = gray_plane(width, frame.bit, bits_x)          # (width,)
        plane = np.broadcast_to(line[None, :], (height, width))
    else:
        line = gray_plane(height, frame.bit, bits_y)         # (height,)
        plane = np.broadcast_to(line[:, None], (height, width))

    if frame.inverted:
        plane = 1 - plane
    return np.where(plane.astype(bool), hi, lo).astype(np.uint8)


def generate(width: int, height: int,
             cfg: PatternConfig | None = None) -> Iterator[tuple[Frame, np.ndarray]]:
    """Yield ``(frame, image)`` for the whole set, in capture order."""
    cfg = cfg or PatternConfig()
    manifest = build_manifest(width, height, cfg)
    for frame in manifest.frames:
        yield frame, render_frame(frame, width, height, manifest.bits_x,
                                  manifest.bits_y, cfg)


def write_patterns(out_dir: str | Path, width: int, height: int,
                   cfg: PatternConfig | None = None) -> Manifest:
    """Write every frame as a PNG at exact projector resolution, plus the manifest.

    Filenames are zero-padded and sort in capture order, which is what the
    ``folder`` capture backend relies on.
    """
    import cv2

    cfg = cfg or PatternConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(width, height, cfg)
    for frame in manifest.frames:
        img = render_frame(frame, width, height, manifest.bits_x, manifest.bits_y, cfg)
        if not cv2.imwrite(str(out / frame.filename), img):
            raise OSError(f"failed to write {out / frame.filename}")
    manifest.write(out / "manifest.json")
    return manifest
