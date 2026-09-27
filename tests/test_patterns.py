"""Stage 1: Gray-code pattern generation."""

from __future__ import annotations

import numpy as np
import pytest

from facade_scan.config import PatternConfig
from facade_scan.patterns import (
    Manifest,
    binary_to_gray,
    build_manifest,
    generate,
    gray_plane,
    gray_to_binary,
    num_bits,
    render_frame,
    write_patterns,
)


@pytest.mark.parametrize(
    "size,expected", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (1080, 11), (1920, 11), (2048, 11), (2049, 12)]
)
def test_num_bits(size, expected):
    assert num_bits(size) == expected
    assert 2 ** num_bits(size) >= size


def test_gray_roundtrip_is_exact():
    v = np.arange(1 << 14, dtype=np.uint32)
    assert np.array_equal(gray_to_binary(binary_to_gray(v), 14), v)


def test_gray_adjacent_codes_differ_by_exactly_one_bit():
    """The defining property. A plain binary code would fail this."""
    v = np.arange(2048, dtype=np.uint32)
    g = binary_to_gray(v)
    diff = g[:-1] ^ g[1:]
    popcount = np.array([bin(int(d)).count("1") for d in diff])
    assert np.all(popcount == 1)


def test_plain_binary_would_fail_the_same_property():
    """Sanity-check the test itself: binary really does flip many bits at once."""
    v = np.arange(2048, dtype=np.uint32)
    diff = v[:-1] ^ v[1:]
    popcount = np.array([bin(int(d)).count("1") for d in diff])
    assert popcount.max() > 1


def test_gray_planes_reconstruct_every_column():
    """Stack all bit planes back into a coordinate and check it is the identity."""
    width, bits = 1920, num_bits(1920)
    gray = np.zeros(width, dtype=np.uint32)
    for bit in range(bits):
        plane = gray_plane(width, bit, bits).astype(np.uint32)
        gray |= plane << np.uint32(bits - 1 - bit)
    assert np.array_equal(gray_to_binary(gray, bits), np.arange(width, dtype=np.uint32))


def test_finest_plane_has_two_pixel_stripes_and_coarsest_has_one_transition():
    """Gray code's finest plane is 2px wide, not 1px as plain binary's LSB is.

    That is a free win: the hardest-to-resolve plane is twice as wide as the
    equivalent binary one, so it survives projector defocus better.
    """
    bits = num_bits(1024)
    finest = gray_plane(1024, bits - 1, bits)
    coarsest = gray_plane(1024, 0, bits)
    assert np.count_nonzero(np.diff(finest.astype(int))) == 512
    assert np.array_equal(finest[:8], [0, 1, 1, 0, 0, 1, 1, 0])
    assert np.count_nonzero(np.diff(coarsest.astype(int))) == 1
    assert np.argmax(np.diff(coarsest.astype(int)) != 0) == 511


def test_binary_lsb_would_be_half_as_wide():
    """Contrast: the binary LSB alternates every single pixel."""
    v = np.arange(1024, dtype=np.uint32)
    binary_lsb = (v & 1).astype(int)
    assert np.count_nonzero(np.diff(binary_lsb)) == 1023


def test_frame_count_for_1920x1080_is_46():
    m = build_manifest(1920, 1080)
    assert (m.bits_x, m.bits_y) == (11, 11)
    assert m.num_frames == 11 * 2 + 11 * 2 + 2 == 46


def test_manifest_has_an_inverse_for_every_pattern():
    m = build_manifest(1280, 800)
    for axis in ("x", "y"):
        normal = m.frames_for(axis, inverted=False)
        inverse = m.frames_for(axis, inverted=True)
        assert [f.bit for f in normal] == [f.bit for f in inverse]
        assert len(normal) == (m.bits_x if axis == "x" else m.bits_y)


def test_frames_sort_by_filename_in_capture_order():
    m = build_manifest(1920, 1080)
    names = [f.filename for f in m.frames]
    assert names == sorted(names)
    assert [f.index for f in m.frames] == list(range(m.num_frames))


def test_rendered_frames_are_exact_projector_resolution_and_two_valued():
    cfg = PatternConfig()
    for _frame, img in generate(640, 400, cfg):
        assert img.shape == (400, 640)
        assert img.dtype == np.uint8
        assert set(np.unique(img)).issubset({cfg.black_level, cfg.white_level})


def test_pattern_and_inverse_are_complementary_everywhere():
    m = build_manifest(640, 400)
    for axis in ("x", "y"):
        for normal, inverse in zip(m.frames_for(axis, False), m.frames_for(axis, True)):
            a = render_frame(normal, 640, 400, m.bits_x, m.bits_y)
            b = render_frame(inverse, 640, 400, m.bits_x, m.bits_y)
            assert np.array_equal(a.astype(int) + b.astype(int),
                                  np.full(a.shape, 255, dtype=int))


def test_vertical_planes_are_constant_down_columns_and_horizontal_across_rows():
    m = build_manifest(256, 128)
    x_frame = m.frames_for("x", False)[3]
    y_frame = m.frames_for("y", False)[3]
    xi = render_frame(x_frame, 256, 128, m.bits_x, m.bits_y)
    yi = render_frame(y_frame, 256, 128, m.bits_x, m.bits_y)
    assert np.all(xi == xi[0:1, :])   # vertical stripes: every row identical
    assert np.all(yi == yi[:, 0:1])   # horizontal stripes: every column identical


def test_white_and_black_frames_exist_and_are_uniform():
    m = build_manifest(320, 200)
    w = render_frame(m.frame_by_role("white"), 320, 200, m.bits_x, m.bits_y)
    b = render_frame(m.frame_by_role("black"), 320, 200, m.bits_x, m.bits_y)
    assert np.all(w == 255) and np.all(b == 0)


def test_write_patterns_round_trip(tmp_path):
    import cv2

    m = write_patterns(tmp_path, 320, 200)
    reread = Manifest.read(tmp_path / "manifest.json")
    assert reread == m
    files = sorted(p.name for p in tmp_path.glob("*.png"))
    assert files == [f.filename for f in m.frames]
    img = cv2.imread(str(tmp_path / m.frames[2].filename), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (200, 320)
    assert np.array_equal(img, render_frame(m.frames[2], 320, 200, m.bits_x, m.bits_y))
