"""Music-driven animation: audio analysis, layers, rendering."""

from __future__ import annotations

import colorsys
from itertools import pairwise

import numpy as np
import pytest

from facade_scan.animate import LayerConfig, Layers, analyse, mask_outline
from facade_scan.animate.audio import (
    HOP,
    SAMPLE_RATE,
    WINDOW,
    AudioAnalysis,
    decode_to_mono,
    estimate_tempo,
    onset_strength,
    spectrogram,
    track_beats,
)
from facade_scan.animate.layers import _load_sprite
from facade_scan.config import Config

# What `analyse` adds to a raw frame index to get a wall-clock time. Imported
# rather than restated so the test cannot drift from the pipeline it checks.
FRAME_OFFSET = (WINDOW - HOP) / SAMPLE_RATE


def click_track(bpm: float, seconds: float = 12.0,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A synthetic metronome: the one signal whose tempo we know exactly."""
    samples = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    period = int(sample_rate * 60.0 / bpm)
    decay = np.exp(-np.arange(600) / 90.0).astype(np.float32)
    tone = np.sin(2 * np.pi * 1400 * np.arange(600) / sample_rate).astype(np.float32)
    for start in range(0, samples.size - 600, period):
        samples[start:start + 600] += decay * tone
    return samples


def analysis_of(samples: np.ndarray) -> tuple[np.ndarray, float]:
    magnitude = spectrogram(samples)
    return onset_strength(magnitude), SAMPLE_RATE / HOP


# --------------------------------------------------------------------------- #
# Audio analysis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
def test_tempo_is_recovered_from_a_click_track(bpm):
    onset, frame_rate = analysis_of(click_track(bpm))
    assert estimate_tempo(onset, frame_rate) == pytest.approx(bpm, rel=0.05)


@pytest.mark.parametrize("bpm", [100.0, 120.0, 140.0])
def test_beats_land_on_the_clicks(bpm):
    """Each beat is compared against the click it is nearest to.

    Measuring against the true click times, rather than against a multiple of
    the tracker's own first beat, is what makes this catch a constant timing
    lead -- the earlier version of this test could not tell a perfectly spaced
    run of beats sitting 65 ms early from a correct one.
    """
    period = 60.0 / bpm
    seconds = 16.0
    onset, frame_rate = analysis_of(click_track(bpm, seconds=seconds))
    beats = track_beats(onset, frame_rate, estimate_tempo(onset, frame_rate))
    beats = beats + FRAME_OFFSET
    clicks = np.arange(0, seconds - 0.5, period)

    assert beats.size > 20
    error = np.array([b - clicks[np.argmin(np.abs(clicks - b))] for b in beats])
    # The 23 ms hop is the resolution floor; this should be well inside it, and
    # unbiased rather than merely close.
    assert abs(np.median(error)) < 0.012, "beats sit consistently off the clicks"
    assert np.median(np.abs(error)) < 0.015
    assert np.std(np.diff(beats)) < 0.03


def test_onset_strength_spikes_when_a_note_starts():
    """Spectral flux counts energy appearing, so a note starting stands well
    clear of the steady tone that follows it.

    Note it also fires somewhat when a note stops abruptly -- a hard cutoff is
    itself a broadband transient -- so this compares a start against
    steady state rather than against an ending.
    """
    quiet = np.zeros(SAMPLE_RATE, np.float32)
    loud = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    onset = onset_strength(spectrogram(np.concatenate([quiet, loud])))

    boundary = int(SAMPLE_RATE / HOP)
    at_start = onset[boundary - 2:boundary + 3].max()
    steady = np.median(onset[boundary + 10:])
    assert at_start > 20 * max(steady, 1e-6)


def test_silence_does_not_produce_a_confident_tempo():
    onset, frame_rate = analysis_of(np.zeros(SAMPLE_RATE * 4, np.float32))
    tempo = estimate_tempo(onset, frame_rate)
    assert 60.0 <= tempo <= 200.0, "must still return something usable"


def test_beat_phase_and_index_track_the_grid():
    beats = np.arange(0.0, 10.0, 0.5)
    a = AudioAnalysis(path=None, duration=10.0, tempo=120.0, beats=beats,  # type: ignore[arg-type]
                      onset=np.zeros(10), energy=np.zeros(10),
                      brightness=np.zeros(10), frame_rate=10.0)
    assert a.beat_phase(1.0) == pytest.approx(0.0, abs=1e-6)
    assert a.beat_phase(1.25) == pytest.approx(0.5, abs=1e-6)
    assert a.beat_index(1.1) == 3
    assert a.beat_period == pytest.approx(0.5)


def test_curve_sampling_interpolates():
    a = AudioAnalysis(path=None, duration=1.0, tempo=120.0, beats=np.zeros(0),  # type: ignore[arg-type]
                      onset=np.array([0.0, 1.0]), energy=np.zeros(2),
                      brightness=np.zeros(2), frame_rate=1.0)
    assert a.at(a.onset, 0.0) == pytest.approx(0.0)
    assert a.at(a.onset, 0.5) == pytest.approx(0.5)
    assert a.at(a.onset, 99.0) == pytest.approx(1.0), "clamps past the end"


def test_a_missing_audio_file_says_so():
    with pytest.raises(FileNotFoundError):
        analyse("/no/such/song.mp3")


# --------------------------------------------------------------------------- #
# Layers
#: Every drawing layer. Named once so a new layer has to be added here, and
#: the tests that mean "only this layer" keep meaning it.
ALL_LAYERS = ("wash", "heat", "bulbs", "snow", "accents", "sparkles", "santa",
              "flyer", "icicles", "star", "window")


def only(*keep, **overrides):
    """A config with every layer off except the named ones.

    The edge margin is off unless asked for: it erodes the mask, which would
    quietly move the edges that layer tests sample. Tests about the margin set
    it explicitly.
    """
    off = {name: False for name in ALL_LAYERS if name not in keep}
    off.setdefault("edge_margin_px", 0.0)
    # Time gates off too: a layer withheld until late in the song would
    # otherwise look, to a test, exactly like a layer that is broken.
    off.setdefault("star_from_frac", 0.0)
    off.setdefault("finale", False)
    # The build schedule withholds layers until their section, which to a test
    # is indistinguishable from the layer being broken.
    off.setdefault("build", False)
    return LayerConfig(**{**off, **overrides})


# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_audio():
    """A steady 120 BPM track with envelopes that sweep the whole range.

    Two sections, quiet then loud, so anything driven by the song's structure
    -- the arc, the build, how far the light reaches -- actually varies here
    instead of sitting inert at its default.
    """
    return AudioAnalysis(
        path=None, duration=8.0, tempo=120.0,  # type: ignore[arg-type]
        beats=np.arange(0.0, 8.0, 0.5),
        onset=np.linspace(0, 1, 80), energy=np.linspace(0, 1, 80),
        brightness=np.full(80, 0.5), frame_rate=10.0,
        sections=np.array([0.0, 4.0]))


@pytest.fixture
def house_mask_and_audio(synthetic_audio):
    mask = np.zeros((240, 320), bool)
    ys, xs = np.mgrid[0:240, 0:320]
    mask[(ys > 120) & (xs > 60) & (xs < 260)] = True           # body
    mask[(ys > 60) & (ys <= 120) & (abs(xs - 160) < (ys - 60) * 2)] = True  # gable
    return mask, synthetic_audio


def test_light_never_lands_outside_the_mask(house_mask_and_audio):
    """The whole point: the background stays dark."""
    mask, audio = house_mask_and_audio
    layers = Layers(mask=mask, audio=audio)
    for t in (0.0, 1.7, 4.0, 7.5):
        frame = layers.frame(t)
        assert frame[~mask].max() == 0, f"light escaped the mask at t={t}"
        assert frame[mask].max() > 0, "nothing lit at all"


def test_the_outline_follows_the_silhouette(house_mask_and_audio):
    mask, _ = house_mask_and_audio
    outline = mask_outline(mask, 120)
    assert len(outline) == 120
    import cv2

    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    inside = dilated[np.clip(outline[:, 1].astype(int), 0, mask.shape[0] - 1),
                     np.clip(outline[:, 0].astype(int), 0, mask.shape[1] - 1)]
    assert inside.mean() > 0.95, "outline points should hug the mask edge"


def test_a_loud_passage_is_brighter_than_a_quiet_one(house_mask_and_audio):
    mask, audio = house_mask_and_audio
    # Finale off: the end of the track is a deliberate blackout, so the last
    # moments are dark by design and would mask what this is checking.
    layers = Layers(mask=mask, audio=audio, cfg=LayerConfig(finale=False))
    quiet = layers.frame(0.2)[mask].mean()      # energy near 0
    loud = layers.frame(7.8)[mask].mean()       # energy near 1
    assert loud > 1.8 * quiet, "the music should drive the intensity"


def test_feature_sizes_scale_with_the_subject(house_mask_and_audio):
    """A bulb sized in panel pixels is a speck on a small subject."""
    mask, audio = house_mask_and_audio
    small = Layers(mask=mask, audio=audio)

    big_mask = np.zeros((960, 1280), bool)
    big_mask[240:900, 200:1100] = True
    big = Layers(mask=big_mask, audio=audio)
    assert big.bulb_radius > small.bulb_radius


def test_layers_can_be_switched_off_individually(house_mask_and_audio):
    mask, audio = house_mask_and_audio
    assert Layers(mask=mask, audio=audio, cfg=only()).frame(2.0).max() == 0
    for layer in ALL_LAYERS:
        if layer == "heat":
            continue          # heat only shows on top of the wash
        # Wind every periodic layer down to a period that fits the window,
        # so "it drew nothing" means the layer is broken rather than that its
        # turn simply had not come round.
        cfg = only(layer, santa_every_beats=1.0, sparkle_every_beats=1.0,
                   flyer_every_beats=2.0, flyer_cross_beats=1.5,
                   flyer_phase_beats=0.0)
        # Across the whole envelope: accents need a loud onset, and the
        # sparkle sweep and Santa are only on screen for part of their cycle.
        frames = [Layers(mask=mask, audio=audio, cfg=cfg).frame(float(t))
                  for t in np.linspace(0.2, 7.8, 20)]
        assert max(f.max() for f in frames) > 0, f"{layer} drew nothing at all"


def test_an_empty_mask_renders_black_rather_than_crashing():
    audio = AudioAnalysis(path=None, duration=2.0, tempo=120.0,  # type: ignore[arg-type]
                          beats=np.arange(0.0, 2.0, 0.5), onset=np.zeros(20),
                          energy=np.zeros(20), brightness=np.zeros(20),
                          frame_rate=10.0)
    frame = Layers(mask=np.zeros((80, 100), bool), audio=audio).frame(1.0)
    assert frame.shape == (80, 100, 3)
    assert frame.max() == 0


def test_frames_differ_over_time(house_mask_and_audio):
    """Static output would mean nothing is actually animating."""
    mask, audio = house_mask_and_audio
    layers = Layers(mask=mask, audio=audio)
    a, b = layers.frame(1.0), layers.frame(1.4)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.5


# --------------------------------------------------------------------------- #
# Configuration reaches the pixels
# --------------------------------------------------------------------------- #
def test_animation_settings_survive_a_toml_round_trip(tmp_path):
    """Every animation threshold is tunable from the same file as the rest.

    The look has to be adjustable on site, in the dark, without editing Python
    -- so these live in the one config tree and must survive being written out
    and read back, tuples included. TOML has no tuple type, so pairs and
    colour triples both come back as lists unless the loader converts them.
    """
    cfg = Config()
    cfg.animate.bulb_count = 77
    cfg.animate.wash_uplight = 1.4
    cfg.animate.bulb_twinkle_hz = (2.0, 5.0)
    cfg.animate.bulb_colour = (10.0, 20.0, 30.0)

    path = tmp_path / "rig.toml"
    path.write_text(cfg.to_toml())
    back = Config.from_toml(path)

    assert back.animate.bulb_count == 77
    assert back.animate.wash_uplight == pytest.approx(1.4)
    assert back.animate.bulb_twinkle_hz == (2.0, 5.0)
    assert back.animate.bulb_colour == (10.0, 20.0, 30.0)


def test_turning_layers_off_actually_darkens_the_frame(synthetic_audio):
    """Guards the wiring: a setting nothing reads is worse than no setting."""
    mask = np.zeros((240, 320), bool)
    mask[60:200, 80:240] = True

    lit = Layers(mask=mask, audio=synthetic_audio, cfg=LayerConfig()).frame(4.0)
    dark = Layers(mask=mask, audio=synthetic_audio, cfg=only()).frame(4.0)
    assert lit.max() > 0
    assert dark.max() == 0, "layers were drawn despite being switched off"


def test_the_wash_is_a_shape_not_a_fill(synthetic_audio):
    """Most of the subject is left dark, and the hue never shifts.

    Filling the whole silhouette evenly is the single most costly mistake
    available: it leaves a coloured panel with a decorated fringe, no focal
    point, and nowhere dark for a bright element to read against. So the wash
    is anchored at the base and falls away well before the roofline.

    It is also a gain on one colour, never a blend towards another: blending
    saturated hues in RGB muds, and crimson into holly lays olive across the
    middle. A gain cannot shift the hue, so the channel ratios at the base
    must match those at the top even though the level does not.
    """
    mask = np.zeros((240, 320), bool)
    mask[60:200, 80:240] = True
    layers = Layers(mask=mask, audio=synthetic_audio, cfg=only("wash", zones=False))

    # The fixture's loudness ramps from 0 to 1 across its eight seconds, so
    # these are a quiet moment and a loud one.
    quiet = layers.frame(0.6)
    loud = layers.frame(7.6)

    top = quiet[70, 120:200].mean(axis=0).astype(float)
    bottom = quiet[190, 120:200].mean(axis=0).astype(float)
    assert bottom.sum() > top.sum() * 2.0, "the base should be much the brighter end"

    # Quiet: most of the subject genuinely dark, not merely graded.
    inside = quiet.max(axis=2)[mask]
    assert float((inside < 0.25 * max(inside.max(), 1)).mean()) > 0.3, \
        "too much of the subject is lit; this is a fill, not a shape"

    # Loud: the light has climbed the building. Measured as how much of the
    # subject the wash actually covers, not as the topmost lit row -- the
    # roofline carries a little light at all times so that it always reads as
    # a building, which pins the topmost row and hides the climb.
    def covered(frame):
        return float((frame.max(axis=2)[mask] > 90).mean())
    assert covered(loud) > covered(quiet) + 0.15, "the light never climbed"

    # Hue held across the pool: a gain cannot shift it, and a blend towards
    # another colour would. Measured on the loud frame, where there is signal
    # to measure -- the quiet one is a few counts above black by design, and
    # there 8-bit rounding swamps any ratio.
    rows = loud[:, 120:200].mean(axis=1).sum(axis=1)
    lit_rows = np.nonzero(rows > rows.max() * 0.35)[0]
    assert lit_rows.size, "nothing is lit at all"
    # Two rows that are both lit and unclipped, found rather than assumed:
    # once a channel saturates the ratios describe the clip, not the hue, and
    # where the pool is bright enough to clip moves with the settings.
    usable = []
    for row in range(loud.shape[0]):
        band = loud[row, 120:200].astype(float)
        keep = (band.max(axis=1) < 250) & (band.sum(axis=1) > 40)
        if keep.sum() >= 8:
            px = band[keep].mean(axis=0) / 255.0
            usable.append((row, colorsys.rgb_to_hsv(px[2], px[1], px[0])[0] * 360.0))

    assert len(usable) >= 2, "nowhere lit and unclipped to compare"
    apart = abs(usable[0][1] - usable[-1][1]) % 360.0
    apart = min(apart, 360.0 - apart)
    assert apart < 14.0, (
        f"hue shifted {apart:.0f} degrees between rows "
        f"{usable[0][0]} and {usable[-1][0]}")


def test_lights_hang_on_the_roof_not_round_the_whole_shape(synthetic_audio):
    """Verges and eaves only.

    A chain round the entire silhouette runs down both walls and straight
    across the ground, which flattens the building into a neon sign and puts
    the brightest band in the frame along the floor -- exactly inverting the
    focal hierarchy.
    """
    mask = np.zeros((240, 320), bool)
    ys, xs = np.mgrid[0:240, 0:320]
    mask[(ys > 120) & (xs > 60) & (xs < 260)] = True
    mask[(ys > 60) & (ys <= 120) & (abs(xs - 160) < (ys - 60) * 2)] = True

    layers = Layers(mask=mask, audio=synthetic_audio,
                    cfg=only("bulbs", bulb_verges_only=True))
    lit = layers.frame(2.0).max(axis=2) > 0
    rows = np.nonzero(lit.any(axis=1))[0]
    base = np.nonzero(mask.any(axis=1))[0][-1]

    assert rows.size, "no lights drawn"
    assert rows.max() < base - 10, "lights are sitting on the ground line"


def test_the_panel_around_the_subject_costs_nothing(synthetic_audio):
    """The same subject renders identically however much panel surrounds it.

    Compositing happens inside the subject's bounding box, because the
    projector usually covers far more than the thing being lit and every pixel
    outside the mask is multiplied away at the end anyway. That is only safe if
    the surrounding panel changes nothing, so: the same subject placed on a
    panel four times the area must come out pixel-identical, and the extra
    panel must stay black.
    """
    shape = (120, 160)
    subject = np.zeros(shape, bool)
    subject[30:100, 40:120] = True

    big = np.zeros((shape[0] * 2, shape[1] * 2), bool)
    big[:shape[0], :shape[1]] = subject

    small_frame = Layers(mask=subject, audio=synthetic_audio).frame(3.0)
    big_frame = Layers(mask=big, audio=synthetic_audio).frame(3.0)

    assert np.array_equal(small_frame, big_frame[:shape[0], :shape[1]])
    assert big_frame[shape[0]:].max() == 0
    assert big_frame[:, shape[1]:].max() == 0


def test_no_light_escapes_the_mask(synthetic_audio):
    """The whole point of the scan: light lands on the subject and nowhere else.

    Checked over a stretch of the timeline rather than one frame, because the
    layers that move -- snow drifting, the chase coming round -- are the ones
    that would leak.
    """
    mask = np.zeros((200, 260), bool)
    ys, xs = np.mgrid[0:200, 0:260]
    mask[(ys > 100) & (xs > 50) & (xs < 210)] = True
    mask[(ys > 40) & (ys <= 100) & (abs(xs - 130) < (ys - 40) * 2)] = True
    layers = Layers(mask=mask, audio=synthetic_audio)

    for t in np.linspace(0.0, 7.5, 24):
        frame = layers.frame(float(t))
        assert frame[~mask].max() == 0, f"light fell outside the subject at t={t:.1f}"
        assert frame[mask].max() > 0, f"subject went dark at t={t:.1f}"


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _write_wav(path, samples, rate, channels=1):
    import wave
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def test_wav_decodes_without_ffmpeg(tmp_path, monkeypatch):
    """The fallback the error message recommends has to actually work.

    The message used to say "supply a 16-bit mono WAV instead" while the code
    refused before ever trying to read one, so the advice was a dead end.
    """
    monkeypatch.setattr("facade_scan.animate.audio.shutil.which", lambda _: None)
    tone = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
    path = tmp_path / "tone.wav"
    _write_wav(path, tone, SAMPLE_RATE)

    out = decode_to_mono(path)
    assert out.size == pytest.approx(SAMPLE_RATE, rel=0.01)
    assert np.abs(out).max() > 0.9


def test_wav_is_downmixed_and_resampled(tmp_path, monkeypatch):
    """Stereo at another rate still has to come back as mono at ours."""
    monkeypatch.setattr("facade_scan.animate.audio.shutil.which", lambda _: None)
    rate = 44100
    t = np.arange(rate) / rate
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=-1)
    path = tmp_path / "stereo.wav"
    _write_wav(path, stereo.ravel(), rate, channels=2)

    out = decode_to_mono(path, SAMPLE_RATE)
    assert out.ndim == 1
    assert out.size == pytest.approx(SAMPLE_RATE, rel=0.01), "should be resampled"
    assert np.abs(out).max() > 0.9, "downmixing identical channels must not cancel"


def test_compressed_audio_without_ffmpeg_explains_the_way_out(tmp_path, monkeypatch):
    monkeypatch.setattr("facade_scan.animate.audio.shutil.which", lambda _: None)
    path = tmp_path / "track.mp3"
    path.write_bytes(b"not really an mp3")

    with pytest.raises(RuntimeError) as excinfo:
        decode_to_mono(path)
    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "WAV" in message, "should name the fallback that needs no ffmpeg"


# --------------------------------------------------------------------------- #
# Sparkles and Santa
# --------------------------------------------------------------------------- #
def test_the_sparkle_sweep_crosses_and_then_clears(house_mask_and_audio):
    """It should travel, and it should leave -- a permanent trail is texture."""
    mask, audio = house_mask_and_audio
    cfg = only("sparkles", sparkle_every_beats=8.0, sparkle_cross_beats=3.0)
    layers = Layers(mask=mask, audio=audio, cfg=cfg)

    # Per sample: the centre of the lit columns, or None when nothing is lit.
    track = []
    for t in np.linspace(0.0, 4.0, 40):
        xs = np.nonzero(layers.frame(float(t)).max(axis=2).max(axis=0))[0]
        track.append(xs.mean() if xs.size else None)

    assert any(c is not None for c in track), "the sweep never appeared"
    assert any(c is None for c in track), "the sweep never cleared"

    # Measure within one sweep. Across the whole window the cycle repeats, so
    # the last sample is a *new* sweep back at the left -- comparing it with
    # the first would compare two different sweeps and look like no motion.
    first = next(i for i, c in enumerate(track) if c is not None)
    run = []
    for centre in track[first:]:
        if centre is None:
            break
        run.append(centre)

    assert len(run) >= 4, "the sweep was on screen for almost no time"
    assert run[-1] > run[0] + mask.shape[1] * 0.1, "it flickered in place"
    # Travelling one way, not jittering.
    assert all(b >= a - 1.0 for a, b in pairwise(run))


def test_santa_rises_and_ducks_back(house_mask_and_audio):
    """He comes up from the bottom edge, holds, and goes back down."""
    mask, audio = house_mask_and_audio
    cfg = only("santa", santa_every_beats=8.0, santa_rise_beats=2.0,
               santa_hold_beats=2.0, santa_duck_beats=2.0)
    layers = Layers(mask=mask, audio=audio, cfg=cfg)

    heights = []
    for t in np.linspace(0.0, 4.0, 40):
        lit = np.nonzero(layers.frame(float(t)).max(axis=2).any(axis=1))[0]
        heights.append(0.0 if not lit.size else float(mask.shape[0] - lit.min()))

    assert max(heights) > 0, "Santa never appeared"
    peak = int(np.argmax(heights))
    assert heights[0] < max(heights), "he should start down"
    assert min(heights[peak:]) < max(heights), "he should duck back down"


def test_sprites_composite_opaquely_and_silhouettes_additively(synthetic_audio):
    """Santa occludes what is behind him; the sleigh adds light to it.

    The two are composited oppositely and both are deliberate. Santa is a
    character standing in front of the wash, so his dark features have to
    replace what is behind them -- added, his eyes and the shadow under the
    hat brim simply would not appear and the face would not read as a face.
    The sleigh is the reverse: the art is a black silhouette, and black is the
    one colour a projector cannot produce, so compositing it would leave a
    sleigh-shaped hole. Its alpha is filled with light instead.

    Tested on the compositor rather than on a rendered frame, because the wash
    is now dark by design: Santa's darkest feature is brighter than the ground
    he lands on, so "did anything get darker" no longer decides it.
    """
    mask = np.zeros((80, 80), bool)
    mask[:, :] = True
    layers = Layers(mask=mask, audio=synthetic_audio, cfg=only())

    bright = np.full((40, 40, 3), 0.9, np.float32)
    dark_sprite = np.zeros((10, 10, 3), np.float32)
    solid = np.ones((10, 10), np.float32)

    opaque = bright.copy()
    layers._blit(opaque, dark_sprite, solid, 5, 5, add=False)
    assert opaque[10, 10].sum() == pytest.approx(0.0), "did not occlude"

    added = bright.copy()
    layers._blit(added, dark_sprite, solid, 5, 5, add=True)
    assert added[10, 10].sum() == pytest.approx(bright[10, 10].sum()), \
        "an added black sprite must leave the background untouched"

    lit_sprite = np.full((10, 10, 3), 0.5, np.float32)
    glow = np.zeros((40, 40, 3), np.float32)
    layers._blit(glow, lit_sprite, solid, 5, 5, add=True)
    assert glow[10, 10].sum() > 0, "an added lit sprite must add light"


def test_a_clip_rectangle_frames_a_sprite(synthetic_audio):
    """Santa is framed by the window, so the blit has to respect its edges."""
    mask = np.ones((80, 80), bool)
    layers = Layers(mask=mask, audio=synthetic_audio, cfg=only())

    canvas = np.zeros((40, 40, 3), np.float32)
    sprite = np.ones((20, 20, 3), np.float32)
    solid = np.ones((20, 20), np.float32)
    layers._blit(canvas, sprite, solid, 5, 5, add=True, clip=(10, 16, 10, 16))

    lit = canvas.max(axis=2) > 0
    ys, xs = np.nonzero(lit)
    assert ys.min() >= 10 and ys.max() < 16
    assert xs.min() >= 10 and xs.max() < 16


def test_the_edge_margin_keeps_light_off_the_wall_behind(house_mask_and_audio):
    """The subject stands proud of the background, so the lit area stops short.

    A ray aimed just inside the silhouette still clears the edge and lands on
    whatever is behind, so the animation is held back from the true outline.
    """
    mask, audio = house_mask_and_audio
    flush = Layers(mask=mask, audio=audio, cfg=LayerConfig(edge_margin_px=0.0))
    pulled = Layers(mask=mask, audio=audio, cfg=LayerConfig(edge_margin_px=6.0))

    lit_flush = flush.frame(3.0).max(axis=2) > 0
    lit_pulled = pulled.frame(3.0).max(axis=2) > 0
    assert lit_pulled.sum() < lit_flush.sum(), "the margin lit no less area"
    # Everything still lands on the subject, and nothing new appears outside it.
    assert not (lit_pulled & ~mask).any()
    import cv2
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert (lit_pulled & ~eroded).sum() < (lit_flush & ~eroded).sum()


def test_the_bundled_santa_image_is_usable(house_mask_and_audio):
    """The shipped default has to load from anywhere and be a real cutout.

    A bare filename resolves against the package's own assets, so the default
    survives being run from another directory or from an installed package.
    """
    colour, alpha = _load_sprite(LayerConfig().santa_image)
    assert colour.shape[:2] == alpha.shape
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    corners = [alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]]
    assert max(corners) == 0.0, "not cut out; it would composite as a rectangle"
    # Solid fills, not line art: outlines with hollow interiors vanish at
    # throw distance because the wash shows straight through the face.
    assert float((alpha > 0.5).mean()) > 0.2


def test_a_santa_image_without_alpha_is_refused(tmp_path):
    """Better to say so than to project a white rectangle."""
    import cv2
    path = tmp_path / "opaque.png"
    cv2.imwrite(str(path), np.full((40, 30, 3), 200, np.uint8))

    with pytest.raises(ValueError, match="alpha"):
        _load_sprite(str(path))


def test_a_missing_santa_image_is_refused(tmp_path):
    """Silently drawing the built-in figure would be baffling."""
    with pytest.raises(FileNotFoundError):
        _load_sprite(str(tmp_path / "nope.png"))


def test_the_santa_image_enters_from_below(house_mask_and_audio):
    """He slides up, so his hat clears the edge before the rest of him."""
    mask, audio = house_mask_and_audio
    cfg = only("santa", santa_every_beats=8.0, santa_rise_beats=3.0,
               santa_hold_beats=2.0, santa_image=LayerConfig().santa_image)
    layers = Layers(mask=mask, audio=audio, cfg=cfg)

    heights = []
    for t in np.linspace(0.0, 2.6, 14):
        lit = np.nonzero(layers.frame(float(t)).max(axis=2).any(axis=1))[0]
        heights.append(0 if not lit.size else int(mask.shape[0] - lit.min()))

    assert max(heights) > 0, "the image never appeared"
    assert heights[0] < max(heights), "he should start low and come up"
    # Rising steadily rather than fading in on the spot: measured up to the
    # peak only, because he ducks again afterwards.
    peak = int(np.argmax(heights))
    assert peak > 0
    assert all(b >= a for a, b in pairwise(heights[:peak + 1]))


def test_the_flyer_crosses_the_subject_on_an_arc(house_mask_and_audio):
    """It travels the full width and rises through the middle of the crossing."""
    mask, audio = house_mask_and_audio
    cfg = only("flyer", flyer_every_beats=6.0, flyer_cross_beats=4.0,
               flyer_phase_beats=0.0)
    layers = Layers(mask=mask, audio=audio, cfg=cfg)

    track = []
    for t in np.linspace(0.0, 2.0, 24):
        lit = layers.frame(float(t)).max(axis=2) > 0
        if not lit.any():
            continue
        ys, xs = np.nonzero(lit)
        track.append((xs.mean(), ys.mean()))

    assert len(track) >= 6, "the flyer was barely on screen"
    xs = [p[0] for p in track]
    ys = [p[1] for p in track]
    assert xs[-1] > xs[0], "it should travel across"
    # An arc, not a straight line: highest (smallest y) somewhere in the middle.
    peak = int(np.argmin(ys))
    assert 0 < peak < len(ys) - 1, "it flew flat rather than arcing"


def test_the_flyer_is_lit_rather_than_a_hole_in_the_wash(house_mask_and_audio):
    """The art is a black silhouette; black is what a projector cannot make.

    Composited normally it would be a sleigh-shaped dark patch. It has to be
    filled with light instead, so it must be *brighter* than the wash alone.
    """
    mask, audio = house_mask_and_audio
    cfg = only("wash", "flyer", flyer_every_beats=6.0, flyer_cross_beats=4.0,
               flyer_phase_beats=0.0)
    lit_layers = Layers(mask=mask, audio=audio, cfg=cfg)
    wash_only = Layers(mask=mask, audio=audio, cfg=only("wash"))

    # Sampled across a crossing: it is off-canvas at either end, so a single
    # frame can legitimately show nothing.
    seen = False
    for t in np.linspace(0.0, 2.0, 21):
        delta = (lit_layers.frame(float(t)).astype(int).sum(axis=2)
                 - wash_only.frame(float(t)).astype(int).sum(axis=2))
        assert delta.min() >= 0, f"it darkened the wash at t={t:.2f}: a hole, not a light"
        seen = seen or bool(delta.max() > 0)
    assert seen, "the flyer added no light at any point in the crossing"


def test_icicles_hang_from_the_scanned_roofline(house_mask_and_audio):
    """They follow the real edge -- the reason for scanning in the first place.

    Each icicle must start at the silhouette's own top edge in its column, so
    they trace the gable rather than sitting on a straight line across.
    """
    mask, audio = house_mask_and_audio
    layers = Layers(mask=mask, audio=audio, cfg=only("icicles"))
    assert len(layers.icicles) > 5

    top, _, left, _ = layers.box
    for points, _phase in layers.icicles:
        # points are [left corner, right corner, tip]; the tip is the centre.
        column = round(points[2][0]) + left
        rows = np.nonzero(mask[:, column])[0]
        assert rows.size
        assert abs((points[0][1] + top) - rows[0]) <= 1.5, "not on the edge"
        assert points[2][1] > points[0][1], "an icicle hangs downward"

    # A gable means the tops sit at genuinely different heights.
    tops = [p[0][1] for p, _ in layers.icicles]
    assert np.std(tops) > 1.0, "they sat on a flat line, ignoring the roofline"


def test_the_star_sits_on_the_apex(house_mask_and_audio):
    """The highest point of the silhouette, which the scan already knows."""
    mask, audio = house_mask_and_audio
    layers = Layers(mask=mask, audio=audio, cfg=only("star"))

    lit = layers.frame(1.0).max(axis=2) > 0
    assert lit.any(), "no star drawn"
    ys, xs = np.nonzero(lit)

    rows = np.nonzero(mask.any(axis=1))[0]
    apex_cols = np.nonzero(mask[rows[0]])[0]
    assert abs(xs.mean() - apex_cols.mean()) < mask.shape[1] * 0.1
    assert ys.mean() < rows[0] + mask.shape[0] * 0.25, "not near the top"


def test_every_animation_setting_survives_a_toml_round_trip(tmp_path):
    """All of them, not a sampled few.

    The config grew to well over a hundred animation fields, several of them
    tuples of tuples -- palettes, colour states, the order elements arrive in.
    TOML has no tuple type, so anything the loader does not convert comes back
    as a list and a loaded config stops being the same object as a default
    one. Comparing every field catches that; spot-checking three did not.
    """
    import dataclasses

    cfg = Config()
    path = tmp_path / "rig.toml"
    path.write_text(cfg.to_toml())
    back = Config.from_toml(path)

    mismatched = [f.name for f in dataclasses.fields(cfg.animate)
                  if getattr(cfg.animate, f.name) != getattr(back.animate, f.name)]
    assert not mismatched, f"did not survive the round trip: {mismatched}"

    # And the types, not just the values: a list that compares equal to a
    # tuple would still be the wrong thing.
    for field in dataclasses.fields(cfg.animate):
        original = getattr(cfg.animate, field.name)
        loaded = getattr(back.animate, field.name)
        assert type(original) is type(loaded), f"{field.name} changed type"
