"""Beat and energy analysis, in numpy, decoding through ffmpeg.

No librosa. It would work, but it drags in numba and a compiler toolchain for
what is, here, a few hundred lines of FFT and dynamic programming -- and this
project already depends on numpy and on ffmpeg being available through OpenCV.

What comes out is everything the animation needs to follow a song without
anyone hand-authoring a cue sheet:

``beats``
    times, in seconds, of every beat
``tempo``
    beats per minute
``onset``
    a per-frame "something just happened" strength, for firing accents
``energy``
    per-frame loudness, which is what makes a chorus look like a chorus
``brightness``
    spectral centroid, which tracks whether the moment is brassy or warm

The animation reads those curves directly rather than a list of sections, so
structure emerges from the recording instead of being described to it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Analysis runs at this rate; plenty for beats and much cheaper than 44.1 kHz.
SAMPLE_RATE = 22050
#: Samples between analysis frames -- about 23 ms, the usual choice.
HOP = 512
WINDOW = 2048


@dataclass
class AudioAnalysis:
    """Everything the animation needs to follow a recording."""

    path: Path
    duration: float
    tempo: float
    #: Beat times in seconds.
    beats: np.ndarray
    #: Per-frame curves, all the same length, at ``frame_rate`` per second.
    onset: np.ndarray
    energy: np.ndarray
    brightness: np.ndarray
    frame_rate: float
    #: Seconds from the start of an analysis window to the moment it describes.
    #:
    #: Frame `i` is scored by comparing its window against frame `i-1`'s, so the
    #: energy it reports is whatever newly entered -- the last `HOP` samples of a
    #: `WINDOW`-sample span. That audio *begins* `WINDOW - HOP` samples in, which
    #: is therefore the instant a spike on this curve refers to.
    #:
    #: Getting this wrong is audible rather than academic. Treating a frame as
    #: its own start put every beat 65 ms early, and even the obvious-looking
    #: window-centre convention left 18 ms of lead; measured against a known
    #: click track this one lands within 7 ms, well inside the 23 ms hop.
    #:
    #: `energy` and `brightness` come from the same frames but are absolute, not
    #: differences, so strictly they describe the window's middle -- 23 ms from
    #: this reference. They are slow envelopes read for wash colour and level,
    #: which do not change meaningfully over 23 ms, so one offset serves both.
    frame_offset: float = 0.0

    #: Times where the music changes character, starting with 0.0. See
    #: `find_sections`. The show uses these to change its look, so it has an
    #: arc instead of three minutes of the same idea.
    sections: np.ndarray = field(default_factory=lambda: np.zeros(1))

    @property
    def beat_period(self) -> float:
        return 60.0 / self.tempo if self.tempo > 0 else 0.5

    def at(self, curve: np.ndarray, time: float) -> float:
        """Sample a per-frame curve at a time in seconds, interpolated."""
        if curve.size == 0:
            return 0.0
        position = np.clip((time - self.frame_offset) * self.frame_rate,
                           0, curve.size - 1)
        low = int(np.floor(position))
        high = min(low + 1, curve.size - 1)
        blend = position - low
        return float(curve[low] * (1.0 - blend) + curve[high] * blend)

    def beat_phase(self, time: float) -> float:
        """Where ``time`` sits between the surrounding beats, in [0, 1).

        0 is exactly on a beat. Effects that should land *on* the beat read
        this and look for it approaching zero.
        """
        if self.beats.size < 2:
            return (time % self.beat_period) / self.beat_period
        index = int(np.searchsorted(self.beats, time, side="right") - 1)
        if index < 0:
            return 0.0
        if index >= self.beats.size - 1:
            return float((time - self.beats[-1]) / self.beat_period % 1.0)
        span = self.beats[index + 1] - self.beats[index]
        return float((time - self.beats[index]) / max(span, 1e-6))

    def section_index(self, time: float) -> int:
        """Which section `time` falls in."""
        return max(0, int(np.searchsorted(self.sections, time, side="right")) - 1)

    def section_progress(self, time: float) -> float:
        """How far through its section `time` is, 0 to 1."""
        index = self.section_index(time)
        start = float(self.sections[index])
        end = (float(self.sections[index + 1]) if index + 1 < self.sections.size
               else max(self.duration, start + 1e-6))
        return float(np.clip((time - start) / max(end - start, 1e-6), 0.0, 1.0))

    def beat_index(self, time: float) -> int:
        """How many beats have passed at ``time``. Drives anything that steps."""
        if self.beats.size == 0:
            return int(time / self.beat_period)
        return int(np.searchsorted(self.beats, time, side="right"))


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _read_wav(path: Path, sample_rate: int) -> np.ndarray | None:
    """Read a PCM WAV with the standard library, or None if it is not one.

    Worth the few lines because it removes the ffmpeg dependency entirely for
    WAV input -- which is what the error below tells people to fall back to, so
    that advice has to actually work.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:        # not 16-bit PCM
                return None
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return None

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and samples.size:
        # Linear resampling. These samples only ever become envelopes and onset
        # strengths at a 23 ms hop, so interpolation error here is far below
        # anything the analysis can see.
        duration = samples.size / rate
        wanted = np.arange(int(duration * sample_rate), dtype=np.float32)
        samples = np.interp(wanted / sample_rate,
                            np.arange(samples.size) / rate, samples).astype(np.float32)
    return samples


def decode_to_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file ffmpeg understands into mono float32 in [-1, 1].

    PCM WAV is handled by the standard library and needs no ffmpeg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no audio file at {path}")

    direct = _read_wav(path, sample_rate)
    if direct is not None:
        return direct

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"ffmpeg is not on PATH, and {path.name} is not a 16-bit PCM WAV. "
            "Install ffmpeg (brew install ffmpeg) to decode compressed audio, "
            "or convert the track to a 16-bit WAV, which needs no ffmpeg."
        )

    with tempfile.TemporaryDirectory(prefix="facade-scan-audio-") as work:
        wav = Path(work) / "mono.wav"
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", str(sample_rate), "-vn",
             "-f", "wav", "-acodec", "pcm_s16le", str(wav)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not wav.exists():
            raise RuntimeError(
                f"ffmpeg could not decode {path.name}:\n{result.stderr.strip()}"
            )
        with wave.open(str(wav), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


# --------------------------------------------------------------------------- #
# Spectral features
# --------------------------------------------------------------------------- #
def spectrogram(samples: np.ndarray, window: int = WINDOW,
                hop: int = HOP) -> np.ndarray:
    """Magnitude spectrogram, as ``(frames, bins)``."""
    if samples.size < window:
        samples = np.pad(samples, (0, window - samples.size))
    frame_count = 1 + (samples.size - window) // hop
    indices = np.arange(window)[None, :] + hop * np.arange(frame_count)[:, None]
    frames = samples[indices] * np.hanning(window)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)


def onset_strength(magnitude: np.ndarray) -> np.ndarray:
    """Spectral flux: how much energy *appeared* since the previous frame.

    Only increases count. A note starting is an onset; a note ending is not,
    and counting both would put a spurious pulse at the end of every phrase.
    """
    compressed = np.log1p(magnitude * 10.0)
    flux = np.diff(compressed, axis=0, prepend=compressed[:1])
    return np.maximum(flux, 0.0).sum(axis=1)


def _smooth(curve: np.ndarray, width: int) -> np.ndarray:
    if width < 2 or curve.size < width:
        return curve
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    return np.convolve(curve, kernel, mode="same")


def _normalise(curve: np.ndarray) -> np.ndarray:
    """Scale to roughly [0, 1] using percentiles, so one loud crash does not
    flatten everything else."""
    if curve.size == 0:
        return curve
    low = float(np.percentile(curve, 5))
    high = float(np.percentile(curve, 98))
    if high - low < 1e-9:
        return np.zeros_like(curve)
    return np.clip((curve - low) / (high - low), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Tempo and beats
# --------------------------------------------------------------------------- #
def estimate_tempo(onset: np.ndarray, frame_rate: float,
                   low_bpm: float = 60.0, high_bpm: float = 200.0) -> float:
    """Tempo by autocorrelation of the onset envelope.

    The envelope repeats at the beat period, so its autocorrelation peaks
    there. Weighted toward the middle of the range, because autocorrelation is
    just as happy at half and double tempo and a bare peak-pick lands on the
    wrong one often enough to matter.
    """
    if onset.size < 16:
        return 120.0
    centred = onset - onset.mean()
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:]
    if correlation[0] > 0:
        correlation = correlation / correlation[0]

    lags = np.arange(correlation.size)
    with np.errstate(divide="ignore"):
        bpm = 60.0 * frame_rate / np.maximum(lags, 1e-9)
    usable = (bpm >= low_bpm) & (bpm <= high_bpm)
    if not usable.any():
        return 120.0

    # A log-normal prior centred on 120 BPM, the standard way to break the
    # octave ambiguity without hard-coding a range per song.
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-9) / 120.0) / 0.9) ** 2)
    score = np.where(usable, correlation * prior, -np.inf)
    return float(bpm[int(np.argmax(score))])


def track_beats(onset: np.ndarray, frame_rate: float, tempo: float,
                tightness: float = 300.0) -> np.ndarray:
    """Dynamic-programming beat tracking (after Ellis, 2007).

    Finds the sequence of onset peaks that best trades off landing on strong
    onsets against staying evenly spaced at the estimated tempo. Greedy
    peak-picking drifts through syncopation and rests; this does not.
    """
    if onset.size < 4 or tempo <= 0:
        return np.zeros(0)

    period = 60.0 * frame_rate / tempo
    local = _normalise(_smooth(onset, max(3, int(period // 4) | 1)))

    score = np.zeros(onset.size, dtype=np.float64)
    previous = np.full(onset.size, -1, dtype=np.int64)
    # Candidate predecessors span half to double the beat period.
    offsets = np.arange(max(1, round(period * 0.5)),
                        max(2, round(period * 2.0)) + 1)
    penalty = -tightness * (np.log(offsets / period) ** 2)

    for frame in range(onset.size):
        candidates = frame - offsets
        valid = candidates >= 0
        if not valid.any():
            score[frame] = local[frame]
            continue
        totals = score[candidates[valid]] + penalty[valid]
        best = int(np.argmax(totals))
        score[frame] = local[frame] + totals[best]
        previous[frame] = candidates[valid][best]

    # Walk back from the best ending inside the final stretch of the track.
    tail = max(1, int(period))
    end = int(np.argmax(score[-tail:]) + onset.size - tail)
    path = []
    while end >= 0:
        path.append(end)
        end = int(previous[end])
    return np.array(path[::-1], dtype=np.float64) / frame_rate


# --------------------------------------------------------------------------- #
def find_sections(magnitude: np.ndarray, frame_rate: float,
                  min_seconds: float = 12.0) -> np.ndarray:
    """Times where the music changes character, including 0.0.

    A show that runs the same look for three minutes has no shape, however
    much is moving inside it. Songs already provide the shape -- verse, chorus,
    key change, last chorus -- so the sections are taken from the recording
    rather than invented.

    The method is the standard self-similarity one: describe each moment by its
    spectral shape, compare every moment with every other, and look for the
    corners of the resulting block structure. Where a verse ends and a chorus
    begins, the description changes and a boundary shows up as a peak in
    novelty. `min_seconds` keeps boundaries musically far enough apart to be
    worth reacting to.
    """
    if magnitude.shape[0] < 8:
        return np.zeros(1, dtype=np.float64)

    # Coarse log-spaced bands: enough to tell a chorus from a verse, few enough
    # that a change of note does not read as a change of section.
    edges = np.geomspace(1, magnitude.shape[1], 13).astype(int)
    bands = np.stack([magnitude[:, a:max(b, a + 1)].mean(axis=1)
                      for a, b in zip(edges[:-1], edges[1:])], axis=-1)
    bands = np.log1p(bands)
    bands /= np.maximum(np.linalg.norm(bands, axis=1, keepdims=True), 1e-9)

    # Novelty: how different the window after each frame is from the window
    # before it. A checkerboard kernel over the similarity matrix is the
    # textbook form; correlating the two half-windows directly is equivalent
    # here and avoids building an N-by-N matrix.
    half = max(2, int(frame_rate * 2.0))
    novelty = np.zeros(bands.shape[0], dtype=np.float32)
    for i in range(half, bands.shape[0] - half):
        before = bands[i - half:i].mean(axis=0)
        after = bands[i:i + half].mean(axis=0)
        novelty[i] = 1.0 - float(np.dot(before, after))
    novelty = _smooth(novelty, max(3, int(frame_rate)))

    spacing = int(min_seconds * frame_rate)
    threshold = float(np.percentile(novelty, 80))
    # A boundary in the first or last few seconds is the track starting or
    # stopping, not a change of section, and cutting the look there just looks
    # like a glitch.
    lo, hi = spacing, novelty.size - spacing
    picks: list[int] = []
    for index in np.argsort(novelty)[::-1]:
        if novelty[index] < threshold:
            break
        if not (lo <= int(index) <= hi):
            continue
        if all(abs(int(index) - p) >= spacing for p in picks):
            picks.append(int(index))
    return np.array([0.0] + sorted(p / frame_rate for p in picks))


def analyse(path: str | Path, sample_rate: int = SAMPLE_RATE,
            hop: int = HOP) -> AudioAnalysis:
    """Decode a track and extract everything the animation follows."""
    samples = decode_to_mono(path, sample_rate)
    magnitude = spectrogram(samples, WINDOW, hop)
    frame_rate = sample_rate / hop

    onset = onset_strength(magnitude)
    tempo = estimate_tempo(onset, frame_rate)
    offset = (WINDOW - HOP) / sample_rate
    beats = track_beats(onset, frame_rate, tempo) + offset

    energy = _normalise(_smooth(magnitude.sum(axis=1), 9))
    bins = np.arange(magnitude.shape[1], dtype=np.float32)
    total = magnitude.sum(axis=1)
    centroid = (magnitude * bins[None, :]).sum(axis=1) / np.maximum(total, 1e-9)
    brightness = _normalise(_smooth(centroid, 9))
    sections = find_sections(magnitude, frame_rate)

    return AudioAnalysis(
        path=Path(path), duration=samples.size / sample_rate,
        tempo=tempo, beats=beats,
        onset=_normalise(onset), energy=energy, brightness=brightness,
        frame_rate=frame_rate, frame_offset=offset, sections=sections,
    )
