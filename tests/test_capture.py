"""Stage 4: capture backends.

Capture is I/O, not numerics, so these tests use a *fake camera* -- a small
object that behaves like a real ``cv2.VideoCapture``, queue lag and all. The
point is not to mock away the behaviour but to reproduce the specific failure
mode that silently ruins real scans: a driver returning a frame from several
patterns ago.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from facade_scan.capture import (
    PREFLIGHT_CHECKLIST,
    CaptureBackend,
    FolderBackend,
    GPhoto2Backend,
    WebcamBackend,
    build_backend,
    preflight_text,
    run_scan,
)
from facade_scan.config import CaptureConfig, Config
from facade_scan.display import DisplayInfo, NullDisplay, PatternDisplay, resolve_origin
from facade_scan.patterns import build_manifest


# --------------------------------------------------------------------------- #
# A fake camera with a realistic frame queue
# --------------------------------------------------------------------------- #
class LaggyCamera:
    """Stands in for ``cv2.VideoCapture`` on a driver that buffers frames.

    ``read()`` returns whatever the projector was showing ``lag`` reads ago,
    which is exactly what a real UVC camera does.
    """

    def __init__(self, lag: int = 3) -> None:
        self.lag = lag
        self.current = np.zeros((4, 4), np.uint8)
        #: Frames already sitting in the driver's queue, oldest first.
        self.queue: list[np.ndarray] = [self.current.copy() for _ in range(lag)]
        self.properties: dict[int, float] = {}
        self.released = False
        self.reads = 0

    # what the projector is showing right now
    def project(self, image: np.ndarray) -> None:
        self.current = image

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        self.reads += 1
        self.queue.append(self.current)
        return True, self.queue.pop(0)

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.properties.get(prop, 0.0)

    def release(self) -> None:
        self.released = True


def marker_frame(value: int) -> np.ndarray:
    return np.full((4, 4), value, np.uint8)


# --------------------------------------------------------------------------- #
# Webcam backend
# --------------------------------------------------------------------------- #
def test_webcam_flushing_defeats_a_laggy_driver():
    """The bug this guards against produces no error, just a ruined scan."""
    camera = LaggyCamera(lag=3)
    manifest = build_manifest(8, 8)

    def factory(index: int) -> LaggyCamera:
        return camera

    with WebcamBackend(CaptureConfig(flush_frames=5), factory) as backend:
        captured = []
        for value, frame in enumerate(manifest.frames[:6], start=10):
            camera.project(marker_frame(value))
            captured.append(int(backend.capture(frame).ravel()[0]))

    assert captured == list(range(10, 16)), "flushed capture should be in sync"


def test_without_flushing_the_same_camera_returns_stale_frames():
    """Confirm the fake camera really does reproduce the bug, so the test above
    is testing something."""
    camera = LaggyCamera(lag=3)
    manifest = build_manifest(8, 8)
    with WebcamBackend(CaptureConfig(flush_frames=0), lambda i: camera) as backend:
        captured = []
        for value, frame in enumerate(manifest.frames[:6], start=10):
            camera.project(marker_frame(value))
            captured.append(int(backend.capture(frame).ravel()[0]))
    assert captured != list(range(10, 16))
    assert captured[0] < 10, "first frames should come from before the scan started"


def test_webcam_disables_every_automatic_feature():
    camera = LaggyCamera()
    cfg = CaptureConfig(webcam_disable_auto=True, webcam_exposure=-7.0,
                        webcam_focus=0.0, webcam_wb_temperature=4200.0,
                        webcam_gain=12.0, webcam_width=1920, webcam_height=1080)
    with WebcamBackend(cfg, lambda i: camera):
        pass

    assert camera.properties[cv2.CAP_PROP_AUTO_EXPOSURE] == 0.25
    assert camera.properties[cv2.CAP_PROP_AUTO_WB] == 0
    assert camera.properties[cv2.CAP_PROP_AUTOFOCUS] == 0
    assert camera.properties[cv2.CAP_PROP_EXPOSURE] == -7.0
    assert camera.properties[cv2.CAP_PROP_FOCUS] == 0.0
    assert camera.properties[cv2.CAP_PROP_WB_TEMPERATURE] == 4200.0
    assert camera.properties[cv2.CAP_PROP_GAIN] == 12.0
    assert camera.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1920


def test_manual_settings_can_be_left_alone_individually():
    """A None means 'do not touch', for cameras where a property is unsupported."""
    camera = LaggyCamera()
    cfg = CaptureConfig(webcam_disable_auto=False, webcam_exposure=None,
                        webcam_gain=None, webcam_wb_temperature=None,
                        webcam_focus=None)
    with WebcamBackend(cfg, lambda i: camera):
        pass
    assert cv2.CAP_PROP_EXPOSURE not in camera.properties
    assert cv2.CAP_PROP_AUTO_EXPOSURE not in camera.properties


def test_webcam_reports_what_the_driver_actually_accepted():
    """Drivers routinely ignore what you ask for; the operator needs to know."""
    class StubbornCamera(LaggyCamera):
        def get(self, prop):
            return -4.0 if prop == cv2.CAP_PROP_EXPOSURE else super().get(prop)

    camera = StubbornCamera()
    backend = WebcamBackend(CaptureConfig(webcam_exposure=-9.0), lambda i: camera)
    backend.open()
    assert backend.report_settings()["exposure"] == -4.0   # not the -9.0 requested
    backend.close()


def test_webcam_release_happens_on_exit():
    camera = LaggyCamera()
    with WebcamBackend(CaptureConfig(), lambda i: camera):
        pass
    assert camera.released


def test_webcam_reports_a_camera_it_cannot_open():
    class DeadCamera(LaggyCamera):
        def isOpened(self):
            return False

    with pytest.raises(OSError, match="could not open camera"):
        WebcamBackend(CaptureConfig(webcam_index=7), lambda i: DeadCamera()).open()


def test_webcam_can_be_selected_by_name(monkeypatch):
    """Indices renumber when a phone connects; names do not."""
    monkeypatch.setattr("facade_scan.capture.webcam.list_cameras",
                        lambda: ["MacBook Pro Camera", "USB Webcam", "Phone"])
    opened: list[int] = []

    def factory(index):
        opened.append(index)
        return LaggyCamera()

    with WebcamBackend(CaptureConfig(webcam_name="usb webcam"), factory) as backend:
        assert backend.resolved_index == 1
        assert "USB Webcam" in backend.describe()
    assert opened == [1]


def test_an_unmatched_camera_name_lists_what_is_there(monkeypatch):
    monkeypatch.setattr("facade_scan.capture.webcam.list_cameras",
                        lambda: ["MacBook Pro Camera", "USB Webcam"])
    with pytest.raises(OSError, match="USB Webcam"):
        WebcamBackend(CaptureConfig(webcam_name="Logitech"),
                      lambda i: LaggyCamera()).open()


def test_capture_before_open_is_an_error():
    backend = WebcamBackend(CaptureConfig(), lambda i: LaggyCamera())
    with pytest.raises(RuntimeError, match="open"):
        backend.capture(build_manifest(8, 8).frames[0])


# --------------------------------------------------------------------------- #
# Folder backend -- the one that must always work
# --------------------------------------------------------------------------- #
def test_folder_backend_matches_sorted_filenames_to_the_manifest(tmp_path):
    manifest = build_manifest(16, 16)
    shots = tmp_path / "shots"
    shots.mkdir()
    # Camera-style names, in the same order as the projection sequence but with
    # nothing about them tying a file to a particular pattern.
    for i, _ in enumerate(manifest.frames):
        cv2.imwrite(str(shots / f"IMG_{7000 + i}.png"), marker_frame(i))

    cfg = CaptureConfig(backend="folder", folder_path=str(shots))
    with build_backend(cfg) as backend:
        backend.prepare(manifest)
        for i, frame in enumerate(manifest.frames):
            assert int(backend.capture(frame).ravel()[0]) == i


def test_folder_backend_accepts_the_extensions_cameras_actually_produce(tmp_path):
    manifest = build_manifest(8, 8)
    shots = tmp_path / "shots"
    shots.mkdir()
    extensions = [".JPG", ".jpg", ".jpeg", ".PNG", ".png", ".tif", ".tiff", ".bmp"]
    for i, _ in enumerate(manifest.frames):
        cv2.imwrite(str(shots / f"{i:04d}{extensions[i % len(extensions)]}"),
                    marker_frame(200))
    backend = FolderBackend(CaptureConfig(folder_path=str(shots)))
    backend.prepare(manifest)           # must not raise
    assert int(backend.capture(manifest.frames[0]).ravel()[0]) == pytest.approx(200, abs=3)


def test_folder_backend_does_not_drive_the_projector():
    backend = FolderBackend(CaptureConfig(folder_path="."))
    assert backend.drives_display is False


def test_folder_backend_counts_the_photographs(tmp_path):
    manifest = build_manifest(16, 16)
    shots = tmp_path / "shots"
    shots.mkdir()
    for i in range(manifest.num_frames - 1):       # one missing
        cv2.imwrite(str(shots / f"IMG_{7000 + i}.JPG"), marker_frame(i))

    backend = FolderBackend(CaptureConfig(folder_path=str(shots)))
    with pytest.raises(ValueError, match="needs"):
        backend.prepare(manifest)


def test_folder_backend_ignores_non_image_files(tmp_path):
    manifest = build_manifest(16, 16)
    shots = tmp_path / "shots"
    shots.mkdir()
    for i, _ in enumerate(manifest.frames):
        cv2.imwrite(str(shots / f"{i:04d}.png"), marker_frame(i))
    (shots / "notes.txt").write_text("focus was good")
    (shots / "manifest.json").write_text("{}")

    backend = FolderBackend(CaptureConfig(folder_path=str(shots)))
    backend.prepare(manifest)           # must not raise
    assert int(backend.capture(manifest.frames[3]).ravel()[0]) == 3


def test_folder_backend_explains_a_missing_or_unset_directory(tmp_path):
    with pytest.raises(ValueError, match="folder_path"):
        FolderBackend(CaptureConfig()).prepare(build_manifest(8, 8))
    backend = FolderBackend(CaptureConfig(folder_path=str(tmp_path / "nope")))
    with pytest.raises(NotADirectoryError):
        backend.prepare(build_manifest(8, 8))


def test_folder_backend_consumes_a_simulated_scan_unmodified(tmp_path):
    """The simulator writes captures the folder backend can read as-is."""
    from facade_scan.config import SimConfig
    from facade_scan.sim.render import simulate

    cfg = SimConfig()
    cfg.projector.width, cfg.projector.height = 64, 64
    cfg.camera.width, cfg.camera.height = 96, 72
    result = simulate(tmp_path, cfg)

    # manifest.json sits in there too, and must be ignored by the matcher.
    backend = FolderBackend(CaptureConfig(folder_path=str(result.capture_dir)))
    backend.prepare(result.manifest)
    image = backend.capture(result.manifest.frame_by_role("white"))
    assert image.shape[:2] == (72, 96)
    assert image.mean() > 20


# --------------------------------------------------------------------------- #
# gphoto2 backend
# --------------------------------------------------------------------------- #
def test_gphoto2_downloads_and_decodes_a_frame(tmp_path):
    def fake_gphoto2(args, cwd):
        assert "--capture-image-and-download" in args
        assert "--force-overwrite" in args
        cv2.imwrite(str(Path(cwd) / "capture_0000.jpg"), marker_frame(99))
        return subprocess.CompletedProcess(args, 0, "", "")

    backend = GPhoto2Backend(CaptureConfig(), runner=fake_gphoto2)
    with backend:
        image = backend.capture(build_manifest(8, 8).frames[0])
    assert int(image.ravel()[0]) == 99


def test_gphoto2_passes_extra_args_through():
    seen: list[list[str]] = []

    def fake(args, cwd):
        seen.append(args)
        cv2.imwrite(str(Path(cwd) / "x.jpg"), marker_frame(1))
        return subprocess.CompletedProcess(args, 0, "", "")

    cfg = CaptureConfig(gphoto2_extra_args=["--set-config", "iso=400"])
    with GPhoto2Backend(cfg, runner=fake) as backend:
        backend.capture(build_manifest(8, 8).frames[0])
    assert seen[0][-2:] == ["--set-config", "iso=400"]


def test_gphoto2_surfaces_camera_errors(tmp_path):
    def failing(args, cwd):
        return subprocess.CompletedProcess(args, 1, "", "*** Error: Could not claim the USB device")

    with GPhoto2Backend(CaptureConfig(), runner=failing) as backend, \
            pytest.raises(OSError, match="Could not claim"):
        backend.capture(build_manifest(8, 8).frames[0])


def test_gphoto2_reports_a_raw_only_camera():
    def raw_only(args, cwd):
        (Path(cwd) / "capture_0000.cr2").write_bytes(b"not an image opencv knows")
        return subprocess.CompletedProcess(args, 0, "", "")

    with GPhoto2Backend(CaptureConfig(), runner=raw_only) as backend, \
            pytest.raises(OSError, match="Shoot JPEG"):
        backend.capture(build_manifest(8, 8).frames[0])


def test_gphoto2_reports_a_silent_no_op():
    def nothing(args, cwd):
        return subprocess.CompletedProcess(args, 0, "", "")

    with GPhoto2Backend(CaptureConfig(), runner=nothing) as backend, \
            pytest.raises(OSError, match=r"downloaded\s+no file"):
        backend.capture(build_manifest(8, 8).frames[0])


def test_gphoto2_without_the_binary_points_at_the_folder_backend(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(OSError, match="folder"):
        GPhoto2Backend(CaptureConfig(gphoto2_binary="definitely-not-installed")).open()


# --------------------------------------------------------------------------- #
# The scan loop
# --------------------------------------------------------------------------- #
class RecordingDisplay:
    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []

    def show(self, image, wait_ms: int = 1) -> int:
        self.shown.append(image)
        return -1

    def describe(self) -> str:
        return "recording display"


class EchoBackend(CaptureBackend):
    """Captures exactly what the display was last asked to show."""

    name = "echo"

    def __init__(self, display: RecordingDisplay) -> None:
        self.display = display

    def capture(self, frame):
        return self.display.shown[-1]


def test_run_scan_projects_every_frame_and_writes_every_capture(tmp_path):
    config = Config()
    config.capture.settle_ms = 0
    config.projector.width, config.projector.height = 32, 32
    manifest = build_manifest(32, 32)

    display = RecordingDisplay()
    out = run_scan(tmp_path / "caps", manifest, EchoBackend(display), display, config)

    assert len(display.shown) == manifest.num_frames
    written = sorted(p.name for p in out.glob("*.png"))
    assert written == [f.filename for f in manifest.frames]
    assert (out / "manifest.json").exists()

    # And what landed on disk is what was projected.
    first = cv2.imread(str(out / manifest.frames[0].filename), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(first, display.shown[0])


def test_run_scan_leaves_the_projector_alone_for_the_folder_backend(tmp_path):
    config = Config()
    config.capture.settle_ms = 0
    manifest = build_manifest(16, 16)
    shots = tmp_path / "shots"
    shots.mkdir()
    for i, _ in enumerate(manifest.frames):
        cv2.imwrite(str(shots / f"{i:04d}.png"), marker_frame(i))

    display = RecordingDisplay()
    backend = FolderBackend(CaptureConfig(folder_path=str(shots)))
    run_scan(tmp_path / "out", manifest, backend, display, config)
    assert display.shown == []


def test_a_scan_written_by_run_scan_decodes(tmp_path):
    """End to end: simulate -> folder capture -> run_scan -> decode."""
    from facade_scan.config import SimConfig
    from facade_scan.decode import decode_directory
    from facade_scan.sim.render import simulate

    sim_cfg = SimConfig()
    sim_cfg.projector.width, sim_cfg.projector.height = 128, 96
    sim_cfg.camera.width, sim_cfg.camera.height = 192, 128
    result = simulate(tmp_path / "sim", sim_cfg)

    config = Config()
    config.capture.settle_ms = 0
    backend = FolderBackend(CaptureConfig(folder_path=str(result.capture_dir)))
    out = run_scan(tmp_path / "scan", result.manifest, backend, NullDisplay(), config)

    decoded = decode_directory(out)
    gt = np.load(result.ground_truth_path)
    both = decoded.valid & gt["valid"]
    err = np.linalg.norm(decoded.proj_map.astype(float) - gt["proj_uv"], axis=-1)[both]
    assert np.median(err) < 1.0
    assert np.percentile(err, 95) < 3.0


def test_interrupted_scan_keeps_what_it_captured(tmp_path):
    """Captures are written as they arrive, so a failure is debuggable."""
    config = Config()
    config.capture.settle_ms = 0
    manifest = build_manifest(16, 16)
    display = RecordingDisplay()

    class DyingBackend(EchoBackend):
        def capture(self, frame):
            if frame.index == 5:
                raise OSError("camera unplugged")
            return super().capture(frame)

    out = tmp_path / "caps"
    with pytest.raises(OSError, match="unplugged"):
        run_scan(out, manifest, DyingBackend(display), display, config)
    assert len(list(out.glob("*.png"))) == 5


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def test_preflight_covers_every_way_to_ruin_a_scan():
    text = preflight_text(Config(), "webcam", "display 1", 46).lower()
    for topic in ("focus", "keystone", "close to the projector", "manual",
                  "exposure", "wind", "ambient", "do not touch"):
        assert topic in text, f"preflight says nothing about {topic!r}"


def test_preflight_warns_that_touching_the_optics_invalidates_the_scan():
    assert "invalidates the entire scan" in PREFLIGHT_CHECKLIST


def test_preflight_estimates_the_duration():
    text = preflight_text(Config(), "webcam", "display 1", 46)
    assert "46 frames" in text and "400 ms settle" in text and "min of holding still" in text


# --------------------------------------------------------------------------- #
# Display placement
# --------------------------------------------------------------------------- #
def test_display_origin_prefers_an_explicit_override():
    cfg = Config().display
    cfg.origin_x, cfg.origin_y = 3840, 120
    assert resolve_origin(cfg, 1920, 1080) == (3840, 120)


def test_display_can_be_selected_by_name(monkeypatch):
    """macOS renumbers displays when one sleeps; the projector must not move."""
    from facade_scan.display import find_display

    monkeypatch.setattr("facade_scan.display.detect_displays",
                        lambda: [DisplayInfo(0, 0, 0, 1728, 1117, "built-in"),
                                 DisplayInfo(1, -2983, -1080, 1920, 1080, "OTM"),
                                 DisplayInfo(2, 857, -1080, 1920, 1080, "GSM")])
    cfg = Config().display
    cfg.display_name = "otm"
    assert find_display(cfg).x == -2983
    assert resolve_origin(cfg, 1920, 1080) == (-2983, -1080)

    cfg.display_name = "nonexistent"
    assert find_display(cfg) is None


def test_patterns_are_blown_up_to_the_panel_without_resampling():
    """A coarse pattern grid on a fine panel must stay hard-edged."""
    shown: list[np.ndarray] = []

    class Recorder(PatternDisplay):
        def open(self):
            self._open = True

        def show(self, image, wait_ms=1):
            import cv2

            if image.shape[1] != self.native_width:
                image = cv2.resize(image, (self.native_width, self.native_height),
                                   interpolation=cv2.INTER_NEAREST)
            shown.append(image)
            return -1

        def close(self):
            self._open = False

    cfg = Config().display
    cfg.native_width, cfg.native_height = 1920, 1080
    display = Recorder(cfg, 960, 540)
    assert display.upscale == 2.0

    pattern = np.zeros((540, 960), np.uint8)
    pattern[:, ::2] = 255                      # finest possible stripes
    display.show(pattern)
    out = shown[0]
    assert out.shape[:2] == (1080, 1920)
    # Still exactly two-valued: nearest-neighbour introduced no intermediate
    # grey, which any smooth interpolation would have.
    assert set(np.unique(out)) == {0, 255}
    assert np.array_equal(out[0, 0:4], [255, 255, 0, 0])


def test_display_origin_uses_a_detected_display(monkeypatch):
    monkeypatch.setattr("facade_scan.display.detect_displays",
                        lambda: [DisplayInfo(0, 0, 0, 1512, 982),
                                 DisplayInfo(1, 1512, 0, 1920, 1080)])
    cfg = Config().display
    cfg.display_index = 1
    assert resolve_origin(cfg, 1920, 1080) == (1512, 0)


def test_display_origin_falls_back_to_a_left_to_right_guess(monkeypatch):
    monkeypatch.setattr("facade_scan.display.detect_displays", lambda: [])
    cfg = Config().display
    cfg.display_index = 2
    assert resolve_origin(cfg, 1920, 1080) == (3840, 0)


def test_null_display_is_inert():
    with NullDisplay() as display:
        assert display.show(np.zeros((2, 2), np.uint8)) == -1
    assert "by hand" in NullDisplay().describe()


def test_pattern_display_describes_where_it_will_appear(monkeypatch):
    monkeypatch.setattr("facade_scan.display.detect_displays", lambda: [])
    cfg = Config().display
    cfg.display_index = 1
    description = PatternDisplay(cfg, 1920, 1080).describe()
    assert "1920,0" in description and "1920x1080" in description


# --------------------------------------------------------------------------- #
# Native macOS window placement
# --------------------------------------------------------------------------- #
def test_make_display_falls_back_when_native_placement_is_unavailable(monkeypatch):
    from facade_scan.display import PatternDisplay, make_display

    monkeypatch.setattr("facade_scan.display.cocoa_available", lambda: False)
    display = make_display(Config().display, 640, 480)
    assert isinstance(display, PatternDisplay)


def test_make_display_falls_back_when_native_placement_raises(monkeypatch):
    """A broken screen lookup must not stop a scan; the OpenCV path still works."""
    from facade_scan.display import PatternDisplay, make_display

    def explode(*args, **kwargs):
        raise RuntimeError("no screens")

    monkeypatch.setattr("facade_scan.display.cocoa_available", lambda: True)
    monkeypatch.setattr("facade_scan.display.CocoaDisplay", explode)
    assert isinstance(make_display(Config().display, 640, 480), PatternDisplay)


def test_native_placement_can_be_declined():
    from facade_scan.display import PatternDisplay, make_display

    display = make_display(Config().display, 640, 480, prefer_native=False)
    assert isinstance(display, PatternDisplay)


def test_a_sleeping_display_is_labelled_not_silently_dropped():
    """An asleep display is online but not active.

    Asking Quartz only for *active* displays returns nothing at all when the
    screen has slept, which looked like "Quartz is unavailable" and fell through
    to guessing display positions from system_profiler -- with different names,
    so name-based selection stopped matching too. Meanwhile the projector sat
    showing its own "no source" screen.
    """
    awake = DisplayInfo(1, 857, -1080, 1920, 1080, "OTM", 120.0)
    asleep = DisplayInfo(1, 857, -1080, 1920, 1080, "OTM", 120.0, asleep=True)
    assert "ASLEEP" not in awake.describe()
    assert "ASLEEP" in asleep.describe()


def test_displays_command_warns_about_a_sleeping_display(monkeypatch):
    from click.testing import CliRunner

    from facade_scan.cli import main

    monkeypatch.setattr(
        "facade_scan.display.detect_displays",
        lambda: [DisplayInfo(0, 0, 0, 1728, 1117, "built-in", builtin=True, asleep=True),
                 DisplayInfo(1, 857, -1080, 1920, 1080, "OTM", asleep=True)])
    result = CliRunner().invoke(main, ["displays"])
    assert result.exit_code == 0
    assert "ASLEEP" in result.output
    assert "Wake the screen" in result.output


def test_sleep_check_never_blocks_a_scan_when_it_cannot_tell(monkeypatch):
    """If Quartz is unavailable the answer is 'do not know', not 'asleep'."""
    import builtins

    from facade_scan.display import displays_are_asleep

    real_import = builtins.__import__

    def no_quartz(name, *args, **kwargs):
        if name == "Quartz":
            raise ImportError("no Quartz here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_quartz)
    assert displays_are_asleep() is False
