"""Putting pattern frames on the projector's output.

OpenCV has no concept of which monitor is which, so going fullscreen on the
*projector* rather than the laptop panel means knowing where that display sits
in the desktop coordinate space. We try to find out, and let the user override
when we get it wrong -- which, on a multi-monitor Mac, we sometimes will.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .config import DisplayConfig


@dataclass(frozen=True)
class DisplayInfo:
    """A physical display, in desktop coordinates."""

    index: int
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    refresh_hz: float = 0.0
    builtin: bool = False
    #: The display is attached but asleep. Nothing projected onto it arrives:
    #: the machine has stopped driving the output and the projector shows its
    #: own "no source" screen.
    asleep: bool = False
    #: Native panel pixels, which on a Retina display differ from the point
    #: size above. Patterns must be authored at the native size.
    native_width: int = 0
    native_height: int = 0

    @property
    def is_scaled(self) -> bool:
        """True if the desktop is not addressing the panel 1:1.

        A scaled display resamples everything sent to it, which destroys the
        finest Gray-code stripes. You want the projector unscaled.
        """
        return bool(self.native_width) and self.native_width != self.width

    def describe(self) -> str:
        bits = [f"[{self.index}] {self.name or 'display'}",
                f"{self.width}x{self.height} at ({self.x},{self.y})"]
        if self.refresh_hz:
            bits.append(f"{self.refresh_hz:.0f}Hz")
        if self.builtin:
            bits.append("built-in")
        if self.asleep:
            bits.append("ASLEEP")
        if self.is_scaled:
            bits.append(f"SCALED from {self.native_width}x{self.native_height}")
        return "  ".join(bits)


def detect_displays() -> list[DisplayInfo]:
    """Best-effort enumeration of attached displays, most reliable first.

    Quartz gives exact desktop origins on macOS and is the only method that
    gets a multi-monitor arrangement right; xrandr does the same on X11. The
    ``system_profiler`` path is a last resort that knows sizes but has to guess
    the layout, and with more than two displays it will guess wrong.
    """
    for method in (_detect_quartz, _detect_xrandr, _detect_macos):
        displays = method()
        if displays:
            return displays
    return []


#: EDID vendor IDs are three letters packed into 15 bits.
def _pnp_id(vendor: int) -> str:
    try:
        letters = [(vendor >> shift) & 0x1F for shift in (10, 5, 0)]
        return "".join(chr(ord("A") + value - 1) for value in letters)
    except ValueError:
        return ""


def _detect_quartz() -> list[DisplayInfo]:
    """Exact display geometry on macOS, via Quartz.

    ``pyobjc-framework-Quartz`` is an optional extra. Without it we fall back to
    guessing, which is why the projector ends up on the wrong screen.
    """
    try:
        import Quartz
    except ImportError:
        return []
    try:
        err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
        asleep = False
        if err or not count:
            # An asleep display is *online* but not *active*, and asking only
            # for active ones returns nothing -- which looked like "Quartz is
            # unavailable" and fell through to guessing display positions. The
            # difference is worth knowing: you cannot project onto a sleeping
            # display, and the projector shows its own "no source" screen.
            err, ids, count = Quartz.CGGetOnlineDisplayList(16, None, None)
            asleep = True
            if err or not count:
                return []
    except Exception:
        return []

    found: list[DisplayInfo] = []
    for index, display_id in enumerate(ids[:count]):
        bounds = Quartz.CGDisplayBounds(display_id)
        mode = Quartz.CGDisplayCopyDisplayMode(display_id)
        refresh = float(Quartz.CGDisplayModeGetRefreshRate(mode)) if mode else 0.0
        native_w = int(Quartz.CGDisplayModeGetPixelWidth(mode)) if mode else 0
        native_h = int(Quartz.CGDisplayModeGetPixelHeight(mode)) if mode else 0
        vendor = _pnp_id(int(Quartz.CGDisplayVendorNumber(display_id)))
        builtin = bool(Quartz.CGDisplayIsBuiltin(display_id))
        found.append(DisplayInfo(
            index=index,
            x=int(bounds.origin.x), y=int(bounds.origin.y),
            width=int(bounds.size.width), height=int(bounds.size.height),
            name=("built-in" if builtin else vendor or f"display {display_id}"),
            refresh_hz=refresh, builtin=builtin, asleep=asleep,
            native_width=native_w, native_height=native_h,
        ))
    return found


def _detect_xrandr() -> list[DisplayInfo]:
    if not shutil.which("xrandr"):
        return []
    try:
        out = subprocess.run(["xrandr", "--listmonitors"], capture_output=True,
                             text=True, timeout=5.0, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    # " 0: +*HDMI-1 1920/509x1080/286+1920+0  HDMI-1"
    pattern = re.compile(r"^\s*(\d+):\s+\S+\s+(\d+)\S*x(\d+)\S*?\+(\d+)\+(\d+)\s+(\S+)")
    found = []
    for line in out.splitlines():
        m = pattern.match(line)
        if m:
            idx, w, h, x, y, name = m.groups()
            found.append(DisplayInfo(int(idx), int(x), int(y), int(w), int(h), name))
    return found


def _detect_macos() -> list[DisplayInfo]:
    """macOS reports resolutions but not desktop origins.

    We therefore return sizes with origins laid out left to right, which is the
    common arrangement and is right often enough to be worth trying. When it is
    wrong the symptom is unmistakable -- the patterns appear on the wrong screen
    -- and ``display.origin_x`` fixes it.
    """
    if not shutil.which("system_profiler"):
        return []
    try:
        out = subprocess.run(["system_profiler", "-json", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=20.0,
                             check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return []

    found: list[DisplayInfo] = []
    cursor = 0
    for gpu in data.get("SPDisplaysDataType", []):
        for screen in gpu.get("spdisplays_ndrvs", []):
            raw = screen.get("_spdisplays_resolution") or screen.get("spdisplays_resolution")
            if not raw:
                continue
            # macOS writes resolutions with a multiplication sign, not an x.
            m = re.search("(\\d+)\\s*[x\u00d7]\\s*(\\d+)", raw)
            if not m:
                continue
            w, h = int(m.group(1)), int(m.group(2))
            found.append(DisplayInfo(len(found), cursor, 0, w, h,
                                     screen.get("_name", "")))
            cursor += w
    return found


def find_display(cfg: DisplayConfig,
                 displays: list[DisplayInfo] | None = None) -> DisplayInfo | None:
    """The display this config selects, by name if given, else by index."""
    displays = detect_displays() if displays is None else displays
    if cfg.display_name:
        wanted = cfg.display_name.strip().lower()
        for info in displays:
            if wanted in info.name.lower():
                return info
        return None
    if 0 <= cfg.display_index < len(displays):
        return displays[cfg.display_index]
    return None


def resolve_origin(cfg: DisplayConfig, width: int, height: int) -> tuple[int, int]:
    """Desktop coordinates to place the fullscreen window at."""
    if cfg.origin_x is not None and cfg.origin_y is not None:
        return int(cfg.origin_x), int(cfg.origin_y)

    found = find_display(cfg)
    if found is not None:
        return (int(cfg.origin_x) if cfg.origin_x is not None else found.x,
                int(cfg.origin_y) if cfg.origin_y is not None else found.y)

    # Nothing detected: assume displays are laid out left to right, each the
    # width of the projector.
    return (int(cfg.origin_x) if cfg.origin_x is not None else cfg.display_index * width,
            int(cfg.origin_y) if cfg.origin_y is not None else 0)


class PatternDisplay:
    """A fullscreen window on the projector, used as a context manager.

    ``windowed=True`` keeps it as an ordinary window, which is how you rehearse
    a scan at a desk without a projector attached.
    """

    def __init__(self, cfg: DisplayConfig, width: int, height: int,
                 windowed: bool = False) -> None:
        self.cfg = cfg
        #: Size of the frames this display is handed.
        self.width = width
        self.height = height
        #: Size actually sent to the panel. When it is larger, frames are blown
        #: up by nearest-neighbour so a logical pixel becomes a hard-edged block
        #: of native pixels rather than a resampled smear.
        self.native_width = cfg.native_width or width
        self.native_height = cfg.native_height or height
        self.windowed = windowed
        self._open = False
        self.origin = resolve_origin(cfg, self.native_width, self.native_height)

    @property
    def upscale(self) -> float:
        return self.native_width / self.width

    def __enter__(self) -> PatternDisplay:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        import cv2

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.cfg.window_name, self.origin[0], self.origin[1])
        if self.windowed:
            cv2.resizeWindow(self.cfg.window_name, self.native_width, self.native_height)
        else:
            cv2.setWindowProperty(self.cfg.window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        self._open = True

    def show(self, image: np.ndarray, wait_ms: int = 1) -> int:
        """Display a frame. Returns the key pressed, or -1."""
        import cv2

        if not self._open:
            self.open()
        if image.shape[1] != self.native_width or image.shape[0] != self.native_height:
            # INTER_NEAREST, always. Any smoothing here would blur exactly the
            # stripe edges the decoder reads.
            image = cv2.resize(image, (self.native_width, self.native_height),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.cfg.window_name, image)
        return cv2.waitKey(max(1, wait_ms)) & 0xFF

    def poll(self, wait_ms: int = 30) -> int:
        """Process input without redrawing. See CocoaDisplay.poll."""
        import cv2

        return cv2.waitKey(max(1, wait_ms)) & 0xFF

    def close(self) -> None:
        import cv2

        if self._open:
            cv2.destroyWindow(self.cfg.window_name)
            cv2.waitKey(1)
            self._open = False

    def describe(self) -> str:
        mode = "windowed" if self.windowed else "fullscreen"
        text = (f"display {self.cfg.display_index} at desktop origin "
                f"{self.origin[0]},{self.origin[1]} "
                f"({self.native_width}x{self.native_height}, {mode})")
        if self.upscale != 1.0:
            text += (f", patterns {self.width}x{self.height} "
                     f"blown up {self.upscale:.0f}x nearest-neighbour")
        return text


class NullDisplay:
    """Does nothing. Used when the operator drives the projector themselves."""

    def __enter__(self) -> NullDisplay:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def show(self, image: np.ndarray, wait_ms: int = 1) -> int:
        return -1

    def poll(self, wait_ms: int = 30) -> int:
        return -1

    def close(self) -> None:
        return None

    def describe(self) -> str:
        return "no display (patterns are being projected by hand)"


# --------------------------------------------------------------------------- #
# Native macOS window placement
# --------------------------------------------------------------------------- #
class CocoaDisplay:
    """A borderless AppKit window pinned to one physical screen.

    OpenCV's own window handling cannot be relied on here. On macOS its
    ``WND_PROP_FULLSCREEN`` moves the window to the *main* display, and
    ``moveWindow`` works in a coordinate space that matches neither Quartz
    (origin top-left, Y down) nor Cocoa (origin bottom-left of the main screen,
    Y up). The observed result on a four-display desk was patterns appearing on
    every screen except the projector.

    So the window is created directly on the ``NSScreen`` we want, borderless,
    above everything else, with the image drawn 1:1. Nothing here scales the
    image: the window is exactly the panel size and the frame handed to
    :meth:`show` is already that size, so stripe edges survive intact.
    """

    def __init__(self, cfg: DisplayConfig, width: int, height: int,
                 windowed: bool = False) -> None:
        import AppKit

        self.cfg = cfg
        self.width = width
        self.height = height
        self.native_width = cfg.native_width or width
        self.native_height = cfg.native_height or height
        self.windowed = windowed
        self._window = None
        self._view = None
        #: NSScreenNumber the window actually ended up on, once opened.
        self.landed_on: int | None = None
        self._app = AppKit.NSApplication.sharedApplication()
        # Accessory: we can put a window up without becoming a dock application.
        self._app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        self.screen = _ns_screen(cfg)
        frame = self.screen.frame()
        self.origin = (int(frame.origin.x), int(frame.origin.y))

    def __enter__(self) -> CocoaDisplay:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        import AppKit

        if self._window is not None:
            return
        frame = self.screen.frame()
        if self.windowed:
            frame = AppKit.NSMakeRect(frame.origin.x, frame.origin.y,
                                      self.native_width // 2, self.native_height // 2)

        # NOTE: do *not* use the initWithContentRect:...screen: variant. When a
        # screen is passed, AppKit interprets the rect relative to that screen's
        # origin, so a global-coordinate rect gets the screen offset applied a
        # second time. On a display at (-2983, 1117) that put the window at
        # (-5966, 2234) -- off every display, invisible, and silently so.
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, frame.size.width, frame.size.height),
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered, False,
        )
        window.setFrameOrigin_(frame.origin)
        window.setLevel_(AppKit.NSScreenSaverWindowLevel)
        window.setBackgroundColor_(AppKit.NSColor.blackColor())
        window.setOpaque_(True)
        # Show on whichever Space is active rather than switching Spaces.
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        view = AppKit.NSImageView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, frame.size.width, frame.size.height))
        view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
        view.setImageAlignment_(AppKit.NSImageAlignCenter)
        window.setContentView_(view)
        window.orderFrontRegardless()

        self._window, self._view = window, view
        self._pump(0.15)

        # Verify it actually landed where we asked. A window placed off every
        # display is invisible and reports no error, which is the worst possible
        # failure for something that is meant to be projecting patterns.
        landed = window.screen()
        if landed is None:
            raise RuntimeError(
                f"the pattern window was placed at {tuple(window.frame().origin)} "
                "which is not on any display. Set display.origin_x/origin_y "
                "explicitly, or use --windowed to rehearse."
            )
        wanted = self.screen.deviceDescription()["NSScreenNumber"]
        self.landed_on = landed.deviceDescription()["NSScreenNumber"]
        if self.landed_on != wanted:
            raise RuntimeError(
                f"the pattern window landed on display {self.landed_on}, not "
                f"the requested {wanted}. Set display.origin_x/origin_y explicitly."
            )

    def _pump(self, seconds: float) -> None:
        """Let AppKit actually draw. Nothing appears without this."""
        import AppKit

        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(seconds))

    def show(self, image: np.ndarray, wait_ms: int = 1) -> int:
        import AppKit
        import cv2

        if self._window is None or self._view is None:
            self.open()
        assert self._view is not None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[1] != self.native_width or image.shape[0] != self.native_height:
            image = cv2.resize(image, (self.native_width, self.native_height),
                               interpolation=cv2.INTER_NEAREST)

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("could not encode a frame for display")
        data = AppKit.NSData.dataWithBytes_length_(encoded.tobytes(), int(encoded.size))
        self._view.setImage_(AppKit.NSImage.alloc().initWithData_(data))
        self._view.setNeedsDisplay_(True)
        self._pump(max(wait_ms, 1) / 1000.0)
        return -1

    def poll(self, wait_ms: int = 30) -> int:
        """Let the window system breathe without re-uploading the image.

        Re-encoding and re-setting an identical frame thirty times a second
        makes the projected image visibly flicker, as well as being pure waste.
        A preview that is not changing should not be redrawn at all.
        """
        self._pump(max(wait_ms, 1) / 1000.0)
        return -1

    def close(self) -> None:
        if self._window is not None:
            self._window.orderOut_(None)
            self._window = None
            self._view = None
            self._pump(0.05)

    def describe(self) -> str:
        mode = "windowed" if self.windowed else "borderless fullscreen"
        text = (f"{self.screen_name} at Cocoa origin {self.origin[0]},{self.origin[1]} "
                f"({self.native_width}x{self.native_height}, {mode}, native AppKit)")
        if self.native_width != self.width:
            text += (f", patterns {self.width}x{self.height} blown up "
                     f"{self.native_width / self.width:.0f}x nearest-neighbour")
        return text

    @property
    def screen_name(self) -> str:
        try:
            import Quartz

            number = self.screen.deviceDescription()["NSScreenNumber"]
            return _pnp_id(int(Quartz.CGDisplayVendorNumber(number)))
        except Exception:
            return "display"


def _ns_screen(cfg: DisplayConfig):
    """The NSScreen this config selects, by name if given, else by index."""
    import AppKit
    import Quartz

    screens = list(AppKit.NSScreen.screens())
    if not screens:
        raise RuntimeError("no screens reported by AppKit")

    if cfg.display_name:
        wanted = cfg.display_name.strip().lower()
        for screen in screens:
            number = screen.deviceDescription()["NSScreenNumber"]
            name = _pnp_id(int(Quartz.CGDisplayVendorNumber(number)))
            if wanted in name.lower():
                return screen
        available = []
        for screen in screens:
            number = screen.deviceDescription()["NSScreenNumber"]
            available.append(_pnp_id(int(Quartz.CGDisplayVendorNumber(number))))
        raise ValueError(
            f"no display matching {cfg.display_name!r}; available: {available}"
        )

    if 0 <= cfg.display_index < len(screens):
        return screens[cfg.display_index]
    raise ValueError(f"display index {cfg.display_index} out of range "
                     f"(0..{len(screens) - 1})")


def displays_are_asleep() -> bool:
    """True when the machine has stopped driving its outputs."""
    try:
        import Quartz

        return bool(Quartz.CGDisplayIsAsleep(Quartz.CGMainDisplayID()))
    except Exception:
        return False


def cocoa_available() -> bool:
    """True if native macOS window placement can be used."""
    try:
        import AppKit  # noqa: F401
        import Quartz  # noqa: F401
    except ImportError:
        return False
    return True


def make_display(cfg: DisplayConfig, width: int, height: int,
                 windowed: bool = False, prefer_native: bool = True):
    """Build the most reliable display available on this machine."""
    if prefer_native and cocoa_available():
        try:
            return CocoaDisplay(cfg, width, height, windowed=windowed)
        except Exception:
            pass
    return PatternDisplay(cfg, width, height, windowed=windowed)
