"""``facade-scan`` command line interface.

Subcommands mirror the pipeline stages, and each one reads what the previous
stage left on disk. That means you can re-run detection with different
thresholds twenty times without going back out to the house::

    facade-scan patterns  --out scan/patterns
    facade-scan capture   --scan scan --backend folder --folder ./photos
    facade-scan decode    --scan scan
    facade-scan detect    --scan scan
    facade-scan export    --scan scan
    facade-scan preview   --scan scan

or ``facade-scan run --scan scan`` for the lot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .config import Config

CONTEXT = {"help_option_names": ["-h", "--help"]}


def load_config(path: str | None) -> Config:
    if not path:
        return Config()
    return Config.from_toml(path)


def require_awake_displays() -> None:
    """Refuse to project onto a sleeping screen.

    A slept machine stops driving its outputs, so the patterns go nowhere and
    the projector shows its own "no source" card. Nothing errors, the captures
    come back as photographs of an unlit room, and the decode is empty. Better
    to say so than to spend forty frames finding out.
    """
    from .display import displays_are_asleep

    if displays_are_asleep():
        raise click.ClickException(
            "The displays are asleep, so nothing projected will reach the "
            "projector -- it will be showing its own 'no source' screen. Wake "
            "the screen (move the mouse or press a key) and run this again."
        )


def echo_stats(stats: dict[str, float]) -> None:
    coverage = stats.get("coverage", 0.0)
    click.echo(f"  decoded {int(stats.get('valid_pixels', 0)):,} camera pixels "
               f"({coverage:.1%} of frame)")
    click.echo(f"  median per-bit confidence {stats.get('median_min_confidence', 0):.3f}")
    click.echo(f"  likely glass {int(stats.get('likely_glass_pixels', 0)):,} px")
    if coverage < 0.15:
        click.secho("  coverage is very low -- see the troubleshooting section "
                    "of the README before trusting this scan", fg="yellow")


common_config = click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False),
                             help="TOML config file. Anything omitted keeps its default.")
common_scan = click.option("--scan", "-s", type=click.Path(file_okay=False), default="scan",
                           show_default=True, help="Scan directory.")


@click.group(context_settings=CONTEXT)
@click.version_option(__version__, prog_name="facade-scan")
def main() -> None:
    """Structured-light projection mapping for architectural facades."""


# --------------------------------------------------------------------------- #
@main.command()
def displays() -> None:
    """List attached displays, so you can tell which one is the projector."""
    from .display import detect_displays

    found = detect_displays()
    if not found:
        click.secho("Could not enumerate displays.", fg="yellow")
        click.echo("On macOS: pip install pyobjc-framework-Quartz")
        click.echo("Otherwise set display.origin_x / origin_y explicitly.")
        return

    click.echo(f"{len(found)} display(s):")
    for info in found:
        click.echo(f"  {info.describe()}")
        if info.asleep:
            click.secho("      ^ asleep: nothing projected onto it will arrive. "
                        "Wake the screen.", fg="yellow")
        if info.is_scaled:
            click.secho("      ^ scaled: everything sent here is resampled. "
                        "Do not scan through a scaled display.", fg="yellow")
    click.echo()
    click.echo("Use the index in [brackets] as --display-index, or pin it in TOML:")
    click.echo("  [display]\n  origin_x = <x>\n  origin_y = <y>")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True)
@click.option("--seconds", type=float, default=None,
              help="Exit automatically after this long.")
def focus(config: str | None, display_index: int | None, windowed: bool,
          seconds: float | None) -> None:
    """Project a focus target: the finest stripes, plus white and black.

    Focus on the *finest* pattern, not on the white frame. The finest Gray
    plane is two projector pixels wide and it is the first thing defocus
    destroys -- if you cannot resolve it on the wall, those bits will not
    decode no matter what else you do.

    Use the white and black frames to set camera exposure: white bright but not
    clipping, black as close to black as the room allows.

    Keys: space = next target, q = quit.
    """
    import time

    from .display import make_display
    from .patterns import build_manifest, render_frame

    cfg = load_config(config)
    if display_index is not None:
        cfg.display.display_index = display_index

    width, height = cfg.projector.width, cfg.projector.height
    manifest = build_manifest(width, height, cfg.patterns)
    targets = [
        ("finest vertical stripes (focus on this)", manifest.frames_for("x", False)[-1]),
        ("finest horizontal stripes", manifest.frames_for("y", False)[-1]),
        ("mid vertical stripes", manifest.frames_for("x", False)[len(
            manifest.frames_for("x", False)) // 2]),
        ("all white (set exposure here: bright, not clipped)",
         manifest.frame_by_role("white")),
        ("all black (check your ambient floor)", manifest.frame_by_role("black")),
    ]
    images = [(label, render_frame(frame, width, height, manifest.bits_x,
                                   manifest.bits_y, cfg.patterns))
              for label, frame in targets]

    require_awake_displays()
    display = make_display(cfg.display, width, height, windowed=windowed)
    click.echo(f"projecting on {display.describe()}")
    click.echo("  space = next target    q = quit")
    started = time.monotonic()
    index = 0
    with display:
        while True:
            label, image = images[index]
            click.echo(f"  -> {label}", nl=False)
            click.echo("\r", nl=False)
            key = display.show(image, wait_ms=120)
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                index = (index + 1) % len(images)
            if seconds is not None and time.monotonic() - started >= seconds:
                index += 1
                if index >= len(images):
                    break
                started = time.monotonic()
    click.echo()


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@click.option("--out", "-o", type=click.Path(file_okay=False), default="patterns",
              show_default=True)
@click.option("--width", type=int, help="Projector width; overrides the config.")
@click.option("--height", type=int, help="Projector height; overrides the config.")
def patterns(config: str | None, out: str, width: int | None, height: int | None) -> None:
    """Generate the Gray-code pattern set as PNGs plus a manifest."""
    from .patterns import write_patterns

    cfg = load_config(config)
    w = width or cfg.projector.width
    h = height or cfg.projector.height
    manifest = write_patterns(out, w, h, cfg.patterns)
    click.echo(f"{manifest.num_frames} frames for a {w}x{h} projector "
               f"({manifest.bits_x} x-bits + {manifest.bits_y} y-bits, "
               f"each with its inverse, plus white and black)")
    click.echo(f"written to {Path(out).resolve()}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--out", "-o", type=click.Path(file_okay=False),
              help="Where to write the simulation. Defaults to the scan directory.")
@click.option("--projector", nargs=2, type=int, help="Simulated projector WIDTH HEIGHT.")
@click.option("--camera", nargs=2, type=int, help="Simulated camera WIDTH HEIGHT.")
@click.option("--house", type=click.Path(exists=True, dir_okay=False),
              help="House geometry TOML. Defaults to the bundled house.")
@click.option("--noise", type=float, help="Gaussian sensor noise sigma, in [0,1].")
@click.option("--ambient", type=float, help="Ambient light fraction, in [0,1].")
def simulate(config: str | None, scan: str, out: str | None,
             projector: tuple[int, int], camera: tuple[int, int],
             house: str | None, noise: float | None, ambient: float | None) -> None:
    """Render a synthetic scan of a fake house, with ground truth.

    Everything downstream works on the result exactly as it would on a real
    capture, so this is the way to try the tool, and the way its numbers are
    verified.
    """
    from .pipeline import write_config
    from .sim.render import simulate as run_simulation

    cfg = load_config(config)
    sim = cfg.sim
    if projector:
        sim.projector.width, sim.projector.height = projector
    if camera:
        sim.camera.width, sim.camera.height = camera
    if house:
        sim.house_path = house
    if noise is not None:
        sim.noise_sigma = noise
    if ambient is not None:
        sim.ambient = ambient

    target = Path(out or scan)
    click.echo(f"rendering {sim.camera.width}x{sim.camera.height} camera views of a "
               f"{sim.projector.width}x{sim.projector.height} projector...")
    result = run_simulation(target, sim)
    write_config(target, cfg)
    click.echo(f"  {result.manifest.num_frames} captures -> {result.capture_dir}")
    click.echo(f"  ground truth -> {result.ground_truth_path}")
    click.echo(f"next: facade-scan decode --scan {target}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--backend", type=click.Choice(["webcam", "gphoto2", "folder"]),
              help="Capture backend; overrides the config.")
@click.option("--folder", type=click.Path(exists=True, file_okay=False),
              help="For the folder backend: where your photographs are.")
@click.option("--display-index", type=int, help="Which display is the projector.")
@click.option("--windowed", is_flag=True,
              help="Do not go fullscreen. For rehearsing without a projector.")
@click.option("--settle-ms", type=int, help="Delay between projecting and capturing.")
@click.option("--yes", "-y", is_flag=True, help="Skip the preflight confirmation.")
def capture(config: str | None, scan: str, backend: str | None, folder: str | None,
            display_index: int | None, windowed: bool, settle_ms: int | None,
            yes: bool) -> None:
    """Project every pattern and photograph it."""
    from .capture import build_backend, preflight_text, run_scan
    from .display import NullDisplay, make_display
    from .patterns import build_manifest
    from .pipeline import scan_layout, write_config

    cfg = load_config(config)
    if backend:
        cfg.capture.backend = backend
    if folder:
        cfg.capture.folder_path = folder
        if not backend:
            cfg.capture.backend = "folder"
    if display_index is not None:
        cfg.display.display_index = display_index
    if settle_ms is not None:
        cfg.capture.settle_ms = settle_ms

    paths = scan_layout(scan)
    manifest = build_manifest(cfg.projector.width, cfg.projector.height, cfg.patterns)
    device = build_backend(cfg.capture)
    if device.drives_display:
        require_awake_displays()
    display = (make_display(cfg.display, cfg.projector.width, cfg.projector.height,
                            windowed=windowed)
               if device.drives_display else NullDisplay())

    click.echo(preflight_text(cfg, device.describe(), display.describe(),
                              manifest.num_frames))
    if not yes and not click.confirm("Ready?", default=False):
        click.echo("aborted; nothing captured")
        sys.exit(1)

    write_config(scan, cfg)
    with device, display, \
            click.progressbar(length=manifest.num_frames, label="capturing") as bar:
        run_scan(paths["captures"], manifest, device, display, cfg,
                 on_frame=lambda frame, image: bar.update(1))
    click.echo(f"captures -> {paths['captures']}")
    click.echo(f"next: facade-scan decode --scan {scan}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
def decode(config: str | None, scan: str) -> None:
    """Decode the captures into a camera -> projector map."""
    from .pipeline import run_decode, scan_layout

    cfg = load_config(config)
    result = run_decode(scan, cfg)
    click.echo("decoded:")
    echo_stats(result.stats())
    click.echo(f"  -> {scan_layout(scan)['decoded']}")
    click.echo(f"next: facade-scan detect --scan {scan}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--sam/--no-sam", default=None, help="Use Segment Anything, if installed.")
@click.option("--min-segment-px", type=float,
              help="Absolute floor on segment length; raise it to reject more texture.")
def detect(config: str | None, scan: str, sam: bool | None,
           min_segment_px: float | None) -> None:
    """Find architectural lines and regions in the all-white capture."""
    from .pipeline import run_detect, scan_layout

    cfg = load_config(config)
    if sam is not None:
        cfg.detect.use_sam = sam
    if min_segment_px is not None:
        cfg.detect.min_segment_length_px = min_segment_px

    found = run_detect(scan, cfg)
    click.echo(f"detected: {found.raw_segment_count} raw segments -> "
               f"{len(found.segments)} merged edges")
    click.echo(f"  {len(found.vanishing_points)} dominant directions")
    click.echo(f"  {len(found.regions)} regions")
    for region in found.regions[:12]:
        click.echo(f"    {region.label:16s} {region.attributes.get('area_px', 0):>10,.0f} px")
    if len(found.regions) > 12:
        click.echo(f"    ... and {len(found.regions) - 12} more")
    for note in found.notes:
        click.secho(f"  note: {note}", fg="yellow")
    if not found.regions:
        click.secho("  no closed regions found -- the house mask is still usable; "
                    "see the README troubleshooting section", fg="yellow")
    click.echo(f"  -> {scan_layout(scan)['detection']}")
    click.echo(f"next: facade-scan export --scan {scan}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--mask-only", is_flag=True,
              help="Export just the house mask. This alone is most of the value.")
@click.option("--foreground", is_flag=True,
              help="Mask only what stands in front of the scene's dominant "
                   "plane, instead of every surface the projector reaches. Use "
                   "this when the goal is to light the subject and not the "
                   "background behind it.")
@click.option("--inset", type=float, default=None,
              help="Shrink the mask by this many projector pixels, so its edge "
                   "does not spill onto whatever is behind the subject.")
def export(config: str | None, scan: str, mask_only: bool,
           foreground: bool, inset: float | None) -> None:
    """Transfer to projector space and write mask.png, regions.svg and scan.json."""
    from .pipeline import run_export

    cfg = load_config(config)
    if inset is not None:
        cfg.transfer.foreground_inset_px = inset
    outcome = run_export(scan, cfg, mask_only=mask_only, foreground=foreground)
    lit = int(outcome.scan.mask.sum()) if outcome.scan.mask is not None else 0
    total = outcome.scan.projector_width * outcome.scan.projector_height
    click.echo(f"exported for a {outcome.scan.projector_width}x"
               f"{outcome.scan.projector_height} projector:")
    click.echo(f"  house mask covers {lit:,} projector pixels ({lit / total:.1%})")
    click.echo(f"  {len(outcome.scan.regions)} regions transferred")
    for note in outcome.scan.notes:
        click.secho(f"  note: {note}", fg="yellow")
    for name, path in outcome.paths.items():
        click.echo(f"  {name:5s} -> {path}")
    click.echo(f"next: facade-scan preview --scan {scan}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--mode", type=click.Choice(["mask", "outline", "fill", "cycle"]),
              help="What to project back.")
@click.option("--blink-ms", type=int,
              help="Blink period in ms. 0 is steady. Blinking makes a few pixels "
                   "of misalignment obvious from the driveway.")
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True, help="Show in a window instead of fullscreen.")
@click.option("--seconds", type=float, help="Exit automatically after this long.")
@click.option("--save", type=click.Path(dir_okay=False),
              help="Also write the preview frame to a PNG.")
@click.option("--foreground", is_flag=True,
              help="Show only what stands in front of the dominant plane.")
@click.option("--inset", type=float, default=None,
              help="Shrink the mask by this many projector pixels.")
def preview(config: str | None, scan: str, mode: str | None, blink_ms: int | None,
            display_index: int | None, windowed: bool, seconds: float | None,
            save: str | None, foreground: bool, inset: float | None) -> None:
    """Project the computed mask back onto the house, to check alignment.

    Do this before you pack anything away. If the mask edge does not sit on the
    edge of the house, you want to know now.
    """
    from .display import make_display
    from .pipeline import load_decode, load_detection
    from .preview import render_preview, run_preview
    from .transfer import (
        foreground_projector_mask,
        house_mask,
        transfer_regions,
    )

    cfg = load_config(config)
    if mode:
        cfg.preview.mode = mode
    if blink_ms is not None:
        cfg.preview.blink_ms = blink_ms
    if display_index is not None:
        cfg.display.display_index = display_index
    if inset is not None:
        cfg.transfer.foreground_inset_px = inset

    decoded = load_decode(scan)
    mask = (foreground_projector_mask(decoded, cfg.transfer) if foreground
            else house_mask(decoded, cfg.transfer, cfg.decode))
    try:
        regions = transfer_regions(decoded, load_detection(scan), cfg.transfer).regions
    except FileNotFoundError:
        regions = []
    click.echo("  (run `facade-scan verify` to confirm the rig has not moved "
               "since the scan)")

    shape = (decoded.projector_height, decoded.projector_width)
    native = (cfg.display.native_width or decoded.projector_width,
              cfg.display.native_height or decoded.projector_height)
    if save:
        import cv2

        frame = render_preview(mask, regions, shape, cfg.preview, cfg.preview.mode,
                               native_size=native)
        cv2.imwrite(save, frame)
        click.echo(f"preview frame -> {save}")

    if seconds is not None and seconds <= 0:
        # "Write the frame and stop" -- no projector needed, and useful for
        # checking a result without taking over the display.
        return

    require_awake_displays()
    display = make_display(cfg.display, decoded.projector_width,
                           decoded.projector_height, windowed=windowed)
    click.echo(f"previewing on {display.describe()}")
    click.echo("  space = next region   a = all   b = blink   o = mode   q = quit")
    with display:
        run_preview(display, mask, regions, shape, cfg.preview, max_seconds=seconds,
                    native_size=native)


# --------------------------------------------------------------------------- #
@main.command()
@common_config
def cameras(config: str | None) -> None:
    """List cameras without opening any of them.

    Opening a camera to find out what it is has side effects: on macOS it can
    wake an iPhone into Continuity Camera, which is both surprising and slow.
    This reads device metadata only, so nothing is activated.
    """
    from .capture.webcam import list_cameras, resolve_camera_index

    cfg = load_config(config)
    found = list_cameras()
    if not found:
        click.secho("Could not enumerate cameras without opening them.", fg="yellow")
        click.echo("On macOS: pip install pyobjc-framework-AVFoundation")
        click.echo("Otherwise set capture.webcam_index and test it directly.")
        return

    selected = (resolve_camera_index(cfg.capture.webcam_name)
                if cfg.capture.webcam_name else cfg.capture.webcam_index)
    click.echo(f"{len(found)} camera(s) as AVFoundation reports them:")
    for index, name in enumerate(found):
        marker = (click.style("  <-- index will be used", fg="green")
                  if index == selected else "")
        click.echo(f"  [{index}] {name}{marker}")
    click.echo()
    click.secho("These names are a HINT, not a lookup table. OpenCV does not "
                "always index cameras in this order -- on some builds the "
                "device listed here at position 1 is opened as index 0.",
                fg="yellow")
    click.echo("The only reliable test is which camera sees the projected "
               "image. Run `facade-scan check` and look at the lit-area figure.")
    if cfg.capture.webcam_name:
        click.echo(f"\nSelecting by name ({cfg.capture.webcam_name!r}) trusts that "
                   "ordering. Pin a verified capture.webcam_index instead if the "
                   "check shows a tiny lit area.")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@click.option("--axis", type=click.Choice(["x", "y"]), default="x", show_default=True)
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True)
@click.option("--save", type=click.Path(file_okay=False),
              help="Also keep the probe captures here, for inspection.")
def check(config: str | None, axis: str, display_index: int | None,
          windowed: bool, save: str | None) -> None:
    """Measure whether this rig can decode, before committing to a full scan.

    Projects each Gray-code plane with its inverse and reports how much contrast
    survives the round trip. A plane below the confidence threshold will not
    decode, and one bad plane means no coordinate at all for those pixels.

    About twenty frames rather than forty-odd, and it tells you which knob to
    turn rather than leaving you to guess after a failed scan.
    """
    from .capture import build_backend
    from .display import NullDisplay, make_display
    from .patterns import build_manifest
    from .rigcheck import check_rig

    cfg = load_config(config)
    if display_index is not None:
        cfg.display.display_index = display_index

    device = build_backend(cfg.capture)
    if device.drives_display:
        require_awake_displays()
    display = (make_display(cfg.display, cfg.projector.width, cfg.projector.height,
                            windowed=windowed)
               if device.drives_display else NullDisplay())
    manifest = build_manifest(cfg.projector.width, cfg.projector.height, cfg.patterns)
    bits = manifest.bits_x if axis == "x" else manifest.bits_y
    total = 2 + bits * 2

    with device, display:
        click.echo(f"projecting on {display.describe()}")
        click.echo(f"capturing with {device.describe()}")
        with click.progressbar(length=total, label="probing") as bar:
            result = check_rig(cfg, device, display, axis=axis,
                               capture_dir=save, on_frame=lambda f: bar.update(1))

    click.echo()
    click.echo(f"  pattern grid   {result.projector_grid[0]}x{result.projector_grid[1]}"
               f"  ->  panel {result.native_grid[0]}x{result.native_grid[1]}")
    click.echo(f"  camera         {result.camera_size[0]}x{result.camera_size[1]}"
               f"   ({result.oversampling:.2f}x the pattern grid)")
    click.echo(f"  white / black  {result.white_level:.3f} / {result.black_level:.3f}"
               f"   clipped {result.clipped_fraction:.1%}")
    click.echo(f"  lit area       {result.lit_fraction:.1%} of frame")
    click.echo()
    click.echo(result.table())
    click.echo()
    coverage = result.predicted_coverage
    colour = "green" if coverage > 0.9 else ("yellow" if coverage > 0.5 else "red")
    click.secho(f"  pixels clearing EVERY {result.measured_axis}-plane: "
                f"{coverage:.1%} of the lit area", fg=colour, bold=True)
    click.echo(f"  a full scan also needs the other axis, so expect roughly "
               f"{result.predicted_scan_coverage:.0%} or a little better")
    for note in result.advice():
        click.echo(f"    - {note}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--tolerance", type=float, default=3.0, show_default=True,
              help="Camera pixels of error to accept before calling it misaligned.")
@click.option("--spacing", type=int, default=160, show_default=True,
              help="Fiducial spacing in camera pixels.")
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True)
def verify(config: str | None, scan: str, tolerance: float, spacing: int,
           display_index: int | None, windowed: bool) -> None:
    """Check that light actually lands where this scan says it should.

    Everything else in this tool is open-loop: it measures a correspondence and
    trusts it. This closes the loop. Fiducial squares are drawn at known camera
    coordinates, pushed through the scan's own correspondence into projector
    space, projected, and photographed. The gap between where they were aimed
    and where the light landed is the end-to-end error of the whole pipeline.

    Run it before you commit to a design, and again if anything has been
    touched. The failure this catches -- the rig shifting after the scan --
    produces no error anywhere else: the mask still exports and the preview
    still projects, simply in the wrong place.
    """
    from .capture import build_backend
    from .display import NullDisplay, make_display
    from .pipeline import load_decode
    from .verify import verify_alignment

    cfg = load_config(config)
    if display_index is not None:
        cfg.display.display_index = display_index

    decoded = load_decode(scan)
    device = build_backend(cfg.capture)
    if device.drives_display:
        require_awake_displays()
    display = (make_display(cfg.display, decoded.projector_width,
                            decoded.projector_height, windowed=windowed)
               if device.drives_display else NullDisplay())

    with device, display:
        click.echo(f"projecting on {display.describe()}")
        click.echo(f"capturing with {device.describe()}")
        with click.progressbar(length=3, label="verifying") as bar:
            report = verify_alignment(decoded, device, display, cfg,
                                      spacing=spacing, tolerance_px=tolerance,
                                      on_step=lambda: bar.update(1))

    click.echo()
    click.echo(f"  camera view has moved {report.drift[0]:+.1f}, "
               f"{report.drift[1]:+.1f} px since the scan "
               f"(confidence {report.drift_confidence:.2f})")
    click.echo(f"  {len(report.measured)} of {len(report.fiducials)} fiducials "
               "returned light")
    if report.measured:
        click.echo(f"  offset: median {report.median_offset_px:.2f} px   "
                   f"bias ({report.bias[0]:+.2f}, {report.bias[1]:+.2f})   "
                   f"scatter {report.scatter_px:.2f} px")
        click.echo(f"  judged against {report.tolerance_px:.1f} px "
                   f"(one projector pixel spans {report.quantisation_px:.1f} "
                   "camera pixels on this grid)")
    click.echo()
    for i, note in enumerate(report.verdict()):
        click.secho(f"  {note}", fg="green" if report.aligned else "yellow",
                    bold=(i == 0))


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False),
              help="The track to follow. Anything ffmpeg can decode.")
@click.option("--out", "-o", type=click.Path(dir_okay=False),
              help="Render an MP4 here, with the audio muxed back in.")
@click.option("--play", is_flag=True, help="Play it live on the projector.")
@click.option("--foreground/--whole-scene", default=True, show_default=True,
              help="Light only what stands in front of the background.")
@click.option("--fps", type=int, default=30, show_default=True)
@click.option("--start", type=float, default=0.0, help="Skip in this far, in seconds.")
@click.option("--seconds", type=float, default=None,
              help="Only this long, for quick iteration.")
@click.option("--inset", type=float, default=None,
              help="Shrink the mask by this many projector pixels.")
@click.option("--gain", type=float, default=None,
              help="Overall output level. Below 1 for a close or bright rig.")
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True)
def animate(config: str | None, scan: str, audio: str, out: str | None,
            play: bool, foreground: bool, fps: int, start: float,
            seconds: float | None, inset: float | None, gain: float | None,
            display_index: int | None, windowed: bool) -> None:
    """Animate a song through this scan's mask.

    The track is analysed for tempo, beats, loudness and brightness, and the
    animation follows those directly -- so the structure comes from the
    recording rather than from a cue sheet anyone had to write. Loud passages
    light up, quiet ones fall away, and the lights chase in time.

    Light only ever lands inside the mask, so the subject is lit and the
    background stays dark.
    """
    from .animate import analyse, play_live, render_video
    from .display import make_display
    from .pipeline import load_decode
    from .preview import upscale_smooth
    from .transfer import foreground_projector_mask, house_mask

    cfg = load_config(config)
    if display_index is not None:
        cfg.display.display_index = display_index
    if inset is not None:
        cfg.transfer.foreground_inset_px = inset
    if gain is not None:
        cfg.animate.master_gain = gain
    if not out and not play:
        raise click.ClickException("nothing to do: pass --out, --play, or both")

    decoded = load_decode(scan)
    mask = (foreground_projector_mask(decoded, cfg.transfer) if foreground
            else house_mask(decoded, cfg.transfer, cfg.decode))
    native = (cfg.display.native_width or decoded.projector_width,
              cfg.display.native_height or decoded.projector_height)
    # Redraw the mask's outline at panel resolution: the correspondence is
    # coarse, but the animation need not be.
    mask = upscale_smooth(mask, native)
    if not mask.any():
        raise click.ClickException(
            "the mask is empty -- nothing would be lit. Check coverage, and "
            "try --whole-scene if the subject is not in front of a background."
        )

    click.echo(f"analysing {Path(audio).name}...")
    track = analyse(audio)
    click.echo(f"  {track.duration:.1f}s at {track.tempo:.1f} BPM, "
               f"{track.beats.size} beats")
    click.echo(f"  mask {mask.sum():,} of {mask.size:,} panel pixels "
               f"({mask.mean():.1%})")

    if out:
        length = (track.duration - start) if seconds is None else seconds
        with click.progressbar(length=max(1, int(length * fps)),
                               label="rendering") as bar:
            result = render_video(mask, track, out, cfg=cfg.animate, fps=fps,
                                  start=start, duration=seconds,
                                  on_frame=lambda i, n: bar.update(1))
        click.echo(f"  -> {result.path}  ({result.seconds:.1f}s, "
                   f"{result.frames} frames"
                   + (", audio muxed" if result.has_audio else ", silent") + ")")

    if play:
        require_awake_displays()
        display = make_display(cfg.display, native[0], native[1], windowed=windowed)
        click.echo(f"playing on {display.describe()}   (q to stop)")
        click.secho("  start the music now", fg="green", bold=True)
        with display:
            drawn = play_live(display, mask, track, cfg=cfg.animate, fps=fps,
                              seconds=seconds, start=start, native_size=native)
        click.echo(f"  {drawn} frames shown")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--out", "-o", type=click.Path(file_okay=False),
              help="Where to write the report. Defaults to <scan>/report.")
@click.option("--title", default=None, help="Title card text.")
@click.option("--notes", default="", help="One line of context for the summary.")
def report(config: str | None, scan: str, out: str | None, title: str | None,
           notes: str) -> None:
    """Render a finished scan as a video plus a summary.

    Works on real and simulated scans alike. Where a ground truth exists the
    summary scores against it; where it does not, it falls back on measures
    that need no ground truth -- chiefly the planarity residual, which is your
    measurement noise in projector pixels, obtained from nothing but a flat
    surface in the scene.
    """
    from .report import build_report

    cfg = load_config(config)
    target = Path(out) if out else Path(scan) / "report"
    built = build_report(scan, target, cfg, title or f"facade-scan: {Path(scan).name}",
                         notes)

    summary = built.summary
    click.echo(f"report -> {target}")
    click.echo(f"  video    {built.video_path}  "
               f"({summary['video']['seconds']}s, {summary['video']['frames']} frames)")
    click.echo(f"  stills   {len(built.still_paths)}")
    click.echo(f"  coverage {summary['decode']['coverage']:.1%}")
    if "planarity_residual_px" in summary:
        fraction = summary.get("planarity_inlier_fraction", 1.0)
        click.echo(f"  planarity residual {summary['planarity_residual_px']:.2f} px "
                   f"over the {fraction:.0%} of pixels lying on one plane")
        if fraction < 0.9:
            click.secho("    the scene holds more than one surface, so this "
                        "describes the largest plane, not the whole scan",
                        fg="yellow")
    if "ground_truth" in summary:
        g = summary["ground_truth"]
        click.echo(f"  vs ground truth: median {g['median_error_px']:.2f} px, "
                   f"p95 {g['p95_error_px']:.2f} px, "
                   f"coverage of decodable {g['coverage_of_decodable']:.1%}")


# --------------------------------------------------------------------------- #
@main.command()
@common_config
@common_scan
@click.option("--backend", type=click.Choice(["webcam", "gphoto2", "folder"]))
@click.option("--folder", type=click.Path(exists=True, file_okay=False))
@click.option("--simulate", "use_simulator", is_flag=True,
              help="Render a synthetic scan instead of capturing, then run the rest.")
@click.option("--display-index", type=int)
@click.option("--windowed", is_flag=True)
@click.option("--yes", "-y", is_flag=True, help="Skip the preflight confirmation.")
@click.option("--no-preview", is_flag=True, help="Do not project the result back.")
@click.pass_context
def run(ctx: click.Context, config: str | None, scan: str, backend: str | None,
        folder: str | None, use_simulator: bool, display_index: int | None,
        windowed: bool, yes: bool, no_preview: bool) -> None:
    """Capture, decode, detect, transfer, export -- the whole pipeline."""
    if use_simulator:
        ctx.invoke(simulate, config=config, scan=scan, out=scan,
                   projector=(), camera=(), house=None, noise=None, ambient=None)
    else:
        ctx.invoke(capture, config=config, scan=scan, backend=backend, folder=folder,
                   display_index=display_index, windowed=windowed, settle_ms=None,
                   yes=yes)
    ctx.invoke(decode, config=config, scan=scan)
    ctx.invoke(detect, config=config, scan=scan, sam=None, min_segment_px=None)
    ctx.invoke(export, config=config, scan=scan, mask_only=False)
    if not no_preview:
        ctx.invoke(preview, config=config, scan=scan, mode=None, blink_ms=None,
                   display_index=display_index, windowed=windowed, seconds=None,
                   save=None)


if __name__ == "__main__":
    main()
