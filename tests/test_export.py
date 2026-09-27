"""Stage 7: export artifacts and the Exporter interface."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from facade_scan.config import ExportConfig
from facade_scan.detect import Region
from facade_scan.export import (
    SCHEMA_VERSION,
    Exporter,
    JsonExporter,
    MaskExporter,
    ScanExport,
    SvgExporter,
    available_exporters,
    export_all,
    get_exporter,
    register_exporter,
    scan_to_dict,
)

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def scan() -> ScanExport:
    wall = Region(
        polygon=np.array([[20.0, 20.0], [380.0, 20.0], [380.0, 280.0], [20.0, 280.0]]),
        label="main_wall",
        holes=[np.array([[80.0, 80.0], [160.0, 80.0], [160.0, 160.0], [80.0, 160.0]])],
        attributes={"area_px": 87200.0, "glass_fraction": 0.02},
    )
    window = Region(
        polygon=np.array([[80.0, 80.0], [160.0, 80.0], [160.0, 160.0], [80.0, 160.0]]),
        label="window_01",
        attributes={"area_px": 6400.0, "glass_fraction": 0.97},
    )
    mask = np.zeros((300, 400), bool)
    mask[20:280, 20:380] = True
    mask[80:160, 80:160] = False
    return ScanExport(projector_width=400, projector_height=300,
                      regions=[wall, window], mask=mask,
                      stats={"coverage": 0.61, "valid_pixels": 12345.0},
                      source={"tool": "facade-scan test"},
                      notes=["a note"])


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
def test_exporter_is_abstract():
    with pytest.raises(TypeError):
        Exporter()  # type: ignore[abstract]


def test_the_three_documented_formats_are_registered():
    assert set(available_exporters()) >= {"mask", "svg", "json"}
    assert isinstance(get_exporter("svg"), SvgExporter)


def test_unknown_format_lists_what_is_available():
    with pytest.raises(ValueError, match="unknown export format"):
        get_exporter("madmapper")


def test_a_third_party_adapter_can_be_added(scan, tmp_path):
    """The documented extension point: a format adapter is a serialiser."""

    class ToyExporter(Exporter):
        name = "toy"
        extension = ".toy"

        def export(self, scan_export, path):
            from pathlib import Path

            path = Path(path)
            path.write_text(f"{scan_export.projector_width}x"
                            f"{scan_export.projector_height}:"
                            + ",".join(r.label for r in scan_export.regions))
            return path

    register_exporter(ToyExporter())
    assert "toy" in available_exporters()
    written = get_exporter("toy").export(scan, tmp_path / "out.toy")
    assert written.read_text() == "400x300:main_wall,window_01"
    assert get_exporter("toy").default_filename() == "toy.toy"


def test_camera_space_geometry_is_rejected(scan):
    """An exporter handed the wrong coordinate space would write a file that
    looks fine and is useless."""
    scan.mask = np.zeros((900, 1440), bool)
    with pytest.raises(ValueError, match="already be in projector space"):
        scan.validate()


def test_a_zero_sized_projector_is_rejected(scan):
    scan.projector_width = 0
    with pytest.raises(ValueError, match="resolution must be positive"):
        scan.validate()


# --------------------------------------------------------------------------- #
# mask.png
# --------------------------------------------------------------------------- #
def test_mask_png_is_projector_sized_and_white_means_project_here(scan, tmp_path):
    path = MaskExporter().export(scan, tmp_path / "mask.png")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (300, 400)
    assert set(np.unique(image)) == {0, 255}
    assert image[150, 300] == 255      # wall
    assert image[120, 120] == 0        # window opening
    assert image[5, 5] == 0            # outside the house


def test_mask_png_falls_back_to_rasterising_the_regions(scan, tmp_path):
    scan.mask = None
    path = MaskExporter().export(scan, tmp_path / "mask.png")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image[150, 300] == 255
    assert image[5, 5] == 0


# --------------------------------------------------------------------------- #
# regions.svg
# --------------------------------------------------------------------------- #
def test_svg_canvas_is_exactly_the_projector_panel(scan, tmp_path):
    path = SvgExporter().export(scan, tmp_path / "regions.svg")
    root = ET.parse(path).getroot()
    assert root.get("width") == "400" and root.get("height") == "300"
    assert root.get("viewBox") == "0 0 400 300"


def test_svg_has_one_labelled_path_per_region(scan, tmp_path):
    path = SvgExporter().export(scan, tmp_path / "regions.svg")
    root = ET.parse(path).getroot()
    groups = root.findall(f".//{SVG_NS}g/{SVG_NS}g")
    assert [g.get("id") for g in groups] == ["main_wall", "window_01"]
    for group in groups:
        element = group.find(f"{SVG_NS}path")
        assert element is not None
        assert element.find(f"{SVG_NS}title").text == group.get("id")


def test_svg_holes_become_subpaths_with_evenodd_fill(scan, tmp_path):
    path = SvgExporter().export(scan, tmp_path / "regions.svg")
    root = ET.parse(path).getroot()
    wall = root.find(f".//{SVG_NS}g[@id='main_wall']/{SVG_NS}path")
    assert wall.get("fill-rule") == "evenodd"
    assert wall.get("d").count("M ") == 2, "hole should be a second subpath"

    window = root.find(f".//{SVG_NS}g[@id='window_01']/{SVG_NS}path")
    assert window.get("d").count("M ") == 1


def test_svg_coordinates_are_projector_pixels(scan, tmp_path):
    path = SvgExporter().export(scan, tmp_path / "regions.svg")
    root = ET.parse(path).getroot()
    d = root.find(f".//{SVG_NS}g[@id='window_01']/{SVG_NS}path").get("d")
    numbers = [float(v) for pair in d.replace("M ", "").replace(" Z", "").split()
               for v in pair.split(",")]
    assert min(numbers) == pytest.approx(80.0)
    assert max(numbers) == pytest.approx(160.0)


def test_svg_labels_can_be_switched_off(scan, tmp_path):
    with_labels = SvgExporter(ExportConfig(svg_label_regions=True)).export(
        scan, tmp_path / "a.svg")
    without = SvgExporter(ExportConfig(svg_label_regions=False)).export(
        scan, tmp_path / "b.svg")
    assert ET.parse(with_labels).getroot().findall(f".//{SVG_NS}text")
    assert not ET.parse(without).getroot().findall(f".//{SVG_NS}text")


def test_svg_escapes_labels(tmp_path):
    scan = ScanExport(projector_width=100, projector_height=100, regions=[
        Region(polygon=np.array([[1.0, 1], [9, 1], [9, 9]]), label='a<b&c"d')
    ])
    path = SvgExporter().export(scan, tmp_path / "x.svg")
    root = ET.parse(path).getroot()        # would raise on malformed XML
    assert root.find(f".//{SVG_NS}g/{SVG_NS}g").get("id") == 'a<b&c"d'


def test_svg_is_well_formed_with_no_regions(tmp_path):
    scan = ScanExport(projector_width=64, projector_height=48)
    path = SvgExporter().export(scan, tmp_path / "empty.svg")
    assert ET.parse(path).getroot().get("width") == "64"


# --------------------------------------------------------------------------- #
# scan.json
# --------------------------------------------------------------------------- #
def test_scan_json_carries_resolution_regions_labels_and_confidence(scan, tmp_path):
    path = JsonExporter().export(scan, tmp_path / "scan.json")
    data = json.loads(path.read_text())

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["projector"] == {"width": 400, "height": 300}
    assert data["coordinate_space"] == "projector_pixels"
    assert [r["label"] for r in data["regions"]] == ["main_wall", "window_01"]
    assert data["confidence"]["coverage"] == pytest.approx(0.61)
    assert data["regions"][0]["holes"]
    assert data["regions"][0]["area_px"] == pytest.approx(360 * 260 - 80 * 80)
    assert data["regions"][1]["attributes"]["glass_fraction"] == pytest.approx(0.97)
    assert data["notes"] == ["a note"]
    assert data["source"]["tool"] == "facade-scan test"


def test_scan_json_is_serialisable_without_numpy_types(scan):
    json.dumps(scan_to_dict(scan))     # would raise on a numpy scalar


def test_scan_json_polygons_round_trip_to_geometry(scan, tmp_path):
    data = json.loads(JsonExporter().export(scan, tmp_path / "s.json").read_text())
    ring = np.array(data["regions"][1]["polygon"])
    assert Region(polygon=ring).area == pytest.approx(6400.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# export_all
# --------------------------------------------------------------------------- #
def test_export_all_writes_the_three_documented_artifacts(scan, tmp_path):
    written = export_all(scan, tmp_path)
    assert set(written) == {"mask", "svg", "json"}
    assert {p.name for p in written.values()} == {"mask.png", "regions.svg", "scan.json"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_export_filenames_are_configurable(scan, tmp_path):
    cfg = ExportConfig(mask_filename="project_here.png", svg_filename="shapes.svg",
                       json_filename="result.json")
    written = export_all(scan, tmp_path, cfg)
    assert written["mask"].name == "project_here.png"
    assert written["svg"].name == "shapes.svg"
    assert written["json"].name == "result.json"


def test_export_creates_missing_directories(scan, tmp_path):
    written = export_all(scan, tmp_path / "deep" / "nested")
    assert all(p.exists() for p in written.values())


# --------------------------------------------------------------------------- #
# End to end, from a simulated scan
# --------------------------------------------------------------------------- #
def test_exported_mask_matches_what_the_projector_can_see(sim_hires, decoded_hires,
                                                          tmp_path):
    from facade_scan.export import MaskExporter
    from facade_scan.sim import build_scene_cache
    from facade_scan.transfer import house_mask

    from .conftest import GLASS_SURFACES

    mask = house_mask(decoded_hires)
    scan = ScanExport(projector_width=decoded_hires.projector_width,
                      projector_height=decoded_hires.projector_height,
                      mask=mask, stats=decoded_hires.stats())
    path = MaskExporter().export(scan, tmp_path / "mask.png")

    written = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
    projector_view = build_scene_cache(sim_hires.scene, sim_hires.projector,
                                       sim_hires.projector, sim_hires.cfg)
    glass = [sim_hires.scene.index_of(n) for n in GLASS_SURFACES]
    truth = projector_view.hit & ~np.isin(projector_view.surface, glass)

    assert (written & truth).sum() / (written | truth).sum() > 0.97
