"""Music-driven animation, projected through a scan's mask."""

from .audio import AudioAnalysis, analyse
from .layers import LayerConfig, Layers, mask_outline
from .render import RenderResult, play_live, render_video

__all__ = [
    "AudioAnalysis",
    "LayerConfig",
    "Layers",
    "RenderResult",
    "analyse",
    "mask_outline",
    "play_live",
    "render_video",
]
