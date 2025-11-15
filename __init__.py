"""Event analysis tools for event-camera data.

This package provides utilities for estimating rotation speeds (RPM) and
detecting moving objects in event camera recordings.  Functions here are
designed to operate on streaming data from ``.dat`` files produced by
Prophesee event cameras and are optimized for real‑time use.  They build
upon the minimal ``evio`` library, decoding the raw 32‑bit words into
coordinate and polarity arrays, computing frequency estimates from
timestamps and clustering events spatially to locate drones or other
rotating objects.

Modules
-------
rpm
    Core routines for estimating rotational speed and finding bounding
    boxes around clusters of events.
visualization
    A streaming viewer similar to ``scripts/play_dat.py`` that overlays
    computed RPM values and bounding boxes on top of the rendered
    frames.

The package is self‑contained and does not modify the upstream ``evio``
repository.  It can be installed as a module by adding the parent
directory to your ``PYTHONPATH`` or by using a proper packaging tool.
"""

from .rpm import (
    decode_window,
    estimate_rpm,
    analyze_window,
    detect_clusters,
)
from .visualization import visualize_dat

__all__ = [
    "decode_window",
    "estimate_rpm",
    "analyze_window",
    "detect_clusters",
    "visualize_dat",
]