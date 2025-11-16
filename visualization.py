"""Real‑time visualization of event data with RPM and bounding boxes.

This module provides a ``visualize_dat`` function that plays back
Prophesee ``.dat`` recordings in real time while overlaying
computed rotational speeds and cluster bounding boxes on the rendered
frames.  It draws heavily from the upstream ``scripts/play_dat.py``
example but extends it with analytics from ``event_tools.rpm``.

Example usage::

    from event_tools.visualization import visualize_dat
    from event_tools.main import stream_dat_file  # helper producing EventWindow objects

    # Play a recording with 100 ms display windows and RPM estimates
    stream = stream_dat_file("path/to/file.dat", window_ms=100)
    visualize_dat(stream, rpm_range=(1000, 7000))

The display can be interrupted by pressing ``Esc`` or ``q``.  When
running in a headless environment the function can be invoked with
``display=False`` to suppress GUI creation; in that case it simply
iterates through the windows and performs the analysis without
rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from statistics import mean
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

import cv2

from rpm import ClusterResult, analyze_window


WINDOW_NAME = "Event Viewer"


@dataclass(frozen=True)
class EventWindow:
    """Represents a single slice of streamed event data."""

    x_coords: np.ndarray
    y_coords: np.ndarray
    polarities: np.ndarray
    timestamps_us: np.ndarray
    width: int
    height: int
    start_ts_us: int
    end_ts_us: int
    metadata: Optional[object] = None


def _get_frame(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarities: np.ndarray,
    width: int,
    height: int,
    base_color: Tuple[int, int, int] = (127, 127, 127),
    on_color: Tuple[int, int, int] = (255, 255, 255),
    off_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render a frame from event coordinates and polarities.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Pixel coordinates for events in the current window.
    polarities : np.ndarray
        Boolean array indicating ON (True) or OFF (False) events.
    width, height : int
        Full resolution of the sensor.
    base_color, on_color, off_color : tuple of int, optional
        RGB colors used for the background, ON events and OFF events.

    Returns
    -------
    np.ndarray
        A 3‑channel ``uint8`` image suitable for display with OpenCV.
    """
    frame = np.full((height, width, 3), base_color, np.uint8)
    # ON events to white, OFF events to black (or other colors)
    frame[y_coords[polarities], x_coords[polarities]] = on_color
    frame[y_coords[~polarities], x_coords[~polarities]] = off_color
    return frame


def _draw_overlay(
    frame: np.ndarray,
    hud_lines: Sequence[str],
    clusters: Sequence[ClusterResult],
    blade_count: int,
    *,
    hud_color: Tuple[int, int, int] = (0, 0, 0),
    box_colors: Iterable[Tuple[int, int, int]] = (
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
    ),
    text_color: Tuple[int, int, int] = (255, 0, 255),
    label_offset: int = 12,
) -> None:
    """Overlay HUD, bounding boxes and RPM text onto the frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame onto which annotations are drawn (modified in place).
    hud_lines : Sequence[str]
        Lines of text rendered at the top-left corner acting as the HUD.
    clusters : Sequence[ClusterResult]
        Per-cluster results defining bounding boxes and RPM values.
    hud_color : tuple, optional
        Color used for the HUD text.
    box_colors : iterable of tuple, optional
        Colors cycled through for drawing bounding boxes.
    text_color : tuple, optional
        Color used for drawing per‑cluster RPM text.
    label_offset : int, optional
        Vertical pixel offset for cluster labels relative to the top
        of the bounding box.
    """
    # HUD text lines
    for idx, line in enumerate(hud_lines):
        y = 20 + idx * 20
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    # Cycle through provided colors for clusters
    color_cycle = itertools.cycle(box_colors)
    for cluster, color in zip(clusters, color_cycle):
        x, y, w, h = map(int, cluster.box)
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        # Draw RPM text above the box
        rpm_text = f"{cluster.rpm:0.0f} RPM" if not np.isnan(cluster.rpm) else "? RPM"
        cv2.putText(frame, rpm_text, (x, max(0, y - label_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def visualize_dat(
    windows: Iterable[EventWindow],
    blade_count: int = 2,
    *,
    rpm_range: Tuple[int, int] = (1000, 7000),
    min_cluster_size: int = 50,
    min_fill_ratio: float = 0.15,
    max_aspect_ratio: Optional[float] = None,
    min_events_per_pixel: Optional[int] = None,
    display: bool = True,
    **rpm_kwargs: object,
) -> None:
    """Render streamed event windows with RPM and bounding box overlays.

    Parameters
    ----------
    windows : Iterable[EventWindow]
        Stream of pre-decoded event windows.  Each window provides
        coordinates, polarities, timestamps and the frame resolution.
        The iterable can be backed by a live stream or by an offline
        ``.dat`` file reader.
    blade_count : int, optional
        Number of blades or repeating features per revolution.  Used to
        convert event RPM values to mechanical RPM.  Must be positive.
    rpm_range : tuple, optional
        Allowed range of RPM values.  The estimator will clip to this
        interval.  Defaults to (1000, 7000) which spans typical
        rotating fan and drone propeller speeds.
    min_cluster_size : int, optional
        Minimum cluster size for drone detection.  Clusters smaller
        than this many occupied pixels are ignored.
    min_fill_ratio : float, optional
        Minimum fraction of active pixels inside a bounding box.
        Sparse blobs (typical of foliage) are rejected.
    max_aspect_ratio : float, optional
        If provided, clusters whose width/height exceeds this ratio are
        dropped.  Use this to reject very thin streaks.
    min_events_per_pixel : int, optional
        Minimum number of hits required per pixel before it participates
        in cluster formation.  Helps suppress sparse noise.
    display : bool, optional
        If ``True`` (default), opens an OpenCV window and renders
        annotated frames.  If ``False``, no GUI is created and the
        function simply iterates through the data, computing the
        analytics without displaying anything.  This is useful for
        headless environments or benchmarking.
    **rpm_kwargs :
        Extra parameters forwarded to the RPM estimator.  See
        ``estimate_rpm`` and ``analyze_window``.
    """
    if blade_count <= 0:
        raise ValueError("blade_count must be positive")

    window_iter = iter(windows)
    try:
        first_window = next(window_iter)
    except StopIteration:
        return

    chain_iter: Iterator[EventWindow] = itertools.chain([first_window], window_iter)
    rec_start_ts = first_window.start_ts_us
    wall_start = time.perf_counter()

    if display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for idx, window in enumerate(chain_iter):
        clusters = analyze_window(
            window.x_coords,
            window.y_coords,
            window.timestamps_us,
            window.width,
            window.height,
            rpm_range=rpm_range,
            min_cluster_size=min_cluster_size,
            min_fill_ratio=min_fill_ratio,
            max_aspect_ratio=max_aspect_ratio,
            min_events_per_pixel=min_events_per_pixel,
            **rpm_kwargs,
        )

        if not display:
            continue

        frame = _get_frame(window.x_coords, window.y_coords, window.polarities, window.width, window.height)
        rpm_values = [
            cluster.rpm / blade_count
            for cluster in clusters
            if not np.isnan(cluster.rpm)
        ]
        mean_rpm = mean(rpm_values) if rpm_values else float("nan")
        wall_time_s = time.perf_counter() - wall_start
        rec_time_s = max(0.0, (window.end_ts_us - rec_start_ts) / 1e6)
        hud_lines = (
            f"frames={idx:05d}  wall={wall_time_s:7.3f}s",
            f"rec={rec_time_s:7.3f}s  Mean RPM={mean_rpm:7.1f}",
        )
        _draw_overlay(frame, hud_lines, clusters, blade_count)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    if display:
        cv2.destroyWindow(WINDOW_NAME)
