"""Real‑time visualization of event data with RPM and bounding boxes.

This module provides a ``visualize_dat`` function that plays back
Prophesee ``.dat`` recordings in real time while overlaying
computed rotational speeds and cluster bounding boxes on the rendered
frames.  It draws heavily from the upstream ``scripts/play_dat.py``
example but extends it with analytics from ``event_tools.rpm``.

Example usage::

    from event_tools.visualization import visualize_dat

    # Play a recording with 100 ms display windows and RPM estimates
    visualize_dat("path/to/file.dat", window_ms=100, rpm_range=(1000, 7000))

The display can be interrupted by pressing ``Esc`` or ``q``.  When
running in a headless environment the function can be invoked with
``display=False`` to suppress GUI creation; in that case it simply
iterates through the windows and performs the analysis without
rendering.
"""

from __future__ import annotations

import itertools
import time
from statistics import mean
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

import cv2


from evio.core.pacer import Pacer
from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

from rpm import ClusterResult, analyze_window, decode_window


WINDOW_NAME = "Event Viewer"


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
    pacer: Pacer,
    batch_range: object,
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
    pacer : Pacer
        Pacer instance used for playback timing; used to compute wall
        time and recording time.
    batch_range : BatchRange
        Represents the current window range; used to compute the
        recording time offset displayed in the HUD.
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
    # Draw HUD similar to play_dat.py
    if pacer._t_start is not None and pacer._e_start is not None:
        wall_time_s = time.perf_counter() - pacer._t_start
        rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)
        if pacer.force_speed:
            first_row = (
                f"speed={pacer.speed:.2f}x  drops/ms={pacer.instantaneous_drop_rate:.2f}  "
                f"avg(drops/ms)={pacer.average_drop_rate:.2f}"
            )
        else:
            first_row = (
                f"(target) speed={pacer.speed:.2f}x  force_speed=False, no drops"
            )
        rpm_values = [
            cluster.rpm / blade_count
            for cluster in clusters
            if not np.isnan(cluster.rpm)
        ]
        mean_rpm = mean(rpm_values) if rpm_values else float("nan")
        second_row = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s  Mean RPM={mean_rpm:7.1f}"
        cv2.putText(frame, first_row, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
        cv2.putText(frame, second_row, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
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
    dat_path: str,
    blade_count: int = 2,
    *,
    window_ms: float = 10.0,
    rpm_range: Tuple[int, int] = (1000, 7000),
    speed: float = 1.0,
    force_speed: bool = False,
    min_cluster_size: int = 50,
    min_fill_ratio: float = 0.15,
    max_aspect_ratio: Optional[float] = None,
    min_events_per_pixel: Optional[int] = None,
    display: bool = True,
    **rpm_kwargs: object,
) -> None:
    """Stream a ``.dat`` file with RPM and bounding box overlays.

    Parameters
    ----------
    dat_path : str
        Path to the ``.dat`` recording to play.
    blade_count : int, optional
        Number of blades or repeating features per revolution.  Used to
        convert event RPM values to mechanical RPM.  Must be positive.
    window_ms : float, optional
        Length of the display window in milliseconds.  This controls
        how many events are grouped into each frame and implicitly
        determines the time period used for RPM estimation.  Larger
        values produce smoother RPM estimates at the cost of latency.
    rpm_range : tuple, optional
        Allowed range of RPM values.  The estimator will clip to this
        interval.  Defaults to (1000, 7000) which spans typical
        rotating fan and drone propeller speeds.
    speed : float, optional
        Playback speed multiplier.  ``1.0`` renders in real time; larger
        values accelerate playback.
    force_speed : bool, optional
        Whether to drop windows to maintain the target playback speed.
        See ``evio.core.pacer.Pacer`` for details.
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

    Notes
    -----
    The underlying ``DatFileSource`` always constructs windows of
    ``window_ms`` duration.  If smoother RPM estimates are desired
    without increasing display latency, consider accumulating events
    across multiple frames outside of this function and passing them
    into ``analyze_window`` separately.
    """
    if blade_count <= 0:
        raise ValueError("blade_count must be positive")
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    # Convert milliseconds to microseconds
    window_us = int(window_ms * 1000)
    # Load the recording once to get timestamps, width and height
    rec = open_dat(dat_path, width=1280, height=720)  # default resolution
    width = int(rec.width)
    height = int(rec.height)
    # Use DatFileSource to precompute window ranges and event_words/order
    src = DatFileSource(dat_path, window_length_us=window_us, width=width, height=height)
    # Use Pacer to enforce playback speed
    pacer = Pacer(speed=speed, force_speed=force_speed)
    
    # Setup display window you want to view the calculations
    if display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Iterate over windows
    for batch_range in pacer.pace(src.ranges()):
        # Decode events and polarities for this window
        x_coords, y_coords, polarities = decode_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        # Retrieve sorted timestamps for the same slice
        # rec.timestamps is sorted; slice with batch indices since ranges are in sorted order
        ts_us = rec.timestamps[batch_range.start : batch_range.stop]
        # Compute analytics: global RPM and per‑cluster bounding boxes & RPM
        clusters = analyze_window(
            x_coords,
            y_coords,
            ts_us,
            width,
            height,
            rpm_range=rpm_range,
            min_cluster_size=min_cluster_size,
            min_fill_ratio=min_fill_ratio,
            max_aspect_ratio=max_aspect_ratio,
            min_events_per_pixel=min_events_per_pixel,
            **rpm_kwargs,
        )
        
        if display:
            # Build frame
            frame = _get_frame(x_coords, y_coords, polarities, width, height)
            # Draw overlay text and boxes
            _draw_overlay(frame, pacer, batch_range, clusters, blade_count)
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            # Poll key; break on Esc or q
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        # If not displaying, still respect potential drop by pacer
    # Cleanup
    if display:
        cv2.destroyWindow(WINDOW_NAME)
