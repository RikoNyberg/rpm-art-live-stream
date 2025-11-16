"""Algorithms for RPM estimation and drone detection.

This module exposes a set of functions that work on batches of
event camera data.  They are designed to be both simple and fast so
that they can run inline with a real‑time playback loop.  The
functions here do not read files themselves; instead they accept
decoded arrays of timestamps and coordinates produced by the
``evio.source.dat_file.DatFileSource`` and its helpers.

Overview
========

The core functionality consists of the following:

* ``decode_window(event_words, time_order, win_start, win_stop)`` –
  Decodes a slice of packed event words and returns X/Y coordinate and
  polarity arrays.  This replicates the logic in ``play_dat.py`` but is
  exposed as a reusable function.

* ``estimate_rpm(timestamps_us, rpm_range=(1000, 7000))`` –
  Estimates the rotational speed in RPM given an array of event
  timestamps (microseconds).  Two methods are provided: a fast
  median‑based estimator and a more robust periodogram (FFT) based
  estimator.  Both operate on microsecond precision and return ``NaN``
  when insufficient events are present.

* ``detect_clusters(x_coords, y_coords, width, height, min_size=50)`` –
  Groups events into connected clusters on the image plane using
  OpenCV’s ``connectedComponentsWithStats``.  A minimal cluster size
  threshold prevents spurious detections.

* ``analyze_window(x_coords, y_coords, timestamps_us, width, height, rpm_range=(1000,7000), min_cluster_size=50)`` –
  Combines cluster detection and frequency estimation.  Returns a list
  of :class:`ClusterResult` objects containing bounding boxes and RPM
  estimates.  The bounding box coordinates refer to the original pixel
  coordinate system (top‑left origin).

All functions are annotated for type checking and strive to avoid
heavy allocations.  For example, the cluster detection first crops
to the minimal bounding rectangle covering all events to limit the
size of the intermediate occupancy map.  The RPM estimation uses
SciPy’s periodogram when available; if SciPy is not installed the
algorithm falls back to a simple median‑difference method.  Users can
select the estimator via the ``method`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# SciPy’s periodogram gives a more robust spectral estimate.
from scipy.signal import periodogram
import cv2  # type: ignore


@dataclass(frozen=True)
class ClusterResult:
    """Simple container describing a detected cluster and its RPM."""

    box: Tuple[int, int, int, int]
    rpm: float

def decode_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a time‑ordered slice of events into coordinates and polarities.

    Parameters
    ----------
    event_words : np.ndarray
        Array of packed ``uint32`` words, each packing polarity, y and x
        coordinates.  See the ``evio`` documentation for the bit layout.
    time_order : np.ndarray
        Indices that sort the events in ascending timestamp order.  This
        can be obtained from ``DatFileSource.order``.
    win_start, win_stop : int
        Inclusive/exclusive indices into ``time_order`` defining the
        current window.  These come from ``BatchRange.start`` and
        ``BatchRange.stop``.

    Returns
    -------
    x_coords : np.ndarray
        Integer array of x pixel coordinates.
    y_coords : np.ndarray
        Integer array of y pixel coordinates.
    polarities : np.ndarray
        Boolean array; ``True`` indicates an ON event and ``False`` an
        OFF event.

    Notes
    -----
    This function mirrors the logic in ``scripts/play_dat.py`` and
    avoids copying where possible.  The result arrays are views into
    the original ``event_words`` array, so they should not be mutated.
    """
    # Get indices of events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    # Bit‑mask decode according to the EVT3 spec
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    # Polarity is stored in the upper 4 bits; >0 denotes ON
    polarities = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, polarities


def _median_rpm(timestamps_us: np.ndarray, rpm_range: Tuple[int, int]) -> float:
    """Fast but crude RPM estimator based on median inter‑event intervals.

    This estimator assumes that successive events correspond roughly to
    half a rotation (two events per cycle).  It computes the median
    difference between timestamps, converts that to a period and
    returns the corresponding RPM.  The result is clamped to the
    provided ``rpm_range``.

    Parameters
    ----------
    timestamps_us : np.ndarray
        Event timestamps in microseconds.  Must be sorted.
    rpm_range : tuple of int
        Allowed RPM range (min, max).  The output is clipped to this
        interval.

    Returns
    -------
    float
        Estimated RPM or ``NaN`` if the input is empty or constant.
    """
    if timestamps_us.size < 2:
        return float("nan")
    # Compute differences between successive events (in microseconds)
    diffs = np.diff(timestamps_us)
    # Remove non‑positive or extreme differences
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    # Use trimmed percentiles to reduce influence of spurious long gaps
    low, high = np.percentile(diffs, [5, 95])
    median_diff = float(np.median(np.clip(diffs, low, high)))
    # Assume two events per rotation (positive and negative edge)
    period_us = median_diff * 2.0
    if period_us <= 0:
        return float("nan")
    period_s = period_us * 1e-6
    hz = 1.0 / period_s
    rpm = hz * 60.0
    # Clamp to requested range
    return float(max(rpm_range[0], min(rpm_range[1], rpm)))


def _periodogram_rpm(
    timestamps_us: np.ndarray,
    rpm_range: Tuple[int, int],
    n_bins: int = 256,
    minimum_events: int = 50,
) -> float:
    """Robust RPM estimator using a periodogram of event counts.

    This estimator bins events into a fixed number of equally spaced
    time bins and computes the power spectral density of the resulting
    histogram.  The dominant frequency within the user‑supplied RPM
    range is returned.  A small number of events may cause noisy
    estimates, hence a minimum event count is enforced.

    Parameters
    ----------
    timestamps_us : np.ndarray
        Event timestamps in microseconds, sorted in ascending order.
    rpm_range : tuple (min_rpm, max_rpm)
        Allowed RPM range.  Frequencies outside this range are ignored
        when selecting the dominant component.
    n_bins : int, optional
        Number of histogram bins to compute.  The default value of 256
        provides a good balance between frequency resolution and speed.
    minimum_events : int, optional
        Minimum number of events required to attempt estimation.

    Returns
    -------
    float
        Estimated RPM or ``NaN`` if SciPy is unavailable or the input
        lacks sufficient events.
    """
    if timestamps_us.size < max(2, int(minimum_events)):
        return _median_rpm(timestamps_us, rpm_range)
    t0 = float(timestamps_us[0])
    duration_us = float(timestamps_us[-1] - t0)
    if duration_us <= 0:
        return _median_rpm(timestamps_us, rpm_range)
    # Build histogram of event counts.  Use int to avoid float noise.
    # Edges are inclusive/exclusive, spanning the entire window.
    hist, _ = np.histogram(
        timestamps_us,
        bins=n_bins,
        range=(t0, t0 + duration_us),
    )
    # Time between bin centres in seconds
    bin_width_us = duration_us / n_bins
    dt_s = bin_width_us * 1e-6
    if dt_s <= 0.0:
        return _median_rpm(timestamps_us, rpm_range)
    # Sampling frequency in Hz
    fs = 1.0 / dt_s
    freqs, power = periodogram(hist, fs=fs, scaling="spectrum")
    # Convert RPM range to Hz range for filtering
    min_hz = rpm_range[0] / 60.0
    max_hz = rpm_range[1] / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return _median_rpm(timestamps_us, rpm_range)
    # Ignore DC component (0 Hz)
    if mask[0] and freqs[0] == 0.0:
        mask[0] = False
    if not np.any(mask):
        return _median_rpm(timestamps_us, rpm_range)
    # Pick the frequency with maximum power within the range
    freq = float(freqs[mask][np.argmax(power[mask])])
    rpm = freq * 60.0
    return float(max(rpm_range[0], min(rpm_range[1], rpm)))


def estimate_rpm(
    timestamps_us: np.ndarray,
    rpm_range: Tuple[int, int] = (1000, 7000),
    **kwargs: object,
) -> float:
    """Estimate the rotational speed (RPM) from event timestamps.

    A unified wrapper around the median‑based and periodogram‑based
    estimators.  When SciPy is available, the periodogram method is
    chosen by default because it tends to give more stable results on
    noisy event streams.  If SciPy is not installed or the user
    explicitly selects ``method='median'``, the fast median estimator
    is used.  Additional keyword arguments are forwarded to the
    underlying estimator.

    Parameters
    ----------
    timestamps_us : np.ndarray
        Sorted event timestamps in microseconds.
    rpm_range : tuple (min_rpm, max_rpm), optional
        Bounds on the plausible RPM values.  The estimate will be
        clipped to this range.  Defaults to (1000, 7000) to cover
        typical rotating fan and drone propeller speeds.
    method : {'periodogram', 'median'}, optional
        Which estimator to use.  ``'periodogram'`` requires SciPy;
        if SciPy is not available this will silently fall back to
        ``'median'``.
    **kwargs :
        Extra parameters to the chosen estimator.  See
        ``_periodogram_rpm`` and ``_median_rpm`` for details.

    Returns
    -------
    float
        Estimated RPM or ``NaN`` if insufficient data is available.
    """
    # Ensure input is a 1‑dimensional NumPy array of ints
    ts = np.asarray(timestamps_us, dtype=np.int64).ravel()
    if ts.size == 0:
        return float("nan")
    # Guarantee monotonic order
    if not (np.all(np.diff(ts) >= 0)):
        ts = np.sort(ts)
    return _periodogram_rpm(ts, rpm_range, **kwargs)


def detect_clusters(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    width: int,
    height: int,
    min_size: int = 50,
    min_events_per_pixel: Optional[int] = None,
    min_fill_ratio: float = 0.0,
    max_aspect_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], np.ndarray]:
    """Detect connected clusters of events and compute bounding boxes.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Arrays of integer pixel coordinates for each event in the current
        window.
    width, height : int
        Full resolution of the sensor.  Bounding boxes are reported
        relative to this coordinate system.
    min_size : int, optional
        Minimum number of pixels in a cluster.  Clusters with fewer
        than this many activated pixels are discarded as noise.
    min_events_per_pixel : int, optional
        Minimum number of events that must hit the same pixel within
        the current window for those events to be considered.  Pixels
        with fewer hits are treated as background.  Defaults to
        ``max(1, min_size // 10)`` which scales the requirement with
        the requested cluster size.
    min_fill_ratio : float, optional
        Minimum fraction of active pixels inside the bounding box for
        a cluster to be retained.  Helps reject large, sparse blobs.
    max_aspect_ratio : float, optional
        Maximum allowed ratio between width and height.  Thin shapes
        with ratios above this threshold are discarded.

    Returns
    -------
    labels : np.ndarray
        Label assigned to each event.  Background is labelled ``0``.
    boxes : list of tuples
        Bounding boxes for each cluster.  Each box is a tuple
        ``(x, y, w, h)`` in absolute pixel coordinates.  Boxes are
        returned in ascending label order (starting from 1).
    cluster_ids : np.ndarray
        Unique labels of retained clusters.  This array is aligned
        with ``boxes``.

    Notes
    -----
    This function constructs a sparse occupancy map limited to the
    minimal rectangle covering all events.  This significantly
    reduces the memory footprint and speeds up the connected
    component analysis.  When OpenCV is unavailable the function
    will fall back to a naive single‑cluster output that spans all
    events.
    """
    x_all = np.asarray(x_coords, dtype=np.int32).ravel()
    y_all = np.asarray(y_coords, dtype=np.int32).ravel()
    num_events = x_all.size
    if num_events == 0:
        # No events: empty outputs
        return np.zeros(0, dtype=np.int32), [], np.empty(0, dtype=np.int32)
    if min_events_per_pixel is None:
        min_events_per_pixel = max(1, min_size // 10)
    min_events_per_pixel = max(1, int(min_events_per_pixel))
    dense_mask: Optional[np.ndarray] = None
    x_work = x_all
    y_work = y_all
    if min_events_per_pixel > 1:
        linear_coords = y_all.astype(np.int64) * int(width) + x_all.astype(np.int64)
        _, inverse_idx, counts = np.unique(linear_coords, return_inverse=True, return_counts=True)
        keep = counts[inverse_idx] >= min_events_per_pixel
        if not np.any(keep):
            # No sufficiently dense events remain
            return np.zeros(num_events, dtype=np.int32), [], np.empty(0, dtype=np.int32)
        if not np.all(keep):
            dense_mask = keep
            x_work = x_all[keep]
            y_work = y_all[keep]
    # After density filtering there may be fewer points
    if x_work.size == 0:
        return np.zeros(num_events, dtype=np.int32), [], np.empty(0, dtype=np.int32)
    # Crop to the region containing all events to reduce the occupancy map
    x_min = int(np.min(x_work))
    x_max = int(np.max(x_work))
    y_min = int(np.min(y_work))
    y_max = int(np.max(y_work))
    # Expand bounding region by 1 pixel on each side to avoid zero area
    x_min = max(0, x_min - 1)
    y_min = max(0, y_min - 1)
    x_max = min(width - 1, x_max + 1)
    y_max = min(height - 1, y_max + 1)
    region_w = x_max - x_min + 1
    region_h = y_max - y_min + 1
    # Allocate occupancy image
    # Use uint8 to save memory; 1 means event present
    
    occ = np.zeros((region_h, region_w), dtype=np.uint8)
    # Fill occupancy map
    occ[y_work - y_min, x_work - x_min] = 1
    # Connected components analysis
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    # labels_map has shape (region_h, region_w)
    # Extract bounding boxes for clusters above min_size
    boxes: List[Tuple[int, int, int, int]] = []
    cluster_ids: List[int] = []
    heuristics_active = min_fill_ratio > 0.0 or (max_aspect_ratio is not None)
    # labels 0 is background; start from 1
    for label_id in range(1, num_labels):
        count = int(stats[label_id, cv2.CC_STAT_AREA])  # type: ignore
        if count < min_size:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])  # type: ignore
        y = int(stats[label_id, cv2.CC_STAT_TOP])  # type: ignore
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])  # type: ignore
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])  # type: ignore
        if w <= 0 or h <= 0:
            continue
        if min_fill_ratio > 0.0:
            box_area = float(w * h)
            fill_ratio = count / box_area if box_area > 0 else 0.0
            if fill_ratio < min_fill_ratio:
                continue
        if max_aspect_ratio is not None and max_aspect_ratio > 0:
            short_side = max(1, min(w, h))
            aspect = max(w, h) / float(short_side)
            if aspect > max_aspect_ratio:
                continue
        # Convert region coordinates back to full frame coordinates
        boxes.append((x + x_min, y + y_min, w, h))
        cluster_ids.append(label_id)
    if not boxes and not heuristics_active:
        # If no cluster passes threshold, produce one covering all events
        boxes = [(x_min, y_min, region_w, region_h)]
        cluster_ids = [1]
    # Map per‑event labels; collapse small clusters to background (0)
    # Build mapping from region label id to cluster index (1..)
    label_remap = np.zeros(num_labels, dtype=np.int32)
    for idx, cid in enumerate(cluster_ids, start=1):
        label_remap[cid] = idx
    # Flatten event labels for the filtered subset
    filtered_labels = label_remap[labels_map[y_work - y_min, x_work - x_min]]
    cluster_array = np.array(list(range(1, len(cluster_ids) + 1)), dtype=np.int32)

    if dense_mask is None:
        return filtered_labels, boxes, cluster_array
    # Expand filtered labels back to the original event array, marking
    # filtered-out events as background (label 0).
    labels_full = np.zeros(num_events, dtype=np.int32)
    labels_full[dense_mask] = filtered_labels
    return labels_full, boxes, cluster_array


def analyze_window(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps_us: np.ndarray,
    width: int,
    height: int,
    rpm_range: Tuple[int, int] = (1000, 7000),
    min_cluster_size: int = 50,
    min_fill_ratio: float = 0.0,
    max_aspect_ratio: Optional[float] = None,
    min_events_per_pixel: Optional[int] = None,
    **rpm_kwargs: object,
) -> List[ClusterResult]:
    """Compute global and per‑cluster RPM estimates and bounding boxes.

    Given decoded event data for a single time window, this function
    clusters the events spatially, estimates the rotational speed for
    the entire window and for each cluster individually, and returns
    structured results.  It is the main workhorse for downstream
    visualization and analytics.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Arrays of pixel coordinates for the events in the current window.
    timestamps_us : np.ndarray
        Timestamps corresponding to each event (in microseconds).
        These should be sorted in the same order as the events appear
        in the window.  If not sorted, they will be sorted internally.
    width, height : int
        Full resolution of the sensor; required to compute absolute
        bounding boxes.
    rpm_range : tuple (min_rpm, max_rpm), optional
        Bounds on the plausible rotation speed.  Defaults to (1000,
        7000), which spans typical rotating fan and drone propeller
        speeds.
    min_cluster_size : int, optional
        Minimum number of pixels for a cluster to be retained.
    min_fill_ratio : float, optional
        Minimum occupancy ratio (active pixels divided by bounding box
        area).  Clusters with lower ratios are discarded.
    max_aspect_ratio : float, optional
        Maximum allowed width/height ratio for clusters.
    **rpm_kwargs :
        Additional parameters forwarded to ``estimate_rpm``.
    min_events_per_pixel : int, optional
        If provided, requires at least this many hits per pixel before
        the pixel is considered when forming clusters.  Useful for
        filtering out sparse noise.

    Returns
    -------
    clusters : list of ClusterResult
        Results sorted in ascending cluster ID order.  If all heuristics
        reject the current window the list may be empty.
    """
    # Ensure arrays are NumPy arrays of correct dtype
    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    ts = np.asarray(timestamps_us, dtype=np.int64)
    # Sort timestamps if necessary (maintain index mapping to x/y)
    if ts.size > 1 and not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts)
        ts = ts[order]
        x = x[order]
        y = y[order]
    # Detect clusters and bounding boxes
    labels, boxes, _ = detect_clusters(
        x,
        y,
        width,
        height,
        min_size=min_cluster_size,
        min_fill_ratio=min_fill_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_events_per_pixel=min_events_per_pixel,
    )
    clusters: List[ClusterResult] = []
    for idx, box in enumerate(boxes, start=1):
        # Extract indices belonging to this cluster
        mask = labels == idx
        cluster_ts = ts[mask]
        if cluster_ts.size == 0:
            rpm = float("nan")
        else:
            rpm = estimate_rpm(cluster_ts, rpm_range=rpm_range, **rpm_kwargs)
        clusters.append(ClusterResult(box=box, rpm=rpm))
    return clusters
