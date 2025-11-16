from __future__ import annotations

from typing import Iterator

from evio.core.pacer import Pacer
from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

from rpm import decode_window
from visualization import EventWindow


def stream_dat_file(
    dat_path: str,
    *,
    window_ms: float = 10.0,
    speed: float = 1.0,
    force_speed: bool = False,
    width: int = 1280,
    height: int = 720,
) -> Iterator[EventWindow]:
    """Yield ``EventWindow`` objects by reading a ``.dat`` file incrementally."""
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")

    window_us = int(window_ms * 1000)
    recording = open_dat(dat_path, width=width, height=height)
    width = int(recording.width)
    height = int(recording.height)
    src = DatFileSource(
        dat_path, window_length_us=window_us, width=width, height=height
    )
    pacer = Pacer(speed=speed, force_speed=force_speed)

    for batch_range in pacer.pace(src.ranges()):
        x_coords, y_coords, polarities = decode_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        ts_us = recording.timestamps[batch_range.start : batch_range.stop]
        start_ts_us = (
            int(ts_us[0]) if ts_us.size else int(getattr(batch_range, "start_ts_us", 0))
        )
        end_ts_us = (
            int(ts_us[-1])
            if ts_us.size
            else int(getattr(batch_range, "end_ts_us", start_ts_us))
        )
        yield EventWindow(
            x_coords=x_coords,
            y_coords=y_coords,
            polarities=polarities,
            timestamps_us=ts_us,
            width=width,
            height=height,
            start_ts_us=start_ts_us,
            end_ts_us=end_ts_us,
            metadata={"batch_range": batch_range},
        )

