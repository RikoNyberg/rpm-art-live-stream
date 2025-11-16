from __future__ import annotations

from typing import Iterator, Literal

from evio.core.pacer import Pacer
from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

from rpm import decode_window
from visualization import EventWindow, visualize_dat


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


configs = {
    "fan_const_rpm": {
        "dat_path": "data/fan_const_rpm.dat",
        "blade_count": 3,
        "window_ms": 100,
        "min_cluster_size": 50,
    },
    "drone_moving": {
        "dat_path": "data/drone_moving.dat",
        "blade_count": 2,
        "window_ms": 10,
        "min_cluster_size": 50,
    },
    "fred_events": {
        "dat_path": "data/fred_events.dat",
        "blade_count": 2,
        "window_ms": 10,
        "min_cluster_size": 10,
    },
}

if __name__ == "__main__":
    config: Literal["fan_const_rpm", "drone_moving", "fred_events"] = "drone_moving"
    dat_stream = stream_dat_file(
        dat_path=configs[config]["dat_path"],
        window_ms=configs[config]["window_ms"],
        speed=1.0,
        force_speed=True,
    )
    visualize_dat(
        dat_stream,
        blade_count=configs[config]["blade_count"],
        rpm_range=(1000, 70000),
        min_fill_ratio=0.15,
        min_cluster_size=configs[config]["min_cluster_size"],
    )
