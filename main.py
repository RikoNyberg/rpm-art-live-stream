from __future__ import annotations

from typing import Literal
from simulate_streaming import stream_dat_file
from visualization import visualize_dat


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
