from visualization import visualize_dat

# Display with 100 ms windows and RPM estimation
visualize_dat(
    # "data/fan_const_rpm.dat",
    "data/drone_moving.dat",
    window_ms=10,
    rpm_range=(1000, 70000),
    force_speed=True,
    # min_cluster_size=50,
    # display=True,
)
