from visualization import visualize_dat

# Display with 100 ms windows and RPM estimation
visualize_dat(
    # "data/fan_const_rpm.dat",
    # blade_count=3,
    # window_ms=100,
    # rpm_range=(1000, 20000),
    
    dat_path="data/drone_moving.dat",
    blade_count=2,
    window_ms=10,
    rpm_range=(1000, 70000),
    force_speed=True,
)
