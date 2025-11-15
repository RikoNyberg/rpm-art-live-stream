from visualization import visualize_dat

# Display with 100 ms windows and RPM estimation
visualize_dat("data/drone_idle.dat", window_ms=100, rpm_range=(1000, 7000))
