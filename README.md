# Senso Radar

A lightweight toolkit for replaying `.dat` recordings from a Prophesee
event camera, clustering rotor blade events, and estimating the RPM of
fans or drones in real time. The `main.py` entry point streams a
recording window-by-window, decodes packed events, and feeds them into
`visualization.py`, which overlays bounding boxes and RPM estimates.

## Setup

1. (Optional) create a virtual environment.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/ahtihelminen/evio.git
   ```
   The second command pulls the `evio` utilities used for decoding
   Prophesee `.dat` files.

## Usage

1. Place your `.dat` recordings in the `data/` directory or update the
   paths in `main.py::configs`.
2. Pick one of the predefined configurations (`fan_const_rpm`,
   `drone_moving`, or `fred_events`) or add your own dictionary entry.
3. Run the visualization loop:
   ```bash
   python main.py
   ```
   The script decodes the stream with the selected window size, estimates
   blade RPM within the configured range, and renders interactive output.

You can tune `window_ms`, `blade_count`, `min_cluster_size`, and other
parameters inside `main.py` to match new datasets or hardware.
