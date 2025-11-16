# RPM-ART: Live Stream Radar

 [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/etFZYg8TpOo/0.jpg)](https://www.youtube.com/watch?v=etFZYg8TpOo)

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

## Large File Storage (LFS)

The sample recordings in `data/` are tracked with Git LFS. If you do not
have it installed, follow the official instructions for your platform:
https://git-lfs.com/. After installation, run:

```bash
git lfs install
git lfs pull
```

`git lfs install` enables LFS hooks for the repository, and `git lfs pull`
downloads the tracked data blobs into `data/`.

## Usage

1. After `git lfs pull` your `.dat` recordings are in the `data/` directory.
2. Run the visualization loop:
   ```bash
   python main.py
   ```
   The script decodes the stream with the selected window size, estimates
   blade RPM within the configured range, and renders interactive output.

You can select between predefined configurations (`fan_const_rpm`, `drone_moving`, or `fred_events`) or set your own configurations and/or set your own data to `data/` folder

You can tune `window_ms`, `blade_count`, `min_cluster_size`, and other
parameters inside `main.py` to match new datasets or hardware.
