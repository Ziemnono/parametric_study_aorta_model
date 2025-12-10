# Parametric Study of Aorta Model

Generate wall shear stress (WSS) statistics and publication-ready figures for a parametric sweep of simplified aortic geometries.  The pipeline reads ParaView `.pvd/.vtp` outputs, aggregates the WSS time history, and exports:

- Fractional residence time (FRT) grouped bar charts across diameters and stroke volumes
- Percentile summaries for each simulated case
- Combined linear/logarithmic histograms with inset view highlighting low, intermediate, and high WSS regions

The repository is designed to be published directly to GitHub with a reproducible Python environment and links to the archived datasets on Zenodo.

## Repository Layout

- `data/` – sample CFD post-processing outputs (`*.pvd` + `*.vtp`) and per-case `parameters.json`
- `figures/` – auto-populated with plots produced by `main.py`
- `main.py` – end-to-end script that loads the ParaView results and generates every figure
- `scripts/setup_venv.sh` – helper to create a local virtual environment
- `requirements.txt` – pinned Python dependencies, including the ParaView Python modules

## Prerequisites

### System Requirements

- Python 3.10 or newer
- ParaView 5.11+ (mandatory: the script imports `paraview.simple`)

Install ParaView using one of the following approaches:

1. **Linux (Ubuntu/Debian)**
   ```bash
   sudo apt update
   sudo apt install paraview
   ```
2. **macOS / Windows** – download the official package from the [ParaView Download Center](https://www.paraview.org/download/), install it, and ensure `pvpython` or `paraview` is on your `PATH`.
3. **Headless environments** – install the ParaView Python wheels from PyPI (already listed in `requirements.txt`). This is sufficient for non-interactive runs but still requires the VTK dependencies provided by the wheel.

> **Tip:** if you install ParaView separately from the Python environment, set `PYTHONPATH` or run the script via `pvpython` to reuse the ParaView runtime libraries.

## Python Environment

Create and populate a virtual environment directly from the repository:

```bash
# from the repository root
git clone <this repo>
cd parametric_study_aorta_model
bash scripts/setup_venv.sh
source .venv/bin/activate
```

Alternatively, manage the environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (`source .venv/bin/activate`).
2. Place your ParaView exports (same directory layout as the provided `data/diameter_*` examples) under `data/`.
3. Run the analysis:
   ```bash
   python main.py
   ```
4. Retrieve the generated images and logs in `figures/`.

### Customising a run

- **Time window:** edit `DEFAULT_TIME_STEP_RANGE` in `main.py` to choose the time steps used for the statistics.
- **Thresholds:** update `LOW_WSS_THRESHOLD` and `HIGH_WSS_THRESHOLD` at the top of `main.py` if your definition of low/intermediate/high WSS differs.
- **Additional metadata:** extend the per-case `parameters.json` with extra keys (e.g., Reynolds number). They are automatically surfaced in the legends if you add them to the plotting logic.

## Outputs

Running the script creates the following deliverables inside `figures/`:

- `wss_distribution.png` – grouped bar chart showing the FRT percentage for each case (sorted by diameter).
- `WSS_histogram_diameter_<X>_strokevolume_<Y>.png` – combined linear/log histograms per case. The inset displays the same distribution on a log scale, including annotated percentages.

## Data Availability (Zenodo)

All CFD-derived surfaces, parameters, and supplementary case descriptions are permanently archived on Zenodo:

- DOI: [10.5281/zenodo.17873230](https://zenodo.org/records/17873231)

Download the archive, extract it next to this repository, and point the `data/` directory to the extracted folder if you need the full-resolution meshes.

> Replace the DOI placeholder above with your published Zenodo record before making the repository public.

## Citation

If you use this pipeline or the accompanying dataset in your research, please cite the Zenodo record and this GitHub repository. A sample BibTeX entry is available from the Zenodo page once the DOI is minted.

## License

Distributed under the [Apache License 2.0](LICENSE). Contributions are welcome via pull requests.
