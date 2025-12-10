import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from paraview.simple import XMLPolyDataReader, servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy

LOW_WSS_THRESHOLD = 0.070477
HIGH_WSS_THRESHOLD = 7.867869
LOG_HIST_BINS = 1000
LINEAR_BIN_WIDTH = 0.1
DEFAULT_TIME_STEP_RANGE = (20, 39)


def _validate_time_step_range(time_step_range: Tuple[int, int]) -> Tuple[int, int]:
    """Ensure the requested time-step window is sensible."""
    start, end = time_step_range
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid time_step_range {time_step_range}. "
            "Expected non-negative start and end > start."
        )
    return start, end


def _compute_frt_percentages(wss_values: np.ndarray) -> Dict[str, float]:
    """Return fractional residence time percentages for the standard thresholds."""
    total_count = wss_values.size
    if total_count == 0:
        return {"low": 0.0, "intermediate": 0.0, "high": 0.0}

    low_count = np.sum(wss_values < LOW_WSS_THRESHOLD)
    mid_count = np.sum(
        (wss_values >= LOW_WSS_THRESHOLD) & (wss_values <= HIGH_WSS_THRESHOLD)
    )
    high_count = np.sum(wss_values > HIGH_WSS_THRESHOLD)
    return {
        "low": low_count / total_count * 100.0,
        "intermediate": mid_count / total_count * 100.0,
        "high": high_count / total_count * 100.0,
    }


def _split_wss_by_range(wss_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split wall shear stress values into the three canonical ranges."""
    low = wss_values[wss_values <= LOW_WSS_THRESHOLD]
    mid = wss_values[
        (wss_values > LOW_WSS_THRESHOLD) & (wss_values < HIGH_WSS_THRESHOLD)
    ]
    high = wss_values[wss_values >= HIGH_WSS_THRESHOLD]
    return low, mid, high


def numpy_from_pvd_paraview(case: str, input_directory: Path) -> np.ndarray:
    """Load the full wall shear stress time series for a case using ParaView."""

    pvd_file = (
        input_directory
        / case
        / "surface"
        / "artery_opened_ma"
        / "artery_opened_ma.pvd"
    )
    if not pvd_file.exists():
        raise FileNotFoundError(f"PVD file not found at {pvd_file}")

    import xml.etree.ElementTree as ET

    tree = ET.parse(pvd_file)
    root = tree.getroot()
    collection = root.find("Collection")
    if collection is None:
        for child in root:
            if str(child.tag).endswith("Collection"):
                collection = child
                break
    if collection is None:
        raise ValueError("No 'Collection' element found in the PVD file")

    datasets = list(collection.findall("DataSet"))
    if not datasets:
        datasets = [c for c in collection if str(c.tag).endswith("DataSet")]
    if not datasets:
        raise ValueError("No DataSet entries found in the PVD file")

    def _get_time(ds):
        t = ds.get("timestep", ds.get("TimeStep", "0"))
        try:
            return float(t)
        except Exception:
            return 0.0

    datasets.sort(key=_get_time)

    wall_shear_series = []
    base_dir = pvd_file.parent
    first_shape = None

    for ds in datasets:
        file_rel = ds.get("file")
        if not file_rel:
            continue
        file_path = base_dir / file_rel
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue

        # Read VTP file using ParaView
        reader = XMLPolyDataReader(FileName=[str(file_path)])
        reader.UpdatePipeline()

        # Get output data object
        data = servermanager.Fetch(reader)

        # Extract wall_shear
        pd = data.GetPointData()
        wss_vtk = pd.GetArray("wall_shear")
        if wss_vtk is None:
            print(f"Warning: 'wall_shear' array not found in {file_path}. Skipping.")
            continue
        wall_shear = vtk_to_numpy(wss_vtk)

        if first_shape is None:
            first_shape = wall_shear.shape
        if wall_shear.shape != first_shape:
            print(
                f"Warning: WSS shape changed from {first_shape} to {wall_shear.shape}. Skipping."
            )
            continue

        wall_shear_series.append(wall_shear)
    if not wall_shear_series:
        raise ValueError(f"No valid 'wall_shear' data found for case {case}")

    wall_shear_array = np.stack(wall_shear_series, axis=0)

    return wall_shear_array


def examination(data_dict: Dict[str, Dict], output_directory: Path, time_step_range):
    """Create the grouped FRT bar plot summarising every case."""

    bar_data = []
    start_time, end_time = _validate_time_step_range(time_step_range)

    for case, metadata in data_dict.items():
        wall_shear = metadata["wall_shear_all"][start_time:end_time]
        frt = _compute_frt_percentages(wall_shear)
        bar_data.append(
            {
                "case": case,
                "diameter": metadata.get("diameter", "unknown"),
                "stroke_volume": metadata.get("SV", "unknown"),
                "cfd_max_mesh_size": metadata.get("cfd_max_mesh_size", "unknown"),
                **frt,
            }
        )

    # Sort the data by diameter for plotting
    bar_data_sorted = sorted(bar_data, key=lambda x: x["diameter"])
    labels = [f"{d['diameter']}mm, {d['stroke_volume']}mL" for d in bar_data_sorted]
    low_vals = [d["low"] for d in bar_data_sorted]
    intermediate_vals = [d["intermediate"] for d in bar_data_sorted]
    high_vals = [d["high"] for d in bar_data_sorted]

    # Create the grouped barplot
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width,
        low_vals,
        width,
        label=f"Low WSS < {round(LOW_WSS_THRESHOLD, 2)} Pa",
        color="yellow",
        edgecolor="black",
    )
    plt.bar(
        x,
        intermediate_vals,
        width,
        label=f"Intermediate WSS",
        color="green",
        edgecolor="black",
    )
    plt.bar(
        x + width,
        high_vals,
        width,
        label=f"High WSS > {round(HIGH_WSS_THRESHOLD, 2)}",
        color="red",
        edgecolor="black",
    )
    plt.ylabel("Fractional Residence Time (%)", fontsize=18)
    plt.xlabel("Surface Mesh Size (mm)", fontsize=18)
    plt.xticks(x, labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, framealpha=0.4, loc="center right")
    # horizontal grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    annotate = False
    if annotate:
        # Add value labels above each bar
        for i, (low, intermediate, high) in enumerate(
            zip(low_vals, intermediate_vals, high_vals)
        ):
            plt.text(
                x[i] - width,
                low + 1,
                f"{low:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
            )
            plt.text(
                x[i],
                intermediate + 1,
                f"{intermediate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
            )
            plt.text(
                x[i] + width,
                high + 1,
                f"{high:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    fig_path = output_directory / "wss_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def threshold_calculator_examination(
    data_dict: Dict[str, Dict], time_step_range: Tuple[int, int]
):
    """Log the 5th and 95th percentile WSS values for every case."""

    start_time_step, end_time_step = _validate_time_step_range(time_step_range)

    for case, metadata in data_dict.items():
        print(f"case: {case}")
        wall_shear = metadata["wall_shear_all"]
        selected_wall_shear = wall_shear[start_time_step:end_time_step].flatten()
        p5 = np.percentile(selected_wall_shear, 5)
        p95 = np.percentile(selected_wall_shear, 95)
        print(f"5th Percentile: {p5:.6f} Pa")
        print(f"95th Percentile: {p95:.6f} Pa")


def wss_histogram_linear_log_plot(
    data_dict,
    filename,
    output_directory,
    time_step_range,
):
    """Save per-case histograms summarising the linear and log WSS distribution."""

    frt_summary = []  # for later grouped bar charts if needed

    start_time_step, end_time_step = _validate_time_step_range(time_step_range)

    for case, data in data_dict.items():
        # ---- Extract & slice WSS ----
        if "wall_shear_all" not in data:
            print(f"[WARN] 'wall_shear_all' missing for case {case}; skipping.")
            continue

        wss_all = np.asarray(data["wall_shear_all"])  # expected shape: (time, points)
        if wss_all.ndim != 2:
            print(
                f"[WARN] wall_shear_all not 2D for {case}; got shape {wss_all.shape}; skipping."
            )
            continue

        # slice time range (end exclusive, consistent with your second snippet)
        wss_sel = wss_all[start_time_step:end_time_step]  # shape ~ (Tsel, Npts)

        # Flatten over time and space for hist/FRT and drop NaN/inf/negative
        wss_flat = wss_sel.reshape(-1)
        mask = np.isfinite(wss_flat) & (wss_flat >= 0.0)
        wss_flat = wss_flat[mask]

        if wss_flat.size == 0:
            print(f"[WARN] No valid WSS after filtering for {case}; skipping.")
            continue

        frt = _compute_frt_percentages(wss_flat)

        # keep for later bar charts
        frt_summary.append(
            {
                "case": case,
                "diameter": data_dict[case].get("diameter", "unknown"),
                "stroke_volume": data_dict[case].get("SV", "unknown"),
                **frt,
            }
        )

        # ---- Prepare histogram splits (linear) ----
        low_data, mid_data, high_data = _split_wss_by_range(wss_flat)

        # To keep x-axes consistent across cases for comparability:
        x_min_linear = 0.0
        x_max_linear = 10.0  # you already set xlim(0,10)
        custom_bins = np.arange(x_min_linear, x_max_linear + LINEAR_BIN_WIDTH, LINEAR_BIN_WIDTH)

        # weights to plot percentages directly
        total_count = wss_flat.size
        w_low = (
            np.ones_like(low_data) * (100.0 / total_count)
            if low_data.size
            else np.array([])
        )
        w_mid = (
            np.ones_like(mid_data) * (100.0 / total_count)
            if mid_data.size
            else np.array([])
        )
        w_high = (
            np.ones_like(high_data) * (100.0 / total_count)
            if high_data.size
            else np.array([])
        )

        # ---- Figure ----
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(
            [low_data, mid_data, high_data],
            bins=custom_bins,
            stacked=True,
            color=["yellow", "green", "red"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
            align="mid",
            weights=[w_low, w_mid, w_high],
        )

        # thresholds
        ax.axvline(
            LOW_WSS_THRESHOLD,
            color="black",
            linestyle="--",
            label=f"Low WSS = {LOW_WSS_THRESHOLD:.2f} Pa",
        )
        ax.axvline(
            HIGH_WSS_THRESHOLD,
            color="black",
            linestyle="-.",
            label=f"High WSS = {HIGH_WSS_THRESHOLD:.2f} Pa",
        )

        # labels/limits
        ax.set_xlabel("WSS (Pa)", fontsize=20)
        ax.set_ylabel("Relative Frequency (%)", fontsize=20)
        ax.set_xlim(x_min_linear, x_max_linear)
        ax.set_ylim(0, 25)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

        # legend info (keep your swirl ratio; also show max in-range WSS)
        swirl_ratio = data_dict[case].get("swirl_ratio", "unknown")
        max_wss_in_range = float(np.max(wss_flat)) if wss_flat.size else float("nan")
        diameter = data.get("diameter", "unknown")
        stroke_volume = data.get("SV", "unknown")

        def nSV_cal(stroke_volume, diameter):
            result = (float(stroke_volume) * 100) / (
                15 * (np.pi * ((float(diameter) / 2) ** 2))
            )
            return f"{result:.2f}"

        import textwrap

        legend_title = f"Diameter = {diameter} mm\nStroke Volume = {stroke_volume} mL\nnSV = {nSV_cal(stroke_volume, diameter)}"

        ax.legend(
            loc="upper right",
            framealpha=0.95,
            title=legend_title,
            title_fontsize=9.5,
            edgecolor="black",
            fancybox=True,
            fontsize=9.5,
        )

        # ---- Inset: log-scale histogram ----
        inset_ax = inset_axes(
            ax, width="55%", height="55%", loc="upper center", borderpad=2
        )

        # guard against zeros for logspace
        with np.errstate(invalid="ignore"):
            positive_min = (
                np.min(wss_flat[wss_flat > 0]) if np.any(wss_flat > 0) else None
            )
        minpos = (
            positive_min if positive_min is not None else 1e-4
        )  # safe tiny positive
        x_max_log = 70.0
        # fewer bins is fine (3000 is overkill and slow)
        inset_bins = np.logspace(np.log10(minpos), np.log10(x_max_log), 1000)

        inset_counts, _ = np.histogram(wss_flat, bins=inset_bins)
        inset_total = inset_counts.sum()
        inset_rel = (
            (inset_counts / inset_total * 100.0) if inset_total > 0 else inset_counts
        )

        # color each bin by its left edge vs thresholds
        inset_bin_colors = [
            (
                "yellow"
                if edge <= LOW_WSS_THRESHOLD
                else ("red" if edge > HIGH_WSS_THRESHOLD else "green")
            )
            for edge in inset_bins[:-1]
        ]

        inset_ax.bar(
            inset_bins[:-1],
            inset_rel,
            width=np.diff(inset_bins),
            alpha=0.7,
            color=inset_bin_colors,
        )

        inset_ax.axvline(LOW_WSS_THRESHOLD, color="black", linestyle="--")
        inset_ax.axvline(HIGH_WSS_THRESHOLD, color="black", linestyle="-.")

        inset_ax.set_xscale("log")
        inset_ax.set_xlim(minpos, x_max_log)
        inset_ax.set_xticks([minpos, LOW_WSS_THRESHOLD, HIGH_WSS_THRESHOLD, x_max_log])
        inset_ax.get_xaxis().set_major_formatter(ScalarFormatter())

        # annotate FRT percentages from count-based logic
        inset_ax.text(
            0.035,
            0.90,
            f"Low: {frt['low']:.2f}%",
            transform=inset_ax.transAxes,
            fontsize=10,
            ha="left",
        )
        inset_ax.text(
            0.60,
            0.90,
            f"Intermediate: {frt['intermediate']:.2f}%",
            transform=inset_ax.transAxes,
            fontsize=10,
            ha="center",
        )
        inset_ax.text(
            1.0,
            0.90,
            f"High: {frt['high']:.2f}%",
            transform=inset_ax.transAxes,
            fontsize=10,
            ha="right",
        )

        inset_ax.set_xlabel("Log WSS (Pa)", fontsize=16)
        inset_ax.set_ylabel("Rel. Freq. (%)", fontsize=16)
        inset_ax.set_ylim(0, 0.6)  # tweak if needed
        inset_ax.tick_params(axis="x", labelsize=14, rotation=30)
        inset_ax.tick_params(axis="y", labelsize=14)

        # plt.tight_layout()

        # ---- Save ----
        safe_sr = str(swirl_ratio).replace(" ", "_")
        diameter = data.get("diameter", "unknown")
        stroke_volume = data.get("SV", "unknown")
        fig_path_png = os.path.join(
            output_directory,
            f"{filename}_diameter_{diameter}_strokevolume_{stroke_volume}.png",
        )

        inset_ax.set_in_layout(False)  # avoid cutting off inset
        fig.canvas.draw()
        plt.savefig(fig_path_png, dpi=300)  # , bbox_inches="tight")
        plt.close(fig)

    return frt_summary


def main():

    repo_root = Path(__file__).resolve().parent
    input_directory = repo_root / "data"
    output_directory = repo_root / "figures"

    output_directory.mkdir(exist_ok=True)

    cases: Iterable[str] = sorted(
        d.name for d in input_directory.iterdir() if d.is_dir()
    )

    data_dict = {}
    for case in cases:
        try:
            wall_shear_all = numpy_from_pvd_paraview(case, input_directory)
            json_path = input_directory / case / "parameters.json"
            with open(json_path, "r", encoding="utf-8") as f:
                parameters = json.load(f)
            data_dict[case] = {
                "wall_shear_all": wall_shear_all,
                "diameter": parameters.get("ascending_aorta_diameter_numerical"),
                "SV": parameters.get("stroke_volume"),
                "cfd_max_mesh_size": parameters.get("cfd_max_mesh_size"),
            }
        except Exception as e:
            print(f"Error processing case {case}: {e}")
    examination(data_dict, output_directory, time_step_range=DEFAULT_TIME_STEP_RANGE)
    threshold_calculator_examination(
        data_dict, time_step_range=DEFAULT_TIME_STEP_RANGE
    )
    wss_histogram_linear_log_plot(
        data_dict=data_dict,
        filename="WSS_histogram",
        output_directory=output_directory,
        time_step_range=DEFAULT_TIME_STEP_RANGE,
    )


if __name__ == "__main__":
    main()
