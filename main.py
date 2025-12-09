import os
import numpy as np
from paraview.simple import XMLPolyDataReader, servermanager
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from vtkmodules.util.numpy_support import vtk_to_numpy
import json


def numpy_from_pvd_paraview(case, INPUT_DIRECTORY):

    pvd_file = os.path.join(
        INPUT_DIRECTORY,
        case,
        "surface",
        "artery_opened_ma",
        "artery_opened_ma.pvd",
    )
    if not os.path.exists(pvd_file):
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
    base_dir = os.path.dirname(pvd_file)
    first_shape = None

    for ds in datasets:
        file_rel = ds.get("file")
        if not file_rel:
            continue
        file_path = os.path.join(base_dir, file_rel)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue

        # Read VTP file using ParaView
        reader = XMLPolyDataReader(FileName=[file_path])
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


def examination(data_dict, OUTPUT_DIRECTORY, time_step_range):
    bar_data = []
    LOW_WSS_THRESHOLD = 0.070477
    HIGH_WSS_THRESHOLD = 7.867869
    for case in data_dict:

        bar_data = []
        start_time, end_time = time_step_range
    for case in data_dict:
        wall_shear = data_dict[case]["wall_shear_all"][start_time:end_time]
        low_wss_count = np.sum(wall_shear < LOW_WSS_THRESHOLD)
        intermediate_wss_count = np.sum(
            (wall_shear >= LOW_WSS_THRESHOLD) & (wall_shear <= HIGH_WSS_THRESHOLD)
        )
        high_wss_count = np.sum(wall_shear > HIGH_WSS_THRESHOLD)
        total_count = wall_shear.size

        bar_data.append(
            {
                "case": case,
                "diameter": data_dict[case].get("diameter", "unknown"),
                "stroke_volume": data_dict[case].get("SV", "unknown"),
                "cfd_max_mesh_size": data_dict[case].get(
                    "cfd_max_mesh_size", "unknown"
                ),
                "low": low_wss_count / total_count * 100,
                "intermediate": intermediate_wss_count / total_count * 100,
                "high": high_wss_count / total_count * 100,
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

    fig_path = os.path.join(OUTPUT_DIRECTORY, "wss_distribution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def threshold_calculator_examination(data_dict, time_step_range):

    for case in data_dict:
        print(f"case: {case}")
        wall_shear = data_dict[case]["wall_shear_all"]
        start_time_step, end_time_step = time_step_range
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
    LOW_WSS_THRESHOLD = 0.070477
    HIGH_WSS_THRESHOLD = 7.867869

    frt_summary = []  # for later grouped bar charts if needed

    start_time_step, end_time_step = time_step_range

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

        # ---- FRT by counts (your second snippet logic) ----
        low_count = np.sum(wss_flat < LOW_WSS_THRESHOLD)
        mid_count = np.sum(
            (wss_flat >= LOW_WSS_THRESHOLD) & (wss_flat <= HIGH_WSS_THRESHOLD)
        )
        high_count = np.sum(wss_flat > HIGH_WSS_THRESHOLD)
        total_count = wss_flat.size

        # percentages
        low_pct = (low_count / total_count) * 100.0
        mid_pct = (mid_count / total_count) * 100.0
        high_pct = (high_count / total_count) * 100.0

        # keep for later bar charts
        frt_summary.append(
            {
                "case": case,
                "diameter": data_dict[case].get("diameter", "unknown"),
                "stroke_volume": data_dict[case].get("SV", "unknown"),
                "low": low_pct,
                "intermediate": mid_pct,
                "high": high_pct,
            }
        )

        # ---- Prepare histogram splits (linear) ----
        low_data = wss_flat[wss_flat <= LOW_WSS_THRESHOLD]
        mid_data = wss_flat[
            (wss_flat > LOW_WSS_THRESHOLD) & (wss_flat < HIGH_WSS_THRESHOLD)
        ]
        high_data = wss_flat[wss_flat >= HIGH_WSS_THRESHOLD]

        # To keep x-axes consistent across cases for comparability:
        x_min_linear = 0.0
        x_max_linear = 10.0  # you already set xlim(0,10)
        bin_width = 0.1
        custom_bins = np.arange(x_min_linear, x_max_linear + bin_width, bin_width)

        # weights to plot percentages directly
        if total_count == 0:
            print(f"[WARN] total_count==0 for {case}; skipping.")
            continue
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
            f"Low: {low_pct:.2f}%",
            transform=inset_ax.transAxes,
            fontsize=10,
            ha="left",
        )
        inset_ax.text(
            0.60,
            0.90,
            f"Intermediate: {mid_pct:.2f}%",
            transform=inset_ax.transAxes,
            fontsize=10,
            ha="center",
        )
        inset_ax.text(
            1.0,
            0.90,
            f"High: {high_pct:.2f}%",
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

    CUR_DIR = os.path.curdir
    INPUT_DIRECTORY = os.path.join(CUR_DIR, "data")
    OUTPUT_DIRECTORY = os.path.join(CUR_DIR, "figures")

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    cases = [
        d
        for d in os.listdir(INPUT_DIRECTORY)
        if os.path.isdir(os.path.join(INPUT_DIRECTORY, d))
    ]

    data_dict = {}
    for case in cases:
        try:
            wall_shear_all = numpy_from_pvd_paraview(case, INPUT_DIRECTORY)
            json_path = os.path.join(INPUT_DIRECTORY, case, "parameters.json")
            with open(json_path, "r") as f:
                parameters = json.load(f)
            data_dict[case] = {
                "wall_shear_all": wall_shear_all,
                "diameter": parameters.get("ascending_aorta_diameter_numerical"),
                "SV": parameters.get("stroke_volume"),
                "cfd_max_mesh_size": parameters.get("cfd_max_mesh_size"),
            }
        except Exception as e:
            print(f"Error processing case {case}: {e}")
    examination(data_dict, OUTPUT_DIRECTORY, time_step_range=(20, 39))
    threshold_calculator_examination(data_dict, time_step_range=(20, 39))
    wss_histogram_linear_log_plot(
        data_dict=data_dict,
        filename="WSS_histogram",
        output_directory=OUTPUT_DIRECTORY,
        time_step_range=(20, 39),
    )


if __name__ == "__main__":
    main()
