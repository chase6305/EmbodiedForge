"""Render figures from the published study JSON; no SDK or model files needed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "ef-humanoid-integration-20260928"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def render(data, prefix):
    models = ["incumbent-14000", "transfer-v2"]
    names = ["Incumbent 14000", "Transfer v2"]
    colors = ["#526c8a", "#42a5a0", "#df9b42"]
    formats = ["fp32", "fp16", "int8"]
    sizes = {}
    for model, row in zip(
        ["incumbent-14000", "transfer-v2"], data["models"], strict=True
    ):
        sizes[model, "fp32"] = row["bytes"] / 2**20
    for row in data["storage_models"]:
        sizes[row["model"], row["format"]] = row["bytes"] / 2**20
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    x = np.arange(2)
    for i, fmt in enumerate(formats):
        values = [sizes[m, fmt] for m in models]
        bars = axes[0, 0].bar(
            x + (i - 1) * 0.24, values, 0.23, color=colors[i], label=fmt.upper()
        )
        axes[0, 0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0, 0].set(
        xticks=x,
        xticklabels=names,
        ylabel="Serialized ONNX (MiB)",
        ylim=(0, 65),
        title="A. Weight storage; computation stays FP32",
    )
    axes[0, 0].legend(frameon=False, ncol=3)
    for i, model in enumerate(models):
        medians, low, high = [], [], []
        for threads in [1, 2, 4]:
            values = [
                r["p50_us"] / 1000
                for r in data["cpu_benchmarks"]
                if r["threads"] == threads and model in r["model"]
            ]
            med = float(np.median(values))
            medians.append(med)
            low.append(med - min(values))
            high.append(max(values) - med)
        axes[0, 1].errorbar(
            [1, 2, 4],
            medians,
            yerr=[low, high],
            fmt="o-",
            color=colors[i],
            label=names[i],
            capsize=4,
        )
    axes[0, 1].set(
        xticks=[1, 2, 4],
        xlabel="ORT CPU threads",
        ylabel="Batch-1 p50 inference (ms)",
        title="B. Real observations; six independent processes",
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].text(
        0.02,
        0.02,
        "Median of process p50; whiskers show min/max.\nShared CPU, no affinity pinning; 64 warmups excluded.",
        transform=axes[0, 1].transAxes,
        fontsize=8,
    )
    for i, fmt in enumerate(["fp16", "int8"]):
        values = [
            next(
                r["fraction_states_any_fsq_code_change"] * 100
                for r in data["storage_offline"]
                if r["model"] == m and r["storage"] == fmt
            )
            for m in models
        ]
        bars = axes[1, 0].bar(
            x + (i - 0.5) * 0.3, values, 0.29, color=colors[i + 1], label=fmt.upper()
        )
        axes[1, 0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1, 0].set(
        xticks=x,
        xticklabels=names,
        ylabel="States with changed FSQ code (%)",
        ylim=(0, 90),
        title="C. 2,145 captured states per actor",
    )
    axes[1, 0].legend(frameon=False)
    presets = ["bigrun", "raw_frozen", "raw_free"]
    for i, preset in enumerate(presets):
        values = []
        labels = []
        for model in models:
            rows = [
                r
                for r in data["control_episodes"]
                if r["model"] == model and r["preset"] == preset
            ]
            values.append(float(np.mean([r["joint_mae_rad"] for r in rows])))
            labels.append(f"{sum(r['status'] == 'success' for r in rows)}/{len(rows)}")
        bars = axes[1, 1].bar(
            x + (i - 1) * 0.24,
            values,
            0.23,
            color=colors[i],
            label=preset.replace("_", " "),
        )
        axes[1, 1].bar_label(bars, labels=labels, padding=3, fontsize=9)
    axes[1, 1].set(
        xticks=x,
        xticklabels=names,
        ylabel="Mean of episode joint MAE (rad)",
        ylim=(0, 0.31),
        title="D. Control presets; labels show completed clips",
    )
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    axes[1, 1].text(
        0.02,
        0.86,
        "16 cases per bar; one early fall shortens its error window.",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8,
    )
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("SONIC X2: local integration and storage study", fontsize=16)
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    fig.savefig(prefix.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)
    if len(data["perturbed_episodes"]) == 864:
        render_pushes(
            data,
            prefix.with_name(prefix.name + "-pushes"),
            "perturbed_episodes",
            [20, 60, 100],
            [0, 100, 200],
            "864 disturbed episodes + 72 reused zero-force baselines",
        )
    if len(data.get("stress_episodes", [])) == 192:
        render_pushes(
            data,
            prefix.with_name(prefix.name + "-stress"),
            "stress_episodes",
            [200, 400],
            [0],
            "Exploratory: 192 stronger-push episodes + 24 reused baselines",
        )


def render_pushes(data, prefix, field, forces, init_frames, title):
    models = ["incumbent-14000", "transfer-v2"]
    names = ["Incumbent 14000", "Transfer v2"]
    formats = ["fp32", "fp16", "int8"]
    cases_per_cell = 4 * len(init_frames)
    directions = ["x+", "x-", "y+", "y-"]
    labels = ["0 N"] + [
        f"{force} N\n{direction}" for force in forces for direction in directions
    ]
    columns = len(labels)
    matrix = np.zeros((6, columns))
    counts = np.zeros((6, columns), dtype=int)
    row_names = []
    for mi, model in enumerate(models):
        for si, storage in enumerate(formats):
            index = mi * 3 + si
            row_names.append(f"{names[mi]} / {storage.upper()}")
            base = (
                data["basic_episodes"]
                if storage == "fp32"
                else data["storage_episodes"]
            )
            rows = [
                r
                for r in base
                if r["model"] == model
                and r["init_frame"] in init_frames
                and (
                    r.get("threads") == 2
                    if storage == "fp32"
                    else r["storage"] == storage
                )
            ]
            assert len(rows) == cases_per_cell
            groups = [rows] + [
                [
                    r
                    for r in data[field]
                    if r["model"] == model
                    and r["storage"] == storage
                    and r["force_magnitude_N"] == force
                    and r["direction"] == direction
                ]
                for force in forces
                for direction in directions
            ]
            for col, rows in enumerate(groups):
                assert len(rows) == cases_per_cell
                success = sum(r["status"] == "success" for r in rows)
                matrix[index, col] = success / cases_per_cell
                counts[index, col] = success
    fig, ax = plt.subplots(figsize=(14, 5.3), layout="constrained")
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for (r, c), n in np.ndenumerate(counts):
        ax.text(
            c,
            r,
            f"{n}/{cases_per_cell}",
            ha="center",
            va="center",
            color="white" if matrix[r, c] > 0.65 else "#17212b",
            fontsize=9,
        )
    ax.set(
        xticks=np.arange(columns),
        xticklabels=labels,
        yticks=np.arange(6),
        yticklabels=row_names,
        title="Pelvis push at 1.0 s for 0.2 s; world-frame force; model-specific controls",
    )
    for pos in [0.5 + 4 * i for i in range(len(forces))]:
        ax.axvline(pos, color="white", lw=2)
    ax.axhline(2.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, label="Full-clip completion fraction")
    fig.suptitle(title, fontsize=15)
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    fig.savefig(prefix.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "humanoid-integration-study-20260928.json",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.output)
