"""Recreate this study's figures from the adjacent article's archived JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_speed_response(report, lookup, output):
    for entry in report["speed_response_evaluations"]:
        label = entry["run"].split("microduck-speed-response-")[1].split("-v")[0]
        summary = entry["summary"]
        speed = summary["conditions"]["velocity_body_frame"][0]
        lookup[label, speed] = summary
    speeds = [0.0, 0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), sharex=True, sharey=True)
    for seed, axis in enumerate(axes):
        for recipe, name, color in [
            ("baseline", "Default", "#2471a3"),
            ("candidate", "Tracking x2", "#d9791f"),
        ]:
            label = f"{recipe}-s{seed}"
            lookup[label, 0.0] = lookup[label, "stand"]
            lookup[label, 0.2] = lookup[label, "forward"]
            metrics = [
                lookup[label, speed]["metrics"]["vx_mean_actual"] for speed in speeds
            ]
            axis.errorbar(
                speeds,
                [metric["mean"] for metric in metrics],
                yerr=[metric["sample_std"] for metric in metrics],
                color=color,
                marker="o",
                capsize=4,
                label=name,
            )
        axis.plot(speeds, speeds, "--", color="#777777", label="Ideal tracking")
        axis.set_title(f"Post-training seed {seed}", loc="left")
        axis.set_xlabel("Forward command (m/s)")
        axis.set_xticks(speeds)
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Actual mean forward velocity (m/s)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Exploratory speed response: same four post-trained policies", fontsize=14
    )
    fig.text(
        0.08,
        0.035,
        "0.0 and 0.2 m/s: primary evaluations; 0.1 and 0.3 m/s: follow-up diagnostic.\n"
        "3 evaluation seeds x 32 envs x 10 seconds; no pushes. Error bars: sample standard deviation.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.12)
    for suffix in (".png", ".svg"):
        fig.savefig(
            output.with_name(f"{output.name}-speed").with_suffix(suffix), dpi=180
        )
    plt.close(fig)


def main():
    output = Path(__file__).resolve().with_suffix("")
    report = json.loads((output.parent.parent / f"{output.name}.json").read_text())
    lookup = {
        (entry["model"], entry["command"]): entry["summary"]
        for entry in report["evaluations"]
    }
    labels = ["initial", "baseline-s0", "candidate-s0", "baseline-s1", "candidate-s1"]
    names = [
        "Initial",
        "Default\nseed 0",
        "Tracking x2\nseed 0",
        "Default\nseed 1",
        "Tracking x2\nseed 1",
    ]
    colors = ["#777777", "#2471a3", "#d9791f", "#2471a3", "#d9791f"]
    panels = [
        ("forward", "vx_mean_actual", "Forward: actual mean vx", "m/s", 0.2),
        (
            "forward",
            "planar_velocity_rmse_m_s",
            "Forward: planar velocity RMSE",
            "m/s",
            None,
        ),
        ("forward", "wz_rmse", "Forward: yaw RMSE", "rad/s", None),
        (
            "stand",
            "planar_velocity_rmse_m_s",
            "Stand: planar velocity RMSE",
            "m/s",
            None,
        ),
        ("left", "wz_mean_actual", "Left turn: actual mean yaw velocity", "rad/s", 0.5),
        (
            "right",
            "wz_mean_actual",
            "Right turn: actual mean yaw velocity",
            "rad/s",
            -0.5,
        ),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for axis, (command, metric, title, unit, target) in zip(
        axes.flat, panels, strict=True
    ):
        for x, label in enumerate(labels):
            summary = lookup[label, command]
            values = [seed[metric] for seed in summary["per_seed"]]
            mean = summary["metrics"][metric]["mean"]
            std = summary["metrics"][metric]["sample_std"]
            axis.bar(
                x, mean, yerr=std, color=colors[x], width=0.65, alpha=0.78, capsize=4
            )
            axis.scatter(
                np.linspace(x - 0.12, x + 0.12, len(values)),
                values,
                s=17,
                color="#222222",
                zorder=3,
            )
        if target is not None:
            axis.axhline(
                target,
                color="#8b1f34",
                linestyle="--",
                linewidth=1.2,
            )
            axis.annotate(
                f"Command {target:g}",
                xy=(0.98, target),
                xycoords=("axes fraction", "data"),
                xytext=(0, -8 if target > 0 else 8),
                textcoords="offset points",
                ha="right",
                va="top" if target > 0 else "bottom",
                color="#8b1f34",
                fontsize=9,
            )
        axis.set_xticks(range(len(labels)), names)
        axis.set_xlim(-0.6, len(labels) - 0.4)
        axis.set_title(title, loc="left", pad=12)
        axis.set_ylabel(unit)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if "rmse" in metric:
            axis.set_ylim(bottom=0)
    fig.suptitle(
        "Microduck: paired post-training from one shared initial policy",
        x=0.065,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.065,
        0.925,
        "2,000 additional PPO updates per model | Same default evaluation environment | 3 evaluation seeds x 32 envs x 10 seconds per command",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.02,
        "Dots: evaluation-seed means. Error bars: sample standard deviation across those 3 seeds (not confidence intervals).\nPost-training seeds 0 and 1 share the same initial checkpoint; these are not independent pretraining replicates.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(
        top=0.85, bottom=0.14, left=0.065, right=0.98, hspace=0.4, wspace=0.3
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    plot_speed_response(report, lookup, output)


if __name__ == "__main__":
    main()
