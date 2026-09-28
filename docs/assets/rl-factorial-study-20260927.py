"""Recreate the reward-factorial figures from the adjacent archived study JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(__file__).resolve().with_suffix("")
    report = json.loads((output.parent.parent / f"{output.name}.json").read_text())
    plot_microduck(report, output)
    plot_rlinf(report, output.with_name(output.name + "-sac"))


def plot_microduck(report, output):
    if not report["complete_microduck_protocol"]:
        raise ValueError("The four-cell, two-seed factorial study is incomplete")
    lookup = {
        (entry["model"], entry["command"]): entry["summary"]
        for entry in report["evaluations"]
    }
    cells = ["default", "linear", "angular", "both"]
    labels = [
        "Default\n(2, 2)",
        "Linear only\n(4, 2)",
        "Angular only\n(2, 4)",
        "Both\n(4, 4)",
    ]
    colors = ["#2471a3", "#d9791f"]
    panels = [
        ("forward", "vx_mean_actual", "Forward: actual mean speed", "m/s", 0.2),
        ("forward", "planar_velocity_rmse_m_s", "Forward: planar RMSE", "m/s", None),
        ("forward", "wz_rmse", "Forward: yaw RMSE", "rad/s", None),
        ("left", "wz_rmse", "Left turn: yaw RMSE", "rad/s", None),
        ("right", "wz_rmse", "Right turn: yaw RMSE", "rad/s", None),
        (
            "push",
            "survived_full_horizon_count",
            "Pushes: first-episode survival",
            "%",
            None,
        ),
    ]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, (command, metric, title, unit, target) in zip(
        axes.flat, panels, strict=True
    ):
        factor = 100 / 32 if metric == "survived_full_horizon_count" else 1
        for group, cell in enumerate(cells):
            for seed in [0, 1]:
                summary = lookup[f"{cell}-s{seed}", command]
                values = summary["metrics"][metric]
                x = group + (seed - 0.5) * 0.34
                axis.bar(
                    x,
                    values["mean"] * factor,
                    yerr=values["sample_std"] * factor,
                    width=0.30,
                    color=colors[seed],
                    alpha=0.8,
                    capsize=3,
                    label=f"Post-training seed {seed}" if group == 0 else None,
                )
                axis.scatter(
                    [x - 0.055, x, x + 0.055],
                    [row[metric] * factor for row in summary["per_seed"]],
                    s=10,
                    color="#222222",
                    zorder=3,
                )
        if target is not None:
            axis.axhline(target, color="#8b1f34", linestyle="--", linewidth=1.2)
            axis.annotate(
                f"Command {target:g}",
                xy=(0.98, target),
                xycoords=("axes fraction", "data"),
                xytext=(0, -5),
                textcoords="offset points",
                ha="right",
                va="top",
                color="#8b1f34",
                fontsize=9,
            )
        axis.set_xticks(range(4), labels)
        axis.set_ylim(bottom=0)
        if metric == "survived_full_horizon_count":
            axis.set_ylim(top=105)
        axis.set_title(title, loc="left", pad=12)
        axis.set_ylabel(unit)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.95),
        frameon=False,
    )
    fig.suptitle(
        "Microduck: separating linear and angular tracking reward weights",
        x=0.06,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.06,
        0.92,
        "Each policy: 2,000 additional PPO updates from one shared initial checkpoint",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.02,
        "Weight pairs: (linear, angular). Default and Both reuse the preceding study; Linear only and Angular only are new.\n"
        "Dots and error bars: 3 evaluation-seed means and sample standard deviation, not confidence intervals.\n"
        "No-push commands: 10 seconds. Push condition: 15 seconds with separate evaluation seeds. No hardware testing.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.84, bottom=0.17, hspace=0.45, wspace=0.3
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_rlinf(report, output):
    if not report["complete_rlinf_protocol"]:
        raise ValueError("The two-seed SAC study is incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for axis, metric, title in zip(
        axes,
        ["success_once", "success_at_end"],
        ["Success at least once", "Success at episode end"],
        strict=True,
    ):
        for prefix, steps, color, label in [
            (
                "reference",
                [1000, 2000, 3000],
                "#2471a3",
                "Seed 1234 (includes checkpoint recovery)",
            ),
            ("new-4321", [1000, 2000], "#d9791f", "Seed 4321 (fresh training)"),
        ]:
            values = [
                report["rlinf_aggregates"][f"{prefix}-{step}"][metric] * 100
                for step in steps
            ]
            axis.plot(steps, values, "o-", color=color, label=label)
            for step in steps:
                for offset, seed in zip([-35, 0, 35], [4001, 4002, 4003], strict=True):
                    row = report["rlinf"][f"{prefix}-{step}-eval-s{seed}"]
                    value = row["result"]["metrics"][f"eval/{metric}"]["value"] * 100
                    axis.scatter(
                        step + offset,
                        value,
                        marker="x",
                        color=color,
                        alpha=0.6,
                        s=24,
                    )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Collection/update iterations")
        axis.set_ylabel("% of completed 50-step trajectories")
        axis.set_xticks([1000, 2000, 3000])
        axis.set_ylim(0, 105)
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.91))
    fig.suptitle("RLinf PickCube SAC: new evaluation seeds and a fresh training seed")
    fig.text(
        0.08,
        0.02,
        "Each checkpoint: evaluation seeds 4001/4002/4003, 160 trajectories each. Circles: aggregate; crosses: seed means.\n"
        "Lines connect evaluated checkpoints only. The 3,000-iteration budget has one training seed.\n"
        "Seed 1234 used checkpoint recovery; seed 4321 did not. This comparison cannot isolate recovery-path effects.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.70, bottom=0.24, wspace=0.3)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
