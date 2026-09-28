"""Recreate the matched SAC restart study figure from its archived JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(__file__).resolve().with_suffix("")
    report = json.loads((output.parent.parent / f"{output.name}.json").read_text())
    if not report["complete"]:
        raise ValueError("The matched continuation/restart protocol is incomplete")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = {"continuous": "#2471a3", "resume": "#d9791f"}
    for column, seed in enumerate(report["protocol"]["training_seeds"]):
        for row, metric in enumerate(["success_once", "success_at_end"]):
            axis = axes[row, column]
            for mode in ["continuous", "resume"]:
                points = sorted(
                    [
                        entry
                        for entry in report["aggregates"]
                        if entry["training_seed"] == seed and entry["mode"] == mode
                    ],
                    key=lambda entry: entry["step"],
                )
                if mode == "resume":
                    start = next(
                        entry
                        for entry in report["aggregates"]
                        if entry["training_seed"] == seed
                        and entry["mode"] == "continuous"
                        and entry["step"] == 1000
                    )
                    points = [start, *points]
                x = [entry["step"] for entry in points]
                y = [100 * entry[metric]["rate"] for entry in points]
                errors = [
                    100 * entry[metric]["sample_std_across_eval_seeds"]
                    for entry in points
                ]
                axis.errorbar(
                    x,
                    y,
                    yerr=errors,
                    color=colors[mode],
                    marker="o" if mode == "continuous" else "s",
                    linestyle="-" if mode == "continuous" else "--",
                    capsize=4,
                    label="Continuous"
                    if mode == "continuous"
                    else "Restart from step 1000",
                )
                for point in points:
                    offset = -28 if mode == "continuous" else 28
                    axis.scatter(
                        [point["step"] + offset] * 3,
                        [100 * item[metric] for item in point["per_eval_seed"]],
                        s=11,
                        color=colors[mode],
                        alpha=0.6,
                    )
            axis.set(
                title=f"Training seed {seed}: {'success once' if metric == 'success_once' else 'success at end'}",
                xlabel="Completed training iterations",
                ylabel="Successful trajectories (%)",
                ylim=(-4, 104),
                xticks=[1000, 2000, 3000],
            )
            axis.grid(axis="y", alpha=0.2)
            axis.legend(loc="best", fontsize=9)
    fig.suptitle(
        "PickCube SAC: continuous training versus checkpoint restart", fontsize=15
    )
    fig.text(
        0.5,
        0.017,
        "Each point: 480 trajectories across evaluation seeds 7001/7002/7003. Error bars: sample SD across those seeds.\nThe step-1000 model is shared within each training seed. Two training seeds; no significance claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for extension in ["png", "svg"]:
        fig.savefig(output.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
