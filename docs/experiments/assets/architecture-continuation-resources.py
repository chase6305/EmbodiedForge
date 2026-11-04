"""Reproduce resource charts from the observed architecture validation report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = ("#b14b3b", "#237a56")
    for label, samples, color in zip(
        ("Before", "After"),
        report["memory"]["recording_trace_samples"].values(),
        colors,
        strict=True,
    ):
        initial = samples[0]["python_traced_bytes"]
        axes[0].plot(
            [row["segment"] for row in samples],
            [(row["python_traced_bytes"] - initial) / 2**20 for row in samples],
            label=label,
            color=color,
        )
    axes[0].set(
        title="Recording-only Python live allocations (CPython 3.12.0)",
        xlabel="Completed segments",
        ylabel="Live allocation change from first segment (MiB)",
    )
    for case in report["native_recording"]:
        resources = case["resources"]
        warm = resources["warm_rss_bytes"]
        curve = [
            row
            for row in resources["resource_curve_every_five_minutes"]
            if row["seconds"] >= 120
        ]
        axes[1].plot(
            [row["seconds"] / 60 for row in curve],
            [(row["rss_bytes"] - warm) / 2**20 for row in curve],
            label=case["task"].upper() + " (memory-fix snapshot)",
        )
    axes[1].set(
        title="Native paired recording RSS after warm-up",
        xlabel="Elapsed minutes per process",
        ylabel="RSS change from 2–3 minute median (MiB)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
