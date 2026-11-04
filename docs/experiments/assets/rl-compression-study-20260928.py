"""Render the compression study from its archived JSON; no training dependencies."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "rl-compression-study-20260928"

NAMES = {
    "fp32": "FP32",
    "fp16_weights": "FP16 storage",
    "int8_weights": "INT8 storage (DQ)",
    "dynamic_all": "Dynamic: all",
    "dynamic_keep_input": "Dynamic: FP32 input layer",
    "dynamic_keep_output": "Dynamic: FP32 output layer",
    "dynamic_inner": "Dynamic: inner layers",
    "qat_dynamic": "Dynamic + QAT",
}


def render(source, output):
    data = json.loads(source.read_text())
    assert data["complete"], "Only render the completed study"
    primary = data["primary"]
    variants = primary["protocol"]["variants"]
    labels = primary["protocol"]["models"] + data["confirmation"]["protocol"]["models"]
    aggregates = primary["aggregates"] + data["confirmation"]["aggregates"]
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[0, 1]),
    ]
    y = np.arange(len(variants))
    sizes = [
        next(
            r["file_bytes"] for r in primary["benchmark"]["models"] if r["variant"] == v
        )
        / 1024
        for v in variants
    ]
    axes[0].barh(y, sizes, color="#467c9d")
    for j, value in enumerate(sizes):
        axes[0].text(value + 7, j, f"{value:.1f}", va="center", fontsize=9)
    axes[0].set(
        yticks=y,
        yticklabels=[NAMES[v] for v in variants],
        xlabel="ONNX file size (KiB)",
        title="Storage (first original actor)",
        xlim=(0, 650),
    )
    axes[0].invert_yaxis()
    delta = np.array(
        [
            [
                next(
                    r for r in aggregates if r["label"] == label and r["variant"] == v
                )["difference_from_fp32_pp"]["success_at_end"]
                for label in labels
            ]
            for v in variants
        ]
    )
    span = max(1, np.abs(delta).max())
    im = axes[1].imshow(delta, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
    for j in range(len(variants)):
        for k in range(len(labels)):
            axes[1].text(
                k,
                j,
                f"{delta[j, k]:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(delta[j, k]) > span * 0.55 else "black",
                fontsize=9,
            )
    axes[1].set(
        yticks=y,
        yticklabels=[NAMES[v] for v in variants],
        xticks=np.arange(len(labels)),
        xticklabels=[
            "cont.\n1234",
            "resume\n1234",
            "cont.\n4321",
            "resume\n4321",
            "fresh\n2026",
        ],
        title="End success vs own FP32 baseline (pp)\n480 trajectories / actor / variant, batch 16",
    )
    axes[1].axvline(3.5, color="black", linestyle="--", linewidth=1)
    fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.03, label="Percentage points")
    for batch, offset, color in [("1", -0.18, "#467c9d"), ("16", 0.18, "#c5843a")]:
        values = [
            np.mean(
                [
                    r["latency"][batch]["p50_batch_ms"]
                    for r in primary["benchmark"]["models"]
                    if r["variant"] == v
                ]
            )
            * 1000
            for v in variants
        ]
        axes[2].barh(
            y + offset, values, height=0.34, label=f"Batch {batch}", color=color
        )
    axes[2].set(
        yticks=y,
        yticklabels=[],
        xlabel="Microseconds / session.run",
        title="Warmed CPU inference\nMean of four actor medians",
    )
    axes[2].invert_yaxis()
    axes[2].legend()
    axes[0].set_ylim(len(variants) - 0.4, -0.6)
    axes[2].set_ylim(len(variants) - 0.4, -0.6)
    for ax in [axes[0], axes[2]]:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "RLinf PickCube SAC\nStorage, activation quantization and fixed-budget QAT",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        "Shared host; one CPU thread. FP16/INT8 storage use FP32 arithmetic.\nTraining seeds: 1234, 4321, 2026; five actor paths are not five independent seeds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=2, w_pad=2)
    for suffix in ["png", "svg"]:
        fig.savefig(
            output.with_suffix("." + suffix),
            dpi=160,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1.1]}
    )
    for cohort in ["primary", "confirmation"]:
        for row in data[cohort]["qat"]["runs"]:
            points = row["metrics"]
            axes[0].plot(
                [p["update"] for p in points],
                [p["validation_action_mse"] for p in points],
                label=row["label"],
                linestyle="--" if cohort == "confirmation" else "-",
            )
    axes[0].set(
        title="QAT validation action error",
        xlabel="Adam updates",
        ylabel="MSE vs own FP32 teacher",
    )
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(alpha=0.2)
    bv = data["batch_one_protocol"]["variants"]
    delta = np.array(
        [
            [
                next(
                    r
                    for r in data["batch_one"]["aggregates"]
                    if r["label"] == label and r["variant"] == v
                )["success_at_end"]["difference_from_batch16_pp"]
                for label in labels
            ]
            for v in bv
        ]
    )
    span = max(1, np.abs(delta).max())
    im = axes[1].imshow(delta, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
    for j in range(len(bv)):
        for k in range(len(labels)):
            axes[1].text(
                k,
                j,
                f"{delta[j, k]:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(delta[j, k]) > span * 0.55 else "black",
            )
    axes[1].set(
        yticks=np.arange(len(bv)),
        yticklabels=[NAMES[v] for v in bv],
        xticks=np.arange(len(labels)),
        xticklabels=[
            "cont.\n1234",
            "resume\n1234",
            "cont.\n4321",
            "resume\n4321",
            "fresh\n2026",
        ],
        title="End success: batch 1 minus batch 16 (pp)\nSame 16 simulator environments and seeds",
    )
    axes[1].axvline(3.5, color="black", linestyle="--", linewidth=1)
    fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.03)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(
            output.with_name(output.name + "-qat-batch").with_suffix("." + suffix),
            dpi=160,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rl-compression-study-20260928.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().with_suffix("")
    )
    args = parser.parse_args()
    render(args.source, args.output)
