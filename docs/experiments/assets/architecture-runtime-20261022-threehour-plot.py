import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
r = args.root
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), layout="constrained")
red, green, blue = "#b44848", "#23845a", "#356fa8"


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def delta(a):
    return (
        np.array([x["resources"]["rss_bytes"] for x in a])
        - a[0]["resources"]["rss_bytes"]
    ) / 2**20


comparison = json.loads((r / "startup-log-comparison.json").read_text())
ax = axes[0, 0]
x = np.arange(2)
for offset, key, label, color in [
    (-0.19, "startup-log-baseline", "Baseline: failed", red),
    (0.19, "candidate-process-tests", "Fixed: passed", green),
]:
    values = [a["seconds"] * 1000 for a in comparison[key]]
    ax.bar(x + offset, values, width=0.36, label=label, color=color)
    for xx, value in zip(x + offset, values, strict=True):
        ax.text(xx, value * 1.14, f"{value:g}", ha="center", fontsize=8)
ax.set(
    yscale="log",
    ylim=(10, 30000),
    xticks=x,
    xticklabels=["1.06 KiB shutdown log", "2.125 MiB shutdown log"],
    ylabel="Fixture time (ms, log scale)",
    title="Startup cancellation: output integrity",
)
ax.text(-0.19, 55, "log lost", ha="center", color=red, fontsize=8)
ax.text(0.81, 2600, "force-killed", ha="center", color=red, fontsize=8)
ax.legend(loc="upper left", fontsize=8)

ax = axes[0, 1]
for name, label, color, style in [
    ("checkpoint-live", "Baseline (90 min)", red, "-"),
    ("checkpoint-intermediate-stopped", "Intermediate: stopped", "#888888", "--"),
    ("checkpoint-candidate", "Final candidate (80 min)", green, "-"),
]:
    a = rows(r / name / "samples.jsonl")
    ax.plot(
        [v["attempted_publications"] / 1000 for v in a],
        delta(a),
        label=label,
        color=color,
        linestyle=style,
    )
ax.set(
    xlabel="Attempted snapshot publications (thousands)",
    ylabel="RSS change from first sample (MiB)",
    title="Checkpoint publisher: unique name growth",
)
ax.legend(fontsize=8)

ax = axes[0, 2]
for directory, label, color in [
    ("checkpoint-live", "Baseline readers", red),
    ("checkpoint-candidate", "Final readers", green),
]:
    for index in range(4):
        a = rows(r / directory / f"reader-{index}.samples.jsonl")
        ax.plot(
            [v["seconds"] / 60 for v in a],
            delta(a),
            label=label if index == 0 else None,
            color=color,
            alpha=0.7,
            linewidth=1,
        )
ax.set(
    xlabel="Elapsed time (minutes)",
    ylabel="RSS change from first sample (MiB)",
    title="Checkpoint readers: four independent processes",
)
ax.legend(fontsize=8)

ax = axes[1, 0]
measurements = json.loads((r / "path-memory-results.json").read_text())
for item in measurements:
    a = item["result"]["samples"]
    y = (np.array([v["rss_bytes"] for v in a]) - a[0]["rss_bytes"]) / 2**20
    label = f"3.12.{('0' if item['version'] == '3120' else '14')} {item['kind']}"
    ax.plot(
        [v["items"] / 1000 for v in a],
        y,
        label=label,
        linestyle="-" if item["kind"] == "path" else "--",
    )
ax.set(
    xlabel="Unique names (thousands)",
    ylabel="RSS change (MiB)",
    title="Isolated paths: tracemalloc enabled",
)
ax.legend(fontsize=8)

ax = axes[1, 1]
for name, label in [
    ("candidate-sdk-old-gdb-worker", "3.12.0 / GDB / worker ack"),
    ("candidate-sdk-old-direct-parent", "3.12.0 / direct / parent trace"),
    ("candidate-sdk-modern-direct-worker", "3.12.14 / direct / worker ack"),
]:
    a = rows(r / name / "samples.jsonl")
    ax.plot([v["seconds"] / 60 for v in a], delta(a), label=label)
p = r / "final-observer-sdk/samples.jsonl"
if p.exists():
    a = rows(p)
    ax.plot(
        [v["seconds"] / 60 for v in a],
        delta(a),
        label="3.12.0 / final string observer",
        color="#824da6",
        linestyle="--",
        linewidth=2,
    )
ax.set(
    xlabel="Elapsed time (minutes)",
    ylabel="RSS change from first sample (MiB)",
    title="SDK runtime + observer memory",
)
ax.legend(fontsize=8)

ax = axes[1, 2]
for y, mode, _label, color in [
    (1, "force", "Normal 10 s grace", red),
    (0, "late", "Diagnostic extra wait", green),
]:
    d = json.loads(
        (r / f"queued-timed-py3120-{mode}/case/signal-trace.json").read_text()
    )
    events = d["events"]
    start = next(
        e["monotonic_ns"]
        for e in events
        if e["kind"] == "killpg_before" and e["signal"] == 2
    )
    if mode == "force":
        end = next(
            e["monotonic_ns"]
            for e in events
            if e["kind"] == "killpg_before" and e["signal"] == 9
        )
    else:
        end = int(
            next(
                line.split()[0]
                for line in d["worker_acknowledgements"].splitlines()
                if " handler " in line
            )
        )
    seconds = (end - start) / 1e9
    ax.plot([0, seconds], [y, y], color=color, linewidth=3)
    ax.scatter([0], [y], color=blue, s=35)
    ax.scatter(
        [seconds], [y], color=color, s=50, marker="x" if mode == "force" else "o"
    )
    ax.text(seconds + 0.6, y, f"{seconds:.2f} s", va="center", color=color)
    ax.plot([10, 10], [y - 0.1, y + 0.1], color="#444444")
    ax.text(10, y + 0.15, "C wakeup byte observed", ha="center", fontsize=7)
ax.set(
    xlim=(-1, 38),
    ylim=(-0.4, 1.55),
    yticks=[0, 1],
    yticklabels=["Diagnostic extra wait", "Normal 10 s grace"],
    xlabel="Time after forwarded SIGINT (seconds)",
    title="Injected native boundary: delayed Python handler",
)
ax.text(28, 0.35, "Python handler at sleep end", ha="center", fontsize=8, color=green)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.16)
fig.text(
    0.5,
    0.99,
    "EmbodiedForge: startup log capture and checkpoint path validation",
    fontsize=16,
    ha="center",
    va="top",
)
fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.89))
fig.text(
    0.5,
    0.028,
    "Controlled CPU checks; publication counters are snapshot versions, not extra PPO updates. RSS is observed, not a bound. Supplementary status-143 exits are retained separately.",
    ha="center",
    fontsize=8,
    color="#555555",
)
fig.text(
    0.5,
    0.011,
    "The three original SDK observer curves include /proc PID Path interning on old Python; the final observer uses string paths.",
    ha="center",
    fontsize=8,
    color="#555555",
)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output.with_suffix(".svg"))
fig.savefig(args.output.with_suffix(".png"), dpi=145)
