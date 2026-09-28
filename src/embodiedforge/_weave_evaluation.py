"""First-episode clip statistics, sampled before IsaacLab's automatic reset."""

import numpy as np


class ClipEvaluation:
    def __init__(self, names, lengths):
        self.names = names
        self.lengths = np.asarray(lengths)
        self.status = np.full(len(names), "incomplete", dtype="U10")
        self.samples = np.zeros(len(names), dtype=np.int64)
        self.frames = np.zeros(len(names), dtype=np.int64)
        self.sums = {}
        self.causes = [[] for _ in names]

    @property
    def complete(self):
        return bool(np.all(self.status != "incomplete"))

    def update(self, metrics, steps, terminated, time_outs, clip_end, terms):
        active = self.status == "incomplete"
        values = {key: np.asarray(value) for key, value in metrics.items()}
        for key, value in values.items():
            if value.shape != active.shape or not np.isfinite(value[active]).all():
                raise ValueError(f"Invalid evaluation metric: {key}")
        frames = np.where(active, np.asarray(steps) + 1, self.frames)
        failed = active & np.asarray(terminated)
        success = active & ~failed & np.asarray(clip_end)
        if np.any(success & (frames < self.lengths)):
            raise ValueError("clip_end before the last reference frame")
        # Publish a whole valid step, keeping a usable report if collection fails.
        for key, value in values.items():
            self.sums.setdefault(key, np.zeros(len(self.names)))[active] += value[
                active
            ]
        self.samples[active] += 1
        self.frames = frames
        self.status[failed] = "failed"
        self.status[success] = "success"
        self.status[active & ~failed & ~success & np.asarray(time_outs)] = "truncated"
        finished = active & (self.status != "incomplete")
        for i in np.flatnonzero(finished):
            self.causes[i] = [key for key, flags in terms.items() if flags[i]]

    def report(self):
        clips = []
        for i, name in enumerate(self.names):
            clips.append(
                {
                    "name": name,
                    "status": str(self.status[i]),
                    "samples": int(self.samples[i]),
                    "termination_terms": self.causes[i],
                    "completion": min(float(self.frames[i] / self.lengths[i]), 1.0),
                    "metrics": {
                        key: float(values[i] / self.samples[i])
                        if self.samples[i]
                        else None
                        for key, values in self.sums.items()
                    },
                }
            )

        def average(rows):
            return {
                key: float(np.mean([row["metrics"][key] for row in rows]))
                if rows
                else None
                for key in self.sums
            }

        return {
            "status": "complete" if self.complete else "incomplete",
            "success_rate": float(np.mean(self.status == "success")),
            "metric_definition": "Per-step Weave errors averaged per clip, then equally across clips; not RMSE",
            "metrics_all": average([row for row in clips if row["samples"]]),
            "metrics_success": average(
                [row for row in clips if row["status"] == "success"]
            ),
            "clips": clips,
        }
