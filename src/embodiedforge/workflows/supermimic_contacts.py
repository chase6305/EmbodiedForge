"""Read solved MuJoCo-Warp contact forces after each real physics substep.

Adapted from liudi159/supermimic-backflip-5090 scripts/contact_recorder.py,
revision5b313d75e804cdd2ff1ce9bdcfa899a98eca856b (MIT). See the adjacent
LICENSE.supermimic-backflip. CPU forward/inverse dynamics never creates forces.
"""

import json
from pathlib import Path

from ..runtime.hashing import file_sha256


def validate_substep_clock(times, dt):
    """Reject duplicate/missing recorder calls using the actual simulator clock."""
    import numpy as np

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Expected finite actual physics timestamps")
    deltas = np.diff(times)
    # The device clock accumulates float32 values. Allow its measured rounding,
    # while a duplicate or skipped5msstep remains far outside this tolerance.
    if not np.allclose(deltas, dt, rtol=1e-4, atol=1e-6):
        raise ValueError("Actual physics clock contains duplicate or missing substeps")
    return {
        "actual_clock_verified": True,
        "actual_first_time_s": float(times[0]),
        "actual_last_time_s": float(times[-1]),
        "actual_max_step_error_s": float(np.max(np.abs(deltas - dt)))
        if len(deltas)
        else 0.0,
    }


class ContactRecorder:
    def __init__(self, env):
        import mujoco
        import mujoco_warp as mjw
        import numpy as np

        self.env = getattr(env, "raw", env)
        self.sim = self.env.sim
        if self.sim.wp_data.nworld != 1:
            raise ValueError("Contact audit requires exactly one physical world")
        self.model = self.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.env.physics_dt)
        if abs(self.dt - 0.005) > 1e-10:
            raise ValueError("Contact audit requires200Hzphysics")
        self.floor = set(
            np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
        )
        self.feet = []
        for side in ("left", "right"):
            matches = [
                i
                for i in range(self.model.nbody)
                if self.model.body(i).name.endswith(side + "_ankle_roll_link")
            ]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify one {side}foot body")
            self.feet.append(
                set(np.flatnonzero(self.model.geom_bodyid == matches[0]).tolist())
            )
        if not self.floor or not all(self.feet):
            raise ValueError("Missing ground/foot geometry in contact audit")
        self.iteration_flags = int(
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        self.samples, self.debug_contacts, self.nonfoot_debug = [], [], []
        self.original = self.sim.step
        self.original_graph_setting = self.sim.use_cuda_graph
        self.sim.use_cuda_graph = False

        def record_step():
            result = self.original()
            self.record()
            return result

        self.sim.step = record_step

    def record(self):
        import mujoco
        import mujoco_warp as mjw
        import numpy as np
        import warp as wp

        # Copies the actual solved contact constraints before manager forward().
        mjw.get_data_into(self.data, self.model, self.sim.wp_data)
        d, m = self.data, self.model
        forces, counts = np.zeros((2, 3)), np.zeros(2, dtype=int)
        total, nonfoot = np.zeros(3), np.zeros(3)
        bad, geometric_contacts, max_depth = 0, 0, 0.0
        for index in range(d.ncon):
            contact = d.contact[index]
            one, two = map(int, contact.geom)
            if one not in self.floor and two not in self.floor:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, force)
            world = contact.frame.reshape(3, 3).T @ force[:3]
            if two in self.floor:
                world, other = -world, one
            else:
                other = two
            total += world
            side = next((i for i, ids in enumerate(self.feet) if other in ids), None)
            active = np.linalg.norm(world) > 1e-3
            if side is None:
                nonfoot += world
                bad += int(active)
                geometric_contacts += int(contact.dist < 0)
                max_depth = max(max_depth, float(-contact.dist))
                if (active or contact.dist < 0) and len(self.nonfoot_debug) < 12:
                    self.nonfoot_debug.append(
                        {
                            "substep": len(self.samples),
                            "geoms": [m.geom(one).name, m.geom(two).name],
                            "distance_m": float(contact.dist),
                            "world_force_N": world.tolist(),
                        }
                    )
            else:
                forces[side] += world
                counts[side] += int(active)
            if len(self.debug_contacts) < 12 and np.linalg.norm(world) > 1:
                self.debug_contacts.append(
                    {
                        "substep": len(self.samples),
                        "geoms": [m.geom(one).name, m.geom(two).name],
                        "local_force": force.tolist(),
                        "world_force": world.tolist(),
                        "root_constraint": d.qfrc_constraint[:3].tolist(),
                    }
                )
        overflow = int(wp.to_torch(self.sim.wp_data.overflow).cpu().numpy()[0])
        self.samples.append(
            {
                "time_s": (len(self.samples) + 1) * self.dt,
                "actual_data_time_s": float(d.time),
                "foot_forces_N": forces,
                "foot_contact_counts": counts,
                "total_ground_force_N": total,
                "nonfoot_force_N": nonfoot,
                "nonfoot_force_contacts": bad,
                "nonfoot_negative_distance_contacts": geometric_contacts,
                "max_nonfoot_depth_m": max_depth,
                "qpos": d.qpos.copy(),
                "qvel": d.qvel.copy(),
                "root_constraint_N": d.qfrc_constraint[:3].copy(),
                "overflow": overflow,
            }
        )

    def save(self, output):
        import mujoco
        import numpy as np

        output = Path(output)
        if not self.samples:
            raise ValueError("No actual physics substeps recorded")
        arrays = {
            name: np.array([row[name] for row in self.samples])
            for name in self.samples[0]
        }
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite actual contact audit data")
        clock = validate_substep_clock(arrays["actual_data_time_s"], self.dt)
        np.savez_compressed(
            output / "contact-forces-200hz.npz",
            dt=self.dt,
            mass=float(self.model.body_mass.sum()),
            **arrays,
        )
        mujoco.mj_saveModel(self.model, str(output / "physics-model.mjb"))
        difference = np.linalg.norm(
            arrays["total_ground_force_N"] - arrays["root_constraint_N"], axis=1
        )
        scale = np.maximum(np.linalg.norm(arrays["root_constraint_N"], axis=1), 1.0)
        capacity = arrays["overflow"] & ~self.iteration_flags
        audit = {
            "status": "complete",
            "physics_hz": 200,
            "samples": len(self.samples),
            **clock,
            "covered_seconds": len(self.samples) * self.dt,
            "force_source": "Actual GPU-solved constraints copied after each5msstep, before manager forward; no forward/inverse recomputation",
            "cuda_graphs_disabled_for_audit": True,
            "force_balance_max_error_N": float(difference.max()),
            "force_balance_max_relative_error": float((difference / scale).max()),
            "force_balance_verified": bool(np.all(difference <= 0.05 + 1e-4 * scale)),
            "force_balance_absolute_tolerance_N": 0.05,
            "force_balance_relative_tolerance": 1e-4,
            "capacity_overflow_substeps": int(np.count_nonzero(capacity)),
            "solver_iteration_flag_substeps": int(
                np.count_nonzero(arrays["overflow"] & self.iteration_flags)
            ),
            "nonfoot_force_substeps": int(
                np.count_nonzero(arrays["nonfoot_force_contacts"])
            ),
            "nonfoot_negative_distance_substeps": int(
                np.count_nonzero(arrays["nonfoot_negative_distance_contacts"])
            ),
            "max_nonfoot_contact_depth_m": float(arrays["max_nonfoot_depth_m"].max()),
            "max_nonfoot_force_N": float(
                np.linalg.norm(arrays["nonfoot_force_N"], axis=1).max()
            ),
            "max_foot_force_N": float(
                np.linalg.norm(arrays["foot_forces_N"], axis=2).max()
            ),
            "mass_kg": float(self.model.body_mass.sum()),
            "compiled_cpu_geometry_model_sha256": file_sha256(
                output / "physics-model.mjb"
            ),
            "model_scope": "Compiled geometry/collision model for saved-state geometry checks; device-side material randomization is not serialized into this CPU model. Forces are copied from the actual device solve.",
            "debug_contacts": self.debug_contacts,
            "debug_nonfoot_contacts": self.nonfoot_debug,
            "scope": "Read-only solved-force audit of one real rollout. Force balance/zero nonfoot force alone do not establish valid airborne rotation or pose quality.",
        }
        (output / "contact-force-audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        return audit

    def close(self):
        self.sim.step = self.original
        self.sim.use_cuda_graph = self.original_graph_setting
