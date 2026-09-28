"""CPU export of the pinned PickCube SAC policy's deterministic actor."""

import hashlib
import io
import json

from .rlinf import REVISION


def export_actor(
    checkpoint, output, model_cfg, expected_sha256, *, weight_storage="float32"
):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from rlinf.models.embodiment.mlp_policy import get_model

    if weight_storage not in ("float32", "float16", "int8"):
        raise ValueError("RLinf matrix weight storage must be float32, float16 or int8")
    payload = checkpoint.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("RLinf weights changed before actor export")
    with torch.device("meta"):
        model = get_model(model_cfg)
    model.load_state_dict(
        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True),
        strict=True,
        assign=True,
    )
    model.eval()
    # weight_summary has checked the fixed SAC action transform (scale 1, bias 0).
    # Evaluation uses tanh(mean); Q networks and the exploration head are unused.
    actor = torch.nn.Sequential(model.backbone, model.actor_mean, torch.nn.Tanh())
    actor.eval()
    parameters = sum(parameter.numel() for parameter in actor.parameters())
    obs_dim, action_dim = model_cfg.obs_dim, model_cfg.action_dim
    rng = np.random.default_rng(0)
    batches = [
        np.zeros((1, obs_dim), dtype=np.float32),
        np.ones((7, obs_dim), dtype=np.float32),
        rng.normal(size=(32, obs_dim)).astype(np.float32),
    ]
    original_actions = [
        model.predict_action_batch(
            {"states": torch.from_numpy(states)}, mode="eval", calculate_values=False
        )[0]
        .reshape(len(states), action_dim)
        .numpy()
        for states in batches
    ]
    metadata = {
        "task": "PickCube-v1",
        "rlinf_revision": REVISION,
        "checkpoint_sha256": expected_sha256,
        "observation": "ManiSkill obs_mode=state, unchanged 42-element ordering",
        "control_mode": "pd_ee_delta_pos",
        "policy_mode": "eval: tanh(actor_mean), without exploration",
        "action_range": "[-1, 1] normalized controller inputs",
        "deterministic_actor_parameters": str(parameters),
    }
    temporary, target = output / "policy.pending.onnx", output / "policy.onnx"
    try:
        torch.onnx.export(
            actor,
            torch.zeros(1, obs_dim),
            str(temporary),
            input_names=["states"],
            output_names=["actions"],
            dynamic_axes={"states": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        graph = onnx.load(str(temporary))
        if weight_storage != "float32":
            restores = []
            scales = []
            decoded_weights = {}
            for item in graph.graph.initializer:
                array = onnx.numpy_helper.to_array(item)
                if array.ndim != 2:
                    continue
                name = item.name
                stored_name = name + "_" + weight_storage
                if weight_storage == "float16":
                    if np.any(np.abs(array) > np.finfo(np.float16).max):
                        raise ValueError("RLinf weights exceed finite float16 storage")
                    stored = array.astype(np.float16)
                    decoded_weights[name] = stored.astype(np.float32)
                    restores.append(
                        onnx.helper.make_node(
                            "Cast",
                            [stored_name],
                            [name],
                            name=name + "_restore",
                            to=onnx.TensorProto.FLOAT,
                        )
                    )
                else:
                    scale = np.maximum(
                        np.abs(array).max(axis=1, keepdims=True) / 127,
                        np.finfo(np.float32).tiny,
                    )
                    stored = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
                    decoded_weights[name] = stored.astype(np.float32) * scale
                    scale_name, cast_name = name + "_scale", name + "_float32"
                    scales.append(onnx.numpy_helper.from_array(scale, scale_name))
                    # Constant Cast+Mul can fold at session initialization. A
                    # DequantizeLinear node remains a per-call cost in ORT 1.24.
                    restores.extend(
                        [
                            onnx.helper.make_node(
                                "Cast",
                                [stored_name],
                                [cast_name],
                                name=name + "_cast",
                                to=onnx.TensorProto.FLOAT,
                            ),
                            onnx.helper.make_node(
                                "Mul",
                                [cast_name, scale_name],
                                [name],
                                name=name + "_restore",
                            ),
                        ]
                    )
                item.CopyFrom(onnx.numpy_helper.from_array(stored, stored_name))
            if len(decoded_weights) != 4:
                raise ValueError("Expected four RLinf actor weight matrices")
            graph.graph.initializer.extend(scales)
            nodes = list(graph.graph.node)
            del graph.graph.node[:]
            graph.graph.node.extend(restores + nodes)
            # Compare against the same decoded weights in the upstream policy;
            # retain the original FP32 actions separately to measure compression.
            with torch.no_grad():
                for name, parameter in actor.named_parameters():
                    if name in decoded_weights:
                        parameter.copy_(torch.from_numpy(decoded_weights[name]))
            metadata.update(weight_storage=weight_storage, computation="float32")
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(temporary), options, providers=["CPUExecutionProvider"]
        )
        maximum_error = 0.0
        original_error = 0.0
        for states, original in zip(batches, original_actions, strict=True):
            expected, _ = model.predict_action_batch(
                {"states": torch.from_numpy(states)},
                mode="eval",
                calculate_values=False,
            )
            expected = expected.reshape(len(states), action_dim).numpy()
            actual = session.run(["actions"], {"states": states})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid exported RLinf actor output")
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.abs(actual - expected).max()))
            original_error = max(original_error, float(np.abs(actual - original).max()))
        raw = temporary.read_bytes()
        report = {
            "path": str(target),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "file_bytes": len(raw),
            "parameters": parameters,
            "weight_storage": weight_storage,
            "input": {
                "name": "states",
                "shape": ["batch", obs_dim],
                "dtype": "float32",
            },
            "output": {
                "name": "actions",
                "shape": ["batch", action_dim],
                "dtype": "float32",
            },
            "metadata": metadata,
            "validation": {
                "passed": True,
                "reference": "upstream predict_action_batch(mode=eval)",
                "reference_weight_storage": weight_storage,
                "provider": "CPUExecutionProvider",
                "batch_sizes": [len(batch) for batch in batches],
                "samples": sum(len(batch) for batch in batches),
                "atol": 1e-5,
                "rtol": 1e-5,
                "max_absolute_error": maximum_error,
                "original_fp32_max_absolute_error": original_error,
            },
        }
        # Publish only a model that passed structural and upstream-action checks.
        json.dumps(report, allow_nan=False)
        temporary.replace(target)
        return report
    finally:
        temporary.unlink(missing_ok=True)
