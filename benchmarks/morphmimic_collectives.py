"""Numerical GPU check of the single-device, three-robot Gloo transport."""

import argparse
import json
import socket
import sys
from pathlib import Path


def rank_check(rank, rendezvous, workspace, output):
    sys.path[:0] = [
        str(workspace / "src"),
        str(workspace / ".cache/external/morphmimic-source/src"),
    ]
    import torch
    import torch.distributed as dist
    from morphmimic.train.distributed import latent_alignment_loss

    from embodiedforge._morphmimic_worker import cpu_collectives

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3)
    cpu_collectives()
    try:
        for operation, expected in (
            (dist.ReduceOp.SUM, 6),
            (dist.ReduceOp.MIN, 1),
            (dist.ReduceOp.MAX, 3),
        ):
            value = torch.tensor([rank + 1.0], device="cuda")
            dist.all_reduce(value, op=operation)
            assert value.item() == expected
        value = torch.tensor([rank + 1.0], device="cuda")
        dist.broadcast(value, src=2)
        assert value.item() == 3
        values = [torch.zeros(1, device="cuda") for _ in range(3)]
        dist.all_gather(values, torch.tensor([rank + 1.0], device="cuda"))
        assert [v.item() for v in values] == [1, 2, 3]
        z = torch.full((2, 4), rank + 1.0, device="cuda", requires_grad=True)
        loss, distance = latent_alignment_loss(z)
        loss.backward()
        assert loss.item() == 4 * (rank - 1) ** 2
        torch.testing.assert_close(z.grad, torch.full_like(z, rank - 1.0))
        torch.testing.assert_close(distance, torch.tensor(8 / 3, device="cuda"))
        (output / f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    ok=True,
                    latent_loss=loss.item(),
                    distance=distance.item(),
                )
            )
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        rank_check,
        args=(
            f"tcp://127.0.0.1:{port}",
            args.workspace.resolve(),
            args.output.resolve(),
        ),
        nprocs=3,
    )
    print(
        "Three-rank CUDA reductions, broadcast, gather and latent-alignment gradients passed."
    )


if __name__ == "__main__":
    main()
