"""Measure bounded JSON receive with real sockets and preframed sender payloads."""

import argparse
import hashlib
import json
import socket
import statistics
import threading
import time
import tracemalloc
from pathlib import Path

from embodiedforge._live_channel import JsonChannel


def measure(value, packet, count, *, allocation=False):
    """Include receive/JSON decode/equality; exclude input construction and framing."""
    reader, writer = socket.socketpair()
    reader.settimeout(30)
    writer.settimeout(30)
    errors = []

    def send():
        try:
            for _ in range(count):
                writer.sendall(packet)
        except BaseException as error:
            errors.append(error)
        finally:
            writer.close()

    thread = threading.Thread(target=send)
    channel = JsonChannel(reader)
    if allocation:
        tracemalloc.start()
    try:
        thread.start()
        start = time.perf_counter()
        for _ in range(count):
            # Do not retain the previous decoded message during the next read.
            received = channel.receive()
            if received != value:
                raise AssertionError("JSON receive changed message contents")
            del received
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if allocation else None
    finally:
        if allocation:
            tracemalloc.stop()
        reader.close()
        thread.join(timeout=35)
    if thread.is_alive():
        raise RuntimeError("Socket benchmark sender did not exit")
    if errors:
        raise errors[0]
    return {
        "seconds": seconds,
        "seconds_per_message": seconds / count,
        "peak_traced_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--payload-bytes", type=int)
    source.add_argument("--input", type=Path, help="Use a representative JSON message")
    parser.add_argument("--messages", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload_bytes = args.payload_bytes if args.payload_bytes is not None else 256 * 1024
    if payload_bytes < 0 or min(args.messages, args.repeats) < 1:
        parser.error("Require a nonnegative payload and positive counts")
    value = (
        json.loads(args.input.read_text())
        if args.input is not None
        else {"state": ("aZ09-_" * ((payload_bytes + 5) // 6))[:payload_bytes]}
    )
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > JsonChannel.LIMIT:
        parser.error("Encoded JSON exceeds the channel's 4 MiB limit")
    packet = len(data).to_bytes(4, "big") + data
    # Warm imports, thread and JSON paths before separate timing/allocation runs.
    measure(value, packet, 1)
    rows = [measure(value, packet, args.messages) for _ in range(args.repeats)]
    allocation = measure(value, packet, 1, allocation=True)
    report = {
        "scope": "Real local socketpair, preframed sender, receive/JSON decode/equality; tracer includes sender thread Python activity but excludes prebuilt inputs, kernel buffers and RSS. No training or robot throughput claim.",
        "payload_bytes": payload_bytes if args.input is None else None,
        "input_path": str(args.input) if args.input is not None else None,
        "encoded_bytes": len(data),
        "messages_per_repeat": args.messages,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "samples": rows,
        "median_seconds_per_message": statistics.median(
            row["seconds_per_message"] for row in rows
        ),
        "allocation": allocation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("encoded_bytes", "median_seconds_per_message", "allocation")
            }
        )
    )


if __name__ == "__main__":
    main()
