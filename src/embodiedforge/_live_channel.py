"""Bounded JSON messages over a private inherited socket, across Python versions."""

import json
import struct


class JsonChannel:
    LIMIT = 4 * 1024**2

    def __init__(self, sock):
        self.socket = sock

    def send(self, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > self.LIMIT:
            raise ValueError("Live message exceeds 4 MiB")
        self.socket.sendall(struct.pack("!I", len(data)) + data)

    def _read(self, size):
        chunks = bytearray()
        while len(chunks) < size:
            data = self.socket.recv(size - len(chunks))
            if not data:
                raise EOFError("Live policy worker disconnected")
            chunks.extend(data)
        return bytes(chunks)

    def receive(self):
        size = struct.unpack("!I", self._read(4))[0]
        if not 0 < size <= self.LIMIT:
            raise ValueError("Invalid live message length")
        return json.loads(self._read(size))
