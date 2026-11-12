"""Size-bounded pickle messages between trusted, same-Python renderer processes.

All socket waits in a request share an absolute monotonic deadline. Pickling and
unpickling run synchronously: checks before and after them account for elapsed
time, but cannot preempt codec CPU work or guarantee the total peak memory of an
object graph. The encoded byte limit bounds our output buffer and incoming frame
allocation. This channel is not the cross-Python JSON live-policy protocol.
"""

import logging
import math
import socket
import struct
from io import BytesIO
from multiprocessing.reduction import ForkingPickler
from time import monotonic

LOGGER = logging.getLogger(__name__)
_HEADER = struct.Struct("!I")


class _LimitedBuffer(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        with memoryview(value) as view:
            if self.tell() + view.nbytes > self.limit:
                raise ValueError(f"Frame message exceeds {self.limit} bytes")
        return super().write(value)


class FrameChannel:
    """Own one socket; discard the stream after any failed message operation.

    One owner calls the channel serially; each I/O operation sets the socket's
    timeout to the remaining request budget. Do not use an endpoint concurrently.
    ``socket`` is exposed only for multiprocessing's spawn socket reduction.
    ``deadline=None`` is reserved for a worker waiting for its next request;
    parent requests and every send require a finite absolute deadline. No channel
    or process resources are shared with a network client.
    """

    LIMIT = 64 * 1024**2

    def __init__(self, sock):
        self.socket = sock
        self._closed = False
        self._failed = False
        try:
            self.socket.setblocking(False)
        except BaseException:
            self._close_preserving_error()
            raise

    @property
    def closed(self):
        return self._closed

    @property
    def failed(self):
        """A failed stream cannot resume, including when descriptor close is pending."""
        return self._failed

    def close(self):
        if not self._closed:
            self.socket.close()
            # A signal may interrupt close before descriptor release. Let the
            # owner retry rather than declaring an unsuccessful close complete.
            self._closed = True

    def _close_preserving_error(self):
        # Even an interrupted descriptor close must not permit another message
        # on a stream that may already contain a partial frame.
        self._failed = True
        try:
            self.close()
        except BaseException:
            LOGGER.exception(
                "Could not close failed frame channel; preserving its error"
            )

    def _remaining(self, deadline):
        if self.closed:
            raise RuntimeError("Frame channel is closed")
        if self._failed:
            raise RuntimeError("Frame channel failed; close it before discarding it")
        if deadline is None:
            return None
        if isinstance(deadline, bool) or not math.isfinite(deadline):
            raise ValueError(
                "Frame deadline must be finite, or None for an idle receive"
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Frame communication deadline exceeded")
        return remaining

    def _write(self, data, deadline):
        with memoryview(data) as view:
            offset = 0
            while offset < len(view):
                self.socket.settimeout(self._remaining(deadline))
                with view[offset:] as remaining:
                    sent = self.socket.send(remaining)
                if not sent:
                    raise EOFError("Frame renderer disconnected during send")
                offset += sent

    def _read(self, size, deadline):
        self._remaining(deadline)
        data = bytearray(size)
        with memoryview(data) as view:
            offset = 0
            while offset < size:
                self.socket.settimeout(self._remaining(deadline))
                with view[offset:] as remaining:
                    received = self.socket.recv_into(remaining)
                if not received:
                    raise EOFError("Frame renderer disconnected")
                offset += received
        return data

    def send(self, value, *, deadline):
        try:
            if deadline is None:
                raise ValueError("Frame send requires a finite deadline")
            self._remaining(deadline)
            with _LimitedBuffer(self.LIMIT) as output:
                # Match multiprocessing.Connection.send's default pickle
                # protocol and registered reducers for local Python objects.
                ForkingPickler(output).dump(value)
                self._remaining(deadline)
                with output.getbuffer() as payload:
                    self._write(_HEADER.pack(len(payload)), deadline)
                    self._write(payload, deadline)
            self._remaining(deadline)
        except BaseException:
            self._close_preserving_error()
            raise

    def receive(self, *, deadline):
        try:
            size = _HEADER.unpack(self._read(_HEADER.size, deadline))[0]
            if not 0 < size <= self.LIMIT:
                raise ValueError(f"Invalid frame message length: {size} bytes")
            payload = self._read(size, deadline)
            self._remaining(deadline)
            value = ForkingPickler.loads(payload)
            self._remaining(deadline)
            return value
        except BaseException:
            self._close_preserving_error()
            raise


def channel_pair():
    """Create two owned endpoints, releasing both if either wrapper fails."""
    sockets = socket.socketpair()
    try:
        return FrameChannel(sockets[0]), FrameChannel(sockets[1])
    except BaseException:
        for endpoint in sockets:
            for attempt in range(2):
                try:
                    endpoint.close()
                    break
                except BaseException:
                    LOGGER.exception(
                        "Frame channel pair cleanup failed%s; preserving its error",
                        "; retrying" if attempt == 0 else "",
                    )
        raise
