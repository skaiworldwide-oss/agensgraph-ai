"""What a client sends: its round trips, counted at the socket, and its statements, from the driver.

A round trip is a turn of the conversation: the client sends, then the server answers. ``Relay``
sits between the two and counts the turns, so a pipeline of ten statements is one and a ``COPY`` is
two, since the server answers the ``COPY`` before the rows go. Counting at the socket works on any
platform and needs nothing from the code under test.
"""

from __future__ import annotations

import re
import selectors
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from agensgraph import observability
from psycopg.conninfo import conninfo_to_dict, make_conninfo


class Relay:
    """A local port that forwards to a server and counts round trips across every connection."""

    def __init__(self, conninfo: str) -> None:
        parts = conninfo_to_dict(conninfo)
        self._target = (str(parts.get("host") or "127.0.0.1"), int(parts.get("port") or 5432))
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.conninfo = make_conninfo(conninfo, host="127.0.0.1", port=self.port)
        self._lock = threading.Lock()
        self._round_trips = 0
        self._closed = False
        self._threads: list[threading.Thread] = []
        accepting = threading.Thread(target=self._accept, daemon=True)
        accepting.start()
        self._threads.append(accepting)

    @property
    def port(self) -> int:
        return int(self._listener.getsockname()[1])

    @property
    def round_trips(self) -> int:
        with self._lock:
            return self._round_trips

    def reset(self) -> None:
        with self._lock:
            self._round_trips = 0

    def close(self) -> None:
        self._closed = True
        self._listener.close()

    def _accept(self) -> None:
        while not self._closed:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            server = socket.create_connection(self._target)
            for sock in (client, server):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            relaying = threading.Thread(target=self._relay, args=(client, server), daemon=True)
            relaying.start()
            self._threads.append(relaying)

    def _relay(self, client: socket.socket, server: socket.socket) -> None:
        # The client speaks first; each time it speaks after the server has, a new turn starts.
        server_spoke_last = True
        with selectors.DefaultSelector() as selector, client, server:
            selector.register(client, selectors.EVENT_READ, server)
            selector.register(server, selectors.EVENT_READ, client)
            while not self._closed:
                for key, _ in selector.select(timeout=0.5):
                    source: socket.socket = key.fileobj  # type: ignore[assignment]
                    try:
                        data = source.recv(65536)
                    except OSError:
                        return
                    if not data:
                        return
                    from_client = source is client
                    if from_client and server_spoke_last:
                        with self._lock:
                            self._round_trips += 1
                    server_spoke_last = not from_client
                    key.data.sendall(data)


_SPACE = re.compile(r"\s+")


@contextmanager
def statements() -> Iterator[list[str]]:
    """Every statement the driver sends inside the block, whitespace collapsed, in order."""
    sent: list[str] = []

    def record(entry: observability.QueryRecord) -> None:
        sent.append(_SPACE.sub(" ", entry.statement).strip())

    observability.add_query_logger(record)
    try:
        yield sent
    finally:
        observability.remove_query_logger(record)
