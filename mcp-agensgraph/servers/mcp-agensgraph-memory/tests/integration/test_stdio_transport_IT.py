import asyncio
import subprocess

import pytest

from conftest import settings, start_server, stop_server


@pytest.mark.asyncio
async def test_stdio_transport():
    """The server starts over stdio, makes what the graph needs, and waits for a request.

    Exiting is a failure here: the previous shape of this test asserted nothing at all unless
    the process had already stopped, so a server that started and one that never did both
    passed.
    """
    process = await start_server(
        settings(), ["--transport", "stdio"], "stdio server", stdin=subprocess.PIPE
    )
    try:
        await asyncio.sleep(5)
        assert process.returncode is None, (
            f"the server exited with {process.returncode} instead of waiting for a request"
        )
    finally:
        await stop_server(process)
