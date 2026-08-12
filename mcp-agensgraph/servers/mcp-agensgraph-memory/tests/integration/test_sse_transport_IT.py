import aiohttp
import pytest


@pytest.mark.asyncio
async def test_sse_endpoint(sse_server):
    """The SSE endpoint answers, and answers as a stream.

    A test that accepted 404 as well as 200 passed against a server that had not started.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://127.0.0.1:8002/mcp/", timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            assert response.status == 200, f"Unexpected status: {response.status}"
            assert response.headers["content-type"].startswith("text/event-stream")
