import asyncio
import os
import subprocess

import pytest


@pytest.mark.asyncio
async def test_stdio_transport():
    """Test that stdio transport can be started without errors."""
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")
    db_url = f"postgresql://{db_host}:{db_port}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-memory",
        "--transport",
        "stdio",
        "--db-url",
        db_url,
        "--username",
        db_user,
        "--password",
        db_password,
        "--database",
        db_name,
        "--graphname",
        graphname,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    # Give it a moment to start
    await asyncio.sleep(3)

    # If process has exited, check if it was successful startup
    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        # Check for startup success message
        assert "Starting MCP server" in stderr_text or "FastMCP" in stdout_text, \
            f"Server failed to start properly.\nSTDOUT: {stdout_text}\nSTDERR: {stderr_text}"

        # Return code 0 is ok for stdio (means it started and exited cleanly)
        assert process.returncode == 0, f"Server exited with error code {process.returncode}"
    else:
        # Process is still running, terminate it
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
