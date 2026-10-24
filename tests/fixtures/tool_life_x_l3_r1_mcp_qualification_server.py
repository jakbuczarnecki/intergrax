# © Artur Czarnecki. All rights reserved.

"""Subprocess FastMCP streamable-HTTP server for TOOL-LIFE-X L3-R1 qualification."""

from __future__ import annotations

import os
import socket
import sys
import time

from fastmcp import FastMCP
import uvicorn

_MODE = os.environ.get("TOOL_LIFE_X_MCP_MODE", "ok")

mcp = FastMCP("tool-life-x-l3-r1")


@mcp.tool
def echo(message: str = "") -> str:
    if _MODE == "hang":
        time.sleep(10_000)
    if _MODE == "slow":
        time.sleep(3.0)
        return "slow-ok"
    if _MODE == "huge":
        return "x" * 100_000
    if _MODE == "crash_mid":
        time.sleep(0.3)
        os._exit(3)
    return f"echo:{message}"


def main() -> None:
    if _MODE == "unavailable":
        sys.exit(2)
    app = mcp.http_app(path="/")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    print(port, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


if __name__ == "__main__":
    main()
