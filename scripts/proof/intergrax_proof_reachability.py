# © Artur Czarnecki. All rights reserved.

"""Shared TCP/HTTP reachability probes for qualification and proof harnesses."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request


def tcp_reachable(host: str, port: int, timeout_seconds: float = 2.0) -> bool:
    """Return whether a TCP connection to ``host:port`` succeeds within the timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def http_reachable(url: str, timeout_seconds: float = 3.0) -> bool:
    """Return whether ``url`` responds with an HTTP status in the 2xx–4xx range."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
