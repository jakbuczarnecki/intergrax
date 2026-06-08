# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Mapping
from urllib.parse import urlparse

import httpx

from intergrax.integrations.contracts.http_client import HttpResponse


class AllowlistHttpClient:
    """HTTP client that permits requests only to configured hostnames."""

    def __init__(self, *, allowed_hosts: frozenset[str], default_timeout_s: float = 30.0) -> None:
        normalized = frozenset(host.strip().lower() for host in allowed_hosts if host.strip())
        self._allowed_hosts = normalized
        self._default_timeout_s = default_timeout_s

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
        timeout_s: float = 30.0,
    ) -> HttpResponse:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        if not host or host not in self._allowed_hosts:
            raise PermissionError(f"host_not_allowed:{host or 'missing'}")

        timeout = timeout_s if timeout_s > 0 else self._default_timeout_s
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.request(
                method.upper(),
                url.strip(),
                headers=dict(headers or {}),
                content=body.encode("utf-8") if body else None,
            )
        return HttpResponse(
            status_code=response.status_code,
            body=response.text,
            headers={str(key): str(value) for key, value in response.headers.items()},
        )
