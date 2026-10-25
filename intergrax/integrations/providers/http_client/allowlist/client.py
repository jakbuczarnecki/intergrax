# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Mapping
from urllib.parse import urljoin, urlparse

import httpx

from intergrax.integrations.contracts.http_client import HttpResponse
from intergrax.integrations.providers.http_client.allowlist.bounded_read import (
    accumulate_response_bytes,
)
from intergrax.integrations.providers.http_client.allowlist.redirect_policy import (
    operation_deadline_monotonic,
    remaining_hop_timeout_s,
    strip_cross_authority_credentials,
)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_MAX_REDIRECTS = 10
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class AllowlistHttpClient:
    """HTTP client that permits requests only to configured hostnames."""

    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str],
        default_timeout_s: float = 30.0,
        max_redirects: int = _DEFAULT_MAX_REDIRECTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized = frozenset(host.strip().lower() for host in allowed_hosts if host.strip())
        self._allowed_hosts = normalized
        self._default_timeout_s = default_timeout_s
        self._max_redirects = max(0, max_redirects)
        self._transport = transport

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
        timeout_s: float = 30.0,
        max_response_bytes: int | None = None,
    ) -> HttpResponse:
        self._assert_host_allowed(url)
        configured_timeout = timeout_s if timeout_s > 0 else self._default_timeout_s
        operation_deadline = operation_deadline_monotonic(configured_timeout)
        with httpx.Client(
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            return self._request_with_redirects(
                client,
                method.upper(),
                url.strip(),
                headers=dict(headers or {}),
                body=body,
                max_response_bytes=max_response_bytes,
                operation_deadline=operation_deadline,
            )

    def _assert_host_allowed(self, url: str) -> None:
        parsed = urlparse(url.strip())
        scheme = (parsed.scheme or "").lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise PermissionError(f"scheme_not_allowed:{scheme or 'missing'}")
        host = (parsed.hostname or "").lower()
        if not host or host not in self._allowed_hosts:
            raise PermissionError(f"host_not_allowed:{host or 'missing'}")

    def _request_with_redirects(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: str,
        max_response_bytes: int | None,
        operation_deadline: float,
    ) -> HttpResponse:
        current_url = url
        current_method = method
        current_body = body
        redirect_count = 0

        while True:
            hop_timeout = remaining_hop_timeout_s(operation_deadline)
            if max_response_bytes is not None and max_response_bytes > 0:
                with client.stream(
                    current_method,
                    current_url,
                    headers=headers,
                    content=current_body.encode("utf-8") if current_body else None,
                    timeout=hop_timeout,
                ) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if location is None:
                            raise PermissionError("redirect_location_missing")
                        redirect_count, current_url, current_method, current_body, headers = (
                            self._advance_redirect(
                                redirect_count=redirect_count,
                                from_url=current_url,
                                location=location,
                                status_code=response.status_code,
                                method=current_method,
                                body=current_body,
                                headers=headers,
                            )
                        )
                        continue
                    raw_body, truncated = accumulate_response_bytes(
                        response.iter_bytes(),
                        max_response_bytes=max_response_bytes,
                    )
                    text = raw_body.decode("utf-8", errors="ignore" if truncated else "replace")
                    return HttpResponse(
                        status_code=response.status_code,
                        body=text,
                        headers={str(key): str(value) for key, value in response.headers.items()},
                        body_truncated_at_transport=truncated,
                        transport_byte_ceiling_enforced=True,
                    )

            response = client.request(
                current_method,
                current_url,
                headers=headers,
                content=current_body.encode("utf-8") if current_body else None,
                timeout=hop_timeout,
            )
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if location is None:
                    raise PermissionError("redirect_location_missing")
                redirect_count, current_url, current_method, current_body, headers = (
                    self._advance_redirect(
                        redirect_count=redirect_count,
                        from_url=current_url,
                        location=location,
                        status_code=response.status_code,
                        method=current_method,
                        body=current_body,
                        headers=headers,
                    )
                )
                continue

            return HttpResponse(
                status_code=response.status_code,
                body=response.text,
                headers={str(key): str(value) for key, value in response.headers.items()},
            )

    def _advance_redirect(
        self,
        *,
        redirect_count: int,
        from_url: str,
        location: str,
        status_code: int,
        method: str,
        body: str,
        headers: dict[str, str],
    ) -> tuple[int, str, str, str, dict[str, str]]:
        if redirect_count >= self._max_redirects:
            raise PermissionError("redirect_limit_exceeded")
        next_url = urljoin(from_url, location.strip())
        self._assert_host_allowed(next_url)
        next_headers = strip_cross_authority_credentials(
            headers,
            from_url=from_url,
            to_url=next_url,
        )
        next_method = method
        next_body = body
        if status_code == 303 or (
            status_code in {301, 302} and method not in frozenset({"GET", "HEAD"})
        ):
            next_method = "GET"
            next_body = ""
        return redirect_count + 1, next_url, next_method, next_body, next_headers
