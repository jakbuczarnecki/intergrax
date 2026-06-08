# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.http_client import HttpClientBackend
from intergrax.tools.providers.http.contracts import HttpRequestInput, HttpRequestOutput
from intergrax.tools.registry.wiring import ToolWiringContext

HTTP_REQUEST_TOOL_ID = "http.request"


def _require_http_client(ctx: ToolWiringContext) -> HttpClientBackend:
    client = ctx.http_client or ctx.extras.get("http_client")
    if client is None:
        raise RuntimeError("http_client_not_configured")
    return client


def http_request(ctx: ToolWiringContext, params: HttpRequestInput) -> HttpRequestOutput:
    client = _require_http_client(ctx)
    method = params.method.strip().upper()
    allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
    if method not in allowed:
        raise ValueError(f"unsupported_http_method:{method}")
    try:
        response = client.request(
            method,
            params.url.strip(),
            headers=dict(params.headers),
            body=params.body,
            timeout_s=params.timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 — tool boundary
        return HttpRequestOutput(success=False, error=str(exc))
    return HttpRequestOutput(
        success=200 <= response.status_code < 400,
        status_code=response.status_code,
        body=response.body,
        headers=dict(response.headers),
    )
