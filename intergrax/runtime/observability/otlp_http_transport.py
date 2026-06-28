# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP HTTP transport for observability export (OBS-EXPORT-4B)."""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporterConfig


class OtlpHttpTransport:
    """POST OTLP JSON payloads to a configured HTTP endpoint via httpx."""

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def send(
        self,
        payload: Mapping[str, Any],
        *,
        config: OtlpObservabilityExporterConfig,
    ) -> None:
        headers = dict(config.headers)
        if not any(key.lower() == "content-type" for key in headers):
            headers["Content-Type"] = "application/json"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=config.timeout_seconds)
        try:
            response = await client.post(
                config.endpoint,
                content=body,
                headers=headers,
            )
            response.raise_for_status()
        finally:
            if owns_client:
                await client.aclose()
