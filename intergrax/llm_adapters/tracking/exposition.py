# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""HTTP-ready metric exposition helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector


def render_prometheus_text() -> str:
    """Prometheus text exposition format (``text/plain; version=0.0.4``)."""
    lines = get_llm_metrics_collector().prometheus_lines()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def render_otlp_json() -> Dict[str, Any]:
    """OTLP-style JSON resource metrics snapshot."""
    return get_llm_metrics_collector().otlp_resource_metrics()


def register_llm_metrics_routes(app: Any, *, prefix: str = "/metrics/llm") -> None:
    """
    Register Tier-3 debug/metrics routes on a FastAPI app.

    - ``GET {prefix}`` — Prometheus text
    - ``GET {prefix}/otlp`` — OTLP-style JSON
    """
    from fastapi import Response
    from fastapi.responses import JSONResponse

    @app.get(prefix)
    def llm_metrics_prometheus() -> Response:
        return Response(
            content=render_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get(f"{prefix}/otlp")
    def llm_metrics_otlp() -> JSONResponse:
        return JSONResponse(render_otlp_json())
