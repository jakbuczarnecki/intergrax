# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Optional Pushgateway export for LLM Prometheus metrics."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Optional

from intergrax.llm_adapters.tracking.exposition import render_prometheus_text


def pushgateway_url() -> Optional[str]:
    raw = os.getenv("INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL", "").strip()
    return raw or None


def push_llm_metrics_to_gateway(
    *,
    job: str = "intergrax_llm",
    grouping_key: Optional[str] = None,
    url: Optional[str] = None,
) -> bool:
    """
    POST Prometheus text exposition to Pushgateway (``/metrics/job/<job>``).

    Returns True when push succeeded, False when disabled or failed (non-fatal).
    """
    base = (url or pushgateway_url() or "").rstrip("/")
    if not base:
        return False

    body = render_prometheus_text()
    if not body.strip():
        return False

    path = f"/metrics/job/{job}"
    if grouping_key:
        path = f"{path}/{grouping_key}"
    endpoint = f"{base}{path}"

    req = urllib.request.Request(
        endpoint,
        data=body.encode("utf-8"),
        method="PUT",
        headers={"Content-Type": "text/plain; version=0.0.4"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
