# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Export document parser pipeline traces to observability backends."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _env_enabled() -> bool:
    return os.environ.get("INTERGRAX_EXPORT_PARSER_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def export_parser_trace(
    *,
    source: str,
    trace: dict[str, Any],
    observability_slug: Optional[str] = None,
) -> None:
    """
    Emit parser attempt metadata to logs and optional Langfuse/Sentry backends.

    Set ``INTERGRAX_EXPORT_PARSER_TRACE=1`` to enable vendor export when
    ``INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND`` is ``langfuse`` or ``sentry``.
    """
    parser_id = trace.get("parser_id")
    attempts = trace.get("attempts") or []
    logger.info(
        "document_parser_trace source=%s parser_id=%s attempts=%s",
        source,
        parser_id,
        len(attempts),
        extra={"integration_parser_trace": trace},
    )

    if not _env_enabled():
        return

    slug = (
        observability_slug
        or os.environ.get("INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND", "").strip().lower()
    )
    if not slug:
        return

    payload = {
        "event": "document_parser_trace",
        "source": source,
        "parser_id": parser_id,
        "trace": trace,
    }

    if slug == "sentry":
        _export_sentry(payload)
    elif slug == "langfuse":
        _export_langfuse(payload)


def _export_sentry(payload: dict[str, Any]) -> None:
    try:
        import sentry_sdk
    except ImportError:
        return
    with sentry_sdk.push_scope() as scope:
        scope.set_context("document_parser", payload)
        sentry_sdk.capture_message(
            f"document_parser:{payload.get('parser_id')}",
            level="info",
        )


def _export_langfuse(payload: dict[str, Any]) -> None:
    try:
        import httpx
    except ImportError:
        return
    base = os.environ.get("INTERGRAX_LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return
    body = {
        "batch": [
            {
                "type": "trace-create",
                "body": {
                    "name": "document_parser_trace",
                    "input": payload,
                    "metadata": {"source": payload.get("source")},
                },
            }
        ]
    }
    try:
        response = httpx.post(
            f"{base}/api/public/ingestion",
            json=body,
            auth=(public_key, secret_key),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.debug("langfuse parser trace export failed: %s", exc)
