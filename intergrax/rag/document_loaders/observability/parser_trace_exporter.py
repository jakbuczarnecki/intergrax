# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Export document parser pipeline traces to structured logs (canonical observability spine)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEPRECATED_VENDOR_EXPORT_ENV = "INTERGRAX_EXPORT_PARSER_TRACE"


def _deprecated_vendor_export_requested() -> bool:
    return os.environ.get(_DEPRECATED_VENDOR_EXPORT_ENV, "").strip().lower() in {
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
    Emit parser attempt metadata to structured logs.

    Parser traces participate in the canonical RuntimeEvent / trace spine when
    tagged as ``integration_parser_trace``. Direct vendor SDK or HTTP ingestion
    is not permitted from this module — use ``ObservabilityExportEnvelope`` routing.
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

    if _deprecated_vendor_export_requested():
        slug = (
            observability_slug
            or os.environ.get("INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND", "").strip().lower()
        )
        logger.warning(
            "deprecated parser trace vendor export env ignored "
            "(env=%s slug=%s); use canonical observability export routing",
            _DEPRECATED_VENDOR_EXPORT_ENV,
            slug or "<unset>",
        )
