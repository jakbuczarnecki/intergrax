# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parse inbound HTTP bodies for interaction adapters."""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import parse_qs


def parse_inbound_http_body(*, content_type: str, body: bytes) -> Dict[str, Any]:
    """
    Normalize JSON or form-encoded webhook bodies to a flat dict.

    Form values use the first entry when vendors send single-value fields.
    """
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type in {"application/x-www-form-urlencoded", "application/x-www-form-urlencoded; charset=utf-8"}:
        parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        return {key: values[0] if values else "" for key, values in parsed.items()}

    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("interaction payload must be a JSON object")
    return payload
