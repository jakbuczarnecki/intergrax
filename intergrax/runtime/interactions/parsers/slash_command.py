# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable slash-command text parsing (Slack, Teams, CLI)."""

from __future__ import annotations

from typing import Optional, Tuple


def _looks_like_capability(token: str) -> bool:
    return "." in token or "-" in token


def parse_slash_command_text(text: str) -> Tuple[Optional[str], str]:
    """
    Parse ``capability remainder`` into capability + message.

    Capability tokens use dotted or dashed names (``echo.basic``, ``research-pipeline``).
    Plain sentences without a capability prefix return ``(None, full_text)``.
    """
    normalized = (text or "").strip()
    if not normalized:
        return None, ""

    if normalized.startswith("/"):
        normalized = normalized[1:].strip()

    parts = normalized.split(None, 1)
    if len(parts) == 1:
        token = parts[0]
        if _looks_like_capability(token):
            return token, ""
        return None, token

    first, rest = parts[0], parts[1]
    if _looks_like_capability(first):
        return first, rest.strip()
    return None, normalized
