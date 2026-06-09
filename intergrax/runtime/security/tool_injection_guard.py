# © Artur Czarnecki. All rights reserved.

"""Tool injection defense for untrusted tool arguments (IDEAL-23.3)."""

from __future__ import annotations

import re

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore\s+previous\s+instructions"),
    re.compile(r"(?i)system\s*:\s*"),
    re.compile(r"(?i)<\s*/?\s*tool_call"),
    re.compile(r"(?i)override\s+policy"),
)


class ToolInjectionError(ValueError):
    """Raised when tool input appears to contain injection patterns."""


def assert_tool_input_safe(value: str) -> None:
    """Reject obvious prompt-injection patterns in tool argument strings."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            raise ToolInjectionError(f"tool input matched injection pattern: {pattern.pattern}")
