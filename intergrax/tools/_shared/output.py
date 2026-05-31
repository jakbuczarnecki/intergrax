# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared tool output helpers (Phase O.7)."""


def limit_tool_output(text: str, limit: int = 16000) -> str:
    """Safely truncate long tool output to avoid overflowing LLM context."""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = "<unserializable tool output>"
    return text if len(text) <= limit else text[:limit] + f"\n[...trimmed {len(text) - limit} chars]"
