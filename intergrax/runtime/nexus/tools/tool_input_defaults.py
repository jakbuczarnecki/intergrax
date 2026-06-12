# © Artur Czarnecki. All rights reserved.

"""Default tool inputs for auto-invoke patterns (TOOL-ENG-25)."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.tools.core.contracts import ToolContract


_QUERY_FIELD_NAMES = frozenset({"query", "text", "message", "q", "search_query", "prompt"})


def default_tool_input(contract: ToolContract, query: str) -> BaseModel | None:
    """Build a minimal valid input model when possible; ``None`` if required fields are missing."""
    fields = contract.input_schema.model_fields
    if not fields:
        return contract.input_schema()
    kwargs: dict[str, object] = {}
    for name in fields:
        if name in _QUERY_FIELD_NAMES:
            kwargs[name] = query
    try:
        return contract.input_schema.model_validate(kwargs)
    except Exception:
        try:
            return contract.input_schema()
        except Exception:
            return None
