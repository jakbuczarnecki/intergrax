# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI function-tool schema export from ``ToolContract`` (Phase O.6)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from intergrax.tools.registry.runtime import RegisteredTool, ToolRegistry


def contract_to_openai_tool(contract: ToolContract, *, compact_description: bool = False) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": contract.tool_id,
            "description": contract.llm_description(compact=compact_description),
            "parameters": pydantic_parameters_schema(contract.input_schema),
        },
    }


def to_openai_tools(
    source: ToolRegistry | Iterable[RegisteredTool] | Iterable[ToolContract],
    *,
    compact_description: bool = False,
) -> list[dict[str, Any]]:
    contracts = _iter_contracts(source)
    return [contract_to_openai_tool(c, compact_description=compact_description) for c in contracts]


def compute_openai_tools_schema_hash(
    tools_schema: Sequence[Mapping[str, Any]],
) -> str:
    """SHA-256 over canonical OpenAI function-tool schema list (sorted by function name)."""
    if not tools_schema:
        digest = hashlib.sha256()
        digest.update(b"")
        return digest.hexdigest()

    canonical_entries = [copy.deepcopy(dict(entry)) for entry in tools_schema]
    canonical_entries.sort(key=lambda item: item["function"]["name"])
    canonical_json = json.dumps(
        canonical_entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256()
    digest.update(canonical_json.encode("utf-8"))
    return digest.hexdigest()


def _iter_contracts(
    source: ToolRegistry | Iterable[RegisteredTool] | Iterable[ToolContract],
) -> Sequence[ToolContract]:
    if isinstance(source, ToolRegistry):
        return [item.contract for item in source.list()]
    items = list(source)
    if not items:
        return []
    first = items[0]
    if isinstance(first, ToolContract):
        return items  # type: ignore[return-value]
    return [item.contract for item in items]  # type: ignore[union-attr]
