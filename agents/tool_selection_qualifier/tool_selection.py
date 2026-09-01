# © Artur Czarnecki. All rights reserved.

"""Tool-selection helpers for DIAG-FUNCTIONAL-Q2 qualification."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.tools.providers.workspace.service import (
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)

DEFAULT_QUALIFICATION_TOOL_IDS: tuple[str, ...] = (
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)


@dataclass(frozen=True, slots=True)
class ToolSelectionCandidate:
    tool_id: str
    rank: int


def artifact_ref_for_tool(tool_id: str) -> str:
    return f"tool:{tool_id}"


def candidates_from_tool_ids(tool_ids: tuple[str, ...]) -> tuple[ToolSelectionCandidate, ...]:
    return tuple(
        ToolSelectionCandidate(tool_id=tool_id, rank=index)
        for index, tool_id in enumerate(tool_ids, start=1)
    )


__all__ = [
    "DEFAULT_QUALIFICATION_TOOL_IDS",
    "ToolSelectionCandidate",
    "artifact_ref_for_tool",
    "candidates_from_tool_ids",
]
