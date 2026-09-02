# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Stable identity types for functional qualification plugins (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_PLUGIN_ID_PATTERN = re.compile(r"^functional\.[a-z][a-z0-9_]*$")
_GATE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FunctionalQualificationPluginId:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise ValueError("functional_qualification_plugin_id_empty")
        if not _PLUGIN_ID_PATTERN.fullmatch(normalized):
            raise ValueError(f"functional_qualification_plugin_id_invalid:{normalized}")
        if normalized != self.value:
            object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


RAG_PLUGIN_ID = FunctionalQualificationPluginId("functional.rag")
TOOL_SELECTION_PLUGIN_ID = FunctionalQualificationPluginId("functional.tool_selection")
WEB_SEARCH_PLUGIN_ID = FunctionalQualificationPluginId("functional.web_search")
MODEL_ROUTING_PLUGIN_ID = FunctionalQualificationPluginId("functional.model_routing")


class QualificationGateId(StrEnum):
    COMPARISON_PASS = "comparison_pass"
    ORACLE_INDEPENDENCE = "oracle_independence"
    EVIDENCE_SCOPE_INTEGRITY = "evidence_scope_integrity"
    PLUGIN_EXECUTION_COMPLETED = "plugin_execution_completed"


def validate_functional_qualification_plugin_id(value: str) -> FunctionalQualificationPluginId:
    return FunctionalQualificationPluginId(value)


def validate_qualification_gate_id(value: str) -> QualificationGateId:
    normalized = value.strip()
    if not normalized:
        raise ValueError("qualification_gate_id_empty")
    if not _GATE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"qualification_gate_id_invalid:{normalized}")
    return QualificationGateId(normalized)


__all__ = [
    "FunctionalQualificationPluginId",
    "MODEL_ROUTING_PLUGIN_ID",
    "QualificationGateId",
    "RAG_PLUGIN_ID",
    "TOOL_SELECTION_PLUGIN_ID",
    "WEB_SEARCH_PLUGIN_ID",
    "validate_functional_qualification_plugin_id",
    "validate_qualification_gate_id",
]
