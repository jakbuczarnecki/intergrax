# © Artur Czarnecki. All rights reserved.

"""Task routing contract — capability tokens only, no class-name routing (ACP-CON-6)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

_AGENT_IMPORT_PATH = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+\.[A-Z][a-zA-Z0-9_]*Agent$")
_AGENT_CLASS_NAME = re.compile(r"^[A-Z][a-zA-Z0-9_]*Agent$")


class TaskRoutingForbiddenKey(StrEnum):
    """Metadata keys that MUST NOT drive Nexus agent selection."""

    AGENT_CLASS = "agent_class"
    AGENT_TYPE = "agent_type"
    REQUIRED_AGENT_CLASS = "required_agent_class"
    AGENT_IMPORT_PATH = "agent_import_path"
    PYTHON_CLASS = "python_class"
    REQUIRED_AGENT = "required_agent"


FORBIDDEN_ROUTING_KEYS: frozenset[str] = frozenset(member.value for member in TaskRoutingForbiddenKey)


class TaskRoutingViolationError(ValueError):
    """Raised when a task payload attempts class-name based routing."""


def _inspect_mapping(mapping: dict[str, Any], *, label: str) -> list[str]:
    violations: list[str] = []
    for key, raw in mapping.items():
        lowered = str(key).lower()
        if lowered in FORBIDDEN_ROUTING_KEYS or lowered.endswith("_agent_class"):
            violations.append(f"{label}.{key} is forbidden for capability routing")
            continue
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if not value:
            continue
        if _AGENT_IMPORT_PATH.match(value):
            violations.append(f"{label}.{key} contains agent import path {value!r}")
        elif _AGENT_CLASS_NAME.match(value):
            violations.append(f"{label}.{key} contains agent class name {value!r}")
    return violations


def validate_task_routing_payload(
    *,
    metadata: dict[str, Any] | None = None,
    context_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Reject task payloads that route by Python class name or import path (§37.6).

    Tier-3 ``AgentBinding`` wires implementations; Nexus tasks use capability tokens.
    """
    violations: list[str] = []
    if metadata:
        violations.extend(_inspect_mapping(metadata, label="metadata"))
    if context_metadata:
        violations.extend(_inspect_mapping(context_metadata, label="context.metadata"))
    if violations:
        raise TaskRoutingViolationError("; ".join(violations))
