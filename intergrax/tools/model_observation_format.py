# © Artur Czarnecki. All rights reserved.

"""Canonical model-visible serialization for tool observations."""

from __future__ import annotations

from intergrax.tools.execution_models import ToolModelObservation

EVIDENCE_REF_PREFIX = "EVIDENCE_REF: "


def format_tool_model_observation_content(observation: ToolModelObservation) -> str:
    """Render one ``role=tool`` message body with optional semantic evidence envelope."""
    reference = observation.evidence_reference
    if reference is None or not reference.strip():
        return observation.content
    return f"{EVIDENCE_REF_PREFIX}{reference.strip()}\n{observation.content}"


def parse_evidence_reference_from_tool_content(content: str) -> str | None:
    """Parse canonical ``EVIDENCE_REF`` envelope prefix when present."""
    if not content:
        return None
    first_line_end = content.find("\n")
    first_line = content[:first_line_end] if first_line_end >= 0 else content
    stripped = first_line.strip()
    if not stripped.startswith(EVIDENCE_REF_PREFIX):
        return None
    reference = stripped[len(EVIDENCE_REF_PREFIX) :].strip()
    return reference or None
