# © Artur Czarnecki. All rights reserved.

"""Test-only scripted native tool planner + reasoning LLM for incident investigation."""

from __future__ import annotations

from platform_proofs.scenarios.ai_incident_investigation.fixtures.lab_planner_llm import (
    FixtureDrivenIncidentInvestigationLLM,
    build_fixture_reasoning_proposal,
)

# Backward-compatible aliases for tests that subclass or reference the scripted double.
ScriptedIncidentInvestigationLLM = FixtureDrivenIncidentInvestigationLLM
build_scripted_reasoning_proposal = build_fixture_reasoning_proposal

__all__ = [
    "ScriptedIncidentInvestigationLLM",
    "build_scripted_reasoning_proposal",
    "FixtureDrivenIncidentInvestigationLLM",
    "build_fixture_reasoning_proposal",
]
