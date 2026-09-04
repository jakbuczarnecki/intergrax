# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed live policy evaluation contexts (ADR-GOVERNED-EXECUTION-001 G1B-1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PreModelPhase(StrEnum):
    """Phase discriminator for pre-model policy evaluation."""

    NEXUS_PLANNING = "nexus_planning"
    AGENT_STEP = "agent_step"


@dataclass(frozen=True, slots=True)
class AgentDecisionPolicyContext:
    """Context for ``evaluate_decision`` live policy evaluation."""

    require_human_on_critical: bool = True
    has_unresolved_critical_interrupt: bool = False


@dataclass(frozen=True, slots=True)
class PreModelPolicyContext:
    """Context for ``evaluate_pre_llm`` (PRE_MODEL) live policy evaluation."""

    phase: PreModelPhase | None = None
    planner_model_id: str = ""
    denied_planner_model_ids: tuple[str, ...] = ()
    model_id: str = ""
    denied_model_ids: tuple[str, ...] = ()
