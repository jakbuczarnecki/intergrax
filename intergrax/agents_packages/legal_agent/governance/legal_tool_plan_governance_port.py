# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Dynamic governance hook for :class:`LegalToolPlan` (abstract base class).

**You do not call these methods from application code manually.** The Legal Agent
wires them inside :func:`~intergrax.agents_packages.legal_agent.pipeline.legal_execution_loop.run_legal_dynamic_execution_loop`
after the LLM tool decision and static org clamp, and **before** the Nexus bridge.

Pipeline order (dynamic pipeline only)::

    decide_legal_tool_plan          # LLM + capability clamp (inside tool_decision_component)
        -> enforce_legal_tool_plan_governance   # LegalAgentConfig: organization_allow_*
        -> legal_tool_plan_governance.adjust_legal_tool_plan   # THIS PORT (optional)
        -> run_legal_tool_runtime_bridge       # RagStep / WebsearchStep / ToolsStep

**Why a separate ABC (not Protocol-only)?** Explicit inheritance and ``isinstance`` checks;
subclasses must implement ``adjust_legal_tool_plan``.

**Platform “dual” wiring:** :class:`~intergrax.runtime.governance.service.GovernanceService`
handles **post-run** ``evaluate()`` only. For the same object to also adjust plans, subclass
both ``GovernanceService`` and ``LegalToolPlanGovernancePort`` and set **both**
:attr:`~intergrax.agents_packages.legal_agent.config.legal_agent_config.LegalAgentConfig.governance_service`
and
:attr:`~intergrax.agents_packages.legal_agent.config.legal_agent_config.LegalAgentConfig.legal_tool_plan_governance`
to that instance.

Ready-made subclasses: :mod:`intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance_impl`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from intergrax.agents_packages.legal_agent.domain.legal_tool_plan import LegalToolPlan
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

if TYPE_CHECKING:
    from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig


class LegalToolPlanGovernancePort(ABC):
    """
    Pre-bridge, hot-path hook. Implementations must stay lightweight (no full replay).
    """

    @abstractmethod
    def adjust_legal_tool_plan(
        self,
        plan: LegalToolPlan,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalToolPlan:
        """Return the plan to execute (possibly ``model_copy`` with degraded layers)."""
        ...

