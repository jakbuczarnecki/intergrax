# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Concrete :class:`LegalToolPlanGovernancePort` subclasses for :class:`LegalAgentConfig`.

**How to use (wiring only — no manual invocation)**

1. Build your :class:`~intergrax.agents_packages.legal_agent.legal_agent_config.LegalAgentConfig`.
2. Set ``legal_tool_plan_governance`` to one of:

   * ``None`` (default) — skip dynamic governance entirely.
   * :class:`PassthroughLegalToolPlanGovernance` — explicit no-op (same ``plan`` object returned).
     Rare; prefer ``None`` unless you need a non-null slot for DI.
   * ``CallableLegalToolPlanGovernance(my_fn)`` — callable signature
     ``(plan, state, legal_config) -> LegalToolPlan`` (positional args).
     returns the plan to run (usually ``plan.model_copy(update={...})``). Quick way to plug
     tenant rules, feature flags, or DB lookups without a new class.
   * Your own subclass of :class:`~intergrax.agents_packages.legal_agent.legal_tool_plan_governance_port.LegalToolPlanGovernancePort`
     for production policy.
   * A subclass of :class:`~intergrax.runtime.governance.service.GovernanceService` that also
     inherits ``LegalToolPlanGovernancePort`` — assign the **same instance** to
     ``governance_service`` and ``legal_tool_plan_governance`` for post-run + pre-bridge in one object.

3. Run the agent via :class:`~intergrax.agents.agent_engine.AgentEngine` as usual.
   :func:`~intergrax.agents_packages.legal_agent.legal_execution_loop.run_legal_dynamic_execution_loop`
   calls ``adjust_legal_tool_plan`` automatically when the field is not ``None``.

**What they must do:** Return a :class:`~intergrax.agents_packages.legal_agent.legal_tool_plan.LegalToolPlan`
(typically degrade ``use_rag`` / ``use_websearch`` / ``use_tools``). Stay lightweight — no full
replay/metrics on the request hot path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_tool_plan import LegalToolPlan
from intergrax.agents_packages.legal_agent.legal_tool_plan_governance_port import LegalToolPlanGovernancePort
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

LegalToolPlanAdjustFn = Callable[
    [LegalToolPlan, RuntimeState, LegalAgentConfig],
    LegalToolPlan,
]


@dataclass(slots=True, frozen=True)
class PassthroughLegalToolPlanGovernance(LegalToolPlanGovernancePort):
    """
    Explicit no-op implementation (returns the same ``plan`` instance).

    Use when you want a non-``None`` slot for DI/testing, or symmetry with other agents.
    Otherwise ``legal_tool_plan_governance=None`` is enough.
    """

    def adjust_legal_tool_plan(
        self,
        plan: LegalToolPlan,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalToolPlan:
        return plan


@dataclass(slots=True)
class CallableLegalToolPlanGovernance(LegalToolPlanGovernancePort):
    """
    Delegates to a plain function or lambda (e.g. tenant lookup, feature flags).

    The callable receives ``(plan, state, legal_config)`` positionally and must return
    a :class:`LegalToolPlan` (typically ``plan.model_copy(update={...})``).
    """

    _fn: LegalToolPlanAdjustFn

    def adjust_legal_tool_plan(
        self,
        plan: LegalToolPlan,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalToolPlan:
        return self._fn(plan, state, legal_config)
