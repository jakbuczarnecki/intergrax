# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Named product SKUs for :class:`~intergrax.agents_packages.legal_agent.config.legal_agent_config.LegalAgentConfig`.

:class:`LegalAgentProductProfile` is a :class:`~enum.StrEnum`: use ``LegalAgentProductProfile.SAFE``,
call :meth:`~LegalAgentProductProfile.make_config` to construct config, or
:meth:`~LegalAgentProductProfile.apply_to` on an existing :class:`LegalAgentConfig` (returns a copy with
SKU fields applied; the passed instance is not mutated).

Memory defaults per SKU set ``memory_policy``; override with ``make_config(..., memory_policy=...)`` or
partial dict accepted by :class:`LegalAgentConfig`. Injectable wiring (RAG, governance, budgets, tools)
remains the host's responsibility; explicit ``make_config`` / ``apply_to`` arguments win over SKU defaults. Typical order: ``profile.make_config(...)`` then
:func:`~intergrax.agents_packages.legal_agent.governance.legal_agent_governance_wiring.with_dual_legal_governance`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.memory.legal_memory_policy import LegalMemoryPolicyPresets
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy
from intergrax.runtime.nexus.session.session_manager import SessionManager


class LegalAgentProductProfile(StrEnum):
    """
    Product-tier preset identifier. Values are stable API strings (config, HTTP, logs).
    """

    STRICT_LEGAL = "strict_legal"
    RESEARCH = "research"
    FAST = "fast"
    SAFE = "safe"

    def _config_updates(self) -> dict[str, Any]:
        match self:
            case LegalAgentProductProfile.FAST:
                return {
                    "legal_loop_max_iterations": 2,
                    "use_legal_run_evaluator": False,
                    "use_legal_route_replanner": False,
                }
            case LegalAgentProductProfile.SAFE:
                return {
                    "organization_allow_websearch": False,
                    "organization_allow_tools": False,
                    "memory_policy": LegalMemoryPolicyPresets.minimal_exposure(),
                    "data_compliance": DataCompliancePolicy(
                        api_trace_export="none",
                        redact_tool_calls_in_api=True,
                    ),
                }
            case LegalAgentProductProfile.RESEARCH:
                return {
                    "organization_allow_rag": True,
                    "organization_allow_websearch": True,
                    "organization_allow_tools": True,
                    "legal_loop_max_iterations": 6,
                    "use_legal_run_evaluator": True,
                    "use_legal_route_replanner": True,
                    "data_compliance": DataCompliancePolicy(
                        api_trace_export="redacted",
                        redact_tool_calls_in_api=False,
                    ),
                }
            case LegalAgentProductProfile.STRICT_LEGAL:
                return {
                    "organization_allow_websearch": False,
                    "organization_allow_tools": False,
                    "legal_loop_max_iterations": 5,
                    "legal_loop_early_exit_min_confidence": 0.95,
                    "use_legal_run_evaluator": True,
                    "use_legal_route_replanner": True,
                    "use_llm_legal_route_planner": True,
                    "memory_policy": LegalMemoryPolicyPresets.strict_legal_workspace(),
                    "data_compliance": DataCompliancePolicy(
                        api_trace_export="redacted",
                        redact_tool_calls_in_api=True,
                    ),
                }

    def make_config(
        self,
        *,
        session_manager: SessionManager,
        llm_adapter: LLMAdapter,
        **overrides: Any,
    ) -> LegalAgentConfig:
        """
        Build :class:`LegalAgentConfig` with this SKU's defaults; ``overrides`` replace any same keys.
        """
        merged: dict[str, Any] = {**self._config_updates(), **overrides}
        return LegalAgentConfig(
            session_manager=session_manager,
            llm_adapter=llm_adapter,
            **merged,
        )

    def apply_to(self, config: LegalAgentConfig) -> LegalAgentConfig:
        """
        Return a new :class:`LegalAgentConfig` based on ``config`` with this SKU's settings applied.

        Does not mutate ``config``; uses :meth:`pydantic.BaseModel.model_copy` internally.
        """
        return config.model_copy(update=self._config_updates())
