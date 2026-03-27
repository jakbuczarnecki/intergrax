# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Product observability contract on :class:`~intergrax.runtime.nexus.responses.response_schema.RouteInfo.extra`.

Hosts aggregate SLA/billing signals from this versioned bucket without scraping full ``trace_events``.
"""

from __future__ import annotations

from typing import Any, Dict

from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


class LegalProductObservability:
    """Stable keys and payload shape for legal run aggregates (Etap 7)."""

    ROUTE_EXTRA_KEY = "legal_product_obs_v1"
    SCHEMA_ID = "intergrax.legal.product_obs.route_extra.v1"

    @staticmethod
    def build_route_extra_payload(
        *,
        agent_state: LegalAgentState,
        state: RuntimeState,
        finalize_empty_fallback: bool,
    ) -> Dict[str, Any]:
        plan = agent_state.last_legal_tool_plan
        tool_plan_obs: Dict[str, Any] | None = None
        if plan is not None:
            tool_plan_obs = {
                "intent": plan.intent,
                "use_rag": plan.use_rag,
                "use_tools": plan.use_tools,
                "use_websearch": plan.use_websearch,
            }
        return {
            "schema": LegalProductObservability.SCHEMA_ID,
            "loop_waves": agent_state.legal_dynamic_loop_waves,
            "finalize_empty_fallback": finalize_empty_fallback,
            "clause_retrieval_outcome": agent_state.clause_extraction_retrieval_outcome,
            "evaluator_degraded": agent_state.legal_run_evaluator_degraded,
            "tool_plan_post_governance": tool_plan_obs,
            "nexus_flags": {
                "used_rag": state.used_rag,
                "used_tools": state.used_tools,
                "used_websearch": state.used_websearch,
            },
        }
