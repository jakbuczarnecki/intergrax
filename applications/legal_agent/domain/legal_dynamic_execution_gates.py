# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Gating rules for :mod:`~legal_agent.pipeline.legal_execution_loop`."""

from __future__ import annotations

from typing import Set

from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.domain.legal_agent_state import LegalAgentState


class LegalDynamicLoopGates:
    """Pure policy checks for dynamic pipeline control flow (early exit, evaluator)."""

    @staticmethod
    def post_wave_early_exit_ok(
        agent_state: LegalAgentState,
        completed: Set[str],
        config: LegalAgentConfig,
    ) -> bool:
        """
        Gates :attr:`~legal_agent.config.legal_agent_config.LegalAgentConfig.legal_loop_early_exit`.

        Contract text lives on ``config.failure_policy`` (low confidence, ESCALATE, blocking issues, violations).
        """
        if "run_decision" not in completed:
            return False
        pol_v = agent_state.policy_violations
        if pol_v and len(pol_v) > 0:
            return False
        d = agent_state.decision
        if d is None:
            return False
        if d.status == "ESCALATE":
            return False
        if d.confidence < config.legal_loop_early_exit_min_confidence:
            return False
        if d.blocking_issues:
            return False
        return True
