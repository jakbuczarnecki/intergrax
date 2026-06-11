# © Artur Czarnecki. All rights reserved.

"""Compliance rollup from Plane B trace (ACP-ORG-4)."""

from __future__ import annotations

from intergrax.contracts.agent_run import ComplianceSummary
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.runtime_policy import PolicyAction


def build_compliance_summary(trace: AgentRunTrace) -> ComplianceSummary:
    deny_count = 0
    warn_count = 0
    rules: set[str] = set()
    for step in trace.steps:
        for verdict in step.policy_verdicts:
            rules.add(verdict.policy_rule_id)
            if verdict.action == PolicyAction.DENY:
                deny_count += 1
            elif verdict.action in {PolicyAction.MODIFY, PolicyAction.ESCALATE}:
                warn_count += 1
    return ComplianceSummary(
        deny_count=deny_count,
        warn_count=warn_count,
        rules_triggered=sorted(rules),
    )
