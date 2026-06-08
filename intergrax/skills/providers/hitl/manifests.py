# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

HITL_APPROVAL_GATE = SkillManifest(
    skill_id="hitl.approval_gate",
    version="1.0.0",
    description="Human-in-the-loop approval: list pending decisions, submit responses, notify stakeholders.",
    tool_ids=(
        "hitl.list_pending",
        "hitl.submit_response",
        "hitl.get_decision",
        "notify.send",
    ),
    prompt_instruction_ids=("hitl.approval_gate.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("hitl", "approval", "governance"),
)
