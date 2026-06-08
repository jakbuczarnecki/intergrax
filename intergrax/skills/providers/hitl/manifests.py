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

HITL_QUEUE_MANAGER = SkillManifest(
    skill_id="hitl.queue_manager",
    version="1.0.0",
    description="HITL queue operations: list task decisions, summarize queue depth, and list pending items.",
    tool_ids=("hitl.list_for_task", "hitl.summarize_queue", "hitl.list_pending"),
    prompt_instruction_ids=("hitl.queue_manager.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("hitl", "queue", "operations"),
)
