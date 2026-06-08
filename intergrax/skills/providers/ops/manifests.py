# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

OPS_TRACE_DEBUG = SkillManifest(
    skill_id="ops.trace_debug",
    version="1.0.0",
    description="Harness trace and log debugging for agent run investigation.",
    tool_ids=("observability.query_traces", "logs.search", "errors.capture"),
    prompt_instruction_ids=("ops.trace_debug.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "observability", "debug"),
)

OPS_INCIDENT_DISPATCH = SkillManifest(
    skill_id="ops.incident_dispatch",
    version="1.0.0",
    description="On-call incident trigger with log context and outbound notification.",
    tool_ids=("pagerduty.trigger_incident", "notify.send", "logs.search"),
    prompt_instruction_ids=("ops.incident_dispatch.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("ops", "incident", "pagerduty"),
)

OPS_SECURITY_AUDIT = SkillManifest(
    skill_id="ops.security_audit",
    version="1.0.0",
    description="Security scan workspace artifacts with search and alert dispatch.",
    tool_ids=("security.scan", "workspace.search", "notify.send"),
    prompt_instruction_ids=("ops.security_audit.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("ops", "security", "audit"),
)

OPS_WORKFLOW_RUNNER = SkillManifest(
    skill_id="ops.workflow_runner",
    version="1.0.0",
    description="Trigger and monitor batch eval or RAG refresh workflow runs.",
    tool_ids=("workflow.trigger", "workflow.poll", "workflow.fetch_logs"),
    prompt_instruction_ids=("ops.workflow_runner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "workflow", "orchestration"),
)

OPS_WORKFLOW_ADMIN = SkillManifest(
    skill_id="ops.workflow_admin",
    version="1.0.0",
    description="Workflow run administration: list runs, cancel in-flight work, and fetch logs.",
    tool_ids=("workflow.list_runs", "workflow.cancel_run", "workflow.fetch_logs"),
    prompt_instruction_ids=("ops.workflow_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "workflow", "admin"),
)

OPS_FINDINGS_REVIEW = SkillManifest(
    skill_id="ops.findings_review",
    version="1.0.0",
    description="Security findings review: scan artifacts, summarize results, and notify owners.",
    tool_ids=("security.summarize_findings", "security.scan", "notify.send"),
    prompt_instruction_ids=("ops.findings_review.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("ops", "security", "findings"),
)
