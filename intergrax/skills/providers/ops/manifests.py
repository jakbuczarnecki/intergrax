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

OPS_LOG_TAIL = SkillManifest(
    skill_id="ops.log_tail",
    version="1.0.0",
    description="Live log tailing with search and error capture for incident response.",
    tool_ids=("logs.tail", "logs.search", "errors.capture"),
    prompt_instruction_ids=("ops.log_tail.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "logs", "tail"),
)


OPS_INCIDENT_ACK = SkillManifest(
    skill_id="ops.incident_ack",
    version="1.0.0",
    description="PagerDuty incident acknowledge with trigger and notify escalation path.",
    tool_ids=("pagerduty.acknowledge_incident", "pagerduty.trigger_incident", "notify.send"),
    prompt_instruction_ids=("ops.incident_ack.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("ops", "incident", "pagerduty"),
)

OPS_ONCALL_RUNBOOK = SkillManifest(
    skill_id="ops.oncall_runbook",
    version="1.0.0",
    description="On-call runbook: logs, traces, and stakeholder notification.",
    tool_ids=("logs.search", "observability.query_traces", "notify.send"),
    prompt_instruction_ids=("ops.oncall_runbook.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "oncall", "runbook"),
)


OPS_POSTMORTEM_WRITER = SkillManifest(
    skill_id="ops.postmortem_writer",
    version="1.0.0",
    description="Postmortem drafting from harness run metadata and logs.",
    tool_ids=("harness.get_run", "logs.search", "workspace.write_file"),
    prompt_instruction_ids=("ops.postmortem_writer.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "postmortem", "writer"),
)


OPS_CHANGE_APPROVER = SkillManifest(
    skill_id="ops.change_approver",
    version="1.0.0",
    description="Change approval loop: HITL pending, notify, workflow poll.",
    tool_ids=("hitl.list_pending", "notify.send", "workflow.poll"),
    prompt_instruction_ids=("ops.change_approver.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("ops", "change", "approval"),
)


OPS_CAPACITY_PLANNER = SkillManifest(
    skill_id="ops.capacity_planner",
    version="1.0.0",
    description="Capacity planning from metrics, cost forecast, and run history.",
    tool_ids=("metrics.query_range", "cost.forecast_spend", "harness.list_runs"),
    prompt_instruction_ids=("ops.capacity_planner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ops", "capacity", "planner"),
)

