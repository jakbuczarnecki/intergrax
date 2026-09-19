# © Artur Czarnecki. All rights reserved.

"""HARNESS-02 — cancellation, deadline & cooperative abort propagation catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Harness02FlowStatus = Literal[
    "CANONICAL",
    "AUTHORIZED_SPECIAL_CASE",
    "GAP",
    "NOT_APPLICABLE",
]

FindingSeverity = Literal[
    "BLOCKER",
    "NON-BLOCKING DEBT",
    "INTENTIONAL DESIGN",
    "FALSE POSITIVE",
]

HARNESS_02_REQUIRED_FLOW_IDS: frozenset[str] = frozenset(
    {
        "H02-root-deadline",
        "H02-child-deadline",
        "H02-grandchild-deadline",
        "H02-retry-deadline",
        "H02-retry-cancellation",
        "H02-tool-pre-effect",
        "H02-llm-provider-boundary",
        "H02-integration-external-operation",
        "H02-background-execution",
        "H02-redelivery-resume",
        "H02-parallel-child-cancellation",
    }
)


@dataclass(frozen=True, slots=True)
class Harness02EvidenceRef:
    pytest_node_id: str


@dataclass(frozen=True, slots=True)
class Harness02PropagationRow:
    flow_id: str
    source: str
    boundary: str
    propagated_state: str
    enforcement: str
    terminal_result: str
    evidence: str
    status: Harness02FlowStatus


@dataclass(frozen=True, slots=True)
class Harness02FindingRow:
    finding: str
    severity: FindingSeverity
    flow: str
    consequence: str
    required_action: str


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_W1 = "tests/unit/runtime/execution/test_enterprise_scale_resilience_w1.py"
_R1 = "tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py"
_P0 = "tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py"
_W4A = "tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_a_cancellation_qualification.py"
_P0C5 = "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py"
_H02 = "tests/qualification/harness_02/test_harness_02_gates.py"
_H02R1 = "tests/unit/runtime/execution/deadline_authority/test_harness_02_r1_qualification.py"
_TOOL_ADM = "tests/unit/runtime/execution/test_tool_protected_work_admission.py"
_H01 = "tests/qualification/harness_01/test_harness_01_gates.py"
_UE9 = "tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py"


HARNESS_02_PROPAGATION_MATRIX: tuple[Harness02PropagationRow, ...] = (
    Harness02PropagationRow(
        flow_id="H02-root-deadline",
        source="bind_root_execution_budget",
        boundary="root ExecutionRuntime",
        propagated_state="ActiveExecutionBudgetState.global_deadline_monotonic",
        enforcement="mint once per bind; retry reads peek_active_execution_global_deadline_monotonic",
        terminal_result="DEADLINE_EXCEEDED / global_deadline_exceeded on retry",
        evidence=_nid(_W1, "test_root_execution_global_deadline_fixed_at_bind"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-child-deadline",
        source="parent ActiveExecutionBudgetState",
        boundary="ChildExecutionRunner",
        propagated_state="inherited global_deadline_monotonic (no extension)",
        enforcement="inheritance only; no pre-admission monotonic check",
        terminal_result="n/a at child gate",
        evidence=_nid(_H02, "test_harness_02_child_inherits_parent_global_deadline"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-grandchild-deadline",
        source="root deadline D",
        boundary="nested ChildExecutionRunner",
        propagated_state="same D at child and grandchild",
        enforcement="inheritance only",
        terminal_result="n/a at child gate",
        evidence=_nid(_H02, "test_harness_02_grandchild_preserves_root_global_deadline"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-retry-deadline",
        source="active execution budget",
        boundary="NexusGraphRunner → ExecutionAttemptRetryService",
        propagated_state="global_deadline_monotonic + now_monotonic + backoff",
        enforcement="evaluate_execution_retry_eligibility fail-closed",
        terminal_result="FAIL global_deadline_exceeded",
        evidence=_nid(_R1, "test_r1_global_deadline_blocks_retry_when_backoff_crosses"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-retry-cancellation",
        source="CancellationCoordinator on task.metadata",
        boundary="graph retry eligibility",
        propagated_state="cancelled=True on ExecutionRetryEligibilityRequest",
        enforcement="evaluate_execution_retry_eligibility → CANCEL",
        terminal_result="no retry transition",
        evidence=_nid(_R1, "test_r1_cancel_before_retry"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-tool-pre-effect",
        source="ExecutionProtectedWorkAdmissionPort + CancellationCoordinator",
        boundary="RuntimeToolInvoker.invoke",
        propagated_state="active deadline projection + cooperative cancel",
        enforcement="admission before idempotency; cancel before first attempt",
        terminal_result="deadline_exceeded / task_cancelled (NOT_STARTED)",
        evidence=_nid(_TOOL_ADM, "test_q09_expired_tool_blocked_before_executor"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-llm-provider-boundary",
        source="ExecutionProtectedWorkAdmissionPort",
        boundary="llm_adapters/contracts/llm_adapter.py",
        propagated_state="active scope projection; provider timeout min(configured, remaining)",
        enforcement="assert_protected_provider_call_allowed before physical provider call",
        terminal_result="ExecutionProtectedWorkDeniedError / bounded timeout",
        evidence=_nid(_H02R1, "test_q10_expired_blocks_llm_execute"),
        status="CANONICAL",
    ),
    Harness02PropagationRow(
        flow_id="H02-integration-external-operation",
        source="external operation attempt records",
        boundary="runtime/external_operations",
        propagated_state="cancellation port + unknown side effect classification",
        enforcement="attempt admission separate from execution global_deadline_monotonic",
        terminal_result="typed physical state / reconciliation",
        evidence="tests/unit/runtime/external_operations (existing suites)",
        status="AUTHORIZED_SPECIAL_CASE",
    ),
    Harness02PropagationRow(
        flow_id="H02-background-execution",
        source="ExecutionRuntime / host task",
        boundary="canonical host port",
        propagated_state="ContextVar budget when bound on same worker",
        enforcement="depends on launch copying context",
        terminal_result="host shutdown vs execution cancel distinct",
        evidence="tests/qualification/harness_01/catalog.py execution.background_host_task",
        status="AUTHORIZED_SPECIAL_CASE",
    ),
    Harness02PropagationRow(
        flow_id="H02-redelivery-resume",
        source="durable RunBudget ledger",
        boundary="worker redelivery / checkpoint resume",
        propagated_state="ledger dimensional remaining; monotonic deadline reminted on bind",
        enforcement="UE-9AR1 token/tool remaining; wall SLA not monotonic-durable",
        terminal_result="retry blocked; resume wall clock may reset with new RuntimeState",
        evidence=_nid(_UE9, "test_attempt_two_gets_remaining_budget_after_attempt_one_consumption"),
        status="GAP",
    ),
    Harness02PropagationRow(
        flow_id="H02-parallel-child-cancellation",
        source="root task.metadata cancellation",
        boundary="GraphExecutor batch loop",
        propagated_state="CancellationCoordinator.is_requested",
        enforcement="mark_pending_graph_nodes_cancelled before next batch",
        terminal_result="graph cancelled / SKIPPED nodes",
        evidence=_nid(_P0C5, "test_graph_runner_persists_terminal_cancellation_before_cleanup"),
        status="CANONICAL",
    ),
)


HARNESS_02_FINDINGS: tuple[Harness02FindingRow, ...] = (
    Harness02FindingRow(
        finding="global_deadline_monotonic is not checked before child admission, tool invoke, or LLM call",
        severity="BLOCKER",
        flow="H02-tool-pre-effect / H02-child-deadline",
        consequence="expired monotonic authority can still start new protected work on paths without enforce_wall_time_budget",
        required_action="contract-driven pre-effect gate using remaining execution deadline (ADR on unification with BudgetEnforcer wall clock)",
    ),
    Harness02FindingRow(
        finding="RuntimeToolInvoker checks cooperative cancellation only for attempt>1 inside _execute_with_policy",
        severity="BLOCKER",
        flow="H02-tool-pre-effect",
        consequence="first physical tool attempt may run after cancellation_requested",
        required_action="check CancellationCoordinator (or should_cancel) in _prepare_invocation before side effect",
    ),
    Harness02FindingRow(
        finding="enforce_wall_time_budget is not called from record_tool_call_and_enforce; only iterative tool loop (iter>=1)",
        severity="NON-BLOCKING DEBT",
        flow="H02-tool-pre-effect",
        consequence="wall-time SLA uneven across tool entry paths",
        required_action="align tool_loop / catalog_dispatch with single pre-effect deadline checkpoint",
    ),
    Harness02FindingRow(
        finding="Runtime wall-time enforcement uses started_at_utc wall clock; global_deadline uses monotonic at bind",
        severity="NON-BLOCKING DEBT",
        flow="H02-root-deadline",
        consequence="clock skew can diverge the two authorities",
        required_action="single monotonic remaining-time contract at enforcement checkpoints",
    ),
    Harness02FindingRow(
        finding="Durable redelivery preserves ledger counters but remints process-local global_deadline_monotonic from full RunBudget",
        severity="BLOCKER",
        flow="H02-redelivery-resume",
        consequence="worker restart/resume can grant fresh monotonic wall allowance inconsistent with global SLA",
        required_action="durable absolute deadline or remaining-wall contract on resume (architecture decision)",
    ),
    Harness02FindingRow(
        finding="CancellationCoordinator remains task-metadata coordination; canonical execution cancel flows through graph_runner + terminal store",
        severity="INTENTIONAL DESIGN",
        flow="H02-parallel-child-cancellation",
        consequence="metadata flag alone insufficient without executor/invoker checkpoints",
        required_action="document + extend enforcement points (see blockers above)",
    ),
    Harness02FindingRow(
        finding="In-flight blocking vendor calls cannot be hard-interrupted without provider support",
        severity="INTENTIONAL DESIGN",
        flow="H02-integration-external-operation",
        consequence="unknown side effect possible until timeout",
        required_action="classify via external_operations reconciliation (existing)",
    ),
)


HARNESS_02_CANCELLATION_SURFACES: tuple[dict[str, str], ...] = (
    {
        "mechanism": "CancellationCoordinator",
        "request_owner": "operator/host via task_control",
        "runtime_authority": "task.metadata cooperative flag",
        "propagation": "propagate(); graph_executor batch checks",
        "terminalization": "graph_runner._handle_cancellation + ExecutionTerminalService",
        "evidence": "runtime/cancellation/coordinator.py",
    },
    {
        "mechanism": "CooperativeCancellationAbort / cooperative_delay_seconds",
        "request_owner": "should_abort callback",
        "runtime_authority": "retry/backoff loops (tool invoker, LLM resilience)",
        "propagation": "poll during sleep",
        "terminalization": "abort retry loop",
        "evidence": _nid(_W4A, "test_cooperative_delay_aborts_when_requested"),
    },
    {
        "mechanism": "asyncio.Task.cancel / host shutdown",
        "request_owner": "host/runtime",
        "runtime_authority": "separate from task cancellation metadata",
        "propagation": "async cancellation",
        "terminalization": "CancelledError / host drain",
        "evidence": _nid(_W1, "test_bounded_concurrent_work_cancellation_propagates"),
    },
    {
        "mechanism": "ExternalOperationCancellationPort",
        "request_owner": "execution + provider bridge",
        "runtime_authority": "external_operations/*",
        "propagation": "per operation_id",
        "terminalization": "mark_observed_cancellation_terminal",
        "evidence": "runtime/external_operations/operation_termination.py",
    },
)
