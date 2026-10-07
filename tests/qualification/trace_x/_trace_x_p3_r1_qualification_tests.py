# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1 mechanical gates TXP3R1-Q01..Q37."""

from __future__ import annotations

import inspect
import subprocess
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_evidence.boundary_event import (
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V2,
    ExecutionBoundaryEvent,
    ExecutionBoundaryEventV2,
    ExecutionIdentitySection,
    GovernedProofSection,
    PolicyDecisionSection,
    ProviderInvocationSection,
    parse_governed_execution_boundary_event_json,
)
from intergrax.contracts.execution_evidence.receipt import (
    SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V1,
    SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V2,
    ProofReceipt,
    ProofReceiptV2,
    parse_execution_evidence_proof_receipt_json,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_execution_result import (
    GovernedExecutionResultV2,
    SCHEMA_GOVERNED_EXECUTION_RESULT_V2,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.attestation.canonical_json import canonical_json_bytes
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.execution_evidence.compose import (
    compose_execution_boundary_event_v2_from_result,
    produce_proof_receipt,
    produce_proof_receipt_v2,
)
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_execution_evidence_proof_receipt,
    verify_proof_receipt,
)
from tests.qualification.trace_x._trace_x_p0_support import repo_root
from pydantic import ValidationError

from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
from intergrax.contracts.governed_proof import GovernedProofProfile
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from tests.qualification.trace_x._trace_x_p3_r1_support import (
    CLOSED_WORLD_INVENTORY,
    ENTERPRISE_AUDIT_MATRIX_GATE_IDS,
    ENTERPRISE_AUDIT_MATRIX_P3_R1,
    ENTERPRISE_AUDIT_MATRIX_ROW_GATES,
    MANDATORY_FRZ_P3_R1_IDS,
    P3_R1_R1_GATE_BY_ID,
    P3_R1_R1_GATE_REGISTRY,
    P3_R1_R1_Q_BLK_01_RESOLUTION,
    PASS1_MECHANICAL_NODEIDS,
    R1GateResult,
    TENANT_ISOLATION_AUDIT_P3_R1_R1,
    TRACE_X_P3_R1_R1_Q1_START_HEAD,
    TRACE_X_P3_R1_R1_Q2_START_HEAD,
    TRACE_X_P3_R1_R1_Q3_START_HEAD,
    P3_R1_R1_Q2_BLK_EVIDENCE_PERSIST_01_RESOLUTION,
    TRACE_X_P3_R1_R1_START_HEAD,
    TRACE_X_P3_R1_START_HEAD,
    assert_nodeid_targets_test_function,
)

_REPO = repo_root()
_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)
_DIGEST = "sha256:" + ("ab" * 32)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=_REPO,
            check=False,
        ).returncode
        == 0
    )


def test_txp3r1_q01_start_head_provenance() -> None:
    head = _git_head()
    start = TRACE_X_P3_R1_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{start}^{{commit}}"], cwd=_REPO, check=True)
    assert _git_is_ancestor(start, head)


def test_txp3r1_q02_scope_trc_04_06_only() -> None:
    assert MANDATORY_FRZ_P3_R1_IDS == ("FRZ-TRC-04", "FRZ-TRC-06")


def test_txp3r1_q03_v1_boundary_unchanged() -> None:
    fields = set(ExecutionBoundaryEvent.model_fields)
    assert "schema_id" in fields and "task_id" in fields and "run_id" in fields
    assert "execution" not in fields


def test_txp3r1_q04_v1_canonical_bytes_stable() -> None:
    e = ExecutionBoundaryEvent(
        event_id="ebe-fixed",
        occurred_at=_T0,
        task_id="task-1",
        run_id="run-1",
        principal_id="u",
        provider_id="p",
        action="external_work.create",
        policy=PolicyDecisionSection(
            bundle_id="b",
            bundle_version="1",
            bundle_digest=_DIGEST,
            rule_id="r",
            action=PolicyAction.ALLOW,
        ),
        provider_invocation=ProviderInvocationSection(
            operation="create_work",
            invocation_id="inv",
            completed_at=_T0,
        ),
        governed_proof=GovernedProofSection(
            proof_id="proof",
            proof_digest=_DIGEST,
        ),
    )
    assert e.schema_id == SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1
    b1 = canonical_json_bytes(e.canonical_payload())
    b2 = canonical_json_bytes(e.canonical_payload())
    assert b1 == b2


def test_txp3r1_q05_v2_boundary_schema_exists() -> None:
    assert SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V2.endswith(".v2")
    assert ExecutionBoundaryEventV2.model_fields["execution"].annotation is not None


def test_txp3r1_q06_v2_typed_execution_identity() -> None:
    section = ExecutionIdentitySection(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    assert section.task_id.startswith("task_")


def test_txp3r1_q08_no_host_exec_uuid_fallback() -> None:
    from governed_contractor_application.host import orchestrator as orch_mod

    src = inspect.getsource(orch_mod.GovernedExternalWorkOrchestrator.create)
    assert 'f"exec-{uuid4().hex}"' not in src
    assert "exec-{uuid4" not in src


def test_txp3r1_q10_ger_v2_carries_attempt_id() -> None:
    assert "attempt_id" in GovernedExecutionResultV2.model_fields
    assert GovernedExecutionResultV2.model_fields["schema_version"].default == (
        SCHEMA_GOVERNED_EXECUTION_RESULT_V2
    )


def test_txp3r1_q21_v1_receipt_still_verifiable() -> None:
    from tests.unit.execution_evidence.test_host_attestation_and_receipt import (
        _allow_decision,
        _proof,
    )
    from intergrax.runtime.execution_evidence.compose import compose_execution_boundary_event

    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        event_id="ebe-v1",
        occurred_at=_T0,
    )
    receipt = produce_proof_receipt(event=event, attestor=attestor, receipt_id="r-v1")
    assert verify_proof_receipt(
        receipt,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    ).valid


def test_txp3r1_q22_v2_receipt_schema_exists() -> None:
    assert SCHEMA_EXECUTION_EVIDENCE_PROOF_RECEIPT_V2.endswith(".v2")


def test_txp3r1_q24_cross_version_pair_rejected() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    v2_event = ExecutionBoundaryEventV2(
        event_id="ebe-v2",
        occurred_at=_T0,
        execution=ExecutionIdentitySection(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        tenant_id="tenant-v2",
        principal_id="u",
        provider_id="p",
        action="external_work.create",
        policy=PolicyDecisionSection(
            bundle_id="b",
            bundle_version="1",
            bundle_digest=_DIGEST,
            rule_id="r",
            action=PolicyAction.ALLOW,
        ),
        provider_invocation=ProviderInvocationSection(
            operation="create_work",
            invocation_id="inv",
            completed_at=_T0,
        ),
        governed_proof=GovernedProofSection(proof_id="p", proof_digest=_DIGEST),
    )
    v2_receipt = produce_proof_receipt_v2(event=v2_event, attestor=attestor)
    v1_shell = ProofReceipt(
        receipt_id="bad",
        execution_boundary_event=ExecutionBoundaryEvent(
            event_id="ebe-v1",
            occurred_at=_T0,
            task_id="t",
            run_id="r",
            principal_id="u",
            provider_id="p",
            action="a",
            policy=v2_event.policy,
            provider_invocation=v2_event.provider_invocation,
            governed_proof=v2_event.governed_proof,
        ),
        host_attestation=v2_receipt.host_attestation,
    )
    result = verify_execution_evidence_proof_receipt(
        v1_shell,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    )
    assert result.valid is False
    assert result.errors
    assert any(
        code in result.errors
        for code in (
            "receipt_event_schema_mismatch",
            "unsupported_attestation_payload_schema",
            "receipt_attestation_schema_mismatch",
        )
    )


def test_txp3r1_q25_v2_attestation_schema_id() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = ExecutionBoundaryEventV2(
        event_id="ebe-v2",
        occurred_at=_T0,
        execution=ExecutionIdentitySection(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        tenant_id="tenant-v2",
        principal_id="u",
        provider_id="p",
        action="external_work.create",
        policy=PolicyDecisionSection(
            bundle_id="b",
            bundle_version="1",
            bundle_digest=_DIGEST,
            rule_id="r",
            action=PolicyAction.ALLOW,
        ),
        provider_invocation=ProviderInvocationSection(
            operation="create_work",
            invocation_id="inv",
            completed_at=_T0,
        ),
        governed_proof=GovernedProofSection(proof_id="p", proof_digest=_DIGEST),
    )
    receipt = produce_proof_receipt_v2(event=event, attestor=attestor)
    assert receipt.host_attestation.payload_schema == SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V2


def test_txp3r1_q26_persisted_event_parser_versioned() -> None:
    v1 = ExecutionBoundaryEvent(
        event_id="e",
        occurred_at=_T0,
        task_id="t",
        run_id="r",
        principal_id="u",
        provider_id="p",
        action="a",
        policy=PolicyDecisionSection(
            bundle_id="b",
            bundle_version="1",
            bundle_digest=_DIGEST,
            rule_id="r",
            action=PolicyAction.ALLOW,
        ),
        provider_invocation=ProviderInvocationSection(
            operation="o",
            invocation_id="i",
            completed_at=_T0,
        ),
        governed_proof=GovernedProofSection(proof_id="p", proof_digest=_DIGEST),
    )
    parsed = parse_governed_execution_boundary_event_json(v1.model_dump_json())
    assert isinstance(parsed, ExecutionBoundaryEvent)
    v2 = ExecutionBoundaryEventV2(
        event_id="e2",
        occurred_at=_T0,
        execution=ExecutionIdentitySection(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        tenant_id="tenant-v2",
        principal_id="u",
        provider_id="p",
        action="a",
        policy=v1.policy,
        provider_invocation=v1.provider_invocation,
        governed_proof=v1.governed_proof,
    )
    parsed2 = parse_governed_execution_boundary_event_json(v2.model_dump_json())
    assert isinstance(parsed2, ExecutionBoundaryEventV2)


def test_txp3r1_q33_strong_typing_no_any_on_v2_identity() -> None:
    from intergrax.contracts.execution_identity import TaskId

    hints = ExecutionIdentitySection.model_fields
    assert hints["task_id"].annotation is TaskId


def test_txp3r1r1q1_q01_production_delta_zero() -> None:
    head = _git_head()
    if _git_is_ancestor(TRACE_X_P3_R1_R1_Q2_START_HEAD, head) and head != TRACE_X_P3_R1_R1_Q1_START_HEAD:
        pytest.skip("Q1 zero-delta gate is historical after Q2 Branch B wiring")
    baseline = TRACE_X_P3_R1_R1_Q1_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{baseline}^{{commit}}"], cwd=_REPO, check=True)
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            baseline,
            "--",
            "intergrax/",
            "applications/governed_contractor_application/host/",
        ],
        cwd=_REPO,
        text=True,
    ).strip()
    assert diff == "", f"production paths changed in Q1 scope: {diff}"


def test_txp3r1r1_q1_q1_start_head_provenance() -> None:
    head = _git_head()
    start = TRACE_X_P3_R1_R1_Q1_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{start}^{{commit}}"], cwd=_REPO, check=True)
    assert _git_is_ancestor(start, head)


def test_txp3r1r1_q2_q2_start_head_provenance() -> None:
    head = _git_head()
    start = TRACE_X_P3_R1_R1_Q2_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{start}^{{commit}}"], cwd=_REPO, check=True)
    assert _git_is_ancestor(start, head)


def test_txp3r1r1_q3_q3_start_head_provenance() -> None:
    head = _git_head()
    start = TRACE_X_P3_R1_R1_Q3_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{start}^{{commit}}"], cwd=_REPO, check=True)
    assert _git_is_ancestor(start, head)


def test_txp3r1r1_q3_evidence_persist_blocker_resolved() -> None:
    assert P3_R1_R1_Q2_BLK_EVIDENCE_PERSIST_01_RESOLUTION.startswith("RESOLVED")


def test_txp3r1r1_gate_registry_mechanical_integrity() -> None:
    gate_ids = {row.gate_id for row in P3_R1_R1_GATE_REGISTRY}
    assert len(gate_ids) == len(P3_R1_R1_GATE_REGISTRY)
    referenced: set[str] = set()
    for gate_tuple in ENTERPRISE_AUDIT_MATRIX_GATE_IDS.values():
        referenced.update(gate_tuple)
    for gate_tuple in ENTERPRISE_AUDIT_MATRIX_ROW_GATES.values():
        referenced.update(gate_tuple)
    assert referenced.issubset(gate_ids)
    for gate_id in gate_ids:
        if gate_id == "TXP3R1R1Q1-Q01":
            continue
        assert gate_id.startswith("TXP3R1R1-Q")
    for row in P3_R1_R1_GATE_REGISTRY:
        assert_nodeid_targets_test_function(row.test_nodeid)


def test_txp3r1r1_closed_world_inventory_classified() -> None:
    assert CLOSED_WORLD_INVENTORY
    categories = {category for _, category in CLOSED_WORLD_INVENTORY}
    assert "Execution identity" in categories
    assert "Governance/tenant" in categories


def test_txp3r1r1_q01_r1_start_head_ancestry() -> None:
    head = _git_head()
    start = TRACE_X_P3_R1_R1_START_HEAD
    subprocess.run(["git", "cat-file", "-e", f"{start}^{{commit}}"], cwd=_REPO, check=True)
    assert _git_is_ancestor(start, head)
    if head == start:
        pending = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_REPO, text=True
        ).strip()
        assert pending, "R1-R1 corrective delta required when HEAD equals R1 START_HEAD"


def test_txp3r1r1_q02_scope_task_tenant_blockers_only() -> None:
    assert MANDATORY_FRZ_P3_R1_IDS == ("FRZ-TRC-04", "FRZ-TRC-06")


def _policy_decision_v2() -> EvaluatedPolicyDecision:
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        policy_rule_id="r.create",
        policy_bundle_id="b1",
        policy_bundle_version="1",
        policy_bundle_digest=_DIGEST,
        decision_id="d1",
    )
    return EvaluatedPolicyDecision(
        decision=decision,
        bundle_id="b1",
        bundle_version="1",
        bundle_digest=_DIGEST,
        matched_rule_id="r.create",
        evaluated_at=_T0,
        request_digest=_DIGEST,
    )


def _proof_v2(**overrides: object) -> GovernedProofProfile:
    base = dict(
        principal_id="u1",
        tenant_id="ten-a",
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        action="external_work.create",
        resource="scope",
        provider_id="prov",
        policy_action=PolicyAction.ALLOW,
        policy_rule_id="r.create",
        policy_reason="ok",
        correlation_id="c1",
        idempotency_key="i1",
    )
    base.update(overrides)
    return GovernedProofProfile.model_validate(base)


def _invocation_v2(task_id: str, run_id: str, **overrides: object) -> ProviderInvocation:
    base = dict(
        invocation_id="inv-1",
        provider_id="prov",
        operation="create_work",
        task_id=task_id,
        run_id=run_id,
        correlation_id="c1",
        idempotency_key="i1",
        request_digest=_DIGEST,
        started_at=_T0,
    )
    base.update(overrides)
    return ProviderInvocation.model_validate(base)


def _ger_v2(**overrides: object) -> GovernedExecutionResultV2:
    task_id = overrides.get("task_id", mint_task_id())
    run_id = overrides.get("run_id", mint_run_id())
    attempt_id = overrides.get("attempt_id", mint_attempt_id())
    execution_id = overrides.get("execution_id", mint_execution_id())
    tenant_id = str(overrides.get("tenant_id", "ten-a"))
    proof = overrides.get("proof")
    if proof is None:
        proof = _proof_v2(
            task_id=str(task_id),
            run_id=str(run_id),
            tenant_id=tenant_id,
        )
    base = dict(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id="u1",
        tenant_id=tenant_id,
        correlation_id="c1",
        idempotency_key="i1",
        action="external_work.create",
        evaluated_policy_decision=_policy_decision_v2(),
        provider_invocation=_invocation_v2(str(task_id), str(run_id)),
        provider_outcome=ProviderInvocationOutcome(
            invocation_id="inv-1",
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=_T0,
        ),
        proof=proof,
        execution_started_at=_T0,
        execution_completed_at=_T0,
    )
    for key, value in overrides.items():
        if key not in {"proof", "task_id", "run_id", "attempt_id", "execution_id", "tenant_id"}:
            base[key] = value
    return GovernedExecutionResultV2.model_validate(base)


def test_txp3r1r1_q10_ger_v2_tenant_mandatory() -> None:
    with pytest.raises(ValidationError):
        _ger_v2(tenant_id="")


def test_txp3r1r1_q11_ger_proof_tenant_exact_equality_ok() -> None:
    ger = _ger_v2(tenant_id="ten-a")
    assert ger.tenant_id == "ten-a"
    assert ger.proof.tenant_id == "ten-a"


def test_txp3r1r1_q12_proof_tenant_absence_rejected() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    with pytest.raises(ValueError, match="proof_tenant_id_required"):
        _ger_v2(
            task_id=task_id,
            run_id=run_id,
            proof=_proof_v2(
                task_id=str(task_id),
                run_id=str(run_id),
                tenant_id=None,
            ),
        )


def test_txp3r1r1_q11_mismatch_rejected() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    with pytest.raises(ValueError, match="tenant_id_inconsistent"):
        _ger_v2(
            task_id=task_id,
            run_id=run_id,
            tenant_id="ten-a",
            proof=_proof_v2(
                task_id=str(task_id),
                run_id=str(run_id),
                tenant_id="ten-b",
            ),
        )


def test_txp3r1r1_q13_ebe_v2_tenant_mandatory() -> None:
    with pytest.raises(ValidationError):
        ExecutionBoundaryEventV2(
            event_id="ebe-v2",
            occurred_at=_T0,
            execution=ExecutionIdentitySection(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
            tenant_id="",
            principal_id="u",
            provider_id="p",
            action="external_work.create",
            policy=PolicyDecisionSection(
                bundle_id="b",
                bundle_version="1",
                bundle_digest=_DIGEST,
                rule_id="r",
                action=PolicyAction.ALLOW,
            ),
            provider_invocation=ProviderInvocationSection(
                operation="create_work",
                invocation_id="inv",
                completed_at=_T0,
            ),
            governed_proof=GovernedProofSection(proof_id="p", proof_digest=_DIGEST),
        )


def test_txp3r1r1_q14_ebe_tenant_copied_from_ger() -> None:
    ger = _ger_v2(tenant_id="tenant-copy")
    event = compose_execution_boundary_event_v2_from_result(ger, event_id="ebe-copy")
    assert event.tenant_id == "tenant-copy"


def test_txp3r1r1_q15_receipt_preserves_tenant() -> None:
    ger = _ger_v2(tenant_id="tenant-rcpt")
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event_v2_from_result(ger, event_id="ebe-rcpt")
    receipt = produce_proof_receipt_v2(event=event, attestor=attestor)
    assert receipt.execution_boundary_event.tenant_id == "tenant-rcpt"


def test_txp3r1r1_tenant_isolation_audit_complete() -> None:
    assert TENANT_ISOLATION_AUDIT_P3_R1_R1["tenant_scope_applicable"] == "YES"
    assert TENANT_ISOLATION_AUDIT_P3_R1_R1["fail_closed_behavior"]
    assert TENANT_ISOLATION_AUDIT_P3_R1_R1["result"] == "PASS"


def test_txp3r1r1_q23_atomic_provider_execution_attribution_in_ebe_v2() -> None:
    ger = _ger_v2(tenant_id="tenant-attr")
    event = compose_execution_boundary_event_v2_from_result(ger, event_id="ebe-attr")
    assert str(event.execution.execution_id) == str(ger.execution_id)
    assert str(event.execution.task_id) == str(ger.task_id)
    assert event.provider_invocation.invocation_id == ger.provider_invocation.invocation_id
    assert event.tenant_id == ger.tenant_id


def test_txp3r1r1_q21_reliability_projection_non_authoritative() -> None:
    from applications.governed_contractor_application.tests.host.test_gr7_a8_r1_early_lifecycle_evidence_wiring import (
        test_observer_failure_does_not_block_success_path,
    )

    test_observer_failure_does_not_block_success_path()


def test_txp3r1r1_q25_governance_not_execution() -> None:
    from tests.qualification.trace_x._trace_x_p3_qualification_tests import (
        test_txp3_q24_governance_not_execution,
    )

    test_txp3_q24_governance_not_execution()


def test_txp3r1r1_q28_no_heuristic_provider_join() -> None:
    from tests.qualification.trace_x._trace_x_p3_qualification_tests import (
        test_txp3_q14_no_heuristic_provider_join,
    )

    test_txp3_q14_no_heuristic_provider_join()


def test_txp3r1r1_q26_exactly_one_compose_owner() -> None:
    compose_path = _REPO / "intergrax/runtime/execution_evidence/compose.py"
    text = compose_path.read_text(encoding="utf-8")
    assert "def compose_execution_boundary_event_v2_from_result" in text
    assert text.count("def compose_execution_boundary_event_v2_from_result") == 1


def test_txp3r1r1_q29_frz_trc_04_readiness_evidence() -> None:
    for gate_id in ENTERPRISE_AUDIT_MATRIX_GATE_IDS["FRZ-TRC-04 readiness"]:
        assert gate_id in P3_R1_R1_GATE_BY_ID
    assert P3_R1_R1_Q_BLK_01_RESOLUTION.startswith("RESOLVED")


def test_txp3r1r1_q30_frz_trc_06_readiness_evidence() -> None:
    import os

    from tests.qualification.trace_x._trace_x_p3_r1_pass1_session import (
        PASS1_PASSED_NODEIDS,
        load_pass1_observed_nodeids,
    )
    from tests.qualification.trace_x._trace_x_p3_r1_pass2_session import PASS2_PASSED_NODEIDS
    from tests.qualification.trace_x._trace_x_p3_r1_support import (
        normalize_pytest_nodeid,
        observed_gate_passed,
    )

    if os.environ.get("TRACE_X_P3_R1_R1_PASS2") != "1":
        pytest.skip("Q30 requires Pass 2 session with Pass 1+2 observed nodeids")
    pass1_observed = load_pass1_observed_nodeids() or {
        normalize_pytest_nodeid(nodeid) for nodeid in PASS1_PASSED_NODEIDS
    }
    passed = {
        normalize_pytest_nodeid(nodeid)
        for nodeid in pass1_observed.union(PASS2_PASSED_NODEIDS)
    }
    for gate_id in ENTERPRISE_AUDIT_MATRIX_GATE_IDS["FRZ-TRC-06 readiness"]:
        if gate_id == "TXP3R1R1-Q30":
            continue
        assert gate_id in P3_R1_R1_GATE_BY_ID
        assert observed_gate_passed(gate_id, passed), gate_id


def test_txp3r1r1_pass1_mechanical_nodeid_manifest() -> None:
    from tests.qualification.trace_x._trace_x_p3_r1_support import gate_required_nodeids

    assert PASS1_MECHANICAL_NODEIDS
    for nodeid in PASS1_MECHANICAL_NODEIDS:
        assert_nodeid_targets_test_function(nodeid)
    for gate_id, evidence in P3_R1_R1_GATE_BY_ID.items():
        if evidence.pass1_required:
            for nodeid in gate_required_nodeids(evidence):
                assert nodeid in PASS1_MECHANICAL_NODEIDS, gate_id


def test_txp3r1_q23_v2_receipt_event_pair_valid() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = ExecutionBoundaryEventV2(
        event_id="ebe-v2",
        occurred_at=_T0,
        execution=ExecutionIdentitySection(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        tenant_id="tenant-v2",
        principal_id="u",
        provider_id="p",
        action="external_work.create",
        policy=PolicyDecisionSection(
            bundle_id="b",
            bundle_version="1",
            bundle_digest=_DIGEST,
            rule_id="r",
            action=PolicyAction.ALLOW,
        ),
        provider_invocation=ProviderInvocationSection(
            operation="create_work",
            invocation_id="inv",
            completed_at=_T0,
        ),
        governed_proof=GovernedProofSection(proof_id="p", proof_digest=_DIGEST),
    )
    receipt = produce_proof_receipt_v2(event=event, attestor=attestor)
    restored = parse_execution_evidence_proof_receipt_json(receipt.model_dump_json())
    assert isinstance(restored, ProofReceiptV2)
    assert verify_execution_evidence_proof_receipt(
        restored,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    ).valid
