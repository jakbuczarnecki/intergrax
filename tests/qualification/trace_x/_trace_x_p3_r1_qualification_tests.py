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
from tests.qualification.trace_x._trace_x_p3_r1_support import (
    ENTERPRISE_AUDIT_MATRIX_P3_R1,
    MANDATORY_FRZ_P3_R1_IDS,
    R1GateResult,
    TRACE_X_P3_R1_START_HEAD,
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


def test_txp3r1_enterprise_audit_matrix_all_pass() -> None:
    assert all(row.result is R1GateResult.PASS for row in ENTERPRISE_AUDIT_MATRIX_P3_R1)


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
