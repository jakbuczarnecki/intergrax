# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import dataclasses
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import get_type_hints

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    DecisionBranchId,
    DecisionProposalRef,
    decision_lineage_ref,
    decision_proposal_ref,
)
from intergrax.contracts.decision_verification import (
    VerificationChallenge,
    VerificationDisposition,
    VerificationFinding,
    VerificationResult,
    VerificationStageOutcome,
    VerificationStageRecord,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_result,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_result,
    verification_stage_record,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_MODULE_PATH = Path(__file__).resolve().parents[3] / "intergrax" / "contracts" / "decision_verification.py"

_CANONICAL_DISPOSITIONS = (
    VerificationDisposition.PASSED,
    VerificationDisposition.CHALLENGED,
)

_FORBIDDEN_DISPOSITION_NAMES = frozenset(
    {
        "ACCEPTED",
        "REJECTED",
        "UNRESOLVED",
        "APPROVED",
        "AUTHORIZED",
        "RETRY",
        "REVISE",
        "ESCALATE_HITL",
    },
)

_FORBIDDEN_IMPORT_FRAGMENTS = (
    "runtime.nexus",
    "runtime.critic",
    "runtime.human",
    "runtime.policy",
    "runtime.governance",
    "runtime.execution",
)

_FORBIDDEN_FIELD_FRAGMENTS = (
    "authorization",
    "authorize",
    "hitl",
    "execution_request",
    "retry",
    "revise",
    "revision",
    "accepted",
    "rejected",
    "finalize",
    "chain_of_thought",
    "reasoning_trace",
    "private_reasoning",
)

_FORBIDDEN_PRODUCTION_PATTERNS = (
    r"\bAny\b",
    r"\bcast\b",
    r"type:\s*ignore",
    r"pyright:\s*ignore",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bhasattr\b",
    r"\binspect\b",
    r"\bexec\b",
    r"\beval\b",
    r"object\.__setattr__",
    r"dict\[str,\s*Any\]",
)

_CONTRACT_TYPES = (
    VerificationFinding,
    VerificationChallenge,
    VerificationStageRecord,
    VerificationResult,
)


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "scope",
    subject: str = "subject-1",
    version: DecisionVersion | None = None,
    decision_id: DecisionId | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _proposal_ref(
    *,
    identity: DecisionIdentity | None = None,
    branch_id: str = "analysis-a",
    version: DecisionVersion | None = None,
) -> DecisionProposalRef:
    resolved_identity = identity or _identity()
    resolved_version = version or resolved_identity.version
    return decision_proposal_ref(
        identity=resolved_identity,
        lineage_ref=decision_lineage_ref(
            resolved_version,
            DecisionBranchId(branch_id),
        ),
    )


def _finding(
    *,
    code: str = "schema.missing_field",
    message: str = "required field summary is missing",
) -> VerificationFinding:
    return verification_finding(
        code=validate_verification_finding_code(code),
        message=message,
    )


def _challenge(
    *,
    proposal_ref: DecisionProposalRef,
    stage: str = "structural_schema",
    requirement_code: str = "artifact.schema",
    finding: VerificationFinding | None = None,
) -> VerificationChallenge:
    return verification_challenge(
        proposal_ref=proposal_ref,
        stage=validate_verification_stage_kind(stage),
        requirement_code=validate_verification_requirement_code(requirement_code),
        finding=finding or _finding(),
    )


def _passed_stage(
    proposal_ref: DecisionProposalRef,
    stage: str = "structural_schema",
) -> VerificationStageRecord:
    return verification_stage_record(
        proposal_ref=proposal_ref,
        stage=validate_verification_stage_kind(stage),
        outcome=VerificationStageOutcome.PASSED,
    )


def _challenged_stage(
    proposal_ref: DecisionProposalRef,
    *,
    stage: str = "deterministic_rules",
) -> VerificationStageRecord:
    return verification_stage_record(
        proposal_ref=proposal_ref,
        stage=validate_verification_stage_kind(stage),
        outcome=VerificationStageOutcome.CHALLENGED,
        challenge=_challenge(proposal_ref=proposal_ref, stage=stage),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_verification_disposition_exact_set() -> None:
    assert tuple(VerificationDisposition) == _CANONICAL_DISPOSITIONS


@pytest.mark.unit
@pytest.mark.gate
def test_verification_disposition_excludes_lifecycle_and_authorization_values() -> None:
    member_names = frozenset(item.name for item in VerificationDisposition)
    assert member_names.isdisjoint(_FORBIDDEN_DISPOSITION_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_passed_result_for_exact_decision_proposal_ref() -> None:
    proposal_ref = _proposal_ref()
    result = verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.PASSED,
        stage_records=(
            _passed_stage(proposal_ref, "structural_schema"),
            _passed_stage(proposal_ref, "deterministic_rules"),
        ),
    )
    assert result.proposal_ref is proposal_ref
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 2
    revalidated = validate_verification_result(result)
    assert revalidated == result
    assert revalidated is not result


@pytest.mark.unit
@pytest.mark.gate
def test_valid_challenged_result() -> None:
    proposal_ref = _proposal_ref()
    challenged = _challenged_stage(proposal_ref)
    result = verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.CHALLENGED,
        stage_records=(_passed_stage(proposal_ref), challenged),
    )
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert challenged.challenge is not None
    assert challenged.challenge.proposal_ref is proposal_ref


@pytest.mark.unit
@pytest.mark.gate
def test_challenge_binds_exact_decision_version_and_branch() -> None:
    identity = _identity(version=DecisionVersion(2))
    proposal_ref = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="A")
    challenge = _challenge(proposal_ref=proposal_ref)
    assert challenge.proposal_ref.identity.decision_id == identity.decision_id
    assert challenge.proposal_ref.identity.version == DecisionVersion(2)
    assert challenge.proposal_ref.lineage_ref.branch_id == "A"


@pytest.mark.unit
@pytest.mark.gate
def test_challenge_for_v1_cannot_be_used_in_result_for_v2() -> None:
    identity = _identity(version=initial_decision_version())
    v1_ref = _proposal_ref(identity=identity, version=initial_decision_version())
    v2_identity = DecisionIdentity(
        decision_id=identity.decision_id,
        version=next_decision_version(initial_decision_version()),
        scope=identity.scope,
        tenant_id=identity.tenant_id,
        execution=identity.execution,
    )
    v2_ref = _proposal_ref(identity=v2_identity, version=v2_identity.version)
    challenged = verification_stage_record(
        proposal_ref=v1_ref,
        stage=validate_verification_stage_kind("deterministic_rules"),
        outcome=VerificationStageOutcome.CHALLENGED,
        challenge=_challenge(proposal_ref=v1_ref),
    )
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=v2_ref,
            disposition=VerificationDisposition.CHALLENGED,
            stage_records=(_passed_stage(v2_ref), challenged),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_sibling_branch_challenge_cannot_aggregate_for_other_branch() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="A")
    ref_b = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="B")
    challenged_for_a = _challenged_stage(ref_a)
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=ref_b,
            disposition=VerificationDisposition.CHALLENGED,
            stage_records=(_passed_stage(ref_b), challenged_for_a),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_mixed_decision_identity_across_stage_records_rejected() -> None:
    ref_a = _proposal_ref()
    ref_b = _proposal_ref()
    challenged = _challenged_stage(ref_a)
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=ref_b,
            disposition=VerificationDisposition.CHALLENGED,
            stage_records=(_passed_stage(ref_b), challenged),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_passed_stage_for_other_proposal_cannot_aggregate() -> None:
    ref_a = _proposal_ref()
    ref_b = _proposal_ref()
    passed_for_a = _passed_stage(ref_a)
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=ref_b,
            disposition=VerificationDisposition.PASSED,
            stage_records=(passed_for_a,),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_passed_stage_for_sibling_branch_cannot_aggregate_for_other_branch() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="A")
    ref_b = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="B")
    passed_for_a = _passed_stage(ref_a)
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=ref_b,
            disposition=VerificationDisposition.PASSED,
            stage_records=(passed_for_a,),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_passed_stage_for_v1_cannot_aggregate_for_v2() -> None:
    identity = _identity(version=initial_decision_version())
    v1_ref = _proposal_ref(identity=identity, version=initial_decision_version())
    v2_identity = DecisionIdentity(
        decision_id=identity.decision_id,
        version=next_decision_version(initial_decision_version()),
        scope=identity.scope,
        tenant_id=identity.tenant_id,
        execution=identity.execution,
    )
    v2_ref = _proposal_ref(identity=v2_identity, version=v2_identity.version)
    passed_for_v1 = _passed_stage(v1_ref)
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_result(
            proposal_ref=v2_ref,
            disposition=VerificationDisposition.PASSED,
            stage_records=(passed_for_v1,),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_challenged_stage_record_rejects_challenge_proposal_mismatch() -> None:
    ref_a = _proposal_ref()
    ref_b = _proposal_ref()
    with pytest.raises(ValueError, match="must match the evaluated Decision proposal reference"):
        verification_stage_record(
            proposal_ref=ref_b,
            stage=validate_verification_stage_kind("deterministic_rules"),
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=_challenge(proposal_ref=ref_a),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_passed_stage_with_challenge_rejected() -> None:
    proposal_ref = _proposal_ref()
    with pytest.raises(ValueError, match="PASSED outcome cannot include challenge"):
        verification_stage_record(
            proposal_ref=proposal_ref,
            stage=validate_verification_stage_kind("structural_schema"),
            outcome=VerificationStageOutcome.PASSED,
            challenge=_challenge(proposal_ref=proposal_ref),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_challenged_stage_without_challenge_rejected() -> None:
    proposal_ref = _proposal_ref()
    with pytest.raises(ValueError, match="CHALLENGED outcome requires challenge"):
        verification_stage_record(
            proposal_ref=proposal_ref,
            stage=validate_verification_stage_kind("structural_schema"),
            outcome=VerificationStageOutcome.CHALLENGED,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_overall_passed_with_challenged_stage_rejected() -> None:
    proposal_ref = _proposal_ref()
    with pytest.raises(ValueError, match="cannot include challenged stage records"):
        verification_result(
            proposal_ref=proposal_ref,
            disposition=VerificationDisposition.PASSED,
            stage_records=(_passed_stage(proposal_ref), _challenged_stage(proposal_ref)),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_overall_challenged_without_challenged_stage_rejected() -> None:
    proposal_ref = _proposal_ref()
    with pytest.raises(ValueError, match="requires at least one challenged stage record"):
        verification_result(
            proposal_ref=proposal_ref,
            disposition=VerificationDisposition.CHALLENGED,
            stage_records=(
                _passed_stage(proposal_ref),
                _passed_stage(proposal_ref, "deterministic_rules"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_empty_stage_records_rejected() -> None:
    proposal_ref = _proposal_ref()
    with pytest.raises(ValueError, match="must contain at least one stage record"):
        verification_result(
            proposal_ref=proposal_ref,
            disposition=VerificationDisposition.PASSED,
            stage_records=(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_challenge_has_no_authorization_or_lifecycle_semantics_fields() -> None:
    field_names = frozenset(
        name
        for contract_type in _CONTRACT_TYPES
        for name in (field.name for field in fields(contract_type))
    )
    lowered = frozenset(name.lower() for name in field_names)
    for fragment in _FORBIDDEN_FIELD_FRAGMENTS:
        assert not any(fragment in name for name in lowered), fragment


@pytest.mark.unit
@pytest.mark.gate
def test_contracts_are_immutable() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for contract_name in (
        "VerificationFinding",
        "VerificationChallenge",
        "VerificationStageRecord",
        "VerificationResult",
    ):
        marker = f"class {contract_name}"
        start = source.index(marker)
        decorator_region = source[max(0, start - 120) : start]
        assert "@dataclass(frozen=True, slots=True)" in decorator_region


@pytest.mark.unit
@pytest.mark.gate
def test_canonical_values_serialize_and_compare_deterministically() -> None:
    assert VerificationDisposition.PASSED.value == "passed"
    assert VerificationDisposition.CHALLENGED.value == "challenged"
    assert VerificationStageOutcome.PASSED.value == "passed"
    assert type(VerificationDisposition.PASSED) is VerificationDisposition


@pytest.mark.unit
@pytest.mark.gate
def test_strict_type_hints_on_core_contracts() -> None:
    result_hints = get_type_hints(VerificationResult)
    assert result_hints["proposal_ref"] is DecisionProposalRef
    assert result_hints["disposition"] is VerificationDisposition
    challenge_hints = get_type_hints(VerificationChallenge)
    assert challenge_hints["proposal_ref"] is DecisionProposalRef
    stage_record_hints = get_type_hints(VerificationStageRecord)
    assert stage_record_hints["proposal_ref"] is DecisionProposalRef


@pytest.mark.unit
@pytest.mark.gate
def test_import_boundary_no_forbidden_layers() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8").lower()
    for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
        assert fragment not in source, f"forbidden fragment {fragment}"


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_production_patterns_absent() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    hits = [
        pattern
        for pattern in _FORBIDDEN_PRODUCTION_PATTERNS
        if re.search(pattern, source)
    ]
    assert hits == [], f"forbidden production patterns found: {hits}"


@pytest.mark.unit
@pytest.mark.gate
def test_replace_preserves_value_semantics() -> None:
    proposal_ref = _proposal_ref()
    result = verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.PASSED,
        stage_records=(_passed_stage(proposal_ref),),
    )
    replaced = replace(result)
    assert replaced == result
    assert replaced is not result
