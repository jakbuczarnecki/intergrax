# © Artur Czarnecki. All rights reserved.

"""MP-4R2 — single canonical human judgment; legacy Approval authority retired."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_APPROVAL_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "approval.py"
_LEGACY_APPROVAL_PACKAGE = _REPO_ROOT / "intergrax" / "approval"
_MULTIPLAYER_ROOT = _REPO_ROOT / "intergrax" / "collaborative_work"
_FORBIDDEN_LEGACY_IMPORT_PREFIXES = (
    "intergrax.contracts.approval",
    "intergrax.approval",
)
_FORBIDDEN_OPERATION_IDS = (
    "approval.request.create",
    "approval.action.execute",
)
_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)


def _production_python_files() -> list[Path]:
    skip_parts = ("docker/runtime-context", "__pycache__", "tests")
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if any(part in rel for part in skip_parts):
                continue
            files.append(path)
    return files


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_mp4r2_legacy_approval_contract_removed() -> None:
    assert not _LEGACY_APPROVAL_CONTRACT.is_file()


def test_mp4r2_legacy_approval_package_removed() -> None:
    assert not _LEGACY_APPROVAL_PACKAGE.exists()


def test_mp4r2_no_production_imports_of_legacy_approval() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_modules(path):
            for prefix in _FORBIDDEN_LEGACY_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{rel} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4r2_no_orphaned_approval_operation_ids_in_production() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for op_id in _FORBIDDEN_OPERATION_IDS:
            if op_id in text:
                violations.append(f"{rel} references {op_id!r}")
    assert not violations, "\n".join(violations)


def test_mp4r2_multiplayer_does_not_define_approval_hitl_authority() -> None:
    if not _MULTIPLAYER_ROOT.is_dir():
        pytest.skip("collaborative_work root missing")
    forbidden_names = (
        "ApprovalService",
        "ApprovalRuntime",
        "MultiplayerHitlEngine",
        "HumanReviewRuntime",
        "ApprovalRepository",
        "MultiplayerApprovalRepository",
    )
    violations: list[str] = []
    for path in _MULTIPLAYER_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for name in forbidden_names:
            if f"class {name}" in text:
                violations.append(f"{rel} defines {name}")
    assert not violations, "\n".join(violations)


def test_mp4r2_canonical_human_review_contracts_importable() -> None:
    from intergrax.contracts.decision_human_review import (
        DecisionHumanReviewOutcome,
        DecisionHumanReviewPort,
        DecisionHumanReviewRequest,
        validate_human_review_decision_for_proposal,
    )

    assert DecisionHumanReviewRequest is not None
    assert DecisionHumanReviewOutcome is not None
    assert DecisionHumanReviewPort is not None
    assert validate_human_review_decision_for_proposal is not None


def test_mp4r2_stale_human_review_decision_rejected_for_revised_proposal() -> None:
    from dataclasses import dataclass

    from intergrax.contracts.decision_human_review import (
        DecisionHumanReviewDecision,
        DecisionHumanReviewMismatchError,
        DecisionHumanReviewOutcome,
        DecisionHumanReviewProvenance,
        DecisionHumanReviewRequest,
        decision_human_review_decision,
        decision_human_review_request,
        validate_human_review_decision_for_proposal,
        verification_challenged_human_review_reason,
    )
    from intergrax.contracts.decision_identity import (
        DecisionExecutionLineage,
        DecisionIdentity,
        DecisionScope,
        initial_decision_version,
        mint_decision_id,
        next_decision_version,
    )
    from intergrax.contracts.decision_record import (
        CandidateDecision,
        DecisionArtifact,
        DecisionProposalRef,
        candidate_decision_ref,
        decision_lineage_ref,
        decision_version_lineage,
        validate_decision_artifact_kind,
        validate_decision_branch_id,
    )
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.contracts.human_approver import local_development_approver_evidence

    @dataclass(frozen=True, slots=True)
    class Payload:
        text: str

    def _execution_lineage() -> DecisionExecutionLineage:
        return DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )

    def _candidate() -> CandidateDecision[Payload]:
        identity = DecisionIdentity(
            decision_id=mint_decision_id(),
            version=initial_decision_version(),
            scope=DecisionScope(namespace="demo", subject="case-1"),
            tenant_id="tenant-a",
            execution=_execution_lineage(),
        )
        lineage = decision_version_lineage(
            current=decision_lineage_ref(
                identity.version,
                validate_decision_branch_id("main"),
            ),
        )
        return CandidateDecision(
            identity=identity,
            artifact=DecisionArtifact(
                kind=validate_decision_artifact_kind("demo.payload"),
                content=Payload(text="draft"),
            ),
            lineage=lineage,
        )

    def _request(proposal_ref: DecisionProposalRef) -> DecisionHumanReviewRequest:
        return decision_human_review_request(
            proposal_ref=proposal_ref,
            reason_code=verification_challenged_human_review_reason(),
        )

    def _decision(request: DecisionHumanReviewRequest) -> DecisionHumanReviewDecision:
        return decision_human_review_decision(
            request=request,
            outcome=DecisionHumanReviewOutcome.APPROVED,
            approver=local_development_approver_evidence(
                tenant_id=request.proposal_ref.identity.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_test123",
                human_request_id=str(request.request_id),
            ),
        )

    candidate_v1 = _candidate()
    approval_v1 = _decision(_request(candidate_decision_ref(candidate_v1)))
    validate_human_review_decision_for_proposal(
        decision=approval_v1,
        proposal_ref=candidate_decision_ref(candidate_v1),
    )
    candidate_v2 = CandidateDecision(
        identity=DecisionIdentity(
            decision_id=candidate_v1.identity.decision_id,
            version=next_decision_version(candidate_v1.identity.version),
            scope=candidate_v1.identity.scope,
            tenant_id=candidate_v1.identity.tenant_id,
            execution=candidate_v1.identity.execution,
        ),
        artifact=candidate_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(next_decision_version(candidate_v1.identity.version)),
            parents=(candidate_decision_ref(candidate_v1).lineage_ref,),
        ),
    )
    with pytest.raises(DecisionHumanReviewMismatchError):
        validate_human_review_decision_for_proposal(
            decision=approval_v1,
            proposal_ref=candidate_decision_ref(candidate_v2),
        )


def test_mp4r2_human_review_outcome_is_not_execution_authorization_type() -> None:
    from intergrax.contracts.decision_authorization import DecisionExecutionAuthorization
    from intergrax.contracts.decision_human_review import DecisionHumanReviewOutcome

    assert DecisionHumanReviewOutcome is not DecisionExecutionAuthorization
    outcome_values = {member.value for member in DecisionHumanReviewOutcome}
    assert "authorized" not in outcome_values
    assert "denied" not in outcome_values


@pytest.mark.asyncio
async def test_mp4r2_governance_require_human_leaves_authorization_none() -> None:
    from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
    from intergrax.contracts.decision_revision import decision_revision_policy
    from intergrax.contracts.decision_record import validate_decision_artifact_kind
    from intergrax.runtime.decision_flow import (
        CanonicalDecisionFlowGate,
        DecisionFlowGateCapabilities,
        DecisionFlowHostAction,
        DecisionFlowRequest,
        DecisionFlowScope,
    )
    from intergrax.runtime.execution.active_decision_lifecycle_host import (
        bind_active_decision_lifecycle_host,
        reset_active_decision_lifecycle_host,
    )
    from intergrax.runtime.execution.decision_lifecycle_host import (
        CanonicalDecisionLifecycleHost,
    )
    from tests.unit.runtime.test_decision_flow import (
        Payload,
        RecordingHumanReviewPort,
        RequireHumanGovernanceEvaluator,
        PassedStage,
        _governance_spec,
        _identity_seed,
        _pipeline,
        evaluator_spec_action,
        evaluator_spec_policy,
    )

    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    try:
        port = RecordingHumanReviewPort()
        gate = CanonicalDecisionFlowGate(
            capabilities=DecisionFlowGateCapabilities(
                verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
                revision_policy=decision_revision_policy(max_revisions=0),
                scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
                governance_spec=_governance_spec(
                    RequireHumanGovernanceEvaluator(
                        action=evaluator_spec_action(),
                        policy_context=evaluator_spec_policy(),
                    ),
                ),
                human_review_port=port,
            ),
        )
        result = await gate.evaluate(
            DecisionFlowRequest(
                identity_seed=_identity_seed(),
                artifact_kind=validate_decision_artifact_kind("test.payload"),
                payload=Payload(text="ok"),
                flow_scope=DecisionFlowScope.UAEP_STEP,
            ),
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert result.host_action is DecisionFlowHostAction.PENDING_HUMAN
    assert result.authorization is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION
    assert port.pending is not None
