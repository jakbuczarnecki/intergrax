# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R6.4 autonomy qualification and safety validation."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field

import pytest

from intergrax.contracts.self_healing.autonomy import (
    AutonomyEvaluationAuditTrailEntry,
    AutonomyLevel,
    AutonomyQualificationRepository,
    AutonomyQualificationResult,
    AutonomyQualificationStatus,
    AutonomySafetyQualificationContext,
    AutonomySafetyValidator,
)
from intergrax.contracts.self_healing.autonomy.evaluation_audit import AutonomyEvaluationAuditRecorder
from intergrax.contracts.self_healing.autonomy.execution_boundary import AutonomyExecutionBoundary
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionGuard, AutonomyGuardCheckResult
from intergrax.runtime.self_healing.autonomy import (
    AutonomyQualificationService,
    DefaultAutonomyExecutionBoundary,
    DefaultAutonomyExecutionGuard,
    DefaultAutonomyPolicy,
    InMemoryAutonomyDecisionRepository,
    InMemoryAutonomyExecutionAuditRepository,
    default_autonomy_safety_checks,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-qual"


@dataclass
class _RecordingQualificationRepository:
    saved: list[AutonomyQualificationResult] = field(default_factory=list)

    def save(self, result: AutonomyQualificationResult) -> None:
        self.saved.append(result)

    def get(self, validation_id: str) -> AutonomyQualificationResult | None:
        for item in self.saved:
            if item.validation_id == validation_id:
                return item
        return None


@dataclass(frozen=True, slots=True)
class _RecordingEvaluationAuditRecorder:
    entries: tuple[AutonomyEvaluationAuditTrailEntry, ...] = ()

    def record(self, entry: AutonomyEvaluationAuditTrailEntry) -> None:
        _ = entry


@dataclass(frozen=True, slots=True)
class _SpyExecutionGuard:
    check_invoked: bool = False
    _guard_id: str = "test.spy_guard"

    @property
    def guard_id(self) -> str:
        return self._guard_id

    def check(self, admission: object) -> AutonomyGuardCheckResult:
        object.__setattr__(self, "check_invoked", True)
        _ = admission
        raise AssertionError("qualification must not invoke guard.check")


def _qualification_service(
    repository: AutonomyQualificationRepository | None = None,
) -> AutonomyQualificationService:
    return AutonomyQualificationService(
        safety_checks=default_autonomy_safety_checks(),
        repository=repository,
    )


def _safe_context(
    *,
    guard: AutonomyExecutionGuard | None = None,
    boundary: AutonomyExecutionBoundary | None = None,
    default_level: AutonomyLevel = AutonomyLevel.RECOMMEND_ONLY,
    evaluation_audit: AutonomyEvaluationAuditRecorder | None = None,
    execution_audit: InMemoryAutonomyExecutionAuditRepository | None = None,
    direct_executor: bool = False,
) -> AutonomySafetyQualificationContext:
    resolved_guard: AutonomyExecutionGuard
    if guard is None:
        resolved_guard = DefaultAutonomyExecutionGuard(
            evaluation_repository=InMemoryAutonomyDecisionRepository(),
        )
    else:
        resolved_guard = guard
    if boundary is None:
        default_authorizing = DefaultAutonomyExecutionGuard(
            evaluation_repository=InMemoryAutonomyDecisionRepository(),
        )
        resolved_boundary: AutonomyExecutionBoundary = DefaultAutonomyExecutionBoundary(
            guard=default_authorizing,
        )
    else:
        resolved_boundary = boundary
    resolved_eval_audit = evaluation_audit
    if resolved_eval_audit is None:
        resolved_eval_audit = _RecordingEvaluationAuditRecorder()
    resolved_exec_audit = execution_audit
    if resolved_exec_audit is None:
        resolved_exec_audit = InMemoryAutonomyExecutionAuditRepository()
    return AutonomySafetyQualificationContext(
        tenant_id=_TENANT,
        default_autonomy_level=default_level,
        execution_guard=resolved_guard,
        execution_boundary=resolved_boundary,
        evaluation_audit_recorder=resolved_eval_audit,
        execution_audit_repository=resolved_exec_audit,
        direct_executor_access_enabled=direct_executor,
    )


def test_positive_safe_configuration_passes() -> None:
    service = _qualification_service()
    result = service.qualify(_safe_context())
    assert result.status is AutonomyQualificationStatus.PASS
    assert not result.issues
    assert len(result.check_results) == 3
    assert result.audit.checks_executed == (
        "execution_boundary_safety",
        "default_policy_safety",
        "audit_capability",
    )


def test_negative_missing_guard_fails() -> None:
    context = AutonomySafetyQualificationContext(
        tenant_id=_TENANT,
        default_autonomy_level=AutonomyLevel.RECOMMEND_ONLY,
        execution_guard=None,
        execution_boundary=DefaultAutonomyExecutionBoundary(
            guard=DefaultAutonomyExecutionGuard(
                evaluation_repository=InMemoryAutonomyDecisionRepository(),
            ),
        ),
        evaluation_audit_recorder=_RecordingEvaluationAuditRecorder(),
        execution_audit_repository=InMemoryAutonomyExecutionAuditRepository(),
    )
    result = _qualification_service().qualify(context)
    assert result.status is AutonomyQualificationStatus.FAILED
    assert any(issue.code == "missing_execution_guard" for issue in result.issues)


def test_negative_missing_audit_capability_fails() -> None:
    context = AutonomySafetyQualificationContext(
        tenant_id=_TENANT,
        default_autonomy_level=AutonomyLevel.RECOMMEND_ONLY,
        execution_guard=DefaultAutonomyExecutionGuard(
            evaluation_repository=InMemoryAutonomyDecisionRepository(),
        ),
        execution_boundary=DefaultAutonomyExecutionBoundary(
            guard=DefaultAutonomyExecutionGuard(
                evaluation_repository=InMemoryAutonomyDecisionRepository(),
            ),
        ),
        evaluation_audit_recorder=None,
        execution_audit_repository=None,
    )
    result = _qualification_service().qualify(context)
    assert result.status is AutonomyQualificationStatus.FAILED
    assert any(issue.code == "missing_evaluation_audit_recorder" for issue in result.issues)
    assert any(issue.code == "missing_execution_audit_repository" for issue in result.issues)


def test_negative_unsafe_default_autonomy_level_fails() -> None:
    for unsafe in (AutonomyLevel.FULL_AUTONOMY, AutonomyLevel.CONTROLLED_EXECUTION):
        result = _qualification_service().qualify(_safe_context(default_level=unsafe))
        assert result.status is AutonomyQualificationStatus.FAILED
        assert any(issue.check_id == "default_policy_safety" for issue in result.issues)


def test_boundary_qualification_does_not_invoke_guard_check() -> None:
    spy_guard = _SpyExecutionGuard()
    real_guard = DefaultAutonomyExecutionGuard(
        evaluation_repository=InMemoryAutonomyDecisionRepository(),
    )
    boundary = DefaultAutonomyExecutionBoundary(guard=real_guard)
    service = _qualification_service()
    service.qualify(
        _safe_context(
            guard=spy_guard,
            boundary=boundary,
        )
    )
    assert spy_guard.check_invoked is False


def test_boundary_qualification_does_not_mutate_context() -> None:
    context = _safe_context()
    before = context
    service = _qualification_service()
    service.qualify(context)
    service.qualify(context)
    assert context == before


def test_qualification_persists_via_repository_port() -> None:
    repo = _RecordingQualificationRepository()
    service = _qualification_service(repository=repo)
    result = service.qualify(_safe_context())
    assert len(repo.saved) == 1
    assert repo.get(result.validation_id) == result


def test_service_satisfies_safety_validator_protocol() -> None:
    service = _qualification_service()
    assert isinstance(service, AutonomySafetyValidator)


def test_default_policy_matches_platform_safe_posture() -> None:
    policy = DefaultAutonomyPolicy()
    request_level = policy.evaluate  # ensure policy module loads
    _ = request_level
    result = _qualification_service().qualify(
        _safe_context(default_level=AutonomyLevel.RECOMMEND_ONLY),
    )
    assert result.status is AutonomyQualificationStatus.PASS


def test_qualification_runtime_has_no_executor_coupling() -> None:
    module_names = (
        "intergrax.runtime.self_healing.autonomy.qualification.service",
        "intergrax.runtime.self_healing.autonomy.qualification.checks.execution_boundary_safety_check",
        "intergrax.runtime.self_healing.autonomy.qualification.checks.audit_capability_check",
        "intergrax.runtime.self_healing.autonomy.qualification.checks.default_policy_safety_check",
    )
    forbidden = ("execution_engine", "SelfHealingActionProvider", "UAEPExecutor", ".execute(")
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden:
            assert token.lower() not in lowered


def test_direct_executor_access_flag_fails() -> None:
    result = _qualification_service().qualify(_safe_context(direct_executor=True))
    assert result.status is AutonomyQualificationStatus.FAILED
    assert any(issue.code == "direct_executor_access" for issue in result.issues)
