# © Artur Czarnecki. All rights reserved.

"""MP-4R3 — canonical ExecutionContinuationPort boundary; no Multiplayer continuation authority."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from intergrax.contracts.decision_human_review import DecisionHumanReviewOutcome
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_engine_continuation_dependencies,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MULTIPLAYER_PRODUCTION_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "intergrax" / "collaborative_work",
)
_FORBIDDEN_NEXUS_PREFIXES = ("intergrax.runtime.nexus",)
_FORBIDDEN_NEXUS_SYMBOLS = frozenset(
    {"GraphExecutor", "NexusLoop", "NexusIntakeRunner"},
)
_FORBIDDEN_CONTINUATION_AUTHORITY_PREFIXES = (
    "MultiplayerContinuation",
    "CollaborativeContinuation",
    "ApprovalContinuation",
    "DecisionContinuation",
    "CollaborativeExecutionState",
    "ApprovalResumeState",
    "DecisionResumeState",
)
_FORBIDDEN_CONTINUATION_STORE_CLASS_NAMES = frozenset(
    {
        "MultiplayerContinuationRepository",
        "CollaborativeContinuationRepository",
        "ApprovalContinuationStore",
        "DecisionContinuationStore",
        "MultiplayerContinuationStore",
    },
)
_FORBIDDEN_EXECUTION_MUTATION_IMPORT_PREFIXES = (
    "intergrax.runtime.execution.continuation",
    "intergrax.runtime.execution.active_execution_continuation",
    "intergrax.runtime.nexus.orchestration.internal_continuation_orchestration",
)
_CANONICAL_CONTINUATION_CONTRACT_PREFIX = "intergrax.contracts.execution_continuation"
_APPROVAL_RESUME_SHORTCUT = re.compile(
    r"DecisionHumanReviewOutcome\.APPROVED[\s\S]{0,400}?\.resume\s*\(",
    re.MULTILINE,
)
_SEED = "mp4r3-continuation-gate"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_mp4r3_gate"
_SCOPE_DIGEST = "sha256:" + "c" * 64


def _production_modules(roots: Iterable[Path]) -> list[Path]:
    modules: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        modules.extend(sorted(path for path in root.rglob("*.py") if path.is_file()))
    return modules


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_class_definitions(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return [
        (node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]


def _nexus_import_violations(modules: list[Path]) -> list[str]:
    violations: list[str] = []
    for module_path in modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, module in _collect_imports(module_path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_NEXUS_PREFIXES
            ):
                violations.append(f"{rel}:{lineno} imports {module}")
            for symbol in _FORBIDDEN_NEXUS_SYMBOLS:
                if symbol in module:
                    violations.append(f"{rel}:{lineno} imports {module}")
    return violations


def _identity(**overrides: object) -> ExecutionContinuationIdentity:
    payload = {
        "task_id": _TASK,
        "run_id": _RUN,
        "attempt_id": _ATTEMPT,
        "execution_id": _EXECUTION,
    }
    payload.update(overrides)
    return ExecutionContinuationIdentity(**payload)  # type: ignore[arg-type]


def _governed_correlation(**overrides: object) -> GovernedContinuationCorrelation:
    payload: dict[str, object] = {
        "continuation_request_id": _CONTINUATION_ID,
        "reason": ContinuationReason.SECURITY,
        "task_id": _TASK,
        "run_id": _RUN,
        "attempt_id": _ATTEMPT,
        "execution_id": _EXECUTION,
        "operation_id": "op_mp4r3",
        "side_effect_scope_id": "scope_mp4r3",
    }
    payload.update(overrides)
    return GovernedContinuationCorrelation.model_validate(payload)


def _pause_request(**overrides: object) -> ExecutionPauseRequest:
    payload: dict[str, object] = {
        "identity": _identity(),
        "continuation_id": _CONTINUATION_ID,
        "reason": ContinuationReason.SECURITY,
        "governed_correlation": _governed_correlation(),
        "pause_id": "pause_mp4r3",
        "human_request_id": "hr_mp4r3",
    }
    payload.update(overrides)
    return ExecutionPauseRequest.model_validate(payload)


def _resolution_command(
    *,
    expected_revision: int,
    verdict: ExecutionHumanVerdict = ExecutionHumanVerdict.APPROVE,
    **overrides: object,
) -> ExecutionContinuationResolutionCommand:
    payload: dict[str, object] = {
        "continuation_id": _CONTINUATION_ID,
        "identity": _identity(),
        "expected_revision": expected_revision,
        "verdict": verdict,
        "approver": local_development_approver_evidence(
            actor_id="op-mp4r3",
            tenant_id="tenant-mp4r3",
        ),
        "human_request_id": "hr_mp4r3",
        "pause_id": "pause_mp4r3",
        "operation_id": "op_mp4r3",
        "side_effect_scope_id": "scope_mp4r3",
        "resolved_at": "2026-09-16T01:00:00Z",
    }
    payload.update(overrides)
    return ExecutionContinuationResolutionCommand.model_validate(payload)


def _assert_four_ids_unchanged(pending: PendingExecutionContinuation) -> None:
    assert pending.identity.task_id == _TASK
    assert pending.identity.run_id == _RUN
    assert pending.identity.attempt_id == _ATTEMPT
    assert pending.identity.execution_id == _EXECUTION


def _drive_to_waiting(
    deps: ExecutionEngineContinuationDependencies,
) -> PendingExecutionContinuation:
    port = deps.continuation
    driver = deps.lifecycle_driver
    port.request_pause(_pause_request())
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    assert waiting.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    return waiting


@pytest.fixture(name="multiplayer_production_modules")
def fixture_multiplayer_production_modules() -> list[Path]:
    return _production_modules(_MULTIPLAYER_PRODUCTION_ROOTS)


def test_mp4r3_multiplayer_production_has_no_public_nexus_dependency(
    multiplayer_production_modules: list[Path],
) -> None:
    violations = _nexus_import_violations(multiplayer_production_modules)
    assert not violations, "\n".join(violations)


def test_mp4r3_no_duplicate_continuation_lifecycle_authority(
    multiplayer_production_modules: list[Path],
) -> None:
    violations: list[str] = []
    for module_path in multiplayer_production_modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, name in _collect_class_definitions(module_path):
            if any(name.startswith(prefix) for prefix in _FORBIDDEN_CONTINUATION_AUTHORITY_PREFIXES):
                violations.append(f"{rel}:{lineno} defines forbidden continuation authority {name}")
    assert not violations, "\n".join(violations)


def test_mp4r3_no_multiplayer_continuation_repository(
    multiplayer_production_modules: list[Path],
) -> None:
    violations: list[str] = []
    for module_path in multiplayer_production_modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, name in _collect_class_definitions(module_path):
            if name in _FORBIDDEN_CONTINUATION_STORE_CLASS_NAMES:
                violations.append(f"{rel}:{lineno} defines forbidden store {name}")
    assert not violations, "\n".join(violations)


def test_mp4r3_contract_only_continuation_dependency(
    multiplayer_production_modules: list[Path],
) -> None:
    violations: list[str] = []
    for module_path in multiplayer_production_modules:
        rel = module_path.relative_to(_REPO_ROOT)
        modules = [module for _, module in _collect_imports(module_path)]
        if not any(
            module == _CANONICAL_CONTINUATION_CONTRACT_PREFIX
            or module.startswith(f"{_CANONICAL_CONTINUATION_CONTRACT_PREFIX}.")
            for module in modules
        ):
            continue
        for module in modules:
            for prefix in _FORBIDDEN_EXECUTION_MUTATION_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(
                        f"{rel} mixes contract continuation with forbidden runtime import {module}",
                    )
    assert not violations, "\n".join(violations)


def test_mp4r3_no_direct_execution_continuation_mutation_imports(
    multiplayer_production_modules: list[Path],
) -> None:
    violations: list[str] = []
    for module_path in multiplayer_production_modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, module in _collect_imports(module_path):
            for prefix in _FORBIDDEN_EXECUTION_MUTATION_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{rel}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4r3_no_approval_to_resume_shortcut_in_multiplayer(
    multiplayer_production_modules: list[Path],
) -> None:
    violations: list[str] = []
    for module_path in multiplayer_production_modules:
        text = module_path.read_text(encoding="utf-8-sig")
        if _APPROVAL_RESUME_SHORTCUT.search(text):
            rel = module_path.relative_to(_REPO_ROOT)
            violations.append(f"{rel} couples DecisionHumanReviewOutcome.APPROVED to resume()")
        if "apply_resolution(" in text and "DecisionHumanReviewOutcome" in text:
            rel = module_path.relative_to(_REPO_ROOT)
            violations.append(f"{rel} mixes decision human review with continuation resolution")
    assert not violations, "\n".join(violations)


def test_mp4r3_multiplayer_has_no_production_continuation_port_caller(
    multiplayer_production_modules: list[Path],
) -> None:
    """Caller audit: collaborative_work does not invoke continuation lifecycle today."""
    markers = (
        "ExecutionContinuationPort",
        "request_pause(",
        "apply_resolution(",
        "advance_to_paused",
        "advance_to_human_wait",
        "cancel_continuation(",
    )
    hits: list[str] = []
    for module_path in multiplayer_production_modules:
        text = module_path.read_text(encoding="utf-8-sig")
        rel = module_path.relative_to(_REPO_ROOT)
        for marker in markers:
            if marker in text:
                hits.append(f"{rel}: {marker}")
    assert not hits, "\n".join(hits)


def test_mp4r3_canonical_continuation_contract_importable() -> None:
    from intergrax.contracts.execution_continuation import PendingExecutionContinuation
    from intergrax.contracts.governed_continuation_correlation import (
        GovernedContinuationCorrelation,
    )

    assert ExecutionContinuationPort is not None
    assert PendingExecutionContinuation is not None
    assert GovernedContinuationCorrelation is not None


def test_mp4r3_human_review_outcome_is_not_resumed_lifecycle_state() -> None:
    resumed_values = {member.value for member in ExecutionContinuationLifecycleState}
    assert DecisionHumanReviewOutcome.APPROVED.value not in resumed_values
    assert ExecutionContinuationLifecycleState.RESUMED.value == "resumed"


def test_mp4r3_two_phase_resume_qualification() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    authorized = port.apply_resolution(_resolution_command(expected_revision=waiting.revision))
    assert authorized.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    assert authorized.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED
    resumed = port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=authorized.revision,
        ),
    )
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED
    _assert_four_ids_unchanged(resumed)


def test_mp4r3_stale_revision_qualification() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    with pytest.raises(ExecutionContinuationError) as stale:
        port.apply_resolution(_resolution_command(expected_revision=waiting.revision - 1))
    assert stale.value.code is ExecutionContinuationErrorCode.STALE_REVISION


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", canonical_task_id_for_tests("mp4r3-other-task")),
        ("run_id", canonical_run_id_for_tests("mp4r3-other-run")),
        ("attempt_id", mint_attempt_id()),
        ("execution_id", mint_execution_id()),
    ],
)
def test_mp4r3_identity_mismatch_qualification(field: str, value: object) -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    with pytest.raises(ExecutionContinuationError) as exc:
        port.apply_resolution(
            _resolution_command(
                expected_revision=waiting.revision,
                identity=_identity(**{field: value}),
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "verdict,terminal",
    [
        (ExecutionHumanVerdict.REJECT, ExecutionContinuationLifecycleState.REJECTED),
        (ExecutionHumanVerdict.ESCALATE, ExecutionContinuationLifecycleState.ESCALATED),
    ],
)
def test_mp4r3_terminal_verdict_blocks_resume(
    verdict: ExecutionHumanVerdict,
    terminal: ExecutionContinuationLifecycleState,
) -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    resolved = port.apply_resolution(
        _resolution_command(expected_revision=waiting.revision, verdict=verdict),
    )
    assert resolved.lifecycle_state is terminal
    with pytest.raises(ExecutionContinuationError):
        port.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=resolved.revision,
            ),
        )


def test_mp4r3_governed_correlation_mismatch_fail_closed() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _drive_to_waiting(deps)
    port = deps.continuation
    with pytest.raises(ExecutionContinuationError) as exc:
        port.apply_resolution(
            _resolution_command(
                expected_revision=waiting.revision,
                operation_id="wrong-operation",
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.SCOPE_MISMATCH

    deps_digest = wire_execution_engine_continuation_dependencies()
    port_digest = deps_digest.continuation
    driver_digest = deps_digest.lifecycle_driver
    digest_correlation = _governed_correlation(side_effect_scope_digest=_SCOPE_DIGEST)
    port_digest.request_pause(_pause_request(governed_correlation=digest_correlation))
    driver_digest.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting_digest = driver_digest.record_ready_for_human_resolution(_CONTINUATION_ID)
    with pytest.raises(ExecutionContinuationError) as digest_exc:
        port_digest.apply_resolution(
            _resolution_command(
                expected_revision=waiting_digest.revision,
                side_effect_scope_digest="sha256:" + "d" * 64,
            ),
        )
    assert digest_exc.value.code is ExecutionContinuationErrorCode.SCOPE_MISMATCH
