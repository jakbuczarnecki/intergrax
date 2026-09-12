# © Artur Czarnecki. All rights reserved.

"""EE-A2-H3 — Identity Authority frozen plane integration & consumer-only certification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_identity_authority import (
    CANONICAL_ATTEMPT_LIFECYCLE_MODULE,
    CANONICAL_IDENTITY_AUTHORITY_MODULE,
    CANONICAL_LIFECYCLE_OWNER_MODULE,
    ExecutionIdentityAuthorityPort,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.identity_authority import (
    DefaultExecutionIdentityAuthority,
    default_execution_identity_authority,
)
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_IDENTITY_AUTHORITY_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_IDENTITY_AUTHORITY_MODEL.md"
)
_RUNTIME_EVENT = _REPO_ROOT / "intergrax" / "runtime" / "events" / "runtime_event.py"
_CHILD_RUNNER = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_PARTIAL_RECOVERY = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "fan_out_partial_recovery.py"
)
_RETRY_SERVICE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "retry" / "service.py"
_LONG_RUNNING_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
_COORDINATOR = _LONG_RUNNING_ROOT / "coordinator.py"
_RUNTIME_CHECKPOINT = _LONG_RUNNING_ROOT / "runtime_checkpoint.py"

_EVIDENCE_PLANE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "events"
_RECOVERY_PLANE_ROOTS = (
    _LONG_RUNNING_ROOT,
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "retry",
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "attempt_lifecycle",
    _PARTIAL_RECOVERY,
)
_CHECKPOINT_PLANE_ROOTS = (
    _LONG_RUNNING_ROOT,
)
_LINEAGE_PLANE_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "lineage",
    _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "execution_lineage_reconstruction.py",
    _REPO_ROOT / "intergrax" / "contracts" / "execution_lineage.py",
)
_GOVERNANCE_PLANE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "governance"

_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
    }
)

_AUTHORITY_DELEGATE_CALLS = frozenset(
    {
        "mint_root_execution_identity",
        "mint_child_execution_id",
        "mint_retry_attempt_id",
    }
)

_FORBIDDEN_PARALLEL_IDENTITY_SERVICES = frozenset(
    {
        "SecondIdentityService",
        "IdentityManager",
        "IdentityFactory",
        "ExecutionIdProvider",
    }
)

_REQUIRED_DOC_SECTIONS = (
    "## Global Freeze Statement",
    "## Frozen Plane Integration Contract",
    "## Identity ownership",
)

_FROZEN_PLANE_AUDIT_ROOTS = (
    _EVIDENCE_PLANE_ROOT,
    _LONG_RUNNING_ROOT,
    _REPO_ROOT / "intergrax" / "runtime" / "execution",
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "lineage",
    _GOVERNANCE_PLANE_ROOT,
    _REPO_ROOT / "intergrax" / "contracts",
)

_MINT_ALLOWED_UNDER_EXECUTION = frozenset(
    {
        CANONICAL_IDENTITY_AUTHORITY_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/identity_authority.py",
        CANONICAL_LIFECYCLE_OWNER_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/child.py",
        CANONICAL_ATTEMPT_LIFECYCLE_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/orchestration.py",
        "intergrax/runtime/execution/orchestration_topology_submission.py",
        "intergrax/runtime/execution/decision_finalization_conformance.py",
        "intergrax/runtime/task/task_run_bridge.py",
    }
)

_CONTRACT_MINT_ALLOWED = frozenset(
    {
        "intergrax/contracts/execution_identity.py",
        CANONICAL_IDENTITY_AUTHORITY_MODULE.replace(".", "/") + ".py",
    }
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if "docker" in path.parts and "runtime-context" in path.parts:
            continue
        paths.append(path)
    return paths


def _collect_forbidden_calls(
    roots: tuple[Path, ...] | Path,
    *,
    forbidden: frozenset[str],
    rel_allowlist: frozenset[str] = frozenset(),
) -> list[str]:
    violations: list[str] = []
    root_list = roots if isinstance(roots, tuple) else (roots,)
    for root in root_list:
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in rel_allowlist:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in forbidden:
                    violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _eligibility_request(
    kind: ExecutionFailureKind,
    *,
    attempt_number: int = 1,
    max_attempts: int = 3,
) -> ExecutionRetryEligibilityRequest:
    return ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(kind=kind),
        attempt_number=attempt_number,
        max_attempts=max_attempts,
    )


def test_single_identity_authority_source() -> None:
    authority: ExecutionIdentityAuthorityPort = default_execution_identity_authority
    assert isinstance(authority, DefaultExecutionIdentityAuthority)
    for root in _FROZEN_PLANE_AUDIT_ROOTS:
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _CONTRACT_MINT_ALLOWED or rel in _MINT_ALLOWED_UNDER_EXECUTION:
                continue
            if rel == "intergrax/contracts/execution_identity.py":
                continue
            text = path.read_text(encoding="utf-8")
            for forbidden in _FORBIDDEN_PARALLEL_IDENTITY_SERVICES:
                assert forbidden not in text, f"{rel}: parallel identity service {forbidden!r}"


def test_no_identity_creation_in_evidence() -> None:
    violations = _collect_forbidden_calls(_EVIDENCE_PLANE_ROOT, forbidden=_MINT_CALLS)
    violations.extend(
        _collect_forbidden_calls(_EVIDENCE_PLANE_ROOT, forbidden=_AUTHORITY_DELEGATE_CALLS),
    )
    assert violations == [], "evidence plane must not mint execution identity: " + ", ".join(
        violations,
    )


def test_no_identity_creation_in_recovery() -> None:
    violations = _collect_forbidden_calls(_RECOVERY_PLANE_ROOTS, forbidden=_MINT_CALLS)
    scoped_delegate = _collect_forbidden_calls(
        _RECOVERY_PLANE_ROOTS,
        forbidden=_AUTHORITY_DELEGATE_CALLS,
        rel_allowlist=frozenset(
            {
                CANONICAL_ATTEMPT_LIFECYCLE_MODULE.replace(".", "/") + ".py",
            },
        ),
    )
    violations.extend(scoped_delegate)
    assert violations == [], "recovery plane must not bypass identity authority: " + ", ".join(
        violations,
    )


def test_no_identity_creation_in_checkpoint() -> None:
    violations = _collect_forbidden_calls(_CHECKPOINT_PLANE_ROOTS, forbidden=_MINT_CALLS)
    violations.extend(
        _collect_forbidden_calls(_CHECKPOINT_PLANE_ROOTS, forbidden=_AUTHORITY_DELEGATE_CALLS),
    )
    assert violations == [], "checkpoint plane must not mint execution identity: " + ", ".join(
        violations,
    )


def test_no_identity_creation_in_lineage() -> None:
    violations = _collect_forbidden_calls(_LINEAGE_PLANE_ROOTS, forbidden=_MINT_CALLS)
    violations.extend(
        _collect_forbidden_calls(_LINEAGE_PLANE_ROOTS, forbidden=_AUTHORITY_DELEGATE_CALLS),
    )
    assert violations == [], "lineage plane must not mint execution identity: " + ", ".join(
        violations,
    )


def test_no_identity_creation_in_governance() -> None:
    violations = _collect_forbidden_calls(_GOVERNANCE_PLANE_ROOT, forbidden=_MINT_CALLS)
    violations.extend(
        _collect_forbidden_calls(_GOVERNANCE_PLANE_ROOT, forbidden=_AUTHORITY_DELEGATE_CALLS),
    )
    assert violations == [], "governance plane must not mint execution identity: " + ", ".join(
        violations,
    )


def test_evidence_plane_never_creates_execution_identity() -> None:
    source = _RUNTIME_EVENT.read_text(encoding="utf-8")
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source
    tree = ast.parse(source, filename=str(_RUNTIME_EVENT))
    runtime_event_class: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeEvent":
            runtime_event_class = node
            break
    assert runtime_event_class is not None
    identity_fields = {"run_id", "attempt_id", "execution_id"}
    for item in runtime_event_class.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if item.target.id not in identity_fields:
            continue
        if item.value is not None:
            raise AssertionError(
                f"RuntimeEvent.{item.target.id} must not default mint execution identity",
            )


def test_retry_does_not_bypass_identity_authority() -> None:
    retry_source = _RETRY_SERVICE.read_text(encoding="utf-8")
    assert "mint_attempt_id" not in retry_source
    assert "mint_retry_attempt_id" not in retry_source
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-ee-a2-h3"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    assert transition.run_id == run_id
    assert transition.active_attempt_id != attempt_a1


def test_resume_preserves_identity_authority() -> None:
    violations = _collect_forbidden_calls(_LONG_RUNNING_ROOT, forbidden=_MINT_CALLS)
    assert violations == [], "resume/long-running must not mint primitives: " + ", ".join(
        violations,
    )
    coordinator_source = _COORDINATOR.read_text(encoding="utf-8")
    assert "resolve_root_task_identity" in coordinator_source or "run_id" in coordinator_source
    checkpoint_source = _RUNTIME_CHECKPOINT.read_text(encoding="utf-8")
    assert "mint_run_id" not in checkpoint_source
    assert "mint_attempt_id" not in checkpoint_source
    assert "mint_execution_id" not in checkpoint_source


def test_partial_recovery_uses_identity_authority() -> None:
    source = _PARTIAL_RECOVERY.read_text(encoding="utf-8")
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source
    assert "run_id=checkpoint.runtime.run_id" in source
    child_source = _CHILD_RUNNER.read_text(encoding="utf-8")
    assert "default_execution_identity_authority" in child_source
    assert "mint_child_execution_identity" in child_source


def test_lineage_is_identity_consumer_only() -> None:
    violations = _collect_forbidden_calls(_LINEAGE_PLANE_ROOTS, forbidden=_MINT_CALLS)
    violations.extend(
        _collect_forbidden_calls(_LINEAGE_PLANE_ROOTS, forbidden=_AUTHORITY_DELEGATE_CALLS),
    )
    assert violations == []


def test_checkpoint_cannot_mint_or_replace_identity() -> None:
    violations = _collect_forbidden_calls(_CHECKPOINT_PLANE_ROOTS, forbidden=_MINT_CALLS)
    assert violations == []
    for path in (_RUNTIME_CHECKPOINT, _LONG_RUNNING_ROOT / "execution_tree_checkpoint.py"):
        text = path.read_text(encoding="utf-8")
        assert "mint_root_execution_identity" not in text
        assert "mint_retry_attempt_id" not in text


def test_governance_is_identity_consumer_only() -> None:
    assert _collect_forbidden_calls(_GOVERNANCE_PLANE_ROOT, forbidden=_MINT_CALLS) == []


def test_ee_a2_h3_frozen_plane_integration_documentation_present() -> None:
    assert _IDENTITY_AUTHORITY_DOC.is_file()
    text = _IDENTITY_AUTHORITY_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "They may **validate**, **persist**, and **reference** identity" in text
    assert "**cannot create, mutate, or replace** execution identity" in text
