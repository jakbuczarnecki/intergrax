# © Artur Czarnecki. All rights reserved.

"""EE-A2-H2 — Identity Authority global freeze & execution identity ownership certification."""

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
_IDENTITY_AUTHORITY_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "execution_identity_authority.py"
)
_IDENTITY_AUTHORITY_RUNTIME = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "identity_authority.py"
_CHILD_RUNNER = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_RUNTIME_OWNER = _REPO_ROOT / Path(CANONICAL_LIFECYCLE_OWNER_MODULE.replace(".", "/") + ".py")
_RETRY_SERVICE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "retry" / "service.py"
_BACKGROUND_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "background_execution"
_LONG_RUNNING_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
_CHECKPOINT_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "checkpoint"
_RECOVERY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "fan_out_partial_recovery.py"

_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
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

_MINT_ALLOWED_FILES = frozenset(
    {
        "intergrax/contracts/execution_identity.py",
        CANONICAL_IDENTITY_AUTHORITY_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/decision_finalization_conformance.py",
        "intergrax/runtime/observability/persistence_conformance.py",
        "intergrax/runtime/diagnostics/persistence_conformance.py",
        "intergrax/runtime/diagnostics/functional_evidence_persistence_conformance.py",
        "intergrax/collaborative_work/repository_qualification_suite.py",
        "intergrax/experiments/workflow.py",
        "intergrax/core/qualification/functional_qualification_runner.py",
        "intergrax/applications/_shared/harness_task_routes.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
    }
)

_BYPASS_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime",
    _REPO_ROOT / "intergrax" / "contracts",
    _REPO_ROOT / "intergrax" / "agents",
    _REPO_ROOT / "intergrax" / "integrations",
    _REPO_ROOT / "intergrax" / "llm_adapters",
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
)

_BACKGROUND_MINT_ALLOWLIST = frozenset(
    {
        "intergrax/runtime/background_execution/bootstrap.py",
        "intergrax/runtime/background_execution/identity_persistence.py",
    }
)

_AUTHORITY_FORBIDDEN_COUPLING = (
    "checkpoint",
    "scheduler",
    "evidence",
    "llm",
    "sqlite",
    "sqlalchemy",
    "provider",
    "tool_execution",
)

_REQUIRED_DOC_SECTIONS = (
    "## Global Freeze Statement",
    "## Identity ownership",
    "## Forbidden patterns",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_python_files(root: Path) -> list[Path]:
    if not root.is_file():
        if not root.is_dir():
            return []
    if root.is_file():
        return [root]
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if "docker" in path.parts and "runtime-context" in path.parts:
            continue
        paths.append(path)
    return paths


def _collect_direct_mint_violations() -> list[str]:
    violations: list[str] = []
    for root in _BYPASS_SCAN_ROOTS:
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _MINT_ALLOWED_FILES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in _MINT_CALLS:
                    violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _collect_forbidden_calls_in_root(root: Path, *, forbidden: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in forbidden:
                violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _module_path(module: str) -> Path:
    return _REPO_ROOT / Path(module.replace(".", "/") + ".py")


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


def test_single_execution_identity_authority_owner() -> None:
    assert _IDENTITY_AUTHORITY_CONTRACT.is_file()
    contract_source = _IDENTITY_AUTHORITY_CONTRACT.read_text(encoding="utf-8")
    assert "ExecutionIdentityAuthorityPort" in contract_source
    assert CANONICAL_IDENTITY_AUTHORITY_MODULE in contract_source
    assert isinstance(default_execution_identity_authority, DefaultExecutionIdentityAuthority)
    runtime_source = _IDENTITY_AUTHORITY_RUNTIME.read_text(encoding="utf-8")
    assert "class DefaultExecutionIdentityAuthority" in runtime_source
    owner_source = _RUNTIME_OWNER.read_text(encoding="utf-8")
    assert "mint_root_execution_identity" in owner_source
    assert "identity_authority" in owner_source
    attempt_source = _module_path(CANONICAL_ATTEMPT_LIFECYCLE_MODULE).read_text(encoding="utf-8")
    assert "mint_retry_attempt_id" in attempt_source
    for root in _BYPASS_SCAN_ROOTS:
        for path in _iter_python_files(root):
            text = path.read_text(encoding="utf-8")
            for forbidden in _FORBIDDEN_PARALLEL_IDENTITY_SERVICES:
                assert forbidden not in text, f"{path}: parallel identity service {forbidden!r}"


def test_no_production_execution_identity_mint_outside_allowlist() -> None:
    violations = _collect_direct_mint_violations()
    assert violations == [], (
        "production execution identity mint outside allowlist: " + ", ".join(violations)
    )


def test_retry_attempt_identity_owned_by_attempt_lifecycle() -> None:
    retry_source = _RETRY_SERVICE.read_text(encoding="utf-8")
    assert "mint_attempt_id" not in retry_source
    assert "mint_retry_attempt_id" not in retry_source
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-ee-a2-h2"
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


def test_resume_preserves_execution_identity() -> None:
    violations = _collect_forbidden_calls_in_root(_LONG_RUNNING_ROOT, forbidden=_MINT_CALLS)
    violations.extend(_collect_forbidden_calls_in_root(_CHECKPOINT_ROOT, forbidden=_MINT_CALLS))
    assert violations == [], "resume/checkpoint must not mint execution identity: " + ", ".join(
        violations,
    )
    coordinator = _LONG_RUNNING_ROOT / "coordinator.py"
    source = coordinator.read_text(encoding="utf-8")
    assert "resolve_root_task_identity" in source or "run_id" in source


def test_child_execution_identity_owned_by_authority() -> None:
    source = _CHILD_RUNNER.read_text(encoding="utf-8")
    assert "default_execution_identity_authority" in source
    assert "mint_child_execution_identity" in source
    tree = ast.parse(source, filename=str(_CHILD_RUNNER))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _MINT_CALLS:
            raise AssertionError("ChildExecutionRunner must not call primitive mint_* helpers")


def test_background_execution_does_not_mint_identity() -> None:
    violations = _collect_forbidden_calls_in_root(_BACKGROUND_ROOT, forbidden=_MINT_CALLS)
    scoped = [v for v in violations if v.split(":")[0] not in _BACKGROUND_MINT_ALLOWLIST]
    assert scoped == [], "background workers must delegate identity to authority: " + ", ".join(
        scoped,
    )


def test_ee_a2_h2_global_freeze_documentation_present() -> None:
    assert _IDENTITY_AUTHORITY_DOC.is_file()
    text = _IDENTITY_AUTHORITY_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "ExecutionIdentityAuthority is the only production authority" in text


def test_ee_a2_h2_identity_authority_port_has_no_storage_or_scheduler_coupling() -> None:
    for path in (_IDENTITY_AUTHORITY_CONTRACT, _IDENTITY_AUTHORITY_RUNTIME):
        lowered = path.read_text(encoding="utf-8").lower()
        for fragment in _AUTHORITY_FORBIDDEN_COUPLING:
            assert fragment not in lowered, f"{path.name} must not couple to {fragment!r}"


def test_ee_a2_h2_root_execution_admits_through_runtime_not_ingress() -> None:
    runtime_source = _RUNTIME_OWNER.read_text(encoding="utf-8")
    assert "mint_root_execution_identity(" in runtime_source
    ingress_violations = _collect_forbidden_calls_in_root(
        _REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters",
        forbidden=_MINT_CALLS | frozenset({"mint_root_execution_identity"}),
    )
    assert ingress_violations == [], "ingress must not mint execution identity: " + ", ".join(
        ingress_violations,
    )


def test_ee_a2_h2_recovery_reentry_does_not_mint_identity() -> None:
    if _RECOVERY_ROOT.is_file():
        violations = _collect_forbidden_calls_in_root(_RECOVERY_ROOT, forbidden=_MINT_CALLS)
        assert violations == [], "recovery must not mint identity: " + ", ".join(violations)


def test_ee_a2_h2_production_mint_outside_allowlist_count_is_zero() -> None:
    assert len(_collect_direct_mint_violations()) == 0


def test_ee_a2_h2_default_authority_satisfies_port() -> None:
    authority: ExecutionIdentityAuthorityPort = default_execution_identity_authority
    minted = authority.mint_execution_identity()
    assert str(minted.run_id).startswith("run_")
    assert str(minted.attempt_id).startswith("attempt_")
    assert str(minted.execution_id).startswith("exec_")
    child = authority.mint_child_execution_identity()
    assert str(child).startswith("exec_")
    assert str(authority.mint_run_identity()).startswith("run_")
    assert str(authority.mint_attempt_identity()).startswith("attempt_")
