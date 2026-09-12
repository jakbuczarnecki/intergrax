# © Artur Czarnecki. All rights reserved.

"""EE-A2-H1 — intake identity convergence certification gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity_authority import (
    CANONICAL_ATTEMPT_LIFECYCLE_MODULE,
    CANONICAL_IDENTITY_AUTHORITY_MODULE,
    CANONICAL_LIFECYCLE_OWNER_MODULE,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.identity_authority import (
    DefaultExecutionIdentityAuthority,
    default_execution_identity_authority,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTAKE_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_IDENTITY_INTAKE_CONVERGENCE_MODEL.md"
)
_ACP_RUN = _REPO_ROOT / "intergrax" / "agents" / "authoring" / "acp_run.py"
_CHILD_RUNNER = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_HARNESS_ROUTES = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "harness_task_routes.py"
_FASTAPI_ADAPTER_ROOT = _REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters"
_BACKGROUND_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "background_execution"
_LONG_RUNNING_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
_IDENTITY_AUTHORITY_RUNTIME = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "identity_authority.py"

_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
    }
)

_INTAKE_FORBIDDEN = _MINT_CALLS | frozenset({"new_run_id", "mint_root_execution_identity"})

_MINT_ALLOWED_FILES = frozenset(
    {
        "intergrax/contracts/execution_identity.py",
        CANONICAL_IDENTITY_AUTHORITY_MODULE.replace(".", "/") + ".py",
        CANONICAL_ATTEMPT_LIFECYCLE_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/decision_finalization_conformance.py",
        "intergrax/runtime/observability/persistence_conformance.py",
        "intergrax/runtime/diagnostics/persistence_conformance.py",
        "intergrax/runtime/diagnostics/functional_evidence_persistence_conformance.py",
        "intergrax/collaborative_work/repository_qualification_suite.py",
        "intergrax/experiments/workflow.py",
        "intergrax/core/qualification/functional_qualification_runner.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
    }
)

_BYPASS_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime",
    _REPO_ROOT / "intergrax" / "agents",
    _REPO_ROOT / "intergrax" / "llm_adapters",
    _REPO_ROOT / "intergrax" / "integrations",
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "intergrax" / "contracts",
)

_REQUIRED_DOC_SECTIONS = (
    "## Ingress points",
    "## Ownership matrix",
    "## Forbidden patterns",
    "## Allowed exceptions",
    "## Lifecycle diagram",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_python_files(root: Path) -> list[Path]:
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


def test_ee_a2_h1_intake_convergence_documentation_complete() -> None:
    assert _INTAKE_DOC.is_file()
    text = _INTAKE_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing section {heading!r}"


def test_only_identity_authority_can_mint_run_id() -> None:
    authority = DefaultExecutionIdentityAuthority()
    run_id = authority.mint_run_identity()
    assert str(run_id).startswith("run_")
    violations = _collect_direct_mint_violations()
    assert violations == [], "direct mint_run_id outside allowlist: " + ", ".join(violations)


def test_only_identity_authority_can_mint_execution_id() -> None:
    authority = DefaultExecutionIdentityAuthority()
    minted = authority.mint_execution_identity()
    assert str(minted.execution_id).startswith("exec_")
    assert len(_collect_direct_mint_violations()) == 0


def test_only_attempt_service_can_create_retry_attempt() -> None:
    attempt_module = _REPO_ROOT / Path(
        CANONICAL_ATTEMPT_LIFECYCLE_MODULE.replace(".", "/") + ".py"
    )
    source = attempt_module.read_text(encoding="utf-8")
    assert "mint_retry_attempt_id" in source


def test_http_ingress_does_not_mint_execution_identity() -> None:
    violations = _collect_forbidden_calls_in_root(
        _FASTAPI_ADAPTER_ROOT,
        forbidden=_INTAKE_FORBIDDEN,
    )
    assert violations == [], "HTTP adapters must not mint execution identity: " + ", ".join(
        violations,
    )


def test_acp_ingress_delegates_identity_creation() -> None:
    source = _ACP_RUN.read_text(encoding="utf-8")
    assert "default_execution_identity_authority" in source
    assert "mint_execution_identity" in source
    assert "mint_root_execution_identity(" not in source
    tree = ast.parse(source, filename=str(_ACP_RUN))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in _MINT_CALLS:
            raise AssertionError("ACP ingress must not call primitive mint_* helpers")


def test_background_ingress_does_not_mint_identity() -> None:
    violations = _collect_forbidden_calls_in_root(
        _BACKGROUND_ROOT,
        forbidden=_MINT_CALLS,
    )
    allowed = {
        (_REPO_ROOT / "intergrax/runtime/background_execution/bootstrap.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax/runtime/background_execution/identity_persistence.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
    }
    scoped = [v for v in violations if v.split(":")[0] not in allowed]
    assert scoped == [], "background ingress must delegate to authority module: " + ", ".join(
        scoped,
    )


def test_child_execution_uses_identity_authority() -> None:
    source = _CHILD_RUNNER.read_text(encoding="utf-8")
    assert "default_execution_identity_authority" in source
    assert "def mint_child_execution_id" in source
    assert "mint_child_execution_identity" in source


def test_retry_creates_attempt_only_through_attempt_lifecycle() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-ee-a2-h1"
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


def test_resume_never_creates_new_identity() -> None:
    violations = _collect_forbidden_calls_in_root(_LONG_RUNNING_ROOT, forbidden=_MINT_CALLS)
    assert violations == [], "resume/long-running must not mint identity: " + ", ".join(violations)


def test_harness_http_routes_do_not_mint_execution_identity() -> None:
    source = _HARNESS_ROUTES.read_text(encoding="utf-8")
    assert "new_run_id" not in source
    tree = ast.parse(source, filename=str(_HARNESS_ROUTES))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in _INTAKE_FORBIDDEN:
            raise AssertionError("harness routes must not mint execution identity at intake")


def test_ee_a2_h1_default_authority_implements_port() -> None:
    assert isinstance(default_execution_identity_authority, DefaultExecutionIdentityAuthority)
    runtime_source = _IDENTITY_AUTHORITY_RUNTIME.read_text(encoding="utf-8")
    assert "class DefaultExecutionIdentityAuthority" in runtime_source
    assert CANONICAL_LIFECYCLE_OWNER_MODULE.replace(".", "/") + ".py"
    lifecycle = _REPO_ROOT / Path(CANONICAL_LIFECYCLE_OWNER_MODULE.replace(".", "/") + ".py")
    assert "identity_authority" in lifecycle.read_text(encoding="utf-8")


def test_ee_a2_h1_production_mint_outside_allowlist_is_zero() -> None:
    assert len(_collect_direct_mint_violations()) == 0
