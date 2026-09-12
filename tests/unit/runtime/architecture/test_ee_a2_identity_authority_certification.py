# © Artur Czarnecki. All rights reserved.

"""EE-A2 — Execution identity lifecycle authority certification gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_identity_authority import (
    CANONICAL_ATTEMPT_LIFECYCLE_MODULE,
    CANONICAL_IDENTITY_AUTHORITY_MODULE,
    CANONICAL_LIFECYCLE_OWNER_MODULE,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
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
_LLM_ADAPTER = _REPO_ROOT / "intergrax" / "llm_adapters" / "contracts" / "llm_adapter.py"
_LONG_RUNNING_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
_CHECKPOINT_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "checkpoint"

_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
    }
)

_MINT_ALLOWED_FILES = frozenset(
    {
        "intergrax/contracts/execution_identity.py",
        CANONICAL_IDENTITY_AUTHORITY_MODULE.replace(".", "/") + ".py",
        "intergrax/runtime/execution/attempt_lifecycle/service.py",
        "intergrax/runtime/execution/decision_finalization_conformance.py",
        "intergrax/runtime/observability/persistence_conformance.py",
        "intergrax/runtime/diagnostics/persistence_conformance.py",
        "intergrax/runtime/diagnostics/functional_evidence_persistence_conformance.py",
        "intergrax/collaborative_work/repository_qualification_suite.py",
        "intergrax/runtime/task/task_run_bridge.py",
        "intergrax/autonomous_work/work_stage_capability_loop.py",
        "intergrax/experiments/workflow.py",
        "intergrax/core/qualification/functional_qualification_runner.py",
        "intergrax/applications/_shared/harness_task_routes.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
    }
)

_BYPASS_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime",
    _REPO_ROOT / "intergrax" / "agents",
    _REPO_ROOT / "intergrax" / "llm_adapters",
    _REPO_ROOT / "intergrax" / "integrations",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "intergrax" / "applications",
)

_REQUIRED_DOC_SECTIONS = (
    "## Identity ownership",
    "## Lifecycle diagram",
    "## Mint rules",
    "## Mutation rules",
    "## Retry semantics",
    "## Resume semantics",
    "## Recovery semantics",
    "## Provider rules",
    "## Forbidden patterns",
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


def _module_path(module: str) -> Path:
    return _REPO_ROOT / Path(module.replace(".", "/") + ".py")


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


def test_ee_a2_identity_authority_documentation_complete() -> None:
    assert _IDENTITY_AUTHORITY_DOC.is_file()
    text = _IDENTITY_AUTHORITY_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "AttemptLifecycleService" in text
    assert "ExecutionRuntime" in text


def test_ee_a2_identity_authority_contract_present() -> None:
    assert _IDENTITY_AUTHORITY_CONTRACT.is_file()
    source = _IDENTITY_AUTHORITY_CONTRACT.read_text(encoding="utf-8")
    assert "ExecutionIdentityAuthorityPort" in source
    assert CANONICAL_IDENTITY_AUTHORITY_MODULE in source


def test_ee_a2_identity_single_authority_runtime_modules() -> None:
    runtime_source = _module_path(CANONICAL_LIFECYCLE_OWNER_MODULE).read_text(encoding="utf-8")
    authority_source = _IDENTITY_AUTHORITY_RUNTIME.read_text(encoding="utf-8")
    assert "mint_root_execution_identity" in authority_source
    assert "identity_authority" in runtime_source


def test_ee_a2_direct_mint_outside_authority_blocked() -> None:
    violations = _collect_direct_mint_violations()
    assert violations == [], (
        "direct execution identity mint outside authority: " + ", ".join(violations)
    )


def test_ee_a2_retry_preserves_run_id_and_creates_new_attempt() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant_id = "tenant-ee-a2"
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


def test_ee_a2_resume_preserves_run_and_attempt_in_process_bind() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        assert peek_active_execution_identity() == (run_id, attempt_id)
    finally:
        reset_active_execution_identity(token)
    assert peek_active_execution_identity() is None


def test_ee_a2_checkpoint_tree_does_not_mint_execution_identity() -> None:
    violations: list[str] = []
    for root in (_LONG_RUNNING_ROOT, _CHECKPOINT_ROOT):
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in _MINT_CALLS:
                    violations.append(f"{rel}:{node.lineno}: {name}()")
    assert violations == [], "checkpoint/resume must not mint identity: " + ", ".join(violations)


def test_ee_a2_provider_boundary_cannot_mint_execution_identity() -> None:
    source = _LLM_ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_LLM_ADAPTER))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _MINT_CALLS:
            rel = _LLM_ADAPTER.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}:{node.lineno}: {name}()")
    assert violations == [], "provider adapter must not mint execution identity: " + ", ".join(
        violations,
    )


def test_ee_a2_attempt_owner_module_uses_retry_mint_only() -> None:
    source = _module_path(CANONICAL_ATTEMPT_LIFECYCLE_MODULE).read_text(encoding="utf-8")
    assert "mint_retry_attempt_id" in source
    assert "AttemptTransitionReason.RETRY" in source


def test_ee_a2_frozen_single_authority_gate_still_present() -> None:
    frozen = _REPO_ROOT / "tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py"
    assert frozen.is_file()


def test_ee_a2_direct_mint_inventory_count_is_zero() -> None:
    assert len(_collect_direct_mint_violations()) == 0
