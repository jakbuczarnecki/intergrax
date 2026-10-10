# © Artur Czarnecki. All rights reserved.

"""Mechanical semantic-owner discovery for TRACE-X-P6 (independent of matrix prose)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_EXCLUDE_PATH_PARTS = (
    "/tests/",
    "\\tests\\",
    "/test_",
    "\\test_",
    "/qualification/",
    "\\qualification\\",
    "/examples/",
    "\\examples\\",
    "/scaffold/",
    "\\scaffold\\",
    "docker/runtime-context",
)

_MINT_ID_FUNCTIONS = frozenset(
    {"mint_task_id", "mint_run_id", "mint_attempt_id", "mint_execution_id"},
)

P6_CANONICAL_OWNER_EXPECTATIONS: Final[dict[str, frozenset[str]]] = {
    "Execution identity owner": frozenset({"intergrax/contracts/execution_identity.py"}),
    "Run identity owner": frozenset({"intergrax/runtime/execution/identity_authority.py"}),
    "Attempt identity owner": frozenset(
        {"intergrax/runtime/execution/attempt_lifecycle/service.py"},
    ),
    "checkpoint persistence semantic contract owner": frozenset(
        {"intergrax/runtime/long_running/persistence_contract.py"},
    ),
    "resumability decision owner": frozenset(
        {"intergrax/runtime/cancellation/resume_admission.py"},
    ),
    "scheduled resume trigger owner": frozenset(
        {"intergrax/runtime/long_running/scheduler.py"},
    ),
    "resume restoration / recovery coordination owner": frozenset(
        {"intergrax/runtime/long_running/coordinator.py"},
    ),
    "resume admission validation owner": frozenset(
        {
            "intergrax/runtime/cancellation/resume_admission.py",
            "intergrax/runtime/background_execution/reentry_admission.py",
        },
    ),
    "execution retry eligibility policy owner": frozenset(
        {"intergrax/runtime/execution/retry/policy.py"},
    ),
    "execution-attempt retry authority owner": frozenset(
        {"intergrax/runtime/execution/retry/service.py"},
    ),
    "run/graph retry scheduling facade owner": frozenset(
        {"intergrax/runtime/nexus/retry/coordinator.py"},
    ),
    "terminal state truth owner": frozenset(
        {"intergrax/runtime/execution/execution_terminal/service.py"},
    ),
    "terminal RuntimeEvent/evidence owner": frozenset(
        {"intergrax/runtime/events/trace_bridge.py"},
    ),
    "failure reconstruction owner": frozenset(
        {"intergrax/runtime/observability/reconstruction/execution_reconstruction.py"},
    ),
    "parent-child causality owner": frozenset({"intergrax/contracts/execution_lineage.py"}),
}

P6_SEMANTIC_OWNER_MATRIX: Final[tuple[tuple[str, frozenset[str]], ...]] = tuple(
    (concern, P6_CANONICAL_OWNER_EXPECTATIONS[concern])
    for concern in P6_CANONICAL_OWNER_EXPECTATIONS
)

_CONCERN_BY_NORMALIZED_KEY: Final[dict[str, str]] = {
    concern.strip().lower(): concern for concern in P6_CANONICAL_OWNER_EXPECTATIONS
}


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_discovery_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in normalized or part in normalized:
            return False
    return True


@lru_cache(maxsize=2048)
def _parse_module(repo_relative_path: str) -> ast.Module | None:
    path = _REPO_ROOT / repo_relative_path
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


@lru_cache(maxsize=1)
def _iter_production_modules() -> tuple[str, ...]:
    paths: list[str] = []
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            paths.append(_normalize_repo_path(path))
    return tuple(sorted(paths))


def _class_names(tree: ast.Module) -> frozenset[str]:
    return frozenset(
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    )


def _top_level_function_names(tree: ast.Module) -> frozenset[str]:
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _class_defines_method(tree: ast.Module, class_name: str, method_name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == method_name:
                    return True
    return False


def _class_defines_methods(
    tree: ast.Module,
    class_name: str,
    method_names: frozenset[str],
) -> bool:
    return all(
        _class_defines_method(tree, class_name, method_name)
        for method_name in method_names
    )


def _class_inherits_name(tree: ast.Module, class_name: str, base_name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == base_name:
                return True
            if isinstance(base, ast.Attribute) and base.attr == base_name:
                return True
    return False


def _function_body_references_attribute(
    tree: ast.Module,
    function_name: str,
    attr_name: str,
) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == attr_name:
                return True
    return False


def _register_candidates(
    index: dict[str, set[str]],
    repo_path: str,
    tree: ast.Module,
) -> None:
    classes = _class_names(tree)
    functions = _top_level_function_names(tree)

    if _MINT_ID_FUNCTIONS.issubset(functions):
        index["execution identity owner"].add(repo_path)
    if "DefaultExecutionIdentityAuthority" in classes:
        index["run identity owner"].add(repo_path)
    if "AttemptLifecycleService" in classes:
        index["attempt identity owner"].add(repo_path)
    if _class_inherits_name(tree, "TaskCheckpointPersistence", "ABC"):
        index["checkpoint persistence semantic contract owner"].add(repo_path)
    if "assert_checkpoint_resumable" in functions:
        index["resumability decision owner"].add(repo_path)
    if _class_defines_methods(
        tree,
        "LongRunningScheduler",
        frozenset({"schedule_resume", "_process_due_schedules", "tick"}),
    ):
        index["scheduled resume trigger owner"].add(repo_path)
    if _class_defines_methods(
        tree,
        "LongRunningCoordinator",
        frozenset(
            {
                "restore_if_resuming",
                "recovery_admission_request_for_checkpoint",
                "admit_task_resume_recovery_handoff",
            },
        ),
    ):
        index["resume restoration / recovery coordination owner"].add(repo_path)
    if "BackgroundExecutionReentryAdmissionError" in classes:
        index["resume admission validation owner"].add(repo_path)
    if "assert_checkpoint_resumable" in functions and _function_body_references_attribute(
        tree,
        "assert_checkpoint_resumable",
        "tenant_id",
    ):
        index["resume admission validation owner"].add(repo_path)
    if "evaluate_execution_retry_eligibility" in functions:
        index["execution retry eligibility policy owner"].add(repo_path)
    if _class_defines_method(tree, "ExecutionAttemptRetryService", "transition_for_retry"):
        index["execution-attempt retry authority owner"].add(repo_path)
    if _class_defines_methods(
        tree,
        "RetryCoordinator",
        frozenset({"should_retry_run", "build_scheduled_event"}),
    ):
        index["run/graph retry scheduling facade owner"].add(repo_path)
    if _class_defines_method(tree, "ExecutionTerminalService", "commit_terminal_outcome"):
        index["terminal state truth owner"].add(repo_path)
    if "trace_event_to_runtime_event" in functions:
        index["terminal runtimeevent/evidence owner"].add(repo_path)
    if "ExecutionReconstructor" in classes:
        index["failure reconstruction owner"].add(repo_path)
    if "ExecutionLineagePersistence" in classes and _class_inherits_name(
        tree,
        "ExecutionLineagePersistence",
        "ABC",
    ):
        index["parent-child causality owner"].add(repo_path)


@lru_cache(maxsize=1)
def _owner_candidate_index() -> dict[str, frozenset[str]]:
    index: dict[str, set[str]] = {
        concern.strip().lower(): set() for concern in P6_CANONICAL_OWNER_EXPECTATIONS
    }
    for repo_path in _iter_production_modules():
        tree = _parse_module(repo_path)
        if tree is None:
            continue
        _register_candidates(index, repo_path, tree)
    return {key: frozenset(paths) for key, paths in index.items()}


def discover_owner_candidates(concern: str) -> frozenset[str]:
    """Independently discover production modules that own a named P6 semantic concern."""
    key = concern.strip().lower()
    if key not in _owner_candidate_index():
        raise KeyError(f"unknown P6 semantic owner concern: {concern}")
    return _owner_candidate_index()[key]


def discover_semantic_owners(concern: str) -> frozenset[str]:
    """Return independently discovered owner anchor module(s) for a P6 concern."""
    return discover_owner_candidates(concern)


@dataclass(frozen=True, slots=True)
class SemanticOwnerGateResult:
    concern: str
    expected_owner_set: frozenset[str]
    discovered_owner_set: frozenset[str]
    ok: bool


def compare_semantic_owner_gate(
    concern: str,
    *,
    extra_discovered: frozenset[str] = frozenset(),
) -> SemanticOwnerGateResult:
    canonical = _CONCERN_BY_NORMALIZED_KEY.get(concern.strip().lower(), concern)
    expected = P6_CANONICAL_OWNER_EXPECTATIONS[canonical]
    discovered = discover_owner_candidates(canonical) | extra_discovered
    return SemanticOwnerGateResult(
        concern=canonical,
        expected_owner_set=expected,
        discovered_owner_set=discovered,
        ok=discovered == expected,
    )
