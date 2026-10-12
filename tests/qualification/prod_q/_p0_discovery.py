# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 mechanical closed-world discovery over bounded production roots."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.prod_q._p0_types import (
    DiscoveryKind,
    ProductionReadinessCandidate,
    ScanFailure,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS: Final[tuple[Path, ...]] = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)

_EXCLUDE_PATH_PARTS: Final[tuple[str, ...]] = (
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
    "testing_support/",
    "docker/runtime-context",
)

_CONTRACT_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax.contracts.execution_capacity_admission",
    "intergrax.contracts.execution_reliability",
    "intergrax.contracts.runtime_execution_admission",
    "intergrax.contracts.functional_evidence",
    "intergrax.contracts.idempotency",
)

_SYMBOL_INDICATORS: Final[tuple[tuple[re.Pattern[str], str, str], ...]] = (
    (re.compile(r"^(lifespan|on_startup|startup|bootstrap_app|bootstrap_)", re.I), "startup", "startup"),
    (re.compile(r"^(shutdown|on_shutdown|drain|teardown|close_runtime)", re.I), "shutdown", "shutdown"),
    (re.compile(r"^(health_snapshot|health_check|liveness|health)$", re.I), "health", "health"),
    (re.compile(r"^(readiness|ready_check|is_ready)$", re.I), "readiness", "readiness"),
    (re.compile(r"degraded", re.I), "degraded", "degraded"),
    (re.compile(r"(terminal_failure|critical_failure|fail_closed)", re.I), "terminal", "terminal"),
    (re.compile(r"(recover|remediation|operator_action)", re.I), "recovery", "recovery"),
    (re.compile(r"(admit|qualification|production_approved)", re.I), "admission", "admission"),
    (re.compile(r"(unsupported|invalid_config|fail_closed_config)", re.I), "unsupported", "unsupported"),
    (re.compile(r"(lab_|_lab|dev_mode|strict_mode|production_mode|permissive)", re.I), "mode", "mode"),
    (re.compile(r"(secret|credential|token_ref)", re.I), "secrets", "secrets"),
    (re.compile(r"(capacity|admission|concurrency|assess_root_execution)", re.I), "capacity", "capacity"),
    (re.compile(r"(backpressure|queue_depth|overload)", re.I), "backpressure", "backpressure"),
    (re.compile(r"(throttle|rate_limit|retry_after)", re.I), "throttle", "throttle"),
    (re.compile(r"(timeout|retry|cancel)", re.I), "lifecycle", "lifecycle"),
    (re.compile(r"idempoten", re.I), "idempotency", "idempotency"),
    (re.compile(r"(materializ|dependency_boundary|bootstrap_dependencies)", re.I), "materialization", "materialization"),
    (re.compile(r"(sandbox|isolation_boundary)", re.I), "sandbox", "sandbox"),
    (re.compile(r"(retention|redact|deletion)", re.I), "retention", "retention"),
    (re.compile(r"(backup|restore)", re.I), "backup", "backup"),
    (re.compile(r"tenant", re.I), "tenant", "tenant"),
)

_CANONICAL_ANCHORS: Final[tuple[tuple[str, str, str], ...]] = (
    ("intergrax/contracts/execution_capacity_admission.py", "ExecutionCapacityAdmissionPort", "capacity"),
    ("intergrax/runtime/execution/capacity.py", "LocalExecutionCapacityAdmission", "capacity"),
    (
        "intergrax/contracts/execution_capacity/admission_decision.py",
        "assess_root_execution_capacity",
        "capacity",
    ),
    ("intergrax/hosting/engine/engine.py", "HostedApplicationEngine", "health"),
    ("intergrax/hosting/engine/engine.py", "health_snapshot", "health"),
    ("intergrax/applications/_shared/auditability_health_wiring.py", "project_host_auditability_health_facts", "readiness"),
    ("intergrax/runtime/observability/auditability_health.py", "build_auditability_health_snapshot", "readiness"),
    ("intergrax/applications/_shared/capability_health/providers/dependency_validation.py", "DependencyValidationCapabilityHealthProvider", "health"),
    (
        "intergrax/contracts/execution_reliability/shutdown_contract.py",
        "ExecutionRuntimeShutdownPhase",
        "shutdown",
    ),
    ("intergrax/integrations/providers/relational_store/sqlite/paths.py", "resolve_sqlite_store_paths", "materialization"),
    ("intergrax/integrations/providers/sandbox_host/e2b/bundle.py", "create_e2b_sandbox_host_integration", "sandbox"),
    (
        "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
        "create_modal_sandbox_host_integration",
        "sandbox",
    ),
    (
        "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
        "create_daytona_sandbox_host_integration",
        "sandbox",
    ),
    ("intergrax/runtime/execution/lab_reference_agent_runtime.py", "LabReferenceAgentRuntime", "mode"),
    ("intergrax/fastapi_core/routers/health.py", "health", "health"),
    ("applications/governed_contractor_application/host/factory.py", "build_host_application", "startup"),
)

_SANDBOX_PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
)


def repo_root() -> Path:
    return _REPO_ROOT


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_candidate_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in normalized or part in normalized:
            return False
    return True


def _candidate_id(module_path: str, symbol: str, kind: DiscoveryKind) -> str:
    return f"{module_path}::{symbol}::{kind.value}"


def _discover_anchors() -> list[ProductionReadinessCandidate]:
    out: list[ProductionReadinessCandidate] = []
    for module_path, symbol, indicator in _CANONICAL_ANCHORS:
        out.append(
            ProductionReadinessCandidate(
                candidate_id=_candidate_id(module_path, symbol, DiscoveryKind.ANCHOR_REGISTRY),
                module_path=module_path,
                symbol=symbol,
                discovery_kind=DiscoveryKind.ANCHOR_REGISTRY,
                line_no=1,
                indicator=indicator,
            )
        )
    for module_path in _SANDBOX_PROVIDER_MODULES:
        out.append(
            ProductionReadinessCandidate(
                candidate_id=_candidate_id(module_path, "sandbox_host_bundle", DiscoveryKind.PROVIDER_SANDBOX_HOST),
                module_path=module_path,
                symbol="sandbox_host_bundle",
                discovery_kind=DiscoveryKind.PROVIDER_SANDBOX_HOST,
                line_no=1,
                indicator="sandbox_provider",
            )
        )
    return out


def _match_symbol_indicator(name: str) -> str | None:
    for pattern, indicator, _ in _SYMBOL_INDICATORS:
        if pattern.search(name):
            return indicator
    return None


def _discover_from_ast(module_path: str, source: str) -> list[ProductionReadinessCandidate]:
    out: list[ProductionReadinessCandidate] = []
    try:
        tree = ast.parse(source, filename=module_path)
    except SyntaxError:
        return out

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            indicator = _match_symbol_indicator(node.name)
            if indicator:
                kind = (
                    DiscoveryKind.AST_LIFESPAN_HANDLER
                    if node.name == "lifespan"
                    else DiscoveryKind.AST_SYMBOL_INDICATOR
                )
                out.append(
                    ProductionReadinessCandidate(
                        candidate_id=_candidate_id(module_path, node.name, kind),
                        module_path=module_path,
                        symbol=node.name,
                        discovery_kind=kind,
                        line_no=node.lineno,
                        indicator=indicator,
                    )
                )
            if any(
                isinstance(dec, ast.Name) and dec.id == "lifespan"
                for dec in node.decorator_list
            ):
                out.append(
                    ProductionReadinessCandidate(
                        candidate_id=_candidate_id(module_path, node.name, DiscoveryKind.AST_LIFESPAN_HANDLER),
                        module_path=module_path,
                        symbol=node.name,
                        discovery_kind=DiscoveryKind.AST_LIFESPAN_HANDLER,
                        line_no=node.lineno,
                        indicator="lifespan",
                    )
                )
        elif isinstance(node, ast.ClassDef):
            indicator = _match_symbol_indicator(node.name)
            if indicator:
                out.append(
                    ProductionReadinessCandidate(
                        candidate_id=_candidate_id(
                            module_path, node.name, DiscoveryKind.AST_SYMBOL_INDICATOR
                        ),
                        module_path=module_path,
                        symbol=node.name,
                        discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
                        line_no=node.lineno,
                        indicator=indicator,
                    )
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            for prefix in _CONTRACT_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    symbol = prefix.split(".")[-1]
                    out.append(
                        ProductionReadinessCandidate(
                            candidate_id=_candidate_id(
                                module_path, f"import:{prefix}", DiscoveryKind.AST_IMPORT_CONTRACT
                            ),
                            module_path=module_path,
                            symbol=f"import:{prefix}",
                            discovery_kind=DiscoveryKind.AST_IMPORT_CONTRACT,
                            line_no=node.lineno,
                            indicator="contract_import",
                        )
                    )
                    break
        if isinstance(node, ast.If):
            for test in ast.walk(node.test):
                if isinstance(test, ast.Constant) and isinstance(test.value, str):
                    val = test.value.lower()
                    if any(tok in val for tok in ("lab", "dev", "strict", "production")):
                        out.append(
                            ProductionReadinessCandidate(
                                candidate_id=_candidate_id(
                                    module_path,
                                    f"branch:{node.lineno}",
                                    DiscoveryKind.AST_STRICT_LAB_BRANCH,
                                ),
                                module_path=module_path,
                                symbol=f"branch:{node.lineno}",
                                discovery_kind=DiscoveryKind.AST_STRICT_LAB_BRANCH,
                                line_no=node.lineno,
                                indicator="mode_branch",
                            )
                        )
                        break

    return out


@lru_cache(maxsize=1)
def production_files_scanned_count() -> int:
    return len(scanned_module_paths())


@lru_cache(maxsize=1)
def scanned_module_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _is_candidate_file(path):
                paths.append(_normalize_repo_path(path))
    return tuple(paths)


@lru_cache(maxsize=1)
def _discovery_scan_failures() -> tuple[ScanFailure, ...]:
    failures: list[ScanFailure] = []
    for module_path in scanned_module_paths():
        full = _REPO_ROOT / module_path
        try:
            source = full.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(ScanFailure(module_path=module_path, reason=str(exc)))
            continue
        try:
            ast.parse(source, filename=module_path)
        except SyntaxError as exc:
            failures.append(ScanFailure(module_path=module_path, reason=f"syntax:{exc}"))
    return tuple(failures)


@lru_cache(maxsize=1)
def raw_discovery_candidates() -> tuple[ProductionReadinessCandidate, ...]:
    candidates: dict[str, ProductionReadinessCandidate] = {}

    for anchor in _discover_anchors():
        candidates[anchor.candidate_id] = anchor

    for module_path in scanned_module_paths():
        if any(f.module_path == module_path for f in _discovery_scan_failures()):
            continue
        full = _REPO_ROOT / module_path
        source = full.read_text(encoding="utf-8")
        for cand in _discover_from_ast(module_path, source):
            candidates.setdefault(cand.candidate_id, cand)

    return tuple(sorted(candidates.values(), key=lambda c: c.candidate_id))


def discovery_scan_failures() -> tuple[ScanFailure, ...]:
    return _discovery_scan_failures()
