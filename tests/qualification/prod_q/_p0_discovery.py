# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 mechanical closed-world discovery over bounded production roots."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.prod_q._p0_types import (
    AnchorResolutionFailure,
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
    (
        "intergrax/runtime/execution/local_execution_capacity_admission.py",
        "LocalExecutionCapacityAdmission",
        "capacity",
    ),
    (
        "intergrax/contracts/execution_capacity/admission_decision.py",
        "assess_root_execution_capacity",
        "capacity",
    ),
    ("intergrax/hosting/engine/engine.py", "HostedApplicationEngine", "health"),
    ("intergrax/hosting/engine/engine.py", "health_snapshot", "health"),
    ("intergrax/applications/_shared/auditability_health_wiring.py", "project_host_auditability_health_facts", "readiness"),
    ("intergrax/runtime/observability/auditability_health.py", "build_auditability_health_snapshot", "readiness"),
    (
        "intergrax/applications/_shared/capability_health/providers/dependency_validation.py",
        "DependencyValidationHealthProvider",
        "health",
    ),
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
    (
        "intergrax/runtime/execution/lab_reference_agent_runtime.py",
        "build_lab_agent_runtime_config_from_merged",
        "mode",
    ),
    ("intergrax/fastapi_core/routers/health.py", "health", "health"),
    (
        "applications/governed_contractor_application/host/factory.py",
        "create_governed_contractor_backend_app",
        "startup",
    ),
)

_ANCHOR_MODULE_SYMBOLS: Final[frozenset[tuple[str, str]]] = frozenset(
    (module_path, symbol) for module_path, symbol, _ in _CANONICAL_ANCHORS
)

_SANDBOX_PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
)

_LAST_DUPLICATE_CANDIDATE_IDS: tuple[str, ...] = ()


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


def _module_defines_symbol(module_path: str, symbol: str) -> tuple[bool, int, str]:
    full = _REPO_ROOT / module_path
    if not full.is_file():
        return False, 0, "missing module file"
    try:
        source = full.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=module_path)
    except OSError as exc:
        return False, 0, f"read:{exc}"
    except SyntaxError as exc:
        return False, 0, f"syntax:{exc}"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == symbol:
                return True, node.lineno, "module-level"
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == symbol:
                    return True, item.lineno, f"class:{node.name}"
    return False, 0, "symbol not found"


@lru_cache(maxsize=1)
def anchor_resolution_failures() -> tuple[AnchorResolutionFailure, ...]:
    failures: list[AnchorResolutionFailure] = []
    for module_path, symbol, _indicator in _CANONICAL_ANCHORS:
        ok, _line, reason = _module_defines_symbol(module_path, symbol)
        if not ok:
            failures.append(
                AnchorResolutionFailure(module_path=module_path, symbol=symbol, reason=reason)
            )
    return tuple(failures)


def _discover_anchors() -> list[ProductionReadinessCandidate]:
    out: list[ProductionReadinessCandidate] = []
    for module_path, symbol, indicator in _CANONICAL_ANCHORS:
        ok, line_no, _reason = _module_defines_symbol(module_path, symbol)
        if not ok:
            continue
        out.append(
            ProductionReadinessCandidate(
                candidate_id=_candidate_id(module_path, symbol, DiscoveryKind.ANCHOR_REGISTRY),
                module_path=module_path,
                symbol=symbol,
                discovery_kind=DiscoveryKind.ANCHOR_REGISTRY,
                line_no=line_no,
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
    out: dict[str, ProductionReadinessCandidate] = {}
    try:
        tree = ast.parse(source, filename=module_path)
    except SyntaxError:
        return []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if (module_path, node.name) in _ANCHOR_MODULE_SYMBOLS:
                continue
            indicator = _match_symbol_indicator(node.name)
            if indicator:
                kind = (
                    DiscoveryKind.AST_LIFESPAN_HANDLER
                    if node.name == "lifespan"
                    else DiscoveryKind.AST_SYMBOL_INDICATOR
                )
                cid = _candidate_id(module_path, node.name, kind)
                out[cid] = ProductionReadinessCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    symbol=node.name,
                    discovery_kind=kind,
                    line_no=node.lineno,
                    indicator=indicator,
                )
            if any(
                isinstance(dec, ast.Name) and dec.id == "lifespan"
                for dec in node.decorator_list
            ):
                cid = _candidate_id(module_path, node.name, DiscoveryKind.AST_LIFESPAN_HANDLER)
                out[cid] = ProductionReadinessCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    symbol=node.name,
                    discovery_kind=DiscoveryKind.AST_LIFESPAN_HANDLER,
                    line_no=node.lineno,
                    indicator="lifespan",
                )
        elif isinstance(node, ast.ClassDef):
            if (module_path, node.name) in _ANCHOR_MODULE_SYMBOLS:
                continue
            indicator = _match_symbol_indicator(node.name)
            if indicator:
                cid = _candidate_id(module_path, node.name, DiscoveryKind.AST_SYMBOL_INDICATOR)
                out[cid] = ProductionReadinessCandidate(
                    candidate_id=cid,
                    module_path=module_path,
                    symbol=node.name,
                    discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
                    line_no=node.lineno,
                    indicator=indicator,
                )
            for item in node.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                method_indicator = _match_symbol_indicator(item.name)
                if method_indicator:
                    symbol = f"{node.name}.{item.name}"
                    cid = _candidate_id(module_path, symbol, DiscoveryKind.AST_METHOD_INDICATOR)
                    out[cid] = ProductionReadinessCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        symbol=symbol,
                        discovery_kind=DiscoveryKind.AST_METHOD_INDICATOR,
                        line_no=item.lineno,
                        indicator=method_indicator,
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            for prefix in _CONTRACT_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    symbol = prefix.split(".")[-1]
                    cid = _candidate_id(module_path, f"import:{prefix}", DiscoveryKind.AST_IMPORT_CONTRACT)
                    out[cid] = ProductionReadinessCandidate(
                        candidate_id=cid,
                        module_path=module_path,
                        symbol=f"import:{prefix}",
                        discovery_kind=DiscoveryKind.AST_IMPORT_CONTRACT,
                        line_no=node.lineno,
                        indicator="contract_import",
                    )
                    break
        if isinstance(node, ast.If):
            for test in ast.walk(node.test):
                if isinstance(test, ast.Constant) and isinstance(test.value, str):
                    val = test.value.lower()
                    if any(tok in val for tok in ("lab", "dev", "strict", "production")):
                        cid = _candidate_id(
                            module_path,
                            f"branch:{node.lineno}",
                            DiscoveryKind.AST_STRICT_LAB_BRANCH,
                        )
                        out[cid] = ProductionReadinessCandidate(
                            candidate_id=cid,
                            module_path=module_path,
                            symbol=f"branch:{node.lineno}",
                            discovery_kind=DiscoveryKind.AST_STRICT_LAB_BRANCH,
                            line_no=node.lineno,
                            indicator="mode_branch",
                        )
                        break

    return list(out.values())


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
    global _LAST_DUPLICATE_CANDIDATE_IDS
    candidates: dict[str, ProductionReadinessCandidate] = {}
    duplicate_ids: list[str] = []

    for anchor in _discover_anchors():
        existing = candidates.get(anchor.candidate_id)
        if existing is not None and existing != anchor:
            duplicate_ids.append(anchor.candidate_id)
        candidates[anchor.candidate_id] = anchor

    for module_path in scanned_module_paths():
        if any(f.module_path == module_path for f in _discovery_scan_failures()):
            continue
        full = _REPO_ROOT / module_path
        source = full.read_text(encoding="utf-8")
        for cand in _discover_from_ast(module_path, source):
            existing = candidates.get(cand.candidate_id)
            if existing is not None and existing != cand:
                duplicate_ids.append(cand.candidate_id)
            elif existing is None:
                candidates[cand.candidate_id] = cand

    _LAST_DUPLICATE_CANDIDATE_IDS = tuple(sorted(frozenset(duplicate_ids)))
    return tuple(sorted(candidates.values(), key=lambda c: c.candidate_id))


def raw_discovery_duplicate_ids() -> tuple[str, ...]:
    raw_discovery_candidates()
    return _LAST_DUPLICATE_CANDIDATE_IDS


def discovery_scan_failures() -> tuple[ScanFailure, ...]:
    return _discovery_scan_failures()


_PRODUCTION_ENTRYPOINT_MODULES: Final[tuple[str, ...]] = (
    "applications/governed_contractor_application/host/factory.py",
    "applications/local_workspace_application/host/main.py",
    "applications/research_application/host/main.py",
    "intergrax/hosting/engine/engine.py",
)


def _resolve_import_to_module_paths(
    importing_module: str,
    import_module: str,
    level: int,
) -> tuple[str, ...]:
    if level > 0:
        base_parts = importing_module.split("/")[:-level]
        if not import_module:
            target = "/".join(base_parts)
        else:
            target = "/".join(base_parts + import_module.split("."))
    else:
        target = import_module.replace(".", "/")
    candidates = (
        f"{target}.py",
        f"{target}/__init__.py",
    )
    scanned = frozenset(scanned_module_paths())
    return tuple(path for path in candidates if path in scanned)


@lru_cache(maxsize=1)
def modules_reachable_from_production_entrypoints() -> frozenset[str]:
    scanned = frozenset(scanned_module_paths())
    reachable: set[str] = set()
    queue: list[str] = []
    for entry in _PRODUCTION_ENTRYPOINT_MODULES:
        if entry in scanned:
            reachable.add(entry)
            queue.append(entry)
    while queue:
        module_path = queue.pop()
        full = _REPO_ROOT / module_path
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"), filename=module_path)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None and node.level == 0:
                    continue
                targets = _resolve_import_to_module_paths(
                    module_path,
                    node.module or "",
                    node.level,
                )
                for target in targets:
                    if target not in reachable:
                        reachable.add(target)
                        queue.append(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    for candidate in (f"{top}/__init__.py", f"{top}.py"):
                        if candidate in scanned and candidate not in reachable:
                            reachable.add(candidate)
                            queue.append(candidate)
    return frozenset(reachable)


@lru_cache(maxsize=1)
def symbol_enclosing_class_by_module() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for module_path in scanned_module_paths():
        full = _REPO_ROOT / module_path
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"), filename=module_path)
        except (OSError, SyntaxError):
            continue
        symbols: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                symbols[node.name] = ""
            elif isinstance(node, ast.ClassDef):
                symbols[node.name] = ""
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        symbols[f"{node.name}.{item.name}"] = node.name
        index[module_path] = symbols
    return index


@lru_cache(maxsize=1)
def module_contract_import_prefixes() -> dict[str, tuple[str, ...]]:
    prefixes_by_module: dict[str, tuple[str, ...]] = {}
    for module_path in scanned_module_paths():
        full = _REPO_ROOT / module_path
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"), filename=module_path)
        except (OSError, SyntaxError):
            continue
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module
            for prefix in _CONTRACT_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    found.append(prefix)
                    break
        if found:
            prefixes_by_module[module_path] = tuple(sorted(frozenset(found)))
    return prefixes_by_module


def _attribute_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _self_method_calls_in_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    called: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        attr = _attribute_call_name(node.func)
        if attr is None:
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "self":
                called.add(attr)
    return frozenset(called)


@lru_cache(maxsize=256)
def class_method_self_calls(module_path: str) -> dict[tuple[str, str], frozenset[str]]:
    """Map (class_name, method_name) -> method names invoked via self.<name> in that method."""
    full = _REPO_ROOT / module_path
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=module_path)
    except (OSError, SyntaxError):
        return {}
    out: dict[tuple[str, str], frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            out[(node.name, item.name)] = _self_method_calls_in_function(item)
    return out


@lru_cache(maxsize=256)
def external_attribute_callers_of_method(
    module_path: str,
    class_name: str,
    method_name: str,
) -> frozenset[tuple[str, str]]:
    """Callers outside ``class_name`` that invoke ``.<method_name>`` (static attribute shape only)."""
    full = _REPO_ROOT / module_path
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=module_path)
    except (OSError, SyntaxError):
        return frozenset()

    callers: set[tuple[str, str]] = set()

    def scan_function(func: ast.FunctionDef | ast.AsyncFunctionDef, scope: str) -> None:
        if scope == class_name:
            return
        for call in ast.walk(func):
            if not isinstance(call, ast.Call):
                continue
            if _attribute_call_name(call.func) != method_name:
                continue
            callers.add((scope, func.name))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            scan_function(node, "<module>")
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    scan_function(item, node.name)
    return frozenset(callers)
