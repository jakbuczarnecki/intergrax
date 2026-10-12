# © Artur Czarnecki. All rights reserved.

"""TENANT-X mechanical tenant-surface discovery (single cached HEAD scan)."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.tenant_x._types import (
    ClosedWorldDiscoveryReport,
    TenantDiscoveryCandidate,
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

_TENANT_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"tenant_id", "tenant", "tenant_scope", "principal_tenant_id"}
)

_AMBIENT_DEFAULT_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r'tenant_id\s*:\s*str\s*=\s*["\']default["\']'), "ambient_default_field"),
    (re.compile(r'tenant_id\s*=\s*Field\s*\(\s*default\s*=\s*["\']default["\']'), "ambient_default_pydantic"),
    (re.compile(r'DEFAULT_TENANT_ID\s*=\s*["\']default["\']'), "ambient_default_constant"),
    (re.compile(r'PREFIX_TENANT_ID["\'],\s*["\']default["\']'), "ambient_default_env_fallback"),
)

_VALIDATOR_MARKERS: Final[tuple[str, ...]] = (
    "require_tenant_id",
    "validate_tenant",
    "_validate_tenant_id",
    "tenant_id must be non-empty",
    "tenant_id is required",
)

_CANONICAL_ANCHORS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "intergrax/runtime/execution/agent_runtime_io.py",
        "RuntimeRequest",
        "canonical.runtime_request",
    ),
    (
        "intergrax/contracts/runtime_execution_admission.py",
        "RootExecutionAuthorityAdmissionRequest",
        "canonical.execution_admission",
    ),
    (
        "intergrax/contracts/agent_runtime_governance.py",
        "GovernanceAuditEvent",
        "canonical.governance_event",
    ),
    (
        "intergrax/contracts/functional_evidence/tenant.py",
        "require_tenant_id_from_exec_ctx",
        "canonical.functional_evidence_tenant",
    ),
    (
        "intergrax/contracts/runtime_inspection/query.py",
        "RuntimeInspectionQuery",
        "canonical.runtime_inspection_query",
    ),
    (
        "intergrax/integrations/credentials/broker.py",
        "ScopedCredentialBroker",
        "canonical.credential_broker",
    ),
    (
        "intergrax/memory/memory_scope_authority.py",
        "assert_memory_scope_authorized",
        "canonical.memory_scope_authority",
    ),
    (
        "intergrax/contracts/tool_runtime_read.py",
        "ToolRuntimeExecutionScope",
        "canonical.tool_runtime_scope",
    ),
    (
        "intergrax/runtime/execution/child.py",
        "ChildExecutionRunner",
        "canonical.child_execution_runner",
    ),
)


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


def _surface_id(module_path: str, symbol: str, kind: str) -> str:
    return f"{module_path}::{symbol}::{kind}"


def _discover_from_ast(module_path: str, source: str) -> list[TenantDiscoveryCandidate]:
    out: list[TenantDiscoveryCandidate] = []
    try:
        tree = ast.parse(source, filename=module_path)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id in _TENANT_FIELD_NAMES:
                        kind = "class.field.tenant"
                        out.append(
                            TenantDiscoveryCandidate(
                                surface_id=_surface_id(module_path, f"{node.name}.{stmt.target.id}", kind),
                                module_path=module_path,
                                symbol=node.name,
                                discovery_kind=kind,
                                line_no=stmt.lineno,
                            )
                        )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in node.args.args:
                if arg.arg in _TENANT_FIELD_NAMES:
                    kind = "function.param.tenant"
                    out.append(
                        TenantDiscoveryCandidate(
                            surface_id=_surface_id(module_path, f"{node.name}.{arg.arg}", kind),
                            module_path=module_path,
                            symbol=node.name,
                            discovery_kind=kind,
                            line_no=node.lineno,
                        )
                    )

    for pattern, kind in _AMBIENT_DEFAULT_PATTERNS:
        for match in pattern.finditer(source):
            line_no = source[: match.start()].count("\n") + 1
            out.append(
                TenantDiscoveryCandidate(
                    surface_id=_surface_id(module_path, f"line{line_no}", kind),
                    module_path=module_path,
                    symbol=f"L{line_no}",
                    discovery_kind=kind,
                    line_no=line_no,
                )
            )

    for marker in _VALIDATOR_MARKERS:
        if marker in source:
            out.append(
                TenantDiscoveryCandidate(
                    surface_id=_surface_id(module_path, marker, "validator.marker"),
                    module_path=module_path,
                    symbol=marker,
                    discovery_kind="validator.marker",
                    line_no=1,
                )
            )
            break

    if "Optional[str]" in source and "tenant_id" in source:
        if "tenant_id: Optional[str]" in source or "tenant_id: str | None" in source:
            out.append(
                TenantDiscoveryCandidate(
                    surface_id=_surface_id(module_path, "optional_tenant_field", "optional.tenant"),
                    module_path=module_path,
                    symbol="optional_tenant",
                    discovery_kind="optional.tenant",
                    line_no=1,
                )
            )

    return out


def _canonical_anchor_candidates() -> tuple[TenantDiscoveryCandidate, ...]:
    anchors: list[TenantDiscoveryCandidate] = []
    for module_path, symbol, kind in _CANONICAL_ANCHORS:
        anchors.append(
            TenantDiscoveryCandidate(
                surface_id=_surface_id(module_path, symbol, kind),
                module_path=module_path,
                symbol=symbol,
                discovery_kind=kind,
                line_no=1,
            )
        )
    return tuple(anchors)


@lru_cache(maxsize=1)
def build_closed_world_discovery_report() -> ClosedWorldDiscoveryReport:
    scanned: list[str] = []
    candidates: dict[str, TenantDiscoveryCandidate] = {}

    for anchor in _canonical_anchor_candidates():
        candidates[anchor.surface_id] = anchor

    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate_file(path):
                continue
            rel = _normalize_repo_path(path)
            scanned.append(rel)
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "tenant" not in source and "Tenant" not in source:
                continue
            for item in _discover_from_ast(rel, source):
                candidates[item.surface_id] = item

    raw = tuple(sorted(candidates.values(), key=lambda c: c.surface_id))
    return ClosedWorldDiscoveryReport(
        module_paths_scanned=tuple(sorted(set(scanned))),
        raw_candidates=raw,
        unclassified_candidate_ids=frozenset(),
    )


def production_files_scanned_count() -> int:
    return len(build_closed_world_discovery_report().module_paths_scanned)


def discover_all_candidates() -> frozenset[TenantDiscoveryCandidate]:
    return frozenset(build_closed_world_discovery_report().raw_candidates)
