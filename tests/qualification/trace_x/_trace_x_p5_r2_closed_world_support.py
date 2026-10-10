# © Artur Czarnecki. All rights reserved.

"""Discovery and parity helpers for TRACE-X-P5-R2 closed-world qualification."""

from __future__ import annotations

import re
from pathlib import Path

from tests.qualification.trace_x._trace_x_p5_r2_closed_world_types import (
    ClosedWorldParityResult,
    ConfiguredExecutionPathClass,
    RegisteredConfiguredExecutionPath,
)

TRACE_X_P5_R2_CLOSED_WORLD_START_HEAD = "d565f36c00582fdadaa6ef3e6d5266265081b44d"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_DISCOVERY_MARKERS: tuple[str, ...] = (
    "CONFIGURED_ADOPTED",
    "ExecutionIntegrationConfigurationAdoption",
    "build_default_configured_relational_store_execution_binding",
    "ExecutionBoundConfiguredRelationalStorePort",
    "ExecutionBoundIntegrationResolution",
    "ConfiguredRelationalStoreExecutionBindingPort",
    "DefaultConfiguredIntegrationToolInvocationProjectionPort",
    "integration_configuration_provenance_requirement",
    "ExecutionIntegrationConfigurationPinningStore",
    "build_production_marketplace_qualified_capability_execution",
    "WorkerConfiguredCapabilityExecutionFulfillmentService",
    "marketplace_qualified_capability_execution_handler",
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
)


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


def discover_configured_execution_path_keys() -> frozenset[tuple[str, str]]:
    """File-level discovery: any production module touching configured/effective provenance seam."""
    keys: set[tuple[str, str]] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not any(marker in text for marker in _DISCOVERY_MARKERS):
                continue
            rel = _normalize_repo_path(path)
            keys.add((rel, "module"))
    return frozenset(keys)


def compare_paths_to_registry(
    discovered: frozenset[tuple[str, str]],
    registry: tuple[RegisteredConfiguredExecutionPath, ...],
) -> ClosedWorldParityResult:
    reg_keys: list[tuple[str, str]] = []
    duplicates: set[tuple[str, str]] = set()
    for row in registry:
        if row.key in reg_keys:
            duplicates.add(row.key)
        reg_keys.append(row.key)
    reg_set = frozenset(reg_keys)
    unknown = discovered - reg_set
    orphan = reg_set - discovered
    bypass = frozenset(
        row.key
        for row in registry
        if row.classification is ConfiguredExecutionPathClass.E_PRODUCTION_BYPASS
    )
    unclassified = frozenset(
        row.key
        for row in registry
        if row.classification is ConfiguredExecutionPathClass.F_UNCLEAR
    )
    return ClosedWorldParityResult(
        unknown=unknown,
        orphan=orphan,
        duplicate_registry_keys=frozenset(duplicates),
        production_bypass=bypass,
        unclassified=unclassified,
    )


def production_intergrax_py_files() -> list[Path]:
    root = _REPO_ROOT / "intergrax"
    return [p for p in root.rglob("*.py") if _is_discovery_candidate(p)]


def grep_production_pattern(pattern: str) -> list[str]:
    rx = re.compile(pattern)
    hits: list[str] = []
    for path in production_intergrax_py_files():
        text = path.read_text(encoding="utf-8")
        if rx.search(text):
            hits.append(_normalize_repo_path(path))
    return sorted(hits)
