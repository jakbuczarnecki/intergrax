# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — Evidence Plane ownership map and static negative checks (qualification only)."""

from __future__ import annotations

import ast
from pathlib import Path

EVIDENCE_PLANE_OWNERSHIP: tuple[tuple[str, str], ...] = (
    ("Execution lifecycle", "ExecutionRuntime"),
    ("Event identity", "RuntimeEvent"),
    ("Durable evidence", "RuntimeEventPersistence"),
    ("Journal projection", "UnifiedRunJournal"),
    ("Export boundary", "ObservabilityExportEnvelope"),
    ("Historical reconstruction", "HistoricalReconstructionService"),
    ("Lineage", "ExecutionLineagePersistence"),
    ("Checkpoint", "RuntimeCheckpoint / TaskCheckpointPersistence"),
    ("Governance", "Canonical Governance Plane"),
    ("Authority", "Canonical Authority Plane"),
)

FROZEN_EVIDENCE_PLANE_CONTRACTS: tuple[tuple[str, str], ...] = (
    ("Durable evidence", "FROZEN"),
    ("Tenant isolation", "FROZEN"),
    ("Event identity", "FROZEN"),
    ("Journal completeness", "FROZEN"),
    ("Export safety", "FROZEN"),
    ("Reconstruction semantics", "FROZEN"),
    ("As-of semantics", "FROZEN"),
    ("Bitemporal semantics", "FROZEN"),
)

_FORBIDDEN_EXECUTION_CONTROL_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "retry",
        "resume",
        "schedule",
        "mint_attempt",
        "mint_run",
        "approve",
        "deny",
        "mutate_execution",
    },
)

_FORBIDDEN_BARE_EXECUTE = "execute"

_EVIDENCE_PLANE_AST_ROOTS: tuple[str, ...] = (
    "intergrax/runtime/events",
    "intergrax/runtime/observability",
    "intergrax/contracts/historical_reconstruction.py",
)


def evidence_plane_static_scan_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for relative in _EVIDENCE_PLANE_AST_ROOTS:
        path = repo_root / relative
        if path.is_dir():
            roots.extend(
                p
                for p in path.rglob("*.py")
                if "__pycache__" not in p.parts
            )
        elif path.is_file():
            roots.append(path)
    return roots


def _call_symbol(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def collect_forbidden_execution_control_calls(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for path in evidence_plane_static_scan_roots(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == _FORBIDDEN_BARE_EXECUTE:
                violations.append(f"{rel}:{node.lineno}:{_FORBIDDEN_BARE_EXECUTE}")
                continue
            symbol = _call_symbol(node.func)
            if symbol in _FORBIDDEN_EXECUTION_CONTROL_ATTRIBUTES:
                violations.append(f"{rel}:{node.lineno}:{symbol}")
    return sorted(violations)
