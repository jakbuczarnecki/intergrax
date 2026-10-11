# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R3-R1 mechanical gates — envelope version authority (qualification only)."""

from __future__ import annotations

import ast
from pathlib import Path

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    parse_runtime_event_payload,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.schema.registry import (
    PREVIEW_RUNTIME_SCHEMA_VERSIONS,
    RUNTIME_SCHEMA_REGISTRY,
    validate_schema_version,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_ROOT = _REPO_ROOT / "intergrax"
_OBSERVABILITY_ROOT = _PRODUCTION_ROOT / "runtime" / "observability"
_CANONICAL_OWNER = _PRODUCTION_ROOT / "contracts" / "runtime_event.py"

_V2_LITERAL = "runtime_event.v2"
_V1_LITERAL = "runtime_event.v1"


def _iter_production_python_files() -> tuple[Path, ...]:
    return tuple(sorted(_PRODUCTION_ROOT.rglob("*.py")))


def production_runtime_event_v2_literal_sites() -> tuple[tuple[str, int], ...]:
    """Every production file/line containing the exact v2 string literal."""
    sites: list[tuple[str, int]] = []
    for path in _iter_production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _V2_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def production_runtime_event_v1_literal_sites() -> tuple[tuple[str, int], ...]:
    sites: list[tuple[str, int]] = []
    for path in _iter_production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _V1_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def count_runtime_event_v2_semantic_authority_literals() -> int:
    """Lines that define the canonical current envelope version string in the contract owner."""
    owner_text = _CANONICAL_OWNER.read_text(encoding="utf-8")
    tree = ast.parse(owner_text)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RUNTIME_EVENT_SCHEMA_VERSION":
                if (
                    isinstance(node.value, ast.Constant)
                    and node.value.value == _V2_LITERAL
                ):
                    count += 1
    return count


def observability_runtime_event_v2_literal_sites() -> tuple[tuple[str, int], ...]:
    sites: list[tuple[str, int]] = []
    if not _OBSERVABILITY_ROOT.is_dir():
        return ()
    for path in sorted(_OBSERVABILITY_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _V2_LITERAL in line:
                sites.append((rel, line_no))
    return tuple(sites)


def registry_runtime_event_version_parity_violations() -> tuple[str, ...]:
    if RUNTIME_SCHEMA_REGISTRY.get("runtime_event") != RUNTIME_EVENT_SCHEMA_VERSION:
        return (
            "RUNTIME_SCHEMA_REGISTRY['runtime_event'] != RUNTIME_EVENT_SCHEMA_VERSION",
        )
    return ()


def synthetic_registry_drift_violations() -> tuple[str, ...]:
    """Probe B: registry current version differs from contract constant."""
    drifted = {**RUNTIME_SCHEMA_REGISTRY, "runtime_event": "runtime_event.v99"}
    if drifted["runtime_event"] == RUNTIME_EVENT_SCHEMA_VERSION:
        return ("probe did not simulate drift",)
    return ()


def synthetic_second_v2_literal_owner(
    extra_site: tuple[str, int] | None = None,
) -> tuple[str, ...]:
    """Probe A: second current-version literal outside canonical owner."""
    sites = list(production_runtime_event_v2_literal_sites())
    if extra_site is not None:
        sites.append(extra_site)
    if len(sites) != 1:
        return (f"expected 1 production v2 literal site, got {len(sites)}: {sites}",)
    rel, _ = sites[0]
    if rel != "intergrax/contracts/runtime_event.py":
        return (f"canonical v2 literal not in contract owner: {rel}",)
    return ()


def discover_persisted_runtime_event_read_paths() -> tuple[str, ...]:
    """Mechanical discovery: production modules calling parse_runtime_event_payload."""
    needle = "parse_runtime_event_payload"
    paths: list[str] = []
    for path in _iter_production_python_files():
        text = path.read_text(encoding="utf-8")
        if needle in text:
            paths.append(path.relative_to(_REPO_ROOT).as_posix())
    return tuple(sorted(set(paths)))


def minimal_runtime_event_payload_dict(
    schema_version: object,
) -> dict[str, object]:
    data: dict[str, object] = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "event_type": RuntimeEventType.TASK_CREATED.value,
        "phase": ExecutionPhase.INTAKE.value,
    }
    if schema_version is not None:
        data["schema_version"] = schema_version
    return data


def runtime_event_default_schema_version() -> str:
    return RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    ).schema_version


__all__ = [
    "RUNTIME_EVENT_SCHEMA_VERSION",
    "count_runtime_event_v2_semantic_authority_literals",
    "discover_persisted_runtime_event_read_paths",
    "minimal_runtime_event_payload_dict",
    "observability_runtime_event_v2_literal_sites",
    "parse_runtime_event_payload",
    "production_runtime_event_v1_literal_sites",
    "production_runtime_event_v2_literal_sites",
    "registry_runtime_event_version_parity_violations",
    "runtime_event_default_schema_version",
    "synthetic_registry_drift_violations",
    "synthetic_second_v2_literal_owner",
    "validate_schema_version",
    "PREVIEW_RUNTIME_SCHEMA_VERSIONS",
]
