# © Artur Czarnecki. All rights reserved.

"""Closed-world discovery parity: candidates = semantic surfaces + evidence-backed exclusions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from intergrax.runtime.events.payload_registry import list_registered_payload_schema_ids
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY

from tests.qualification.compat_x._compat_x_ast_signals import (
    ClassVersionFieldSignal,
    extract_class_version_fields,
    extract_migration_signals,
    extract_module_version_constants,
    extract_public_export_signals,
    extract_shim_path_signal,
    extract_wire_persistence_signals,
    class_has_schema_version_field,
    parse_module,
)
from tests.qualification.compat_x._compat_x_classifiers import classify_migration_module, classify_shim_module
from tests.qualification.compat_x._compat_x_types import (
    CompatDiscoveryExclusion,
    DiscoveredCompatSurface,
    DiscoveryCandidate,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOT_REL_PATHS: Final[tuple[str, ...]] = (
    "intergrax/contracts",
    "intergrax/contracts/migrations",
    "intergrax/runtime/schema",
    "intergrax/runtime/events",
    "intergrax/runtime",
    "intergrax/core/plugins",
    "intergrax/core/distribution",
    "intergrax/integrations/contracts",
    "intergrax/integrations/registry",
    "intergrax/hosting/contracts",
    "intergrax/tools",
    "intergrax/skills",
    "intergrax/marketplace",
    "intergrax/agent_distribution",
    "intergrax/proofs",
    "intergrax/proof_data",
    "intergrax/compat",
    "intergrax/applications",
    "applications/contracts",
)

_EXCLUDE_PATH_PARTS: Final[tuple[str, ...]] = (
    "/tests/",
    "\\tests\\",
    "/qualification/",
    "\\qualification\\",
    "/examples/",
    "\\examples\\",
    "/scaffold/",
    "\\scaffold\\",
)

_MECHANISM_MODULE_MARKERS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "intergrax/core/plugins/package_contract.py",
        "mechanism.platform_plugin_manifest",
        "PlatformPluginManifest",
    ),
    (
        "intergrax/integrations/contracts/external_contract_compatibility.py",
        "mechanism.external_contract_compatibility",
        "ExternalContractCompatibilityAssessment",
    ),
)


@dataclass(frozen=True, slots=True)
class ClosedWorldReport:
    raw_candidates: tuple[DiscoveryCandidate, ...]
    semantic_surfaces: tuple[DiscoveredCompatSurface, ...]
    exclusions: tuple[CompatDiscoveryExclusion, ...]
    discovery_counts_by_kind: dict[str, int]
    unclassified_candidate_ids: frozenset[str]
    resolved_candidate_ids: frozenset[str]


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_candidate_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part in normalized:
            return False
    return True


def _candidate(
    *,
    candidate_id: str,
    path: str,
    discovery_kind: str,
    discovered_signal: str,
    semantic_identity: str,
    current_version: str | None,
    version_source: str | None,
) -> DiscoveryCandidate:
    return DiscoveryCandidate(
        candidate_id=candidate_id,
        path=path,
        discovery_kind=discovery_kind,
        discovered_signal=discovered_signal,
        semantic_identity=semantic_identity,
        current_version=current_version,
        version_source=version_source,
    )


def _registry_contract_candidates() -> list[DiscoveryCandidate]:
    found: list[DiscoveryCandidate] = []
    for entry in CONTRACT_SCHEMA_REGISTRY:
        module_path = entry.module_path.replace(".", "/") + ".py"
        identity = f"registry.contract:{entry.contract_name}"
        found.append(
            _candidate(
                candidate_id=f"registry.contracts.{entry.contract_name}",
                path=module_path,
                discovery_kind="registry.contracts",
                discovered_signal="CONTRACT_SCHEMA_REGISTRY",
                semantic_identity=identity,
                current_version=entry.current_version,
                version_source="intergrax/contracts/migrations/registry.py:CONTRACT_SCHEMA_REGISTRY",
            )
        )
    return found


def _registry_runtime_candidates() -> list[DiscoveryCandidate]:
    found: list[DiscoveryCandidate] = []
    for schema_key, version in RUNTIME_SCHEMA_REGISTRY.items():
        identity = f"registry.runtime:{schema_key}"
        found.append(
            _candidate(
                candidate_id=f"registry.runtime.{schema_key}",
                path="intergrax/runtime/schema/registry.py",
                discovery_kind="registry.runtime",
                discovered_signal="RUNTIME_SCHEMA_REGISTRY",
                semantic_identity=identity,
                current_version=version,
                version_source="intergrax/runtime/schema/registry.py:RUNTIME_SCHEMA_REGISTRY",
            )
        )
    return found


def _registry_event_candidates() -> list[DiscoveryCandidate]:
    found: list[DiscoveryCandidate] = []
    for schema_id in list_registered_payload_schema_ids():
        identity = f"registry.event:{schema_id}"
        found.append(
            _candidate(
                candidate_id=f"event.payload.{schema_id}",
                path="intergrax/runtime/events/payload_registry.py",
                discovery_kind="event.payload",
                discovered_signal="list_registered_payload_schema_ids",
                semantic_identity=identity,
                current_version=schema_id,
                version_source="intergrax/runtime/events/payload_registry.py",
            )
        )
    return found


def _ast_candidates_for_module(module_path: str, source: str) -> list[DiscoveryCandidate]:
    tree = parse_module(module_path, source)
    found: list[DiscoveryCandidate] = []
    for signal in extract_module_version_constants(module_path, tree):
        identity = f"const:{module_path}:{signal.const_name}"
        found.append(
            _candidate(
                candidate_id=f"const.{module_path.replace('/', '.')}::{signal.const_name}",
                path=module_path,
                discovery_kind="version.constant",
                discovered_signal=signal.signal,
                semantic_identity=identity,
                current_version=signal.version_literal,
                version_source=f"{module_path}:{signal.const_name}",
            )
        )
    for signal in extract_class_version_fields(module_path, tree):
        version = signal.version_literal or "unknown"
        identity = f"class.field:{module_path}:{signal.class_name}:{signal.field_name}"
        if signal.version_literal:
            identity = f"schema.literal:{signal.version_literal}"
        found.append(
            _candidate(
                candidate_id=f"class.{module_path.replace('/', '.')}::{signal.class_name}.{signal.field_name}",
                path=module_path,
                discovery_kind="class.field.version",
                discovered_signal=signal.signal,
                semantic_identity=identity,
                current_version=signal.version_literal,
                version_source=f"{module_path}:{signal.class_name}.{signal.field_name}",
            )
        )
    wire_index = 0
    for signal in extract_wire_persistence_signals(module_path, tree):
        wire_index += 1
        class_part = signal.class_name or "module_level"
        identity = f"wire:{module_path}:{class_part}:{signal.call_name}:{wire_index}"
        found.append(
            _candidate(
                candidate_id=f"wire.{module_path.replace('/', '.')}::{class_part}.{signal.call_name}:{signal.lineno}:{wire_index}",
                path=module_path,
                discovery_kind="wire.persistence",
                discovered_signal=signal.signal,
                semantic_identity=identity,
                current_version=None,
                version_source=f"{module_path}:{signal.lineno}",
            )
        )
        if signal.class_name and not class_has_schema_version_field(tree, signal.class_name):
            if _persisted_boundary_without_version(module_path, source, signal.class_name):
                _append_persisted_defect(found, module_path, signal.class_name, source)
        elif (
            signal.call_name == "model_dump"
            and _persisted_boundary_without_version(module_path, source, "")
        ):
            for class_name in _unversioned_model_classes(tree):
                _append_persisted_defect(found, module_path, class_name, source)
    _ast_candidates_public_exports(module_path, tree, found)
    return found


def _unversioned_model_classes(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not class_has_schema_version_field(tree, node.name):
            names.append(node.name)
    return names


def _append_persisted_defect(
    found: list[DiscoveryCandidate],
    module_path: str,
    class_name: str,
    source: str,
) -> None:
    if not _persisted_boundary_without_version(module_path, source, class_name):
        return
    defect_id = f"defect.persisted.{module_path.replace('/', '.')}::{class_name}"
    if any(c.candidate_id == defect_id for c in found):
        return
    defect_identity = f"persisted.without_version:{module_path}:{class_name}"
    found.append(
        _candidate(
            candidate_id=defect_id,
            path=module_path,
            discovery_kind="defect.persisted_without_version",
            discovered_signal=f"missing_schema_version:{class_name}",
            semantic_identity=defect_identity,
            current_version=None,
            version_source=f"{module_path}:{class_name}",
        )
    )


def _ast_candidates_public_exports(module_path: str, tree: ast.Module, found: list[DiscoveryCandidate]) -> None:
    for signal in extract_public_export_signals(module_path, tree):
        identity = f"export:{module_path}:{signal.exported_name}"
        found.append(
            _candidate(
                candidate_id=f"export.{module_path.replace('/', '.')}::{signal.exported_name}",
                path=module_path,
                discovery_kind="public.export",
                discovered_signal=signal.signal,
                semantic_identity=identity,
                current_version=None,
                version_source=module_path,
            )
        )


def _migration_candidates_for_module(module_path: str, source: str) -> list[DiscoveryCandidate]:
    tree = parse_module(module_path, source)
    signals = extract_migration_signals(module_path, tree, source)
    if not signals:
        return []
    migration_class = classify_migration_module(module_path, source)
    identity = f"migration:{module_path}"
    return [
        _candidate(
            candidate_id=f"migration.{module_path.replace('/', '.')}",
            path=module_path,
            discovery_kind="migration.mechanism",
            discovered_signal=signals[0].signal,
            semantic_identity=identity,
            current_version=migration_class.value,
            version_source=module_path,
        )
    ]


def _shim_candidates_for_module(module_path: str, source: str) -> list[DiscoveryCandidate]:
    path_signal = extract_shim_path_signal(module_path)
    shim_class = classify_shim_module(module_path, source)
    from tests.qualification.compat_x._compat_x_types import ShimClass

    if path_signal is None and shim_class == ShimClass.NOT_APPLICABLE:
        return []
    identity = f"shim:{module_path}"
    return [
        _candidate(
            candidate_id=f"shim.{module_path.replace('/', '.')}",
            path=module_path,
            discovery_kind="compat.shim",
            discovered_signal=path_signal.signal if path_signal else "shim.classifier",
            semantic_identity=identity,
            current_version=shim_class.value,
            version_source=module_path,
        )
    ]


def _persisted_boundary_without_version(module_path: str, source: str, class_name: str) -> bool:
    if module_path.startswith("synthetic/"):
        if class_name:
            return class_name in source
        return "model_dump" in source and "persist" in source.lower()
    lowered = source.lower()
    if class_name not in source:
        return False
    if "model_dump" not in lowered:
        return False
    return any(marker in lowered for marker in ("persist_", "save_checkpoint", "put_document", "append_event"))


def _mechanism_candidates() -> list[DiscoveryCandidate]:
    found: list[DiscoveryCandidate] = []
    for module_path, surface_id, identity_name in _MECHANISM_MODULE_MARKERS:
        full = _REPO_ROOT / module_path
        if not full.is_file():
            continue
        found.append(
            _candidate(
                candidate_id=surface_id,
                path=module_path,
                discovery_kind="mechanism.policy_owner",
                discovered_signal=f"policy_owner_module:{identity_name}",
                semantic_identity=f"mechanism:{surface_id}",
                current_version="policy",
                version_source=module_path,
            )
        )
    return found


def discover_ast_candidates_from_repo() -> list[DiscoveryCandidate]:
    found: list[DiscoveryCandidate] = []
    seen_modules: set[str] = set()
    for rel_root in _DISCOVERY_ROOT_REL_PATHS:
        root = _REPO_ROOT / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate_file(path):
                continue
            module_path = _normalize_repo_path(path)
            if module_path in seen_modules:
                continue
            seen_modules.add(module_path)
            if module_path.endswith("registry.py") and "migrations/registry.py" in module_path:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                found.extend(_ast_candidates_for_module(module_path, source))
                found.extend(_migration_candidates_for_module(module_path, source))
                found.extend(_shim_candidates_for_module(module_path, source))
            except SyntaxError:
                continue
    return found


def discover_candidates_from_source(module_path: str, source: str) -> list[DiscoveryCandidate]:
    """Adversarial helper: run the same discovery pipeline on synthetic source."""
    found: list[DiscoveryCandidate] = []
    found.extend(_ast_candidates_for_module(module_path, source))
    found.extend(_migration_candidates_for_module(module_path, source))
    found.extend(_shim_candidates_for_module(module_path, source))
    return found


def _wire_exclusion(candidate: DiscoveryCandidate) -> CompatDiscoveryExclusion | None:
    if candidate.discovery_kind != "wire.persistence":
        return None
    return CompatDiscoveryExclusion(
        exclusion_id=f"exclusion.{candidate.candidate_id}",
        path=candidate.path,
        discovered_signal=candidate.discovered_signal,
        reason="internal_wire_call_not_public_compat_surface",
        evidence=(
            f"{candidate.path}: {candidate.discovered_signal} is a serialization helper; "
            "classified after discovery as non-public/non-authoritative wire evidence"
        ),
    )


def _export_exclusion(candidate: DiscoveryCandidate) -> CompatDiscoveryExclusion | None:
    if candidate.discovery_kind != "public.export":
        return None
    return CompatDiscoveryExclusion(
        exclusion_id=f"exclusion.{candidate.candidate_id}",
        path=candidate.path,
        discovered_signal=candidate.discovered_signal,
        reason="export_list_requires_semantic_classification",
        evidence=f"{candidate.path}: __all__ export discovered; not auto PUBLIC_STABLE without contract evidence",
    )


def _defect_surface(candidate: DiscoveryCandidate) -> DiscoveredCompatSurface | None:
    if candidate.discovery_kind != "defect.persisted_without_version":
        return None
    return DiscoveredCompatSurface(
        surface_id=f"semantic.{candidate.semantic_identity}",
        semantic_identity=candidate.semantic_identity,
        owner_module_path=candidate.path,
        contract_schema_identity=candidate.discovered_signal,
        version_source=candidate.version_source or candidate.path,
        current_version="MISSING",
        discovery_kind=candidate.discovery_kind,
        evidence_kinds=(candidate.discovery_kind,),
    )


def _semantic_surface_from_candidate(candidate: DiscoveryCandidate) -> DiscoveredCompatSurface:
    return DiscoveredCompatSurface(
        surface_id=f"semantic.{candidate.semantic_identity}",
        semantic_identity=candidate.semantic_identity,
        owner_module_path=candidate.path,
        contract_schema_identity=candidate.semantic_identity.split(":")[-1],
        version_source=candidate.version_source or candidate.path,
        current_version=candidate.current_version or "unknown",
        discovery_kind=candidate.discovery_kind,
        evidence_kinds=(candidate.discovery_kind,),
    )


def _merge_surface(existing: DiscoveredCompatSurface, candidate: DiscoveryCandidate) -> DiscoveredCompatSurface:
    kinds = tuple(sorted(set((*existing.evidence_kinds, candidate.discovery_kind))))
    version = existing.current_version
    if candidate.current_version and (version in {"unknown", "MISSING"} or candidate.discovery_kind.startswith("registry")):
        version = candidate.current_version
    return DiscoveredCompatSurface(
        surface_id=existing.surface_id,
        semantic_identity=existing.semantic_identity,
        owner_module_path=existing.owner_module_path,
        contract_schema_identity=existing.contract_schema_identity,
        version_source=existing.version_source,
        current_version=version,
        discovery_kind=existing.discovery_kind,
        evidence_kinds=kinds,
    )


@lru_cache(maxsize=1)
def build_closed_world_report() -> ClosedWorldReport:
    raw: list[DiscoveryCandidate] = []
    raw.extend(_registry_contract_candidates())
    raw.extend(_registry_runtime_candidates())
    raw.extend(_registry_event_candidates())
    raw.extend(discover_ast_candidates_from_repo())
    raw.extend(_mechanism_candidates())

    exclusions: list[CompatDiscoveryExclusion] = []
    semantic: dict[str, DiscoveredCompatSurface] = {}
    unclassified: set[str] = set()
    resolved: set[str] = set()
    counts: dict[str, int] = {}

    def _resolve_semantic(candidate: DiscoveryCandidate) -> None:
        resolved.add(candidate.candidate_id)
        key = candidate.semantic_identity
        if key in semantic:
            semantic[key] = _merge_surface(semantic[key], candidate)
        else:
            semantic[key] = _semantic_surface_from_candidate(candidate)

    for candidate in raw:
        counts[candidate.discovery_kind] = counts.get(candidate.discovery_kind, 0) + 1
        defect = _defect_surface(candidate)
        if defect is not None:
            resolved.add(candidate.candidate_id)
            semantic[defect.semantic_identity] = defect
            continue
        wire_ex = _wire_exclusion(candidate)
        if wire_ex is not None:
            resolved.add(candidate.candidate_id)
            exclusions.append(wire_ex)
            continue
        export_ex = _export_exclusion(candidate)
        if export_ex is not None:
            resolved.add(candidate.candidate_id)
            exclusions.append(export_ex)
            continue
        if candidate.discovery_kind in {"version.constant", "class.field.version", "registry.contracts"}:
            _resolve_semantic(candidate)
            continue
        if candidate.discovery_kind in {
            "registry.runtime",
            "event.payload",
            "migration.mechanism",
            "compat.shim",
            "mechanism.policy_owner",
        }:
            _resolve_semantic(candidate)
            continue
        unclassified.add(candidate.candidate_id)

    return ClosedWorldReport(
        raw_candidates=tuple(raw),
        semantic_surfaces=tuple(sorted(semantic.values(), key=lambda s: s.surface_id)),
        exclusions=tuple(exclusions),
        discovery_counts_by_kind=counts,
        unclassified_candidate_ids=frozenset(unclassified),
        resolved_candidate_ids=frozenset(resolved),
    )


def build_closed_world_report_with_extra_candidates(
    extra: tuple[DiscoveryCandidate, ...],
) -> ClosedWorldReport:
    """Parity helper for synthetic omission probes (extra candidates must classify)."""
    base = build_closed_world_report()
    raw = list(base.raw_candidates) + list(extra)
    exclusions = list(base.exclusions)
    semantic = {s.semantic_identity: s for s in base.semantic_surfaces}
    unclassified: set[str] = set()
    resolved = set(base.resolved_candidate_ids)
    counts = dict(base.discovery_counts_by_kind)

    def _resolve_semantic(candidate: DiscoveryCandidate) -> None:
        resolved.add(candidate.candidate_id)
        key = candidate.semantic_identity
        if key in semantic:
            semantic[key] = _merge_surface(semantic[key], candidate)
        else:
            semantic[key] = _semantic_surface_from_candidate(candidate)

    for candidate in extra:
        counts[candidate.discovery_kind] = counts.get(candidate.discovery_kind, 0) + 1
        defect = _defect_surface(candidate)
        if defect is not None:
            resolved.add(candidate.candidate_id)
            semantic[defect.semantic_identity] = defect
            continue
        wire_ex = _wire_exclusion(candidate)
        if wire_ex is not None:
            resolved.add(candidate.candidate_id)
            exclusions.append(wire_ex)
            continue
        export_ex = _export_exclusion(candidate)
        if export_ex is not None:
            resolved.add(candidate.candidate_id)
            exclusions.append(export_ex)
            continue
        _resolve_semantic(candidate)

    return ClosedWorldReport(
        raw_candidates=tuple(raw),
        semantic_surfaces=tuple(sorted(semantic.values(), key=lambda s: s.surface_id)),
        exclusions=tuple(exclusions),
        discovery_counts_by_kind=counts,
        unclassified_candidate_ids=frozenset(unclassified),
        resolved_candidate_ids=frozenset(resolved),
    )
