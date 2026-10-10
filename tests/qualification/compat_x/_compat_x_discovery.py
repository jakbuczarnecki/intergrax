# © Artur Czarnecki. All rights reserved.

"""COMPAT-X mechanical closed-world discovery (current HEAD)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from intergrax.runtime.events.payload_registry import list_registered_payload_schema_ids
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY

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

_SCHEMA_CONST_RE = re.compile(r"^SCHEMA_[A-Z0-9_]+$")
_VERSION_CONST_SUFFIXES = ("_SCHEMA_VERSION", "_CONTRACT_VERSION")

_MIGRATION_OWNER_PATHS: Final[tuple[str, ...]] = (
    "intergrax/contracts/migrations/registry.py",
    "intergrax/runtime/schema/registry.py",
    "intergrax/runtime/observability/causal_evidence_index.py",
    "intergrax/runtime/events/spine_payload_codec.py",
    "intergrax/runtime/diagnostics/problem_occurrence_migration.py",
    "intergrax/applications/contracts/environment_profile/decision_profile_legacy.py",
    "intergrax/applications/contracts/environment_profile/normalization.py",
    "intergrax/compat/langchain/documents.py",
)

_SHIM_MODULE_PATHS: Final[tuple[str, ...]] = (
    "intergrax/compat/langchain/documents.py",
)


@dataclass(frozen=True, slots=True)
class DiscoveredCompatSurface:
    surface_id: str
    owner_module_path: str
    contract_schema_identity: str
    version_source: str
    current_version: str
    discovery_kind: str


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


def _literal_version(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return str(node.value)
    return None


def _const_surface_id(module_path: str, const_name: str) -> str:
    return f"const.{module_path.replace('/', '.')}::{const_name}"


def _is_version_constant_name(name: str) -> bool:
    if name.startswith("UNSUPPORTED_"):
        return False
    return (
        name == "MANIFEST_SCHEMA_VERSION"
        or any(name.endswith(suffix) for suffix in _VERSION_CONST_SUFFIXES)
        or _SCHEMA_CONST_RE.match(name) is not None
    )


def _surface_from_constant(
    module_path: str, name: str, value_node: ast.AST
) -> DiscoveredCompatSurface | None:
    if not _is_version_constant_name(name):
        return None
    version = _literal_version(value_node)
    if version is None:
        return None
    return DiscoveredCompatSurface(
        surface_id=_const_surface_id(module_path, name),
        owner_module_path=module_path,
        contract_schema_identity=name,
        version_source=f"{module_path}:{name}",
        current_version=version,
        discovery_kind="version.constant",
    )


def _extract_version_constants(module_path: str, tree: ast.Module) -> list[DiscoveredCompatSurface]:
    found: list[DiscoveredCompatSurface] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            surface = _surface_from_constant(module_path, node.targets[0].id, node.value)
            if surface is not None:
                found.append(surface)
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                continue
            surface = _surface_from_constant(module_path, node.target.id, node.value)
            if surface is not None:
                found.append(surface)
    return found


@lru_cache(maxsize=1)
def discover_contract_registry_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: set[DiscoveredCompatSurface] = set()
    for entry in CONTRACT_SCHEMA_REGISTRY:
        surfaces.add(
            DiscoveredCompatSurface(
                surface_id=f"registry.contracts.{entry.contract_name}",
                owner_module_path=entry.module_path.replace(".", "/") + ".py",
                contract_schema_identity=entry.contract_name,
                version_source="intergrax/contracts/migrations/registry.py:CONTRACT_SCHEMA_REGISTRY",
                current_version=entry.current_version,
                discovery_kind="registry.contracts",
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def discover_runtime_registry_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: set[DiscoveredCompatSurface] = set()
    for schema_key, version in RUNTIME_SCHEMA_REGISTRY.items():
        surfaces.add(
            DiscoveredCompatSurface(
                surface_id=f"registry.runtime.{schema_key}",
                owner_module_path="intergrax/runtime/schema/registry.py",
                contract_schema_identity=schema_key,
                version_source="intergrax/runtime/schema/registry.py:RUNTIME_SCHEMA_REGISTRY",
                current_version=version,
                discovery_kind="registry.runtime",
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def discover_event_payload_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: set[DiscoveredCompatSurface] = set()
    for schema_id in list_registered_payload_schema_ids():
        surfaces.add(
            DiscoveredCompatSurface(
                surface_id=f"event.payload.{schema_id}",
                owner_module_path="intergrax/runtime/events/payload_registry.py",
                contract_schema_identity=schema_id,
                version_source="intergrax/runtime/events/payload_registry.py:list_registered_payload_schema_ids",
                current_version=schema_id,
                discovery_kind="event.payload",
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def discover_ast_version_constant_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: dict[str, DiscoveredCompatSurface] = {}
    for rel_root in _DISCOVERY_ROOT_REL_PATHS:
        root = _REPO_ROOT / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate_file(path):
                continue
            module_path = _normalize_repo_path(path)
            if module_path.endswith("registry.py") and "migrations" in module_path:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text, filename=module_path)
            except SyntaxError:
                continue
            for item in _extract_version_constants(module_path, tree):
                surfaces[item.surface_id] = item
    return frozenset(surfaces.values())


@lru_cache(maxsize=1)
def discover_shim_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: set[DiscoveredCompatSurface] = set()
    for module_path in _SHIM_MODULE_PATHS:
        surfaces.add(
            DiscoveredCompatSurface(
                surface_id=f"shim.{module_path.replace('/', '.')}",
                owner_module_path=module_path,
                contract_schema_identity="langchain_document_bridge",
                version_source=module_path,
                current_version="metadata.schema_version:int",
                discovery_kind="compat.shim",
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def discover_migration_mechanism_surfaces() -> frozenset[DiscoveredCompatSurface]:
    surfaces: set[DiscoveredCompatSurface] = set()
    for index, module_path in enumerate(_MIGRATION_OWNER_PATHS):
        surfaces.add(
            DiscoveredCompatSurface(
                surface_id=f"migration.mechanism.{index}",
                owner_module_path=module_path,
                contract_schema_identity=f"migration@{module_path}",
                version_source=module_path,
                current_version="mechanism",
                discovery_kind="migration.mechanism",
            )
        )
    return frozenset(surfaces)


@lru_cache(maxsize=1)
def discover_mechanism_surfaces() -> frozenset[DiscoveredCompatSurface]:
    """Sanctioned compatibility owners (not duplicate registries)."""
    return frozenset(
        {
            DiscoveredCompatSurface(
                surface_id="mechanism.platform_plugin_manifest",
                owner_module_path="intergrax/core/plugins/package_contract.py",
                contract_schema_identity="PlatformPluginManifest",
                version_source="intergrax/core/plugins/package_contract.py:MANIFEST_SCHEMA_VERSION",
                current_version="1",
                discovery_kind="mechanism.plugin",
            ),
            DiscoveredCompatSurface(
                surface_id="mechanism.external_contract_compatibility",
                owner_module_path="intergrax/integrations/contracts/external_contract_compatibility.py",
                contract_schema_identity="ExternalContractCompatibilityAssessment",
                version_source="intergrax/integrations/contracts/external_contract_compatibility.py",
                current_version="advisory",
                discovery_kind="mechanism.integrations",
            ),
        }
    )


@lru_cache(maxsize=1)
def discover_all_compat_surfaces() -> frozenset[DiscoveredCompatSurface]:
    merged: dict[str, DiscoveredCompatSurface] = {}
    for batch in (
        discover_contract_registry_surfaces(),
        discover_runtime_registry_surfaces(),
        discover_event_payload_surfaces(),
        discover_ast_version_constant_surfaces(),
        discover_shim_surfaces(),
        discover_migration_mechanism_surfaces(),
        discover_mechanism_surfaces(),
    ):
        for surface in batch:
            merged[surface.surface_id] = surface
    return frozenset(merged.values())


def discover_all_compat_surface_ids() -> frozenset[str]:
    return frozenset(surface.surface_id for surface in discover_all_compat_surfaces())


# --- Adversarial probe helpers (synthetic; not production mechanisms) ---


def synthetic_probe_omitted_public_contract_id() -> str:
    return "registry.contracts.SyntheticOmittedProbe"


def classify_synthetic_unclassified_surface(surface_id: str) -> str:
    if surface_id.startswith("synthetic.unclassified."):
        return "UNCLASSIFIED"
    return "CLASSIFIED"
