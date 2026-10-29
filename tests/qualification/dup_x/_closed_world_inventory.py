# © Artur Czarnecki. All rights reserved.

"""DUP-X closed-world inventory — responsibility families + mechanical census."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.dup_x._findings_register import DUP_CLASSES
from tests.qualification.ebh_5._closed_world_discovery import (
    COMPOSITION_ROOT_HINTS,
)
from tests.qualification.ebh_5._closed_world_discovery import (
    discover_composition_root_paths as discover_ebh5_composition_root_paths,
)
from tests.qualification.ebh_6._closed_world_discovery import (
    discover_application_wiring_paths,
    discover_sanctioned_production_composition_roots,
)
from tests.qualification.trace_x._trace_x_p4_production_composition_registry import (
    PRODUCTION_COMPOSITION_SITE_REGISTRY,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS: Final[tuple[Path, ...]] = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)

_EXCLUDE_PARTS: Final[tuple[str, ...]] = (
    "/tests/",
    "\\tests\\",
    "/qualification/",
    "\\qualification\\",
)


@dataclass(frozen=True, slots=True)
class ResponsibilityFamilyRow:
    family_id: str
    semantic_responsibility: str
    canonical_owner_doc: str
    implementation_owner_path: str
    composition_root: str
    reachability: str


DUP_X_RESPONSIBILITY_FAMILIES: Final[tuple[ResponsibilityFamilyRow, ...]] = (
    ResponsibilityFamilyRow(
        "RF-EXE-01",
        "Execution ingress / dispatch",
        "docs/project/architecture/EXECUTION.md",
        "intergrax/runtime/execution/",
        "intergrax/integrations/registry/factory.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-BGW-01",
        "Background workers / sync dispatch",
        "docs/project/architecture/EXECUTION.md",
        "agents/",
        "applications/**/host/factory.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-GOV-01",
        "Governance admission",
        "docs/project/architecture/GOVERNANCE.md",
        "intergrax/runtime/governance/",
        "intergrax/applications/_shared/security_wiring.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-CAP-01",
        "Capability / integration selection",
        "docs/project/architecture/INTEGRATIONS.md",
        "intergrax/integrations/registry/factory.py",
        "intergrax/integrations/registry/factory.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-PRV-01",
        "Provider config / bundle materialization",
        "docs/project/architecture/INTEGRATIONS.md",
        "intergrax/integrations/providers/",
        "intergrax/integrations/existing_capability_configuration_facade.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-REG-01",
        "Tool / skill / agent / integration registries",
        "docs/project/architecture/TOOLS.md",
        "intergrax/tools/registry/",
        "intergrax/tools/registry/wiring.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-RCV-01",
        "Retry / recovery / replay",
        "docs/project/architecture/EXECUTION.md",
        "intergrax/runtime/enterprise_reliability/",
        "intergrax/applications/_shared/reliability_wiring.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-STA-01",
        "State / persistence / cache",
        "docs/project/architecture/STATE.md",
        "intergrax/runtime/state/",
        "intergrax/applications/_shared/memory_wiring.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-PRV-02",
        "Provenance / evidence / intents",
        "docs/project/architecture/OBSERVABILITY.md",
        "intergrax/runtime/events/",
        "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-OBS-01",
        "Observability / diagnostics",
        "docs/project/architecture/OBSERVABILITY.md",
        "intergrax/runtime/observability/",
        "intergrax/tools/providers/observability/resolve.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-PLG-01",
        "Plugin lifecycle",
        "docs/project/architecture/INTEGRATIONS.md",
        "intergrax/integrations/registry/plugin_register.py",
        "intergrax/applications/_shared/plugin_bootstrap.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-TEN-01",
        "Tenant / execution context",
        "docs/project/architecture/EXECUTION.md",
        "intergrax/runtime/tenant/",
        "intergrax/integrations/execution_bound_integration_resolution.py",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-LEG-01",
        "Legacy / compat seams",
        "docs/project/maintainers/qualification/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md",
        "intergrax/tools/tools_base.py",
        "N/A — deprecation surface",
        "legacy-reachable",
    ),
    ResponsibilityFamilyRow(
        "RF-CMP-01",
        "Composition roots",
        "docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md",
        "TRACE-X + COMPOSITION_ROOT_HINTS",
        "union(hints, sanctioned sites, *_wiring.py)",
        "production-effective",
    ),
    ResponsibilityFamilyRow(
        "RF-T3-01",
        "Tier-3 platform copies",
        "docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md",
        "applications/**",
        "applications/**/host/factory.py",
        "production-effective — Tier-3 disposition doc (9 hosts)",
    ),
)


@lru_cache(maxsize=1)
def discover_application_host_factory_paths() -> frozenset[str]:
    paths: set[str] = set()
    for path in (_REPO_ROOT / "applications").glob("*/host/factory.py"):
        if path.is_file():
            paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


_CLASS_COVERAGE_HINTS: Final[dict[str, tuple[str, ...]]] = {
    "DUP-01": ("class \\w+Registry", "intergrax/tools/", "intergrax/context/"),
    "DUP-02": ("applications/", "dedupe", "registry"),
    "DUP-03": ("registry_v2", "integrations/registry/", "Factory"),
    "DUP-04": ("dispatch", "VendorKnowledgeSyncDispatcher", "nexus/"),
    "DUP-05": ("sqlite3.connect", "dedup", "InMemory"),
    "DUP-06": ("retry", "redrive", "enterprise_reliability"),
    "DUP-07": ("admit", "governance/", "policy"),
    "DUP-08": ("debug_lab", "bootstrap.py", "bypass"),
    "DUP-09": ("deprecated", "tools_base", "compat"),
    "DUP-10": ("read_integration_slug", "factory.py", "profile"),
    "DUP-11": ("QualifiedTool", "Marketplace", "Resolver"),
    "DUP-12": ("DiagnosticExtensionRegistry", "problem grouping"),
    "DUP-13": ("contract_spec", "registry_v2", "contracts/"),
    "DUP-14": ("COMPOSITION_ROOT_HINTS", "wiring.py", "TRACE-X"),
}


def _is_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    return not any(part in normalized for part in _EXCLUDE_PARTS)


@lru_cache(maxsize=1)
def discover_normative_composition_root_union() -> frozenset[str]:
    """DUP-14 / CONFIG-X SSOT — EBH-5 hints ∪ TRACE-X sanctioned ∪ wiring scan subset."""
    from tests.qualification.config_x._config_x_discovery import (
        discover_composition_root_paths as discover_config_x_heuristic_paths,
    )

    hints = discover_ebh5_composition_root_paths()
    assert hints == frozenset(h.replace("\\", "/") for h in COMPOSITION_ROOT_HINTS if (_REPO_ROOT / h).is_file())
    heuristic = discover_config_x_heuristic_paths()
    sanctioned = discover_sanctioned_production_composition_roots()
    wiring = discover_application_wiring_paths()
    return hints | heuristic | sanctioned | wiring


@lru_cache(maxsize=1)
def discover_registry_class_census() -> frozenset[str]:
    names: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith("Registry"):
                    names.add(node.name)
    return frozenset(names)


@lru_cache(maxsize=1)
def discover_resolver_class_census() -> frozenset[str]:
    names: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith("Resolver"):
                    names.add(node.name)
    return frozenset(names)


@lru_cache(maxsize=1)
def discover_dispatcher_symbol_census() -> frozenset[str]:
    tokens: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "Dispatcher" in text or "dispatch(" in text:
                rel = path.relative_to(_REPO_ROOT).as_posix()
                tokens.add(rel)
    return frozenset(tokens)


@lru_cache(maxsize=1)
def discover_production_tools_base_imports() -> frozenset[str]:
    imports: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_candidate(path):
                continue
            if path.name == "tools_base.py":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "tools_base" in text or "intergrax.tools.tools_base" in text:
                imports.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(imports)


@lru_cache(maxsize=1)
def discover_integration_env_slug_reads_outside_factory() -> frozenset[str]:
    """Shadow selector probe — category slug env reads outside factory + _shared/config."""
    allowed = frozenset(
        {
            "intergrax/integrations/registry/factory.py",
            "intergrax/integrations/_shared/config.py",
            "intergrax/integrations/_shared/__init__.py",
        }
    )
    hits: set[str] = set()
    intergrax_root = _REPO_ROOT / "intergrax"
    for path in intergrax_root.rglob("*.py"):
        if not _is_candidate(path):
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in allowed:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "read_integration_slug_from_env" in text:
            hits.add(rel)
    return frozenset(hits)


_LAB_COMPAT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/runtime/transport/bootstrap.py",
        "intergrax/runtime/execution/debug_lab_nexus_loop.py",
    }
)


def assert_dup_x_closed_world_integrity() -> None:
    assert len(DUP_X_RESPONSIBILITY_FAMILIES) == 15
    assert len(DUP_CLASSES) == 14
    assert discover_ebh5_composition_root_paths()
    assert discover_sanctioned_production_composition_roots()
    assert len(PRODUCTION_COMPOSITION_SITE_REGISTRY) >= 1
    union = discover_normative_composition_root_union()
    assert COMPOSITION_ROOT_HINTS[0] in union
    assert len(discover_registry_class_census()) >= 5
    assert len(discover_resolver_class_census()) >= 5
    assert len(discover_dispatcher_symbol_census()) >= 1
    host_factories = discover_application_host_factory_paths()
    assert len(host_factories) >= 9, "Tier-3 host factory census"
    for rel in COMPOSITION_ROOT_HINTS:
        if (_REPO_ROOT / rel).is_file():
            assert rel in union, rel


def class_coverage_playbook() -> dict[str, tuple[str, ...]]:
    return dict(_CLASS_COVERAGE_HINTS)
