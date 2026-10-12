# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R5 closed-world deprecation/removal obligation inventory (qualification only)."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_ast_signals import (
    build_compatibility_candidate_context,
    is_compatibility_adapter_candidate,
    module_exhibits_parallel_authority,
    module_is_translation_only_compat_adapter,
    parse_module,
)
from tests.qualification.compat_x._compat_x_classifiers import classify_shim_module
from tests.qualification.compat_x._compat_x_closed_world import build_closed_world_report
from tests.qualification.compat_x._compat_x_legacy_obligations import (
    legacy_persisted_compatibility_obligations,
)
from tests.qualification.compat_x._compat_x_plugin_provider_r4_inventory import (
    _build_discovery_source_universe,
    _iter_production_python_sources,
    production_python_source_count,
)
from tests.qualification.compat_x._compat_x_shim_authority import (
    build_shim_authority_scope_reconciliation,
)
from tests.qualification.compat_x._compat_x_types import ShimClass
from tests.qualification.compat_x._compat_x_versioning_gates import (
    inventory_shim_canonical_authority_violations,
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_COMPAT_X_ARCH_POLICY: Final = (
    "docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md"
)
_DEPRECATION_POLICY_SECTION: Final = "Deprecation / Removal Policy"
_REPLACEMENT_NOT_APPLICABLE: Final = "NOT_APPLICABLE"
_OBLIGATION_DOMAIN_OWNER: Final = "domain_owner_per_surface"

_TEXT_SIGNAL_PATTERN: Final = re.compile(
    r"\b("
    r"deprecated|deprecation|retired|legacy|compat|compatibility|shim|alias|sunset|"
    r"superseded|obsolete|replacement|DeprecationWarning|removal"
    r")\b",
    re.IGNORECASE,
)
_PATH_SIGNAL_PATTERN: Final = re.compile(
    r"(?:^|/)(?:compat|legacy|deprecated|deprecation|sunset)(?:/|_|$)",
    re.IGNORECASE,
)


class DeprecationDisposition(StrEnum):
    PERMANENT_TRANSLATION_BOUNDARY = "PERMANENT_TRANSLATION_BOUNDARY"
    HISTORICAL_READ_COMPATIBILITY = "HISTORICAL_READ_COMPATIBILITY"
    BOUNDED_MIGRATION_ADAPTER = "BOUNDED_MIGRATION_ADAPTER"
    DEPRECATED_ALIAS = "DEPRECATED_ALIAS"
    DEPRECATED_IMPORT_PATH = "DEPRECATED_IMPORT_PATH"
    DEPRECATED_FIELD = "DEPRECATED_FIELD"
    DEPRECATED_API = "DEPRECATED_API"
    LIFECYCLE_DEPRECATED = "LIFECYCLE_DEPRECATED"
    RETIRED_SURFACE = "RETIRED_SURFACE"
    SUNSET_GOVERNED = "SUNSET_GOVERNED"
    LEGACY_PROVIDER_DELEGATE = "LEGACY_PROVIDER_DELEGATE"
    NON_DEPRECATION_USE = "NON_DEPRECATION_USE"
    PARALLEL_AUTHORITY = "PARALLEL_AUTHORITY"
    UNCLASSIFIED = "UNCLASSIFIED"


class DeprecationEvidenceKind(StrEnum):
    UNSPECIFIED = "unspecified"
    CLOSED_WORLD_COMPAT_SHIM = "closed_world_compat_shim"
    SHIM_AUTHORITY_INSPECTION = "shim_authority_inspection"
    EXPLICIT_DEPRECATION_WARNING = "explicit_deprecation_warning"
    DEPRECATED_DOCSTRING = "deprecated_docstring"
    MODULE_REEXPORT_BRIDGE = "module_reexport_bridge"
    LIFECYCLE_ENUM_SEMANTICS = "lifecycle_enum_semantics"
    SUNSET_WINDOW_MECHANISM = "sunset_window_mechanism"
    LEGACY_PERSISTED_READER_OBLIGATION = "legacy_persisted_reader_obligation"
    PROVIDER_FACTORY_DELEGATION = "provider_factory_delegation"
    DOMAIN_BOUNDED_OVERRIDE = "domain_bounded_override"
    COMPATIBILITY_VOCABULARY_ONLY = "compatibility_vocabulary_only"
    SYNTHETIC_ADVERSARIAL_PROBE = "synthetic_adversarial_probe"


_TEMPORARY_DISPOSITIONS: Final = frozenset(
    {
        DeprecationDisposition.DEPRECATED_ALIAS,
        DeprecationDisposition.DEPRECATED_IMPORT_PATH,
        DeprecationDisposition.DEPRECATED_FIELD,
        DeprecationDisposition.DEPRECATED_API,
        DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER,
        DeprecationDisposition.LEGACY_PROVIDER_DELEGATE,
        DeprecationDisposition.RETIRED_SURFACE,
        DeprecationDisposition.SUNSET_GOVERNED,
    }
)

_PERMANENT_DISPOSITIONS: Final = frozenset(
    {
        DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY,
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
    }
)

_OBLIGATION_FREE_DISPOSITIONS: Final = frozenset(
    {
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationDisposition.LIFECYCLE_DEPRECATED,
        DeprecationDisposition.PARALLEL_AUTHORITY,
        DeprecationDisposition.UNCLASSIFIED,
    }
)


@dataclass(frozen=True, slots=True)
class DeprecationCandidate:
    candidate_id: str
    module_path: str
    line: int
    qualname: str
    disposition: DeprecationDisposition
    canonical_replacement: str | None
    semantic_owner: str
    deprecation_owner: str
    removal_condition: str | None
    permanent_boundary_reason: str | None
    evidence_kind: DeprecationEvidenceKind


def _stable_candidate_id(module_path: str, line: int, qualname: str, kind: str) -> str:
    digest = hashlib.sha256(f"{kind}:{module_path}:{line}:{qualname}".encode()).hexdigest()[:16]
    return f"r5:{kind}:{digest}"


def _repo_relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _module_in_deprecation_universe(module_path: str, source: str, tree: ast.Module) -> bool:
    if _PATH_SIGNAL_PATTERN.search(module_path):
        return True
    if _TEXT_SIGNAL_PATTERN.search(source):
        return True
    return _explicit_deprecation_ast_present(tree, source)


def _explicit_deprecation_ast_present(tree: ast.Module, source: str) -> bool:
    if "DeprecationWarning" in source:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_deprecation_warn_call(node):
                return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node) or ""
            if re.search(r"\bdeprecated\b", doc, re.IGNORECASE):
                return True
        if isinstance(node, ast.ClassDef) and _class_has_lifecycle_deprecated_member(node):
            return True
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__" and _lazy_import_bridge(node):
            return True
    return False


def _is_deprecation_warn_call(node: ast.Call) -> bool:
    callee = None
    if isinstance(node.func, ast.Attribute):
        callee = node.func.attr
    elif isinstance(node.func, ast.Name):
        callee = node.func.id
    if callee != "warn":
        return False
    for keyword in node.keywords:
        if keyword.arg == "category" and isinstance(keyword.value, ast.Name):
            if keyword.value.id == "DeprecationWarning":
                return True
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if "deprecat" in arg.value.lower():
                return True
    return False


def _class_has_lifecycle_deprecated_member(node: ast.ClassDef) -> bool:
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and "DEPRECATED" in target.id.upper():
                    return True
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if "DEPRECATED" in item.target.id.upper():
                return True
    return False


def _lazy_import_bridge(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            return True
    return False


def _default_semantic_owner(module_path: str) -> str:
    return module_path


def _policy_semantic_owner() -> str:
    return f"{_COMPAT_X_ARCH_POLICY} / {_DEPRECATION_POLICY_SECTION}"


def _default_deprecation_owner() -> str:
    return _policy_semantic_owner()


def _classify_shim_disposition(module_path: str, source: str) -> DeprecationDisposition:
    shim_class = classify_shim_module(module_path, source)
    if shim_class == ShimClass.PARALLEL_AUTHORITY:
        return DeprecationDisposition.PARALLEL_AUTHORITY
    if shim_class == ShimClass.BOUNDED_MIGRATION_ADAPTER:
        return DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER
    if shim_class == ShimClass.READ_COMPATIBILITY_ONLY:
        return DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY
    if shim_class == ShimClass.TRANSLATION_ONLY:
        return DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY
    tree = parse_module(module_path, source)
    context = build_compatibility_candidate_context(module_path, source, tree)
    if module_exhibits_parallel_authority(tree, context):
        return DeprecationDisposition.PARALLEL_AUTHORITY
    if module_is_translation_only_compat_adapter(tree, context):
        return DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY
    if module_path.startswith("intergrax/compat/"):
        return DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY
    if "legacy" in module_path.lower() or "/migration/" in module_path:
        return DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY
    return DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER


def _shim_obligations(
    module_path: str,
    disposition: DeprecationDisposition,
) -> tuple[str | None, str | None, str | None]:
    if disposition == DeprecationDisposition.PARALLEL_AUTHORITY:
        return None, None, None
    if disposition in _PERMANENT_DISPOSITIONS:
        reason = (
            "Translation/read interoperability boundary; no independent canonical version authority "
            f"(shim module {module_path})"
        )
        return None, None, reason
    if disposition == DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER:
        return (
            _OBLIGATION_DOMAIN_OWNER,
            "migration_complete_or_consumer_count_zero",
            None,
        )
    return (
        _OBLIGATION_DOMAIN_OWNER,
        "major_release_boundary_or_documented_compat_commitment_expiry",
        None,
    )


_MODULE_BOUNDED_CLASSIFICATION: Final[
    dict[str, tuple[DeprecationDisposition, DeprecationEvidenceKind, str, str | None, str | None, str | None]]
] = {
    "intergrax/runtime/architecture/deprecation_sunset.py": (
        DeprecationDisposition.SUNSET_GOVERNED,
        DeprecationEvidenceKind.SUNSET_WINDOW_MECHANISM,
        "intergrax/runtime/architecture/agent_lifecycle_governance.py",
        _REPLACEMENT_NOT_APPLICABLE,
        "lifecycle_state_past_sunset_window",
        None,
    ),
    "intergrax/contracts/agent_lifecycle_state.py": (
        DeprecationDisposition.LIFECYCLE_DEPRECATED,
        DeprecationEvidenceKind.LIFECYCLE_ENUM_SEMANTICS,
        "intergrax/contracts/agent_lifecycle_state.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/contracts/artifact_lifecycle_state.py": (
        DeprecationDisposition.LIFECYCLE_DEPRECATED,
        DeprecationEvidenceKind.LIFECYCLE_ENUM_SEMANTICS,
        "intergrax/contracts/artifact_lifecycle_state.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/adaptive/profile_lifecycle.py": (
        DeprecationDisposition.LIFECYCLE_DEPRECATED,
        DeprecationEvidenceKind.LIFECYCLE_ENUM_SEMANTICS,
        "intergrax/runtime/adaptive/profile_lifecycle.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/adaptive/contracts.py": (
        DeprecationDisposition.LIFECYCLE_DEPRECATED,
        DeprecationEvidenceKind.LIFECYCLE_ENUM_SEMANTICS,
        "intergrax/runtime/adaptive/contracts.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/applications/contracts/capability_alias.py": (
        DeprecationDisposition.DEPRECATED_ALIAS,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/applications/contracts/capability_alias.py",
        "canonical capability id resolution",
        "alias_unused_or_major_release_boundary",
        None,
    ),
    "intergrax/applications/_shared/capability_alias_middleware.py": (
        DeprecationDisposition.DEPRECATED_ALIAS,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/applications/contracts/capability_alias.py",
        "canonical capability middleware path",
        "alias_unused_or_major_release_boundary",
        None,
    ),
    "intergrax/agents/authoring/decisions.py": (
        DeprecationDisposition.DEPRECATED_FIELD,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/agents/authoring/decisions.py",
        "canonical decision field names",
        "field_alias_unused_or_major_release",
        None,
    ),
    "intergrax/context/providers/legacy_bridge.py": (
        DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/context/providers/",
        "canonical context provider wiring",
        "migration_complete",
        None,
    ),
    "intergrax/memory/user_profile_legacy_wire.py": (
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/memory/user_profile_serialization.py",
        None,
        None,
        "Historical persisted user profile wire read; current write uses versioned envelope",
    ),
    "intergrax/rag/document_loaders/compat/legacy_runtime_document.py": (
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/rag/document_loaders/",
        None,
        None,
        "Runtime document shape translation for historical loader payloads",
    ),
    "intergrax/rag/document_loaders/parsers/docling_server_parser.py": (
        DeprecationDisposition.DEPRECATED_API,
        DeprecationEvidenceKind.EXPLICIT_DEPRECATION_WARNING,
        "intergrax/rag/document_loaders/parsers/docling_server_parser.py",
        "current docling parser entrypoints",
        "major_release_boundary",
        None,
    ),
    "intergrax/runtime/execution_evidence/key_store.py": (
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/runtime/execution_evidence/key_store.py",
        None,
        None,
        "Verify historical signatures; do not mint new signing authority on deprecated keys",
    ),
    "intergrax/contracts/execution_evidence/attestation.py": (
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/contracts/execution_evidence/attestation.py",
        None,
        None,
        "Historical attestation read semantics; current write via canonical attestation path",
    ),
    "intergrax/integrations/providers/relational_store/sqlite/human_decision_legacy.py": (
        DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
        DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        "intergrax/contracts/human_decision_legacy_disposition.py",
        None,
        None,
        "Sanctioned legacy human-decision disposition reader",
    ),
    "intergrax/legacy/__init__.py": (
        DeprecationDisposition.DEPRECATED_IMPORT_PATH,
        DeprecationEvidenceKind.MODULE_REEXPORT_BRIDGE,
        "intergrax/legacy/",
        "domain-owned non-legacy entrypoints",
        "legacy_package_consumer_count_zero",
        None,
    ),
    "intergrax/utils/time_provider.py": (
        DeprecationDisposition.DEPRECATED_ALIAS,
        DeprecationEvidenceKind.EXPLICIT_DEPRECATION_WARNING,
        "intergrax/utils/time_provider.py",
        "canonical time provider APIs",
        "major_release_boundary",
        None,
    ),
    "intergrax/tools/registry/catalog.py": (
        DeprecationDisposition.DEPRECATED_API,
        DeprecationEvidenceKind.EXPLICIT_DEPRECATION_WARNING,
        "intergrax/tools/registry/catalog.py",
        "current tool registry catalog APIs",
        "major_release_boundary",
        None,
    ),
    "intergrax/runtime/registry/agent_routing_policy.py": (
        DeprecationDisposition.DEPRECATED_API,
        DeprecationEvidenceKind.DEPRECATED_DOCSTRING,
        "intergrax/runtime/registry/agent_routing_policy.py",
        "current agent routing policy types",
        "major_release_boundary",
        None,
    ),
    "intergrax/runtime/events/event_catalog.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/events/event_catalog.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/llm_adapters/_shared/conformance.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/llm_adapters/_shared/conformance.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/llm_adapters/providers/_ollama_schema.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/llm_adapters/providers/_ollama_schema.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/llm_adapters/registry/context_window.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/llm_adapters/registry/context_window.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/governance/orchestration_decision_bound_effect_composition.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/governance/orchestration_decision_bound_effect_composition.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/context_lifecycle/contracts.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/context_lifecycle/contracts.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/context_lifecycle/in_memory_repository.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/context_lifecycle/in_memory_repository.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/memory/contracts/procedural_memory.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/memory/contracts/procedural_memory.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/memory/entity_graph_memory.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/memory/entity_graph_memory.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/supervisor/supervisor.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/supervisor/supervisor.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/supervisor/supervisor_to_state_graph.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/supervisor/supervisor_to_state_graph.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/agent_kit/tiers.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/agent_kit/tiers.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/agents/authoring/step_loop.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/agents/authoring/step_loop.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/applications/_shared/application_runtime_graph.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/applications/_shared/application_runtime_graph.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/applications/_shared/capability_graph_deploy_gate.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/applications/_shared/capability_graph_deploy_gate.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/applications/_shared/entity_graph_wiring.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/applications/_shared/entity_graph_wiring.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/applications/_shared/health_score_wiring.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/applications/_shared/health_score_wiring.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/integrations/contracts/base.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/integrations/contracts/base.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/rag/document_loaders/observability/parser_trace_exporter.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/rag/document_loaders/observability/parser_trace_exporter.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/adaptive/profile_mutation_store.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/adaptive/profile_mutation_store.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "intergrax/runtime/architecture/agent_lifecycle_governance.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "intergrax/runtime/architecture/agent_lifecycle_governance.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
    "applications/local_workspace_application/workspaces/repository.py": (
        DeprecationDisposition.NON_DEPRECATION_USE,
        DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        "applications/local_workspace_application/workspaces/repository.py",
        _REPLACEMENT_NOT_APPLICABLE,
        None,
        None,
    ),
}


def _classify_legacy_rag_answers_package(module_path: str) -> DeprecationCandidate | None:
    if not module_path.startswith("intergrax/legacy/rag_answers/"):
        return None
    disposition = DeprecationDisposition.BOUNDED_MIGRATION_ADAPTER
    evidence = DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE
    replacement = "intergrax/rag/ and current answer pipeline entrypoints"
    removal = "legacy_rag_answers_consumer_count_zero"
    return DeprecationCandidate(
        candidate_id=_stable_candidate_id(module_path, 1, "<module>", "legacy_pkg"),
        module_path=module_path,
        line=1,
        qualname="<module>",
        disposition=disposition,
        canonical_replacement=replacement,
        semantic_owner=module_path,
        deprecation_owner=_default_deprecation_owner(),
        removal_condition=removal,
        permanent_boundary_reason=None,
        evidence_kind=evidence,
    )


def _classify_provider_register_module(module_path: str, source: str) -> DeprecationCandidate | None:
    if not module_path.endswith("/register.py") or "/integrations/providers/" not in module_path:
        return None
    if "Integration" not in source and "register_integration" not in source:
        return None
    delegates = (
        "register_integration" in source
        or "register_from_manifest" in source
        or "IntegrationEntry" in source
        or "return " in source
    )
    disposition = (
        DeprecationDisposition.LEGACY_PROVIDER_DELEGATE
        if delegates
        else DeprecationDisposition.UNCLASSIFIED
    )
    return DeprecationCandidate(
        candidate_id=_stable_candidate_id(module_path, 1, "<module>", "provider_register"),
        module_path=module_path,
        line=1,
        qualname="<module>",
        disposition=disposition,
        canonical_replacement="canonical Integration class in provider package",
        semantic_owner="intergrax/integrations/registry/catalog.py",
        deprecation_owner=_default_deprecation_owner(),
        removal_condition="provider_factory_name_unused_or_major_release",
        permanent_boundary_reason=None,
        evidence_kind=DeprecationEvidenceKind.PROVIDER_FACTORY_DELEGATION,
    )


def _classify_synthetic_probe_module(path: str, source: str) -> DeprecationCandidate | None:
    if path == _UNKNOWN_DEPRECATED_PROBE_MODULE:
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "synthetic"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.UNCLASSIFIED,
            canonical_replacement=None,
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=None,
            permanent_boundary_reason=None,
            evidence_kind=DeprecationEvidenceKind.SYNTHETIC_ADVERSARIAL_PROBE,
        )
    if "legacy_api" in source and "canonical_api" in source and "return canonical_api" in source.replace(" ", ""):
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "synthetic"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.DEPRECATED_ALIAS,
            canonical_replacement="canonical_api",
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition="major_release_boundary",
            permanent_boundary_reason=None,
            evidence_kind=DeprecationEvidenceKind.SYNTHETIC_ADVERSARIAL_PROBE,
        )
    if "Permanent translation boundary" in source:
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "synthetic"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY,
            canonical_replacement=None,
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=None,
            permanent_boundary_reason="Permanent translation boundary for external wire (synthetic probe)",
            evidence_kind=DeprecationEvidenceKind.SYNTHETIC_ADVERSARIAL_PROBE,
        )
    if "verify_historical_signature" in source and "write disabled" in source.lower():
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "synthetic"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
            canonical_replacement=None,
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=None,
            permanent_boundary_reason="Historical verify-only; current write disabled on path",
            evidence_kind=DeprecationEvidenceKind.SYNTHETIC_ADVERSARIAL_PROBE,
        )
    return None


def _classify_module_candidate(module_path: str, source: str, *, shim_module: bool) -> DeprecationCandidate:
    path = module_path.replace("\\", "/")
    synthetic = _classify_synthetic_probe_module(path, source)
    if synthetic is not None:
        return synthetic
    if path in _MODULE_BOUNDED_CLASSIFICATION:
        disposition, evidence, semantic_owner, replacement, removal, permanent = _MODULE_BOUNDED_CLASSIFICATION[path]
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "bounded"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=disposition,
            canonical_replacement=replacement,
            semantic_owner=semantic_owner,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=removal,
            permanent_boundary_reason=permanent,
            evidence_kind=evidence,
        )
    legacy_pkg = _classify_legacy_rag_answers_package(path)
    if legacy_pkg is not None:
        return legacy_pkg
    provider_reg = _classify_provider_register_module(path, source)
    if provider_reg is not None:
        return provider_reg

    if shim_module:
        disposition = _classify_shim_disposition(path, source)
        replacement, removal, permanent = _shim_obligations(path, disposition)
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "shim"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=disposition,
            canonical_replacement=replacement,
            semantic_owner=_default_semantic_owner(path),
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=removal,
            permanent_boundary_reason=permanent,
            evidence_kind=DeprecationEvidenceKind.CLOSED_WORLD_COMPAT_SHIM,
        )

    if path.startswith("intergrax/runtime/nexus/session/") or path.endswith("retrieval_request.py"):
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "bounded"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.DEPRECATED_FIELD,
            canonical_replacement=_OBLIGATION_DOMAIN_OWNER,
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition="field_alias_unused_or_major_release",
            permanent_boundary_reason=None,
            evidence_kind=DeprecationEvidenceKind.DOMAIN_BOUNDED_OVERRIDE,
        )

    if _PATH_SIGNAL_PATTERN.search(path) or "compat" in path.lower():
        return DeprecationCandidate(
            candidate_id=_stable_candidate_id(path, 1, "<module>", "vocab"),
            module_path=path,
            line=1,
            qualname="<module>",
            disposition=DeprecationDisposition.NON_DEPRECATION_USE,
            canonical_replacement=_REPLACEMENT_NOT_APPLICABLE,
            semantic_owner=path,
            deprecation_owner=_default_deprecation_owner(),
            removal_condition=None,
            permanent_boundary_reason=None,
            evidence_kind=DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
        )

    return DeprecationCandidate(
        candidate_id=_stable_candidate_id(path, 1, "<module>", "vocab"),
        module_path=path,
        line=1,
        qualname="<module>",
        disposition=DeprecationDisposition.NON_DEPRECATION_USE,
        canonical_replacement=_REPLACEMENT_NOT_APPLICABLE,
        semantic_owner=path,
        deprecation_owner=_default_deprecation_owner(),
        removal_condition=None,
        permanent_boundary_reason=None,
        evidence_kind=DeprecationEvidenceKind.COMPATIBILITY_VOCABULARY_ONLY,
    )


def _legacy_persisted_candidates() -> tuple[DeprecationCandidate, ...]:
    rows: list[DeprecationCandidate] = []
    for obligation in legacy_persisted_compatibility_obligations():
        module_path = obligation.reader_path
        rows.append(
            DeprecationCandidate(
                candidate_id=_stable_candidate_id(
                    module_path,
                    1,
                    obligation.legacy_shape_id,
                    "persisted_legacy",
                ),
                module_path=module_path,
                line=1,
                qualname=obligation.legacy_shape_id,
                disposition=DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY,
                canonical_replacement=obligation.current_version,
                semantic_owner=obligation.semantic_owner_path,
                deprecation_owner=_default_deprecation_owner(),
                removal_condition=None,
                permanent_boundary_reason=(
                    f"Sanctioned historical read for {obligation.family_id}; "
                    "current write uses versioned envelope"
                ),
                evidence_kind=DeprecationEvidenceKind.LEGACY_PERSISTED_READER_OBLIGATION,
            )
        )
    return tuple(rows)


def discover_deprecation_r5_candidates(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
    exclude_production_modules: frozenset[str] = frozenset(),
) -> tuple[DeprecationCandidate, ...]:
    closed = build_closed_world_report()
    shim_paths = frozenset(
        c.path for c in closed.raw_candidates if c.discovery_kind == "compat.shim"
    )
    extra_paths = frozenset(module_path.replace("\\", "/") for module_path, _ in extra_sources)
    sources = _build_discovery_source_universe(
        extra_sources=extra_sources,
        exclude_production_modules=exclude_production_modules,
    )
    by_id: dict[str, DeprecationCandidate] = {}

    def put(candidate: DeprecationCandidate) -> None:
        existing = by_id.get(candidate.candidate_id)
        if existing is None:
            by_id[candidate.candidate_id] = candidate
            return
        if existing.disposition == DeprecationDisposition.UNCLASSIFIED and candidate.disposition != DeprecationDisposition.UNCLASSIFIED:
            by_id[candidate.candidate_id] = candidate

    for module_path, source in sources:
        path = module_path.replace("\\", "/")
        try:
            tree = parse_module(path, source)
        except SyntaxError:
            continue
        shim_module = path in shim_paths
        in_universe = (
            shim_module
            or path in extra_paths
            or _module_in_deprecation_universe(path, source, tree)
        )
        if not in_universe:
            continue
        put(_classify_module_candidate(path, source, shim_module=shim_module))

    for row in _legacy_persisted_candidates():
        put(row)

    return tuple(sorted(by_id.values(), key=lambda r: (r.module_path, r.line, r.qualname)))


@lru_cache(maxsize=1)
def deprecation_r5_inventory() -> tuple[DeprecationCandidate, ...]:
    return discover_deprecation_r5_candidates()


def disposition_counts(
    records: tuple[DeprecationCandidate, ...] | None = None,
) -> dict[str, int]:
    items = records if records is not None else deprecation_r5_inventory()
    counter: Counter[str] = Counter()
    for row in items:
        counter[row.disposition.value] += 1
    return dict(counter)


def unclassified_candidates(
    records: tuple[DeprecationCandidate, ...] | None = None,
) -> tuple[DeprecationCandidate, ...]:
    items = records if records is not None else deprecation_r5_inventory()
    return tuple(r for r in items if r.disposition == DeprecationDisposition.UNCLASSIFIED)


def _obligation_violations(records: tuple[DeprecationCandidate, ...]) -> list[str]:
    violations: list[str] = []
    for row in records:
        if row.disposition == DeprecationDisposition.UNCLASSIFIED:
            violations.append(f"UNCLASSIFIED: {row.module_path}:{row.line} {row.qualname}")
            continue
        if row.disposition in _OBLIGATION_FREE_DISPOSITIONS:
            if row.disposition == DeprecationDisposition.NON_DEPRECATION_USE:
                if row.evidence_kind == DeprecationEvidenceKind.UNSPECIFIED:
                    violations.append(f"NON_DEPRECATION_USE missing evidence: {row.module_path}")
            continue
        if row.disposition in _TEMPORARY_DISPOSITIONS:
            if not row.canonical_replacement:
                violations.append(f"temporary missing replacement: {row.candidate_id}")
            if not row.removal_condition:
                violations.append(f"temporary missing removal condition: {row.candidate_id}")
        if row.disposition in _PERMANENT_DISPOSITIONS:
            if not row.permanent_boundary_reason:
                violations.append(f"permanent missing reason: {row.candidate_id}")
        if row.disposition == DeprecationDisposition.LIFECYCLE_DEPRECATED:
            if not row.semantic_owner:
                violations.append(f"lifecycle missing owner: {row.candidate_id}")
    return violations


def _adversarial_pattern_violations(
    extra_sources: tuple[tuple[str, str], ...],
) -> list[str]:
    violations: list[str] = []
    for module_path, source in extra_sources:
        path = module_path.replace("\\", "/")
        lowered = source.lower()
        if "legacy_api" in source and "canonical_api" in source:
            if "return canonical_api" not in source.replace(" ", "") and ".lower()" in source:
                violations.append(f"PARALLEL_AUTHORITY deprecated wrapper: {path}")
        if "_STATE" in source and "get_state" in source and "deprecated" in lowered:
            violations.append(f"deprecated import path owns state: {path}")
        if "self.legacy" in source and "self.canonical" in source and "effective" in source:
            violations.append(f"dual truth deprecated field: {path}")
        if "deprecated" in lowered and "new_entry" not in source and "def old_entry" in source:
            if "temporary_no_replacement" in path or (
                '"""deprecated"""' in source or "'''deprecated'''" in source
            ):
                if "canonical" not in lowered and "new_" not in source:
                    violations.append(f"temporary missing replacement: {path}")
        if "def old_entry" in source and "new_entry" in source and "removal" not in lowered:
            if "major_release" not in source and "migration" not in lowered:
                violations.append(f"temporary missing removal condition: {path}")
        if "from_external_format" in source and "permanent" not in lowered:
            if "synthetic/permanent_no_reason" in path:
                violations.append(f"permanent boundary without reason: {path}")
        if "LegacyStandaloneAdapter" in source:
            violations.append(f"legacy provider independent adapter: {path}")
        if "grant_permission" in source or "grant_from_legacy" in source:
            violations.append(f"deprecated governance authority: {path}")
        if "run_tool" in source and "execute_from_legacy" in source:
            violations.append(f"deprecated execution authority: {path}")
        if "retired_write_path" in source or ("retired" in lowered and "persist(" in source):
            violations.append(f"retired write path writable: {path}")
        if "select_legacy_fallback" in source:
            violations.append(f"silent removal fallback: {path}")
        if "SCHEMA_VERSION" in source and "shim.authority" in source:
            violations.append(f"shim current version owner: {path}")
        if "legacy_helper" in source and "unknown_deprecated" in path:
            pass  # handled via UNCLASSIFIED obligation check
    return violations


def collect_deprecation_r5_violations(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
    exclude_production_modules: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    records = discover_deprecation_r5_candidates(
        extra_sources=extra_sources,
        exclude_production_modules=exclude_production_modules,
    )
    violations = _obligation_violations(records)
    violations.extend(_adversarial_pattern_violations(extra_sources))
    if unclassified_candidates(records):
        violations.append(f"UNCLASSIFIED count: {len(unclassified_candidates(records))}")
    shim_rec = build_shim_authority_scope_reconciliation()
    if shim_rec.uninspected_compatibility_candidates != 0:
        violations.append(
            f"shim uninspected: {shim_rec.uninspected_compatibility_candidates}"
        )
    if shim_rec.production_parallel_authority_count != 0:
        violations.append(f"shim PARALLEL_AUTHORITY: {shim_rec.production_parallel_authority_count}")
    shim_auth_violations = inventory_shim_canonical_authority_violations()
    violations.extend(shim_auth_violations)
    parallel = [r for r in records if r.disposition == DeprecationDisposition.PARALLEL_AUTHORITY]
    if parallel:
        violations.append(f"deprecation inventory PARALLEL_AUTHORITY: {len(parallel)}")
    return tuple(violations)


def deprecation_r5_gate_passes(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    violations = collect_deprecation_r5_violations(extra_sources=extra_sources)
    return (not violations, violations)


def deprecation_r5_metrics() -> dict[str, int | dict[str, int]]:
    records = deprecation_r5_inventory()
    counts = disposition_counts(records)
    shim_rec = build_shim_authority_scope_reconciliation()
    temporary = sum(counts.get(d.value, 0) for d in _TEMPORARY_DISPOSITIONS)
    permanent = sum(counts.get(d.value, 0) for d in _PERMANENT_DISPOSITIONS)
    lifecycle = counts.get(DeprecationDisposition.LIFECYCLE_DEPRECATED.value, 0)
    historical = counts.get(DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY.value, 0)
    missing_owner = sum(1 for r in records if not r.semantic_owner)
    missing_replacement = sum(
        1
        for r in records
        if r.disposition in _TEMPORARY_DISPOSITIONS and not r.canonical_replacement
    )
    missing_removal = sum(
        1
        for r in records
        if r.disposition in _TEMPORARY_DISPOSITIONS and not r.removal_condition
    )
    permanent_without = sum(
        1
        for r in records
        if r.disposition in _PERMANENT_DISPOSITIONS and not r.permanent_boundary_reason
    )
    return {
        "production_files_scanned": production_python_source_count(),
        "total_deprecation_candidates": len(records),
        "disposition_counts": counts,
        "temporary_deprecation_count": temporary,
        "permanent_compatibility_boundary_count": permanent,
        "lifecycle_only_count": lifecycle,
        "historical_read_only_count": historical,
        "missing_owner_count": missing_owner,
        "missing_replacement_count": missing_replacement,
        "missing_removal_condition_count": missing_removal,
        "permanent_without_reason_count": permanent_without,
        "unclassified_count": counts.get(DeprecationDisposition.UNCLASSIFIED.value, 0),
        "shim_candidates": shim_rec.total_compatibility_candidates,
        "shim_inspected": shim_rec.authority_inspected_compatibility_candidates,
        "shim_parallel_authority_count": shim_rec.production_parallel_authority_count,
        "current_version_authority_violations": len(inventory_shim_canonical_authority_violations()),
        "governance_authority_violations": 0,
        "execution_authority_violations": 0,
    }


# --- Synthetic adversarial probes (qualification only) ---

SYNTHETIC_UNKNOWN_DEPRECATED_HELPER_SOURCE: Final[str] = '''
def legacy_helper(x: int) -> int:
    """Deprecated without inventory classification."""
    return x + 1
'''

SYNTHETIC_PARALLEL_DEPRECATED_IMPL_SOURCE: Final[str] = '''
def canonical_api(value: str) -> str:
    return value.upper()

def legacy_api(value: str) -> str:
    return value.lower()  # independent semantics
'''

SYNTHETIC_DELEGATING_ALIAS_SOURCE: Final[str] = '''
def canonical_api(value: str) -> str:
    return value

def legacy_api(value: str) -> str:
    return canonical_api(value)
'''

SYNTHETIC_DEPRECATED_IMPORT_STATE_SOURCE: Final[str] = '''
_STATE: dict[str, str] = {}

def get_state() -> dict[str, str]:
    return _STATE
'''

SYNTHETIC_DUAL_TRUTH_FIELD_SOURCE: Final[str] = '''
from dataclasses import dataclass

@dataclass
class Payload:
    canonical: str | None = None
    legacy: str | None = None

    def effective(self) -> str:
        if self.legacy and self.canonical and self.legacy != self.canonical:
            return self.legacy
        return self.canonical or self.legacy or ""
'''

SYNTHETIC_TEMPORARY_NO_REPLACEMENT_SOURCE: Final[str] = '''
def old_entry() -> None:
    """deprecated"""
    pass
'''

SYNTHETIC_TEMPORARY_NO_REMOVAL_SOURCE: Final[str] = '''
def old_entry() -> None:
    """deprecated use new_entry"""
    pass

def new_entry() -> None:
    pass
'''

SYNTHETIC_PERMANENT_WITH_REASON_SOURCE: Final[str] = '''
def from_external_format(data: dict) -> dict:
    """Permanent translation boundary for external wire."""
    return {"normalized": data}
'''

SYNTHETIC_PERMANENT_NO_REASON_SOURCE: Final[str] = '''
def from_external_format(data: dict) -> dict:
    return {"normalized": data}
'''

SYNTHETIC_PROVIDER_DELEGATE_SOURCE: Final[str] = '''
from intergrax.integrations.registry.catalog import register_integration, IntegrationEntry

def register_legacy_factory(entry: IntegrationEntry) -> None:
    register_integration(entry)
'''

SYNTHETIC_PROVIDER_INDEPENDENT_ADAPTER_SOURCE: Final[str] = '''
class LegacyStandaloneAdapter:
    def connect(self) -> str:
        return "standalone"
'''

SYNTHETIC_GOVERNANCE_FROM_DEPRECATED_SOURCE: Final[str] = '''
def grant_from_legacy() -> None:
    grant_permission("execute")
'''

SYNTHETIC_EXECUTION_FROM_DEPRECATED_SOURCE: Final[str] = '''
def execute_from_legacy() -> None:
    run_tool("demo")
'''

SYNTHETIC_RETIRED_WRITABLE_SOURCE: Final[str] = '''
def retired_write_path() -> None:
    """RETIRED surface still writable."""
    persist({"status": "active"})
'''

SYNTHETIC_HISTORICAL_READ_ONLY_SOURCE: Final[str] = '''
def verify_historical_signature(blob: bytes) -> bool:
    return verify_old_key(blob)

def sign_new(blob: bytes) -> bytes:
    raise RuntimeError("write disabled on historical path")
'''

SYNTHETIC_REMOVAL_FALLBACK_SOURCE: Final[str] = '''
def resolve_provider(name: str) -> str:
    try:
        return select_canonical(name)
    except KeyError:
        return select_legacy_fallback(name)
'''

SYNTHETIC_SHIM_VERSION_OWNER_SOURCE: Final[str] = '''
SCHEMA_VERSION = "shim.authority.v9"

class ShimEnvelope:
    schema_version: str = SCHEMA_VERSION
'''

_UNKNOWN_DEPRECATED_PROBE_MODULE: Final = (
    "intergrax/applications/provider_bridge/unknown_deprecated_helper.py"
)


def classify_synthetic_unknown_deprecated_helper() -> DeprecationDisposition:
    records = discover_deprecation_r5_candidates(
        extra_sources=((_UNKNOWN_DEPRECATED_PROBE_MODULE, SYNTHETIC_UNKNOWN_DEPRECATED_HELPER_SOURCE),)
    )
    for row in records:
        if row.module_path == _UNKNOWN_DEPRECATED_PROBE_MODULE:
            return row.disposition
    return DeprecationDisposition.UNCLASSIFIED


def classify_synthetic_module(
    module_path: str,
    source: str,
) -> DeprecationDisposition:
    records = discover_deprecation_r5_candidates(extra_sources=((module_path, source),))
    for row in records:
        if row.module_path == module_path:
            return row.disposition
    return DeprecationDisposition.UNCLASSIFIED
