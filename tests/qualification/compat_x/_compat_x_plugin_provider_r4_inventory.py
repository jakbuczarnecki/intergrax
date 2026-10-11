# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R4 closed-world plugin/provider compatibility inventory (qualification only)."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from tests.qualification.compat_x._compat_x_types import CompatOwnerMatrixRow

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_PRODUCTION_SCAN_ROOTS: Final = ("intergrax", "agents", "applications")
_PRODUCTION_SCAN_EXCLUDE: Final = frozenset(
    {"tests", "testing_support", "docs", "scripts", "__pycache__"}
)

_COMPAT_X_ARCH_POLICY: Final = (
    "docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md"
)
_PACKAGE_COMPAT_SEMANTIC: Final = "intergrax/core/distribution/platform_compatibility.py"
_PACKAGE_MANIFEST: Final = "intergrax/core/plugins/package_contract.py"
_PACKAGE_MANIFEST_IO: Final = "intergrax/core/plugins/manifest_io.py"
_PACKAGE_SEMANTICS: Final = "intergrax/core/plugins/platform_semantics.py"
_PACKAGE_QUALIFICATION: Final = "intergrax/core/plugins/platform_qualification.py"
_PROVIDER_CATALOG: Final = "intergrax/integrations/registry/catalog.py"
_PROVIDER_CONTRACT_SPEC: Final = "intergrax/integrations/registry/contract_spec.py"
_PROVIDER_PLUGIN_REGISTER: Final = "intergrax/integrations/registry/plugin_register.py"
_PROVIDER_FACTORY: Final = "intergrax/integrations/registry/factory.py"
_REGISTRY_V2: Final = "intergrax/runtime/integrations/registry_v2.py"
_EXTERNAL_COMPAT_CONTRACTS: Final = (
    "intergrax/integrations/contracts/external_contract_compatibility.py"
)
_EXTERNAL_COMPAT_EXTENSIONS: Final = (
    "intergrax/integrations/external_contract_compatibility_extensions.py"
)
_EXTERNAL_COMPAT_SERVICE: Final = "intergrax/integrations/external_contract_compatibility_service.py"

_PACKAGE_EVAL_CALLS: Final = frozenset(
    {
        "check_platform_compatibility",
        "require_platform_compatibility",
        "check_manifest_platform_compatibility",
    }
)
_PROVIDER_REG_CALLS: Final = frozenset(
    {
        "register_integration",
        "register_from_manifest",
        "register_integration_plugin",
        "declare_integration_contract",
        "augment_integration_contract_specs",
    }
)
_EXTERNAL_COMPAT_CALLS: Final = frozenset(
    {
        "assess_external_contract_compatibility",
    }
)
_EXTERNAL_COMPAT_TYPE_MARKERS: Final = frozenset(
    {
        "ExternalContractCompatibilityExpectation",
        "ExternalContractCompatibilityEvidence",
        "ExternalContractCompatibilityAssessment",
        "ExternalContractCompatibilityExpectationResolver",
        "ExternalContractEvidenceProvider",
        "ExternalContractCompatibilityEvaluator",
        "ExternalContractCompatibilityExtensions",
        "ExternalContractCompatibilityService",
    }
)


class PluginProviderCompatibilityDisposition(StrEnum):
    PACKAGE_COMPATIBILITY_DECLARATION = "PACKAGE_COMPATIBILITY_DECLARATION"
    PACKAGE_COMPATIBILITY_EVALUATOR = "PACKAGE_COMPATIBILITY_EVALUATOR"
    PACKAGE_COMPATIBILITY_CONSUMER = "PACKAGE_COMPATIBILITY_CONSUMER"
    PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY = "PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY"
    PROVIDER_CONTRACT_DECLARATION = "PROVIDER_CONTRACT_DECLARATION"
    PROVIDER_CATALOG_AUTHORITY = "PROVIDER_CATALOG_AUTHORITY"
    PROVIDER_CATALOG_DELEGATE = "PROVIDER_CATALOG_DELEGATE"
    PROVIDER_PROJECTION_ONLY = "PROVIDER_PROJECTION_ONLY"
    PROVIDER_MATERIALIZATION_CONSUMER = "PROVIDER_MATERIALIZATION_CONSUMER"
    PROVIDER_REGISTRATION_SHADOW_AUTHORITY = "PROVIDER_REGISTRATION_SHADOW_AUTHORITY"
    EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT = "EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT"
    EXTERNAL_COMPATIBILITY_EVIDENCE_PROVIDER = "EXTERNAL_COMPATIBILITY_EVIDENCE_PROVIDER"
    EXTERNAL_COMPATIBILITY_EVALUATOR = "EXTERNAL_COMPATIBILITY_EVALUATOR"
    EXTERNAL_COMPATIBILITY_COMPOSITION = "EXTERNAL_COMPATIBILITY_COMPOSITION"
    SHADOW_EXTERNAL_COMPATIBILITY_AUTHORITY = "SHADOW_EXTERNAL_COMPATIBILITY_AUTHORITY"
    COMPATIBILITY_ACTIVATION_SHORTCUT = "COMPATIBILITY_ACTIVATION_SHORTCUT"
    COMPATIBILITY_GOVERNANCE_AUTHORITY_SHORTCUT = "COMPATIBILITY_GOVERNANCE_AUTHORITY_SHORTCUT"
    COMPATIBILITY_EXECUTION_AUTHORITY_SHORTCUT = "COMPATIBILITY_EXECUTION_AUTHORITY_SHORTCUT"
    IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY = "IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY"
    NON_COMPATIBILITY_USE = "NON_COMPATIBILITY_USE"
    TEST_LAB_ONLY = "TEST_LAB_ONLY"
    UNCLASSIFIED = "UNCLASSIFIED"


class PluginProviderClassificationEvidence(StrEnum):
    UNSPECIFIED = "unspecified"
    CANONICAL_PLATFORM_COMPATIBILITY_MODULE = "canonical_platform_compatibility_module"
    MANIFEST_PLATFORM_COMPATIBILITY_FIELD = "manifest_platform_compatibility_field"
    MANIFEST_IO_COERCION = "manifest_io_coercion"
    MANIFEST_DELEGATING_WRAPPER = "manifest_delegating_wrapper"
    QUALIFICATION_ADMISSION_CONSUMER = "qualification_admission_consumer"
    INTEGRATION_CATALOG_REGISTER = "integration_catalog_register"
    INTEGRATION_CONTRACT_SPEC_TYPED = "integration_contract_spec_typed"
    PLUGIN_REGISTER_CATALOG_DELEGATE = "plugin_register_catalog_delegate"
    CATALOG_BOOTSTRAP_DELEGATE = "catalog_bootstrap_delegate"
    PROVIDER_REGISTER_MODULE_DELEGATE = "provider_register_module_delegate"
    CONTEXT_VIEW_SCOPE_COMPATIBILITY = "context_view_scope_compatibility"
    RUNTIME_PLUGIN_SCHEMA_COMPATIBILITY = "runtime_plugin_schema_compatibility"
    CAPABILITY_GRAPH_MODALITY_COMPATIBILITY = "capability_graph_modality_compatibility"
    ARTIFACT_LIFECYCLE_COMPATIBILITY = "artifact_lifecycle_compatibility"
    DOMAIN_INCOMPATIBILITY_ERROR_SYMBOL = "domain_incompatibility_error_symbol"
    APPLICATION_QUALIFICATION_COMPATIBILITY = "application_qualification_compatibility"
    EXTERNAL_CONTRACT_EXAMPLE_COMPOSITION = "external_contract_example_composition"
    VECTOR_INDEX_ADMIN_COMPATIBILITY_ERROR = "vector_index_admin_compatibility_error"
    LINEAGE_PERSISTENCE_COMPATIBILITY = "lineage_persistence_compatibility"
    LANGCHAIN_COMPAT_SHIM = "langchain_compat_shim"
    WORKER_CAPABILITY_AUTHORITY_COMPATIBILITY = "worker_capability_authority_compatibility"
    REGISTRY_V2_CATALOG_PROJECTION = "registry_v2_catalog_projection"
    FACTORY_MATERIALIZATION_TYPE_GUARD = "factory_materialization_type_guard"
    EXTERNAL_CONTRACT_TYPED_CONTRACTS = "external_contract_typed_contracts"
    EXTERNAL_COMPAT_EXTENSIONS_COMPOSITION = "external_compat_extensions_composition"
    EXTERNAL_COMPAT_SERVICE_COMPOSITION = "external_compat_service_composition"
    RUNTIME_SEMVER_REGISTRY_RESOLUTION = "runtime_semver_registry_resolution"
    UNRELATED_DOMAIN_COMPATIBILITY_NAMING = "unrelated_domain_compatibility_naming"
    OPENAI_API_COMPATIBILITY_ALIAS = "openai_api_compatibility_alias"
    CONTRACT_IDENTITY_ASSERTION = "contract_identity_assertion"
    SYNTHETIC_SHADOW_PROBE = "synthetic_shadow_probe"
    SYNTHETIC_UNCLASSIFIED_PROBE = "synthetic_unclassified_probe"


@dataclass(frozen=True, slots=True)
class PluginProviderCompatibilityCandidateRecord:
    module_path: str
    line: int
    qualname: str
    operation: str
    disposition: PluginProviderCompatibilityDisposition
    semantic_concern: str
    owner_evidence: str
    classification_evidence: PluginProviderClassificationEvidence


def _repo_relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _iter_production_python_sources() -> list[Path]:
    paths: list[Path] = []
    for root_name in _PRODUCTION_SCAN_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for file_path in root.rglob("*.py"):
            rel_parts = file_path.relative_to(_REPO_ROOT).parts
            if any(part in _PRODUCTION_SCAN_EXCLUDE for part in rel_parts):
                continue
            if "docker" in rel_parts and "runtime-context" in rel_parts:
                continue
            paths.append(file_path)
    return sorted(paths)


def _enclosing_qualname(tree: ast.Module, lineno: int) -> str:
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno):
                if best is None or node.lineno >= best[0]:
                    best = (node.lineno, node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.lineno <= lineno <= getattr(item, "end_lineno", item.lineno):
                        qual = f"{node.name}.{item.name}"
                        if best is None or item.lineno >= best[0]:
                            best = (item.lineno, qual)
    return best[1] if best else "<module>"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _function_uses_class_name_compatibility(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for inner in ast.walk(func):
        if isinstance(inner, ast.Attribute) and inner.attr == "__class__":
            return True
        if isinstance(inner, ast.Attribute) and inner.attr == "__name__":
            return True
    return False


def _is_specifier_set_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "SpecifierSet":
        return True
    if isinstance(node.func, ast.Attribute) and node.func.attr == "SpecifierSet":
        return True
    return False


_CATALOG_REGISTRATION_DELEGATE_CALLEES: Final = frozenset(
    {
        "register_from_manifest",
        "register_integration_plugin",
    }
)

_PLATFORM_COMPATIBILITY_SAFE_CONSUMER_MODULES: Final = frozenset(
    {
        _PACKAGE_COMPAT_SEMANTIC,
        _PACKAGE_SEMANTICS,
        _PACKAGE_QUALIFICATION,
        _PACKAGE_MANIFEST,
        _PACKAGE_MANIFEST_IO,
    }
)


def _path_bounded_non_r4_compat_classification(
    path: str,
    name: str,
) -> tuple[
    PluginProviderCompatibilityDisposition,
    PluginProviderClassificationEvidence,
    str,
] | None:
    """Evidence-backed classification for non-R4 compatibility vocabulary (no name-only fallback)."""
    if path == "intergrax/integrations/examples/custom_memory_kv/compatibility.py":
        if name == "CustomMemoryKvSchemaCompatibilityEvaluator":
            return (
                PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_EVALUATOR,
                PluginProviderClassificationEvidence.EXTERNAL_CONTRACT_EXAMPLE_COMPOSITION,
                "external_contract_example_evaluator",
            )
        if name == "custom_memory_kv_compatibility_extensions":
            return (
                PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_COMPOSITION,
                PluginProviderClassificationEvidence.EXTERNAL_CONTRACT_EXAMPLE_COMPOSITION,
                "external_contract_example_composition",
            )
    if path == "intergrax/runtime/plugins/compatibility.py":
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.RUNTIME_PLUGIN_SCHEMA_COMPATIBILITY,
            "runtime_plugin_schema_bundle_compatibility",
        )
    if path == "intergrax/runtime/architecture/capability_graph_compatibility.py":
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.CAPABILITY_GRAPH_MODALITY_COMPATIBILITY,
            "capability_graph_modality_compatibility",
        )
    if path in {
        "intergrax/contracts/context_view_scope_compatibility.py",
        "intergrax/contracts/context_view_composition.py",
        "intergrax/contracts/context_view_source_ports.py",
        "intergrax/collaborative_work/context_view_composition.py",
        "intergrax/context/policy/scope_isolation.py",
    }:
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.CONTEXT_VIEW_SCOPE_COMPATIBILITY,
            "context_view_scope_collaboration_compatibility",
        )
    if path in {
        "intergrax/autonomous_work/capability_acquisition_ports.py",
        "intergrax/contracts/autonomous_work/capability_acquisition.py",
    }:
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.WORKER_CAPABILITY_AUTHORITY_COMPATIBILITY,
            "worker_capability_authority_compatibility",
        )
    if path in {
        "intergrax/runtime/context_lifecycle/contracts.py",
        "intergrax/runtime/context_lifecycle/serialization.py",
    }:
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.ARTIFACT_LIFECYCLE_COMPATIBILITY,
            "artifact_lifecycle_compatibility",
        )
    if path == "intergrax/compat/langchain/documents.py":
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.LANGCHAIN_COMPAT_SHIM,
            "langchain_translation_shim",
        )
    if path == "intergrax/integrations/contracts/vector_index_administration.py":
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.VECTOR_INDEX_ADMIN_COMPATIBILITY_ERROR,
            "vector_index_administration_error",
        )
    if path == "intergrax/runtime/execution/lineage/wiring.py":
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.LINEAGE_PERSISTENCE_COMPATIBILITY,
            "execution_lineage_persistence_compatibility",
        )
    if path.startswith("applications/local_workspace_application/"):
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.APPLICATION_QUALIFICATION_COMPATIBILITY,
            "application_local_workspace_compatibility",
        )
    if path in {
        "intergrax/contracts/npsc5f_compatibility.py",
        "intergrax/core/distribution/errors.py",
        "intergrax/core/qualification/execution.py",
        "intergrax/hosting/contracts/service_registry.py",
        "intergrax/agent_distribution/agent_manager_models.py",
        "intergrax/applications/_shared/wiring.py",
        "intergrax/memory/contracts/entity_temporal_memory.py",
        "intergrax/runtime/architecture/capability_graph.py",
        "intergrax/runtime/nexus/context/graph_assembly.py",
        "intergrax/runtime/self_healing/resolution.py",
        "intergrax/runtime/vendor_knowledge/sync_document_store.py",
        "intergrax/runtime/plugins/default_plugins.py",
        "intergrax/integrations/providers/vector_store/qdrant/rag_store.py",
    }:
        return (
            PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE,
            PluginProviderClassificationEvidence.DOMAIN_INCOMPATIBILITY_ERROR_SYMBOL,
            "domain_local_incompatibility_symbol",
        )
    return None


def _classify_definition(
    module_path: str,
    qualname: str,
    name: str,
    lineno: int,
) -> PluginProviderCompatibilityCandidateRecord:
    path = module_path.replace("\\", "/")
    evidence = PluginProviderClassificationEvidence.UNSPECIFIED
    disposition = PluginProviderCompatibilityDisposition.UNCLASSIFIED
    concern = "plugin_provider_compatibility_candidate"
    owner = "unknown"

    if path == _PACKAGE_COMPAT_SEMANTIC:
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
        evidence = PluginProviderClassificationEvidence.CANONICAL_PLATFORM_COMPATIBILITY_MODULE
        owner = _PACKAGE_COMPAT_SEMANTIC
        concern = "package_platform_version_compatibility"
    elif path == _PACKAGE_MANIFEST and name in {"PlatformCompatibility", "PlatformPluginManifest"}:
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_DECLARATION
        evidence = PluginProviderClassificationEvidence.MANIFEST_PLATFORM_COMPATIBILITY_FIELD
        owner = _PACKAGE_MANIFEST
        concern = "package_platform_compatibility_declaration"
    elif path == _PACKAGE_MANIFEST_IO and "platform_compatibility" in name.lower():
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_DECLARATION
        evidence = PluginProviderClassificationEvidence.MANIFEST_IO_COERCION
        owner = _PACKAGE_MANIFEST_IO
        concern = "package_platform_compatibility_declaration"
    elif path == _PACKAGE_SEMANTICS and name == "check_manifest_platform_compatibility":
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_CONSUMER
        evidence = PluginProviderClassificationEvidence.MANIFEST_DELEGATING_WRAPPER
        owner = _PACKAGE_COMPAT_SEMANTIC
        concern = "package_platform_compatibility"
    elif path == _PACKAGE_QUALIFICATION:
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_CONSUMER
        evidence = PluginProviderClassificationEvidence.QUALIFICATION_ADMISSION_CONSUMER
        owner = _PACKAGE_QUALIFICATION
        concern = "package_qualification_admission"
    elif path == _PROVIDER_CATALOG and name == "register_integration":
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
        evidence = PluginProviderClassificationEvidence.INTEGRATION_CATALOG_REGISTER
        owner = _PROVIDER_CATALOG
        concern = "provider_registration_authority"
    elif path == _PROVIDER_CONTRACT_SPEC or name in {
        "declare_integration_contract",
        "augment_integration_contract_specs",
    }:
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_CONTRACT_DECLARATION
        evidence = PluginProviderClassificationEvidence.INTEGRATION_CONTRACT_SPEC_TYPED
        owner = _PROVIDER_CONTRACT_SPEC
        concern = "provider_contract_identity"
    elif path == _PROVIDER_PLUGIN_REGISTER:
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_DELEGATE
        evidence = PluginProviderClassificationEvidence.PLUGIN_REGISTER_CATALOG_DELEGATE
        owner = _PROVIDER_CATALOG
        concern = "provider_registration_delegate"
    elif path == _PROVIDER_FACTORY:
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_MATERIALIZATION_CONSUMER
        evidence = PluginProviderClassificationEvidence.FACTORY_MATERIALIZATION_TYPE_GUARD
        owner = _PROVIDER_FACTORY
        concern = "provider_materialization"
    elif path == _REGISTRY_V2:
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_PROJECTION_ONLY
        evidence = PluginProviderClassificationEvidence.REGISTRY_V2_CATALOG_PROJECTION
        owner = _REGISTRY_V2
        concern = "provider_contract_projection"
    elif path.startswith("intergrax/integrations/contracts/external_contract") or path in {
        _EXTERNAL_COMPAT_CONTRACTS,
        _EXTERNAL_COMPAT_EXTENSIONS,
        _EXTERNAL_COMPAT_SERVICE,
    }:
        if "EvidenceProvider" in name or name.endswith("EvidenceProvider"):
            disposition = PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_EVIDENCE_PROVIDER
            evidence = PluginProviderClassificationEvidence.EXTERNAL_CONTRACT_TYPED_CONTRACTS
        elif "Evaluator" in name:
            disposition = PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_EVALUATOR
            evidence = PluginProviderClassificationEvidence.EXTERNAL_CONTRACT_TYPED_CONTRACTS
        elif path == _EXTERNAL_COMPAT_EXTENSIONS or path == _EXTERNAL_COMPAT_SERVICE:
            disposition = PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_COMPOSITION
            evidence = (
                PluginProviderClassificationEvidence.EXTERNAL_COMPAT_EXTENSIONS_COMPOSITION
                if path == _EXTERNAL_COMPAT_EXTENSIONS
                else PluginProviderClassificationEvidence.EXTERNAL_COMPAT_SERVICE_COMPOSITION
            )
        else:
            disposition = PluginProviderCompatibilityDisposition.EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT
            evidence = PluginProviderClassificationEvidence.EXTERNAL_CONTRACT_TYPED_CONTRACTS
        owner = _EXTERNAL_COMPAT_CONTRACTS
        concern = "external_contract_semantic_compatibility"
    elif name == "is_compatible_runtime" or path.endswith("runtime/registry/semver_compat.py"):
        disposition = PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE
        evidence = PluginProviderClassificationEvidence.RUNTIME_SEMVER_REGISTRY_RESOLUTION
        owner = "intergrax/runtime/registry/semver_compat.py"
        concern = "runtime_registry_semver_not_plugin_provider_policy"
    elif name == "embed_openai_compatible" or "openai_compatible" in name.lower():
        disposition = PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE
        evidence = PluginProviderClassificationEvidence.OPENAI_API_COMPATIBILITY_ALIAS
        owner = path
        concern = "vendor_api_shape_not_platform_plugin_policy"
    elif name.startswith("assert_") and name.endswith("_compatible"):
        disposition = PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE
        evidence = PluginProviderClassificationEvidence.CONTRACT_IDENTITY_ASSERTION
        owner = path
        concern = "identity_assertion_not_compatibility_policy"
    elif disposition == PluginProviderCompatibilityDisposition.UNCLASSIFIED:
        bounded = _path_bounded_non_r4_compat_classification(path, name)
        if bounded is not None:
            disposition, evidence, concern = bounded
            owner = path

    return PluginProviderCompatibilityCandidateRecord(
        module_path=path,
        line=lineno,
        qualname=qualname,
        operation=name,
        disposition=disposition,
        semantic_concern=concern,
        owner_evidence=owner,
        classification_evidence=evidence,
    )


def _classify_call(
    module_path: str,
    qualname: str,
    callee: str,
    lineno: int,
) -> PluginProviderCompatibilityCandidateRecord:
    path = module_path.replace("\\", "/")
    disposition = PluginProviderCompatibilityDisposition.UNCLASSIFIED
    evidence = PluginProviderClassificationEvidence.UNSPECIFIED
    concern = "compatibility_call_site"
    owner = "unknown"

    if callee in _PACKAGE_EVAL_CALLS:
        if path == _PACKAGE_COMPAT_SEMANTIC:
            disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
            evidence = PluginProviderClassificationEvidence.CANONICAL_PLATFORM_COMPATIBILITY_MODULE
            owner = _PACKAGE_COMPAT_SEMANTIC
        elif path in {_PACKAGE_SEMANTICS, _PACKAGE_QUALIFICATION} or "platform_qualification" in path:
            disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_CONSUMER
            evidence = (
                PluginProviderClassificationEvidence.MANIFEST_DELEGATING_WRAPPER
                if path == _PACKAGE_SEMANTICS
                else PluginProviderClassificationEvidence.QUALIFICATION_ADMISSION_CONSUMER
            )
            owner = _PACKAGE_COMPAT_SEMANTIC
        else:
            disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_CONSUMER
            evidence = PluginProviderClassificationEvidence.QUALIFICATION_ADMISSION_CONSUMER
            owner = _PACKAGE_COMPAT_SEMANTIC
        concern = "package_platform_compatibility"
    elif callee in _PROVIDER_REG_CALLS:
        if callee in {"declare_integration_contract", "augment_integration_contract_specs"}:
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_CONTRACT_DECLARATION
            evidence = PluginProviderClassificationEvidence.INTEGRATION_CONTRACT_SPEC_TYPED
            owner = _PROVIDER_CONTRACT_SPEC
            concern = "provider_contract_identity"
        elif path == _PROVIDER_CATALOG and callee == "register_integration":
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
            evidence = PluginProviderClassificationEvidence.INTEGRATION_CATALOG_REGISTER
            owner = _PROVIDER_CATALOG
            concern = "provider_registration"
        elif path == _REGISTRY_V2:
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_PROJECTION_ONLY
            evidence = PluginProviderClassificationEvidence.REGISTRY_V2_CATALOG_PROJECTION
            owner = _REGISTRY_V2
            concern = "provider_registration"
        elif callee == "register_integration" and (
            path.startswith("intergrax/integrations/registry/")
            and path != _PROVIDER_CATALOG
            or path in {
                _PROVIDER_PLUGIN_REGISTER,
                "intergrax/core/catalog_bootstrap.py",
            }
            or (path.endswith("/register.py") and "/integrations/providers/" in path)
        ):
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_DELEGATE
            evidence = (
                PluginProviderClassificationEvidence.CATALOG_BOOTSTRAP_DELEGATE
                if path == "intergrax/core/catalog_bootstrap.py"
                else PluginProviderClassificationEvidence.PLUGIN_REGISTER_CATALOG_DELEGATE
                if path == _PROVIDER_PLUGIN_REGISTER
                else PluginProviderClassificationEvidence.PROVIDER_REGISTER_MODULE_DELEGATE
            )
            owner = _PROVIDER_CATALOG
            concern = "provider_registration_delegate"
        elif callee in _CATALOG_REGISTRATION_DELEGATE_CALLEES or path in {
            "intergrax/integrations/registry/bootstrap.py",
        }:
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_DELEGATE
            evidence = PluginProviderClassificationEvidence.PLUGIN_REGISTER_CATALOG_DELEGATE
            owner = _PROVIDER_CATALOG
            concern = "provider_registration_delegate"
        else:
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_REGISTRATION_SHADOW_AUTHORITY
            evidence = PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE
            owner = "shadow"
            concern = "provider_registration"
    elif callee in _EXTERNAL_COMPAT_CALLS:
        disposition = PluginProviderCompatibilityDisposition.EXTERNAL_COMPATIBILITY_COMPOSITION
        evidence = PluginProviderClassificationEvidence.EXTERNAL_COMPAT_EXTENSIONS_COMPOSITION
        owner = _EXTERNAL_COMPAT_EXTENSIONS
        concern = "external_contract_semantic_compatibility"
    elif callee == "is_compatible_runtime":
        disposition = PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE
        evidence = PluginProviderClassificationEvidence.RUNTIME_SEMVER_REGISTRY_RESOLUTION
        owner = "intergrax/runtime/registry/semver_compat.py"
        concern = "runtime_registry_semver_not_plugin_provider_policy"

    return PluginProviderCompatibilityCandidateRecord(
        module_path=path,
        line=lineno,
        qualname=qualname,
        operation=callee,
        disposition=disposition,
        semantic_concern=concern,
        owner_evidence=owner,
        classification_evidence=evidence,
    )


def _classify_specifier_set_call(
    module_path: str,
    qualname: str,
    lineno: int,
) -> PluginProviderCompatibilityCandidateRecord:
    path = module_path.replace("\\", "/")
    if path == _PACKAGE_COMPAT_SEMANTIC:
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
        evidence = PluginProviderClassificationEvidence.CANONICAL_PLATFORM_COMPATIBILITY_MODULE
    else:
        disposition = PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY
        evidence = PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE
    return PluginProviderCompatibilityCandidateRecord(
        module_path=path,
        line=lineno,
        qualname=qualname,
        operation="SpecifierSet(...)",
        disposition=disposition,
        semantic_concern="package_platform_version_compatibility",
        owner_evidence=_PACKAGE_COMPAT_SEMANTIC,
        classification_evidence=evidence,
    )


def discover_plugin_provider_compatibility_candidates(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[PluginProviderCompatibilityCandidateRecord, ...]:
    """Mechanical closed-world discovery for R4 (production roots + optional synthetic sources)."""
    records: list[PluginProviderCompatibilityCandidateRecord] = []
    seen: set[tuple[str, int, str]] = set()

    def add(record: PluginProviderCompatibilityCandidateRecord) -> None:
        key = (record.module_path, record.line, record.operation)
        if key in seen:
            return
        seen.add(key)
        records.append(record)

    sources: list[tuple[str, str]] = []
    for file_path in _iter_production_python_sources():
        rel = _repo_relative(file_path)
        sources.append((rel, file_path.read_text(encoding="utf-8")))
    sources.extend(extra_sources)

    for module_path, source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if "compatib" in node.name.lower() and _function_uses_class_name_compatibility(node):
                    add(
                        PluginProviderCompatibilityCandidateRecord(
                            module_path=module_path.replace("\\", "/"),
                            line=node.lineno,
                            qualname=node.name,
                            operation=node.name,
                            disposition=PluginProviderCompatibilityDisposition.IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY,
                            semantic_concern="implementation_type_as_compatibility",
                            owner_evidence="reflection_or_class_name",
                            classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                        )
                    )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if "compatib" not in node.name.lower() and node.name not in _EXTERNAL_COMPAT_TYPE_MARKERS:
                    if node.name in _PACKAGE_EVAL_CALLS or node.name in _PROVIDER_REG_CALLS:
                        qual = _enclosing_qualname(tree, node.lineno)
                        add(_classify_definition(module_path, qual, node.name, node.lineno))
                    continue
                qual = node.name if isinstance(node, ast.ClassDef) else _enclosing_qualname(tree, node.lineno)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    qual = f"{qual}.{node.name}" if qual != node.name else node.name
                add(_classify_definition(module_path, qual, node.name, node.lineno))
            if isinstance(node, ast.Call):
                callee = _call_name(node)
                if callee and (
                    callee in _PACKAGE_EVAL_CALLS
                    or callee in _PROVIDER_REG_CALLS
                    or callee in _EXTERNAL_COMPAT_CALLS
                    or callee == "is_compatible_runtime"
                ):
                    qual = _enclosing_qualname(tree, node.lineno)
                    add(_classify_call(module_path, qual, callee, node.lineno))
                if _is_specifier_set_call(node):
                    qual = _enclosing_qualname(tree, node.lineno)
                    add(_classify_specifier_set_call(module_path, qual, node.lineno))
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "__class__",
                    "__name__",
                }:
                    qual = _enclosing_qualname(tree, node.lineno)
                    parent_fn = qual.split(".")[-1] if qual else ""
                    if "compatib" in parent_fn.lower():
                        add(
                            PluginProviderCompatibilityCandidateRecord(
                                module_path=module_path.replace("\\", "/"),
                                line=node.lineno,
                                qualname=qual,
                                operation=parent_fn,
                                disposition=PluginProviderCompatibilityDisposition.IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY,
                                semantic_concern="implementation_type_as_compatibility",
                                owner_evidence="reflection_or_class_name",
                                classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                            )
                        )

    return tuple(sorted(records, key=lambda r: (r.module_path, r.line, r.operation)))


@lru_cache(maxsize=1)
def plugin_provider_r4_inventory() -> tuple[PluginProviderCompatibilityCandidateRecord, ...]:
    return discover_plugin_provider_compatibility_candidates()


def disposition_counts(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> dict[str, int]:
    items = records if records is not None else plugin_provider_r4_inventory()
    counter: Counter[str] = Counter()
    for row in items:
        counter[row.disposition.value] += 1
    return dict(counter)


def unclassified_records(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> tuple[PluginProviderCompatibilityCandidateRecord, ...]:
    items = records if records is not None else plugin_provider_r4_inventory()
    return tuple(r for r in items if r.disposition == PluginProviderCompatibilityDisposition.UNCLASSIFIED)


def count_package_platform_evaluator_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.module_path == _PACKAGE_COMPAT_SEMANTIC
        and r.operation == "check_platform_compatibility"
        and r.qualname == "check_platform_compatibility"
        and r.disposition == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
    )


def count_package_shadow_evaluators(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY
    )


def count_provider_catalog_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
        and r.module_path == _PROVIDER_CATALOG
        and r.operation == "register_integration"
        and r.qualname == "register_integration"
    )


def count_provider_catalog_delegates(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_DELEGATE
    )


def count_provider_projection_only(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_PROJECTION_ONLY
    )


def count_shadow_external_compat_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.SHADOW_EXTERNAL_COMPATIBILITY_AUTHORITY
    )


def count_compatibility_activation_shortcuts(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.COMPATIBILITY_ACTIVATION_SHORTCUT
    )


def count_compatibility_governance_shortcuts(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition
        in {
            PluginProviderCompatibilityDisposition.COMPATIBILITY_GOVERNANCE_AUTHORITY_SHORTCUT,
            PluginProviderCompatibilityDisposition.COMPATIBILITY_EXECUTION_AUTHORITY_SHORTCUT,
        }
    )


def count_external_semantic_assessment_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    contract_markers = {
        r.operation
        for r in items
        if r.module_path == _EXTERNAL_COMPAT_CONTRACTS
        and r.disposition
        == PluginProviderCompatibilityDisposition.EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT
    }
    return 1 if _EXTERNAL_COMPAT_CONTRACTS and contract_markers else 0


def count_provider_registration_shadow_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_REGISTRATION_SHADOW_AUTHORITY
    )


def count_implementation_based_compatibility_authorities(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> int:
    items = records if records is not None else plugin_provider_r4_inventory()
    return sum(
        1
        for r in items
        if r.disposition
        == PluginProviderCompatibilityDisposition.IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY
    )


def validate_r4_closed_world_classification(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    items = records if records is not None else plugin_provider_r4_inventory()
    unclassified = unclassified_records(items)
    if unclassified:
        violations.append(f"UNCLASSIFIED plugin/provider compatibility candidates: {len(unclassified)}")
    for row in unclassified[:5]:
        violations.append(f"  example: {row.module_path}:{row.line} {row.operation}")
    for row in items:
        if row.disposition == PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE:
            if row.classification_evidence == PluginProviderClassificationEvidence.UNSPECIFIED:
                violations.append(
                    f"NON_COMPATIBILITY_USE without positive evidence: {row.module_path}:{row.line}"
                )
    return tuple(violations)


def qualification_composition_authority_violations(
    matrix: "tuple[CompatOwnerMatrixRow, ...] | None" = None,
) -> tuple[str, ...]:
    from tests.qualification.compat_x._compat_x_owner_discovery import COMPAT_X_OWNER_MATRIX
    from tests.qualification.compat_x._compat_x_types import CompatOwnerMatrixRow, OwnerResponsibilityState

    owner_matrix: tuple[CompatOwnerMatrixRow, ...] = (
        matrix if matrix is not None else COMPAT_X_OWNER_MATRIX
    )
    violations: list[str] = []
    policy_rows = [
        r
        for r in owner_matrix
        if r.concern == "plugin_provider_compatibility_policy"
    ]
    if len(policy_rows) != 1:
        violations.append(
            f"plugin_provider_compatibility_policy owner rows: {len(policy_rows)} (expected 1)"
        )
    for row in policy_rows:
        if row.responsibility_state is not OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER:
            violations.append("plugin_provider_compatibility_policy not CURRENT_CONFIRMED_OWNER")
        if "COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md" not in row.semantic_owner_path:
            violations.append("plugin_provider_compatibility_policy semantic owner drift")
        if row.composition_owner_path.startswith("tests/"):
            violations.append("qualification module listed as composition owner for R4 policy")
    qual_authority = sum(
        1
        for row in owner_matrix
        if row.responsibility_state is OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
        and (
            row.semantic_owner_path.startswith("tests/")
            or row.composition_owner_path.startswith("tests/")
        )
        and row.concern
        in {
            "plugin_provider_compatibility_policy",
            "external_contract_semantic_assessment",
            "platform_plugin_manifest_compatibility",
            "provider_contract_compatibility",
        }
    )
    if qual_authority != 0:
        violations.append(f"qualification semantic/composition authority rows: {qual_authority}")
    external_rows = [
        r for r in owner_matrix if r.concern == "external_contract_semantic_assessment"
    ]
    if len(external_rows) != 1:
        violations.append(
            f"external_contract_semantic_assessment owner rows: {len(external_rows)} (expected 1)"
        )
    elif external_rows[0].semantic_owner_path != _EXTERNAL_COMPAT_CONTRACTS:
        violations.append("external_contract_semantic_assessment semantic owner drift")
    return tuple(violations)


def _bounded_platform_compatibility_consumer_violations() -> tuple[str, ...]:
    violations: list[str] = []
    for file_path in _iter_production_python_sources():
        rel = _repo_relative(file_path)
        if rel in _PLATFORM_COMPATIBILITY_SAFE_CONSUMER_MODULES:
            continue
        source = file_path.read_text(encoding="utf-8")
        if "check_platform_compatibility(" not in source:
            continue
        violations.append(
            f"unclassified platform compatibility consumer outside bounded surface: {rel}"
        )
    return tuple(violations)


def _function_has_compatible_activation_shortcut(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    activation_calls = frozenset(
        {
            "activate_plugin",
            "load_plugin",
            "register_integration",
            "register_integration_plugin",
            "execute_plugin",
        }
    )
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Attribute)
            and test.attr == "compatible"
            and isinstance(test.value, ast.Name)
        ):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _call_name(child)
                if callee in activation_calls:
                    return True
    return False


def _function_grants_governance_from_compatible(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    governance_calls = frozenset(
        {
            "grant_permission",
            "approve",
            "authorize",
            "mint_execution_authority",
        }
    )
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Attribute) and test.attr == "compatible"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _call_name(child)
                if callee in governance_calls:
                    return True
    return False


def _is_shadow_external_assessment_class(node: ast.ClassDef) -> bool:
    method_names = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    return {"assess", "collect_evidence", "resolve_expectation"}.issubset(method_names)


def _module_defines_rogue_provider_registry(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rogue_"):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign) and isinstance(inner.targets[0], ast.Subscript):
                    return True
    return False


def _discover_shortcut_and_shadow_records(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[PluginProviderCompatibilityCandidateRecord, ...]:
    records: list[PluginProviderCompatibilityCandidateRecord] = []
    for module_path, source in extra_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if _module_defines_rogue_provider_registry(tree):
            records.append(
                PluginProviderCompatibilityCandidateRecord(
                    module_path=module_path.replace("\\", "/"),
                    line=1,
                    qualname="<module>",
                    operation="rogue_local_provider_registry",
                    disposition=PluginProviderCompatibilityDisposition.PROVIDER_REGISTRATION_SHADOW_AUTHORITY,
                    semantic_concern="provider_registration_shadow",
                    owner_evidence="shadow",
                    classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                )
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_shadow_external_assessment_class(node):
                records.append(
                    PluginProviderCompatibilityCandidateRecord(
                        module_path=module_path.replace("\\", "/"),
                        line=node.lineno,
                        qualname=node.name,
                        operation=node.name,
                        disposition=PluginProviderCompatibilityDisposition.SHADOW_EXTERNAL_COMPATIBILITY_AUTHORITY,
                        semantic_concern="shadow_external_contract_assessment",
                        owner_evidence="shadow",
                        classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                    )
                )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if _function_has_compatible_activation_shortcut(node):
                    records.append(
                        PluginProviderCompatibilityCandidateRecord(
                            module_path=module_path.replace("\\", "/"),
                            line=node.lineno,
                            qualname=node.name,
                            operation=node.name,
                            disposition=PluginProviderCompatibilityDisposition.COMPATIBILITY_ACTIVATION_SHORTCUT,
                            semantic_concern="compatibility_activation_shortcut",
                            owner_evidence=module_path,
                            classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                        )
                    )
                if _function_grants_governance_from_compatible(node):
                    records.append(
                        PluginProviderCompatibilityCandidateRecord(
                            module_path=module_path.replace("\\", "/"),
                            line=node.lineno,
                            qualname=node.name,
                            operation=node.name,
                            disposition=PluginProviderCompatibilityDisposition.COMPATIBILITY_GOVERNANCE_AUTHORITY_SHORTCUT,
                            semantic_concern="compatibility_governance_shortcut",
                            owner_evidence=module_path,
                            classification_evidence=PluginProviderClassificationEvidence.SYNTHETIC_SHADOW_PROBE,
                        )
                    )
    return tuple(records)


def collect_plugin_provider_compatibility_r4_violations(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[str, ...]:
    records = discover_plugin_provider_compatibility_candidates(extra_sources=extra_sources)
    records = tuple(records) + _discover_shortcut_and_shadow_records(extra_sources=extra_sources)
    violations = list(validate_r4_closed_world_classification(records))
    violations.extend(qualification_composition_authority_violations())
    violations.extend(_bounded_platform_compatibility_consumer_violations())
    if count_package_shadow_evaluators(records) != 0:
        violations.append(
            f"package/platform shadow evaluator count {count_package_shadow_evaluators(records)} (expected 0)"
        )
    eval_defs = sum(
        1
        for r in records
        if r.module_path == _PACKAGE_COMPAT_SEMANTIC
        and r.operation == "check_platform_compatibility"
        and r.disposition == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
        and r.qualname == "check_platform_compatibility"
    )
    if eval_defs != 1:
        violations.append(f"canonical check_platform_compatibility definitions: {eval_defs} (expected 1)")
    if count_implementation_based_compatibility_authorities(records) != 0:
        violations.append("implementation-based compatibility authority detected")
    if count_provider_catalog_authorities(records) != 1:
        violations.append(
            f"canonical Integration Catalog authority count {count_provider_catalog_authorities(records)} (expected 1)"
        )
    if count_provider_registration_shadow_authorities(records) != 0:
        violations.append(
            f"provider registration shadow authority count {count_provider_registration_shadow_authorities(records)}"
        )
    if count_shadow_external_compat_authorities(records) != 0:
        violations.append("parallel external contract compatibility authority detected")
    if count_compatibility_activation_shortcuts(records) != 0:
        violations.append("compatibility_to_activation_shortcut detected")
    if count_compatibility_governance_shortcuts(records) != 0:
        violations.append("compatibility_to_governance_or_execution_authority_shortcut detected")
    if count_package_platform_evaluator_authorities(records) != 1:
        violations.append(
            f"package/platform evaluator semantic authority count {count_package_platform_evaluator_authorities(records)} (expected 1)"
        )
    registry_v2_authority = sum(
        1
        for r in records
        if r.module_path == _REGISTRY_V2
        and r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
    )
    if registry_v2_authority != 0:
        violations.append("registry_v2 registration authority must remain projection-only")
    projection_only = all(
        r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_PROJECTION_ONLY
        for r in records
        if r.module_path == _REGISTRY_V2 and r.operation == "register_integration"
    )
    v2_regs = [
        r
        for r in records
        if r.module_path == _REGISTRY_V2 and r.operation == "register_integration"
    ]
    if v2_regs and not projection_only:
        violations.append("registry_v2 register_integration is not projection-only disposition")
    return tuple(violations)


def plugin_provider_compatibility_r4_gate_passes(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    violations = collect_plugin_provider_compatibility_r4_violations(extra_sources=extra_sources)
    return (not violations, violations)


# --- Synthetic adversarial fixtures (qualification probes; not production) ---

SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE: Final[str] = '''
from packaging.specifiers import SpecifierSet

def decide_plugin_compat(declared: str, platform: str) -> bool:
    return platform in SpecifierSet(declared)
'''

SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE: Final[str] = '''
def provider_contract_compatible(left: str, right: str) -> bool:
    return left == right
'''

_UNKNOWN_COMPAT_PROBE_MODULE: Final = (
    "intergrax/applications/provider_bridge/provider_contract_compatible.py"
)

SYNTHETIC_SANCTIONED_CATALOG_DELEGATE_SOURCE: Final[str] = '''
from intergrax.integrations.registry.catalog import IntegrationEntry, register_integration

def register_via_catalog_delegate(entry: IntegrationEntry) -> None:
    register_integration(entry)
'''

SYNTHETIC_ROGUE_LOCAL_PROVIDER_REGISTRY_SOURCE: Final[str] = '''
from intergrax.integrations.registry.catalog import IntegrationEntry

_ROGUE_LOCAL_REGISTRY: dict[str, IntegrationEntry] = {}

def rogue_local_register(entry: IntegrationEntry) -> None:
    _ROGUE_LOCAL_REGISTRY[entry.slug] = entry
'''

SYNTHETIC_SHADOW_EXTERNAL_ASSESSMENT_SOURCE: Final[str] = '''
class RogueExternalContractAssessment:
    def resolve_expectation(self, subject: object) -> object:
        return subject

    def collect_evidence(self, request: object) -> tuple[object, ...]:
        return ()

    def assess(self, expectation: object, evidence: tuple[object, ...]) -> bool:
        return True
'''

SYNTHETIC_COMPATIBILITY_ACTIVATION_SHORTCUT_SOURCE: Final[str] = '''
def admit_on_platform_compatibility(compatibility: object) -> None:
    if compatibility.compatible:
        activate_plugin("demo")
'''

SYNTHETIC_COMPATIBILITY_GOVERNANCE_SHORTCUT_SOURCE: Final[str] = '''
def grant_on_platform_compatibility(compatibility: object) -> None:
    if compatibility.compatible:
        grant_permission("execute")
'''

SYNTHETIC_REGISTRY_V2_SHADOW_REGISTER_SOURCE: Final[str] = '''
from intergrax.integrations.registry.catalog import IntegrationEntry, register_integration

def rogue_register(entry: IntegrationEntry) -> None:
    register_integration(entry)
'''

SYNTHETIC_IMPLEMENTATION_COMPATIBILITY_SOURCE: Final[str] = '''
def provider_is_compatible(impl: object) -> bool:
    return impl.__class__.__name__.endswith("Integration")
'''


def classify_synthetic_shadow_package_evaluator() -> PluginProviderCompatibilityDisposition:
    records = discover_plugin_provider_compatibility_candidates(
        extra_sources=(
            ("intergrax/synthetic/shadow_package_eval.py", SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE),
        )
    )
    shadow = [
        r
        for r in records
        if r.module_path == "intergrax/synthetic/shadow_package_eval.py"
        and r.disposition == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY
    ]
    return shadow[0].disposition if shadow else PluginProviderCompatibilityDisposition.UNCLASSIFIED


def classify_synthetic_unknown_helper() -> PluginProviderCompatibilityDisposition:
    records = discover_plugin_provider_compatibility_candidates(
        extra_sources=(
            (_UNKNOWN_COMPAT_PROBE_MODULE, SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE),
        )
    )
    for row in records:
        if row.module_path == _UNKNOWN_COMPAT_PROBE_MODULE and row.operation == "provider_contract_compatible":
            return row.disposition
    return PluginProviderCompatibilityDisposition.UNCLASSIFIED


def classify_synthetic_sanctioned_catalog_delegate() -> PluginProviderCompatibilityDisposition:
    module = "intergrax/integrations/registry/delegate_probe.py"
    records = discover_plugin_provider_compatibility_candidates(
        extra_sources=((module, SYNTHETIC_SANCTIONED_CATALOG_DELEGATE_SOURCE),)
    )
    for row in records:
        if row.module_path == module and row.operation == "register_integration":
            return row.disposition
    return PluginProviderCompatibilityDisposition.UNCLASSIFIED


def classify_synthetic_registry_shadow_register() -> PluginProviderCompatibilityDisposition:
    records = discover_plugin_provider_compatibility_candidates(
        extra_sources=(
            ("intergrax/synthetic/rogue_catalog_register.py", SYNTHETIC_REGISTRY_V2_SHADOW_REGISTER_SOURCE),
        )
    )
    for row in records:
        if row.module_path == "intergrax/synthetic/rogue_catalog_register.py":
            return row.disposition
    return PluginProviderCompatibilityDisposition.UNCLASSIFIED


def classify_synthetic_implementation_compatibility() -> PluginProviderCompatibilityDisposition:
    records = discover_plugin_provider_compatibility_candidates(
        extra_sources=(
            (
                "intergrax/synthetic/impl_class_compat.py",
                SYNTHETIC_IMPLEMENTATION_COMPATIBILITY_SOURCE,
            ),
        )
    )
    for row in records:
        if row.module_path == "intergrax/synthetic/impl_class_compat.py" and row.operation == "provider_is_compatible":
            return row.disposition
    return PluginProviderCompatibilityDisposition.UNCLASSIFIED


def provider_r4_authority_report(
    records: tuple[PluginProviderCompatibilityCandidateRecord, ...] | None = None,
) -> dict[str, int]:
    items = records if records is not None else plugin_provider_r4_inventory()
    return {
        "authority_count": count_provider_catalog_authorities(items),
        "delegate_count": count_provider_catalog_delegates(items),
        "projection_count": count_provider_projection_only(items),
        "shadow_count": count_provider_registration_shadow_authorities(items),
    }


__all__ = [
    "PluginProviderCompatibilityDisposition",
    "PluginProviderCompatibilityCandidateRecord",
    "PluginProviderClassificationEvidence",
    "collect_plugin_provider_compatibility_r4_violations",
    "plugin_provider_compatibility_r4_gate_passes",
    "discover_plugin_provider_compatibility_candidates",
    "plugin_provider_r4_inventory",
    "disposition_counts",
    "unclassified_records",
    "validate_r4_closed_world_classification",
    "classify_synthetic_shadow_package_evaluator",
    "classify_synthetic_unknown_helper",
    "classify_synthetic_registry_shadow_register",
    "classify_synthetic_implementation_compatibility",
    "classify_synthetic_sanctioned_catalog_delegate",
    "qualification_composition_authority_violations",
    "provider_r4_authority_report",
    "count_provider_catalog_delegates",
    "SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE",
    "SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE",
    "SYNTHETIC_SANCTIONED_CATALOG_DELEGATE_SOURCE",
    "SYNTHETIC_ROGUE_LOCAL_PROVIDER_REGISTRY_SOURCE",
    "SYNTHETIC_SHADOW_EXTERNAL_ASSESSMENT_SOURCE",
    "SYNTHETIC_COMPATIBILITY_ACTIVATION_SHORTCUT_SOURCE",
    "SYNTHETIC_COMPATIBILITY_GOVERNANCE_SHORTCUT_SOURCE",
    "_UNKNOWN_COMPAT_PROBE_MODULE",
]
