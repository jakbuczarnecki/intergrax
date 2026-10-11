# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R4 closed-world plugin/provider compatibility inventory (qualification only)."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

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
    PROVIDER_PROJECTION_ONLY = "PROVIDER_PROJECTION_ONLY"
    PROVIDER_MATERIALIZATION_CONSUMER = "PROVIDER_MATERIALIZATION_CONSUMER"
    PROVIDER_REGISTRATION_SHADOW_AUTHORITY = "PROVIDER_REGISTRATION_SHADOW_AUTHORITY"
    EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT = "EXTERNAL_CONTRACT_COMPATIBILITY_CONTRACT"
    EXTERNAL_COMPATIBILITY_EVIDENCE_PROVIDER = "EXTERNAL_COMPATIBILITY_EVIDENCE_PROVIDER"
    EXTERNAL_COMPATIBILITY_EVALUATOR = "EXTERNAL_COMPATIBILITY_EVALUATOR"
    EXTERNAL_COMPATIBILITY_COMPOSITION = "EXTERNAL_COMPATIBILITY_COMPOSITION"
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
        disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
        evidence = PluginProviderClassificationEvidence.PLUGIN_REGISTER_CATALOG_DELEGATE
        owner = _PROVIDER_CATALOG
        concern = "provider_registration_authority"
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
    elif name == "is_compatible" and "/synthetic/" in path:
        disposition = PluginProviderCompatibilityDisposition.UNCLASSIFIED
        evidence = PluginProviderClassificationEvidence.SYNTHETIC_UNCLASSIFIED_PROBE
        owner = path
        concern = "unknown_compatibility_helper"
    elif "compatib" in name.lower():
        disposition = PluginProviderCompatibilityDisposition.NON_COMPATIBILITY_USE
        evidence = PluginProviderClassificationEvidence.UNRELATED_DOMAIN_COMPATIBILITY_NAMING
        owner = path
        concern = "domain_local_compatibility_vocabulary"

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
        elif path == _PROVIDER_PLUGIN_REGISTER or path.endswith("/register.py") or path.endswith(
            "contract_spec.py"
        ) or "registry/bootstrap" in path or path.endswith("catalog_bootstrap.py"):
            disposition = PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
            evidence = PluginProviderClassificationEvidence.PLUGIN_REGISTER_CATALOG_DELEGATE
            owner = _PROVIDER_CATALOG
            concern = "provider_registration"
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
        if r.disposition == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_EVALUATOR
        and r.operation in {"check_platform_compatibility", "require_platform_compatibility", "PlatformCompatibility"}
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
    modules = {
        r.module_path
        for r in items
        if r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
        and r.module_path == _PROVIDER_CATALOG
        and r.operation == "register_integration"
    }
    return len(modules)


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


def collect_plugin_provider_compatibility_r4_violations(
    *,
    extra_sources: tuple[tuple[str, str], ...] = (),
) -> tuple[str, ...]:
    records = discover_plugin_provider_compatibility_candidates(extra_sources=extra_sources)
    violations = list(validate_r4_closed_world_classification(records))
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
    if count_provider_registration_shadow_authorities(records) != 0:
        violations.append("provider registration shadow authority detected")
    if count_implementation_based_compatibility_authorities(records) != 0:
        violations.append("implementation-based compatibility authority detected")
    catalog_reg = sum(
        1
        for r in records
        if r.module_path == _PROVIDER_CATALOG
        and r.operation == "register_integration"
        and r.disposition == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_AUTHORITY
    )
    if catalog_reg < 1:
        violations.append("canonical Integration Catalog register_integration authority missing")
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
def is_compatible(plugin_id: str, host: str) -> bool:
    return plugin_id == host
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
            ("intergrax/synthetic/new_plugin_compatibility.py", SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE),
        )
    )
    for row in records:
        if row.module_path == "intergrax/synthetic/new_plugin_compatibility.py":
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
    "SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE",
    "SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE",
]
