# © Artur Czarnecki. All rights reserved.

"""FRZ-CFG-05 — closed-world semantic production selection discovery."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.config_x._config_x_concern_inventory import (
    CONFIG_X_CONCERN_INVENTORY,
)
from tests.qualification.config_x._config_x_path_evidence import (
    expand_provider_surface_glob,
    primary_repo_paths_from_text,
    repo_root,
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
    "docker/runtime-context",
)

_SEMANTIC_PARAM_NAMES: Final[frozenset[str]] = frozenset(
    {
        "tenant_id",
        "default_model",
        "model",
        "endpoint",
        "api_key",
        "region",
        "backend",
        "provider",
        "base_url",
        "secret_ref",
        "url",
        "host",
        "http_host",
    },
)

_CONFIG_SURFACE_TOKENS: Final[tuple[str, ...]] = (
    "/config.py",
    "/configs.py",
    "/schema.py",
    "llm_profile.py",
    "embedding_profile.py",
    "routing_profile.py",
)

_FORBIDDEN_TEXT_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ('tenant_id: str = "default"', "ambient default tenant literal"),
    ('tenant_id = "default"', "ambient default tenant literal"),
    ('DEFAULT_TENANT_ID = "default"', "ambient default tenant constant"),
    ('PREFIX_TENANT_ID", "default")', "ambient default tenant env fallback"),
    (', "default").strip() or "default"', "ambient default tenant env fallback"),
)

_AMBIENT_LOCALHOST_URL_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ('"http://localhost', "ambient localhost base URL literal"),
    ("'http://localhost", "ambient localhost base URL literal"),
)

_LOCALHOST_SANCTIONED_REL_PATHS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/integrations/_shared/config.py",
    },
)

# Explicit FRZ-CFG-05 evidence: per-surface sanctioned transport/deployment defaults.
# Provider activation remains explicit; env/configuration overrides these defaults.
_SANCTIONED_INTEGRATION_TRANSPORT_DEFAULT_SURFACES: Final[frozenset[str]] = frozenset(
    {
        "intergrax/integrations/providers/vector_store/qdrant/config.py",
        "intergrax/integrations/providers/vector_store/chroma/config.py",
        "intergrax/integrations/providers/key_value_cache/redis/config.py",
        "intergrax/integrations/providers/message_bus/rabbitmq/config.py",
        "intergrax/integrations/providers/relational_store/mysql/config.py",
        "intergrax/integrations/providers/relational_store/postgresql/config.py",
    },
)

_SANCTIONED_INTEGRATION_TRANSPORT_DETAIL: Final[str] = (
    "sanctioned integration transport default (localhost/local endpoint) — explicit "
    "provider selection required; host/url/port are deployment wiring only; "
    "INTERGRAX_* env and typed overrides supersede; presence of provider code "
    "does not activate the backend"
)


class SemanticProductionFindingClass(StrEnum):
    APPROVED_TYPED_CONFIGURATION = "approved_typed_configuration"
    SANCTIONED_DEPLOYMENT_DEFAULT = "sanctioned_deployment_default"
    DEPLOYMENT_ENVIRONMENT_CONSTANT = "deployment_environment_constant"
    PROTOCOL_FORMAT_CONSTANT = "protocol_format_constant"
    REFERENCE_LAB_TEST_ONLY = "reference_lab_test_only"
    HARD_CODED_SEMANTIC_PRODUCTION_SELECTION = "hard_coded_semantic_production_selection"
    NAMED_CONSTANT_SEMANTIC_BLIND_SPOT = "named_constant_semantic_blind_spot"


@dataclass(frozen=True, slots=True)
class SemanticProductionFinding:
    repo_path: str
    finding_class: SemanticProductionFindingClass
    detail: str
    concern_id: str | None = None


def _is_excluded_path(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in rel or part in normalized:
            return True
    if normalized.startswith("applications/lab_application/"):
        return True
    return False


def _is_configuration_semantics_surface(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    if any(token in normalized for token in _CONFIG_SURFACE_TOKENS):
        return True
    if normalized.endswith("integrations/_shared/p3/configs.py"):
        return True
    return False


def _chokepoint_paths() -> frozenset[str]:
    paths: set[str] = set()
    for row in CONFIG_X_CONCERN_INVENTORY:
        paths.update(primary_repo_paths_from_text(row.composition_owner))
        paths.update(primary_repo_paths_from_text(row.effective_resolution_owner))
    paths.update(
        {
            "intergrax/applications/_shared/harness_task_routes.py",
            "intergrax/applications/_shared/trace_explorer_routes.py",
            "intergrax/multimedia/image_smart_loader.py",
            "intergrax/integrations/_shared/p3/configs.py",
            "intergrax/tools/providers/observability/resolve.py",
            "intergrax/tokenizers/registry/tokenizer_registry.py",
            "intergrax/integrations/registry/factory.py",
            "intergrax/llm_adapters/llm_provider_registry.py",
        },
    )
    return frozenset(sorted(paths))


@lru_cache(maxsize=1)
def discover_provider_surface_path_to_concerns() -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for row in CONFIG_X_CONCERN_INVENTORY:
        concern_id = row.concern_id
        expanded = expand_provider_surface_glob(row.provider_surface)
        for path in expanded:
            mapping.setdefault(path, []).append(concern_id)
        for path in primary_repo_paths_from_text(row.provider_surface):
            if (repo_root() / path).is_file():
                mapping.setdefault(path, []).append(concern_id)
    return {path: tuple(concerns) for path, concerns in mapping.items()}


@lru_cache(maxsize=1)
def discover_inventory_provider_surface_paths() -> frozenset[str]:
    return frozenset(discover_provider_surface_path_to_concerns().keys())


def _scan_paths_for_inventory_surfaces() -> frozenset[str]:
    paths: set[str] = set(_chokepoint_paths())
    paths.update(discover_inventory_provider_surface_paths())
    return frozenset(sorted(paths))


def _primary_concern_for_path(rel: str) -> str | None:
    concerns = discover_provider_surface_path_to_concerns().get(rel)
    if not concerns:
        return None
    return concerns[0]


def _lab_reference_provider_surface_paths() -> frozenset[str]:
    return frozenset(
        path
        for path in discover_inventory_provider_surface_paths()
        if path.replace("\\", "/").startswith("applications/lab_application/")
    )


def _normalize_production_finding(finding: SemanticProductionFinding) -> SemanticProductionFinding:
    if finding.finding_class is not SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION:
        return finding
    rel = finding.repo_path.replace("\\", "/")
    if "api.elevenlabs.io" in finding.detail:
        return SemanticProductionFinding(
            repo_path=finding.repo_path,
            finding_class=SemanticProductionFindingClass.PROTOCOL_FORMAT_CONSTANT,
            detail="vendor REST API base URL — not ambient production endpoint selection",
            concern_id=finding.concern_id,
        )
    if "/observability_backend/" in rel and ("localhost" in finding.detail or "base_url" in finding.detail):
        return SemanticProductionFinding(
            repo_path=finding.repo_path,
            finding_class=SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT,
            detail=(
                "local observability stack reference default — explicit profile/env binding required for activation"
            ),
            concern_id=finding.concern_id,
        )
    return finding


def _module_level_string_constants(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    bindings[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                bindings[node.target.id] = node.value.value
    return bindings


def _resolve_string_constant(
    expr: ast.expr | None,
    bindings: dict[str, str],
) -> tuple[str | None, bool]:
    """Return (resolved literal, unresolved_name_reference)."""
    if expr is None:
        return None, False
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value, False
    if isinstance(expr, ast.Name):
        if expr.id in bindings:
            return bindings[expr.id], False
        return None, True
    return None, False


def _is_environ_get_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
    )


def _collect_env_get_default_name_nodes(expr: ast.expr) -> tuple[ast.Name, ...]:
    names: list[ast.Name] = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and _is_environ_get_call(node):
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                names.append(node.args[1])
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for value in node.values:
                if isinstance(value, ast.Name):
                    names.append(value)
    return tuple(names)


def _classify_resolved_semantic_string_default(
    rel: str,
    field_name: str,
    value: str,
    *,
    context: str,
) -> SemanticProductionFindingClass:
    normalized = rel.replace("\\", "/")
    stripped = value.strip()
    if not stripped:
        return SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION
    if stripped == "default" and field_name == "tenant_id":
        return SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION
    if "localhost" in stripped and field_name in {"base_url", "url", "host", "http_host"}:
        if normalized in _SANCTIONED_INTEGRATION_TRANSPORT_DEFAULT_SURFACES:
            return SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
        return SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION
    return SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION


def _finding_for_semantic_string_default(
    rel: str,
    field_name: str,
    value: str,
    *,
    context: str,
    concern_id: str | None,
) -> SemanticProductionFinding | None:
    finding_class = _classify_resolved_semantic_string_default(
        rel,
        field_name,
        value,
        context=context,
    )
    if finding_class is SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION:
        return None
    detail = (
        _SANCTIONED_INTEGRATION_TRANSPORT_DETAIL
        if finding_class is SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
        else f"{context} defaults to semantic string {value!r}"
    )
    return SemanticProductionFinding(
        repo_path=rel,
        finding_class=finding_class,
        detail=detail,
        concern_id=concern_id,
    )


def _classify_path_context(rel: str) -> SemanticProductionFindingClass | None:
    if _is_excluded_path(rel):
        return SemanticProductionFindingClass.REFERENCE_LAB_TEST_ONLY
    if rel in _LOCALHOST_SANCTIONED_REL_PATHS or rel.endswith("integrations/_shared/config.py"):
        return SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
    if "/contracts/" in rel or "/core/" in rel:
        return SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION
    if rel.endswith("USAGE.md"):
        return SemanticProductionFindingClass.REFERENCE_LAB_TEST_ONLY
    return None


def _class_assign_defaults_on_semantic_fields(
    tree: ast.Module,
    rel: str,
    *,
    bindings: dict[str, str] | None = None,
) -> list[SemanticProductionFinding]:
    if bindings is None:
        bindings = _module_level_string_constants(tree)
    findings: list[SemanticProductionFinding] = []
    concern_id = _primary_concern_for_path(rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                target_name: str | None = None
                default_node: ast.expr | None = None
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    target_name = stmt.target.id
                    default_node = stmt.value
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            target_name = target.id
                            default_node = stmt.value
                            break
                if target_name is None or target_name not in _SEMANTIC_PARAM_NAMES:
                    continue
                if default_node is None:
                    continue
                resolved, unresolved = _resolve_string_constant(default_node, bindings)
                if unresolved:
                    const_ref = default_node.id if isinstance(default_node, ast.Name) else ast.dump(default_node)
                    findings.append(
                        SemanticProductionFinding(
                            repo_path=rel,
                            finding_class=SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT,
                            detail=(
                                f"{node.name}.{target_name} uses unresolved named constant "
                                f"{const_ref!r} as semantic default"
                            ),
                            concern_id=concern_id,
                        ),
                    )
                    continue
                if resolved is None or not resolved.strip():
                    continue
                item = _finding_for_semantic_string_default(
                    rel,
                    target_name,
                    resolved,
                    context=f"{node.name}.{target_name}",
                    concern_id=concern_id,
                )
                if item is not None:
                    findings.append(item)
    return findings


def _from_env_semantic_constant_fallbacks(
    tree: ast.Module,
    rel: str,
    *,
    bindings: dict[str, str],
) -> list[SemanticProductionFinding]:
    findings: list[SemanticProductionFinding] = []
    concern_id = _primary_concern_for_path(rel)
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for stmt in class_node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if stmt.name != "from_env":
                continue
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.Assign) or len(inner.targets) != 1:
                    continue
                target = inner.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                field_name = target.id
                if field_name not in _SEMANTIC_PARAM_NAMES:
                    continue
                for name_node in _collect_env_get_default_name_nodes(inner.value):
                    resolved, unresolved = _resolve_string_constant(name_node, bindings)
                    if unresolved:
                        findings.append(
                            SemanticProductionFinding(
                                repo_path=rel,
                                finding_class=SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT,
                                detail=(
                                    f"{class_node.name}.from_env {field_name} env fallback "
                                    f"references unresolved constant {name_node.id!r}"
                                ),
                                concern_id=concern_id,
                            ),
                        )
                        continue
                    if resolved is None or not resolved.strip():
                        continue
                    item = _finding_for_semantic_string_default(
                        rel,
                        field_name,
                        resolved,
                        context=f"{class_node.name}.from_env {field_name} env fallback",
                        concern_id=concern_id,
                    )
                    if item is not None:
                        findings.append(item)
    return findings


def classify_semantic_defaults_in_module_source(
    source: str,
    rel: str,
) -> tuple[SemanticProductionFinding, ...]:
    """Qualification helper — classify semantic defaults in a synthetic module snippet."""
    tree = ast.parse(source)
    bindings = _module_level_string_constants(tree)
    findings: list[SemanticProductionFinding] = []
    findings.extend(_class_assign_defaults_on_semantic_fields(tree, rel, bindings=bindings))
    findings.extend(_from_env_semantic_constant_fallbacks(tree, rel, bindings=bindings))
    return tuple(_normalize_production_finding(f) for f in findings)


def _literal_default_on_semantic_param(tree: ast.Module, rel: str) -> list[SemanticProductionFinding]:
    findings: list[SemanticProductionFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = list(node.args.args) + list(node.args.kwonlyargs)
        defaults = list(node.args.defaults) + list(node.args.kw_defaults)
        if len(defaults) < len(args):
            pad = [None] * (len(args) - len(defaults))
            defaults = pad + defaults
        for arg, default in zip(args, defaults, strict=False):
            if default is None or arg.arg not in _SEMANTIC_PARAM_NAMES:
                continue
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                if default.value.strip():
                    findings.append(
                        SemanticProductionFinding(
                            repo_path=rel,
                            finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                            detail=(f"{node.name} parameter {arg.arg!r} defaults to string literal {default.value!r}"),
                            concern_id=_primary_concern_for_path(rel),
                        ),
                    )
    return findings


def _scan_file_semantic_findings(rel: str) -> tuple[SemanticProductionFinding, ...]:
    path = repo_root() / rel
    if not path.is_file():
        return ()
    context = _classify_path_context(rel)
    if context == SemanticProductionFindingClass.REFERENCE_LAB_TEST_ONLY:
        return ()

    text = path.read_text(encoding="utf-8")
    findings: list[SemanticProductionFinding] = []
    concern_id = _primary_concern_for_path(rel)
    is_config_surface = _is_configuration_semantics_surface(rel) or rel in _chokepoint_paths()

    for marker, detail in _FORBIDDEN_TEXT_MARKERS:
        if marker in text:
            findings.append(
                SemanticProductionFinding(
                    repo_path=rel,
                    finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                    detail=detail,
                    concern_id=concern_id,
                ),
            )

    if is_config_surface and rel not in _LOCALHOST_SANCTIONED_REL_PATHS:
        for marker, detail in _AMBIENT_LOCALHOST_URL_MARKERS:
            if marker in text and "get(" in text and "localhost" in text:
                findings.append(
                    SemanticProductionFinding(
                        repo_path=rel,
                        finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                        detail=detail,
                        concern_id=concern_id,
                    ),
                )

    try:
        tree = ast.parse(text)
    except SyntaxError:
        if findings:
            return tuple(findings)
        if is_config_surface:
            return (
                SemanticProductionFinding(
                    repo_path=rel,
                    finding_class=SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
                    detail="configuration surface — syntax parse skipped",
                    concern_id=concern_id,
                ),
            )
        return ()

    module_bindings = _module_level_string_constants(tree)

    if is_config_surface:
        for item in _class_assign_defaults_on_semantic_fields(tree, rel, bindings=module_bindings):
            findings.append(item)
        for item in _from_env_semantic_constant_fallbacks(tree, rel, bindings=module_bindings):
            findings.append(item)

    for item in _literal_default_on_semantic_param(tree, rel):
        if context in (
            SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT,
            SemanticProductionFindingClass.DEPLOYMENT_ENVIRONMENT_CONSTANT,
            SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
        ):
            continue
        findings.append(item)

    findings = [_normalize_production_finding(f) for f in findings]

    blocking = [
        f
        for f in findings
        if f.finding_class
        in (
            SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
            SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT,
        )
    ]
    if blocking:
        return tuple(blocking)

    if is_config_surface:
        evidence = [
            f for f in findings if f.finding_class is SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
        ]
        evidence.append(
            SemanticProductionFinding(
                repo_path=rel,
                finding_class=SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
                detail="provider/configuration surface mechanically inspected",
                concern_id=concern_id,
            ),
        )
        return tuple(evidence)

    if rel in discover_inventory_provider_surface_paths():
        return (
            SemanticProductionFinding(
                repo_path=rel,
                finding_class=SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
                detail=(
                    "provider implementation surface — configuration activation owned by "
                    "typed config / selection chokepoint"
                ),
                concern_id=concern_id,
            ),
        )

    if context is not None:
        return (
            SemanticProductionFinding(
                repo_path=rel,
                finding_class=context,
                detail="classified by path context",
                concern_id=concern_id,
            ),
        )

    return ()


@lru_cache(maxsize=1)
def discover_semantic_production_findings() -> tuple[SemanticProductionFinding, ...]:
    surfaces = _scan_paths_for_inventory_surfaces()
    all_findings: list[SemanticProductionFinding] = []
    for rel in sorted(surfaces):
        if _is_excluded_path(rel):
            continue
        all_findings.extend(_scan_file_semantic_findings(rel))
    for rel in sorted(_lab_reference_provider_surface_paths()):
        all_findings.append(
            SemanticProductionFinding(
                repo_path=rel,
                finding_class=SemanticProductionFindingClass.REFERENCE_LAB_TEST_ONLY,
                detail="lab application host wiring — not production activation surface",
                concern_id=_primary_concern_for_path(rel),
            ),
        )
    return tuple(all_findings)


@lru_cache(maxsize=1)
def discover_classified_provider_surface_paths() -> frozenset[str]:
    return frozenset(finding.repo_path for finding in discover_semantic_production_findings())


@lru_cache(maxsize=1)
def discover_unclassified_provider_surface_paths() -> frozenset[str]:
    inventory = discover_inventory_provider_surface_paths()
    classified = discover_classified_provider_surface_paths()
    lab_reference = _lab_reference_provider_surface_paths()
    return frozenset(path for path in inventory if path not in classified and path not in lab_reference)


@lru_cache(maxsize=1)
def discover_semantic_i_blocker_paths() -> frozenset[str]:
    paths: set[str] = set()
    for finding in discover_semantic_production_findings():
        if finding.finding_class in (
            SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
            SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT,
        ):
            paths.add(finding.repo_path)
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_named_constant_semantic_blind_spot_paths() -> frozenset[str]:
    paths: set[str] = set()
    for finding in discover_semantic_production_findings():
        if finding.finding_class is SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT:
            paths.add(finding.repo_path)
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_sanctioned_vector_store_localhost_transport_paths() -> frozenset[str]:
    paths: set[str] = set()
    for finding in discover_semantic_production_findings():
        if (
            finding.finding_class is SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
            and finding.repo_path.replace("\\", "/") in _SANCTIONED_INTEGRATION_TRANSPORT_DEFAULT_SURFACES
            and "integration transport default" in finding.detail
        ):
            paths.add(finding.repo_path.replace("\\", "/"))
    return frozenset(paths)


def frz_cfg_05_semantic_i_blocker_count() -> int:
    return len(discover_semantic_i_blocker_paths())


def frz_cfg_05_unclassified_provider_surface_count() -> int:
    return len(discover_unclassified_provider_surface_paths())


def frz_cfg_05_named_constant_blind_spot_count() -> int:
    return len(discover_named_constant_semantic_blind_spot_paths())


# --- FC3 vendor-default discovery (T-IGX-043-F / FRZ-CFG-08 scoped extension) ---

FC3_SCOPED_PHYSICAL_PATHS: Final[tuple[str, ...]] = (
    "intergrax/rag/embedding/registry/provider_authority.py",
    "intergrax/rag/embedding/registry/profile.py",
    "intergrax/rag/embedding/runtime/resolver.py",
    "intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py",
    "intergrax/runtime/notifications/backend_contract.py",
    "intergrax/runtime/interactions/verification/factory.py",
    "intergrax/integrations/providers/llm_guardrail/_factory.py",
    "intergrax/model_inference/adapters/triton_vision.py",
    "intergrax/model_inference/adapters/huggingface_inference_vision.py",
)

_FC3_SEMANTIC_PARAM_NAMES: Final[frozenset[str]] = _SEMANTIC_PARAM_NAMES | frozenset(
    {"mode", "slug", "store", "channel"}
)

_FC3_KNOWN_ENUMS: dict[str, type] = {}

_LAB_SELECTOR_OPT_IN_FQ: Final[str] = "intergrax.integrations.contracts.base.LabSelectorOptIn"
_LAB_SELECTOR_OPT_IN_SIMPLE: Final[str] = "LabSelectorOptIn"
_VISION_INFERENCE_ADAPTER_FQ: Final[str] = "intergrax.model_inference.contracts.VisionInferenceAdapter"
_VISION_INFERENCE_ADAPTER_SIMPLE: Final[str] = "VisionInferenceAdapter"

_FC3_KNOWN_ENUM_MODULES: Final[dict[str, str]] = {
    "NotificationBackend": "intergrax.runtime.notifications.backend_contract",
    "InboundVerifierMode": "intergrax.runtime.interactions.verification.factory",
    "VisionProvider": "intergrax.model_inference.registry.vision_provider",
}


@dataclass
class _Fc3ScanContext:
    local_enums: dict[str, dict[str, str]] = field(default_factory=dict)
    enum_aliases: dict[str, str] = field(default_factory=dict)
    lab_imports: dict[str, str] = field(default_factory=dict)
    vision_imports: dict[str, str] = field(default_factory=dict)


class Fc3VendorDisposition(StrEnum):
    BLOCKING_VENDOR_DEFAULT = "blocking_vendor_default"
    PROVIDER_IDENTITY = "provider_identity"
    VALIDATION_ONLY = "validation_only"
    OPEN_SYNTACTIC_RESIDUAL = "open_syntactic_residual"
    EXPECTED_RESIDUAL = "expected_residual"
    LAB_REFUSED = "lab_refused"
    OUTSIDE_SCOPE_RECORD = "outside_scope_record"


@dataclass(frozen=True, slots=True)
class Fc3VendorDefaultFinding:
    repo_path: str
    enclosing_symbol: str
    semantic_field: str
    expression: str
    normalized_expression: str
    resolved_vendor_value: str
    detection_mechanism: str
    disposition: Fc3VendorDisposition
    source_line: int
    detail: str = ""

    @property
    def fingerprint(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.repo_path.replace("\\", "/"),
            self.enclosing_symbol,
            self.semantic_field,
            self.normalized_expression,
            self.resolved_vendor_value.strip().lower(),
            self.detection_mechanism,
        )


EXPECTED_RESIDUAL_FINGERPRINTS: Final[tuple[tuple[str, str, str, str, str, str], ...]] = (
    (
        "intergrax/runtime/interactions/verification/factory.py",
        "InboundVerifierSettings",
        "mode",
        "InboundVerifierMode.NONE",
        "none",
        "class_field_default:local_enum_member",
    ),
    (
        "intergrax/rag/embedding/registry/provider_authority.py",
        "lab_embedding_provider_slug",
        "_LAB_EMBEDDING_PROVIDER",
        "_LAB_EMBEDDING_PROVIDER",
        "ollama",
        "lab_named_constant",
    ),
    (
        "intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py",
        "lab_inmemory_vector_store_slug",
        "_LAB_INMEMORY_VECTOR_STORE_SLUG",
        "_LAB_INMEMORY_VECTOR_STORE_SLUG",
        "inmemory",
        "lab_named_constant",
    ),
)


def _fc3_build_scan_context(tree: ast.Module) -> _Fc3ScanContext:
    return _Fc3ScanContext(
        local_enums=_module_str_enum_members(tree),
        enum_aliases=_enum_aliases(tree),
        lab_imports=_fc3_lab_selector_import_bindings(tree),
        vision_imports=_fc3_vision_adapter_import_bindings(tree),
    )


def _fc3_lab_selector_import_bindings(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                if alias.name != "LabSelectorOptIn":
                    continue
                local = alias.asname or alias.name
                out[local] = f"{mod}.LabSelectorOptIn" if mod else _LAB_SELECTOR_OPT_IN_SIMPLE
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("LabSelectorOptIn") or alias.name == _LAB_SELECTOR_OPT_IN_FQ:
                    local = alias.asname or "LabSelectorOptIn"
                    out[local] = alias.name
    return out


def _fc3_vision_adapter_import_bindings(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                if alias.name != "VisionInferenceAdapter":
                    continue
                local = alias.asname or alias.name
                out[local] = f"{mod}.VisionInferenceAdapter" if mod else _VISION_INFERENCE_ADAPTER_SIMPLE
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("VisionInferenceAdapter"):
                    local = alias.asname or "VisionInferenceAdapter"
                    out[local] = alias.name
    return out


def _fc3_annotation_is_lab_selector(ann: ast.expr | None, ctx: _Fc3ScanContext) -> bool:
    if ann is None:
        return False
    if isinstance(ann, ast.Constant) and ann.value is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id in ctx.lab_imports or ann.id == _LAB_SELECTOR_OPT_IN_SIMPLE
    if isinstance(ann, ast.Attribute) and ann.attr == "LabSelectorOptIn":
        return True
    if isinstance(ann, ast.Subscript):
        return _fc3_annotation_is_lab_selector(ann.value, ctx) or _fc3_annotation_is_lab_selector(ann.slice, ctx)
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _fc3_annotation_is_lab_selector(ann.left, ctx) or _fc3_annotation_is_lab_selector(ann.right, ctx)
    return False


def _fc3_ancestor_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _fc3_mark_guarded_stmt_exprs(stmts: list[ast.stmt], guarded: set[int]) -> None:
    for stmt in stmts:
        for child in ast.walk(stmt):
            if isinstance(child, ast.expr):
                guarded.add(id(child))


def _fc3_is_validation_only_lab_load(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _Fc3ScanContext,
) -> bool:
    if not _lab_opt_in_param(func, ctx):
        return False
    cur: ast.AST | None = node
    while cur is not None and cur is not func:
        if isinstance(cur, ast.Compare):
            return True
        if isinstance(cur, ast.JoinedStr):
            return True
        if isinstance(cur, ast.Call):
            fn = cur.func
            if isinstance(fn, ast.Name) and fn.id.endswith("Error"):
                return True
            if isinstance(fn, ast.Attribute) and fn.attr.endswith("Error"):
                return True
        cur = parents.get(cur)
    return False


def _scan_cross_module_lab_imports(
    tree: ast.Module,
    rel: str,
    findings: list[_Fc3RawFinding],
) -> None:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.name
                if name.startswith("_LAB_"):
                    findings.append(
                        _Fc3RawFinding(
                            repo_path=rel,
                            line=_line_of(node),
                            enclosing_symbol="<import>",
                            semantic_field=name,
                            expression=name,
                            resolved_vendor_value="",
                            detection_mechanism="lab_named_constant",
                            lab_exemption="REFUSED",
                            reason="cross_module_lab_constant_import",
                        )
                    )


def _fc3_is_excluded_scan_path(rel: str) -> bool:
    return _is_excluded_path(rel)


def _fc3_bootstrap_known_enums() -> dict[str, type[StrEnum]]:
    if _FC3_KNOWN_ENUMS:
        return _FC3_KNOWN_ENUMS
    from intergrax.model_inference.registry.vision_provider import VisionProvider
    from intergrax.runtime.interactions.verification.factory import InboundVerifierMode
    from intergrax.runtime.notifications.backend_contract import NotificationBackend

    _FC3_KNOWN_ENUMS.update(
        {
            "NotificationBackend": NotificationBackend,
            "InboundVerifierMode": InboundVerifierMode,
            "VisionProvider": VisionProvider,
        }
    )
    return _FC3_KNOWN_ENUMS


def derive_fc3_vendor_values() -> tuple[frozenset[str], dict[str, str | int]]:
    """Derive vendor slug authority in a child process so catalog bootstrap cannot leak."""
    import hashlib

    _fc3_vendor_values_cached.cache_clear()
    script = """
import json
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import list_slugs
reset_default_integrations_state()
register_default_integrations(preset="full", override=True)
print(json.dumps([s.strip().lower() for s in list_slugs() if s and s.strip()]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(repo_root()),
    )
    slugs = frozenset(json.loads(proc.stdout))
    enums = _fc3_bootstrap_known_enums()
    enum_vals: set[str] = set()
    for cls in enums.values():
        for member in cls:
            enum_vals.add(member.value.strip().lower())
    vendor = frozenset(slugs | enum_vals)
    counts: dict[str, str | int] = {
        "catalog_slug_count": len(slugs),
        "enum_value_count": len(enum_vals),
        "unique_vendor_value_count": len(vendor),
    }
    counts["vendor_value_digest"] = hashlib.sha256("\n".join(sorted(vendor)).encode()).hexdigest()
    return vendor, counts


def _fc3_vision_adapter_identity_bases(tree: ast.Module, ctx: _Fc3ScanContext) -> frozenset[str]:
    targets = frozenset({_VISION_INFERENCE_ADAPTER_FQ, _VISION_INFERENCE_ADAPTER_SIMPLE, *ctx.vision_imports.values()})
    bases: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name):
                fq = ctx.vision_imports.get(base.id, base.id)
                if fq in targets:
                    bases.add(node.name)
            elif isinstance(base, ast.Attribute) and base.attr == _VISION_INFERENCE_ADAPTER_SIMPLE:
                bases.add(node.name)
    return frozenset(bases)


def _fc3_is_provider_identity_slug(
    *,
    class_name: str,
    field_name: str,
    literal_value: str,
    tree: ast.Module,
    vision_bases: frozenset[str],
) -> bool:
    if field_name != "slug" or class_name not in vision_bases:
        return False
    return literal_value.strip().lower() in _fc3_vendor_values_cached()


@lru_cache(maxsize=1)
def _fc3_vendor_values_cached() -> frozenset[str]:
    vendor, _ = derive_fc3_vendor_values()
    return vendor


def _fc3_normalize_expression(expression: str) -> str:
    return " ".join(expression.split())


def _fc3_disposition_for_finding(
    raw: Fc3VendorDefaultFinding,
    *,
    tree: ast.Module | None = None,
    field_name: str = "",
    ctx: _Fc3ScanContext | None = None,
    vision_bases: frozenset[str] | None = None,
    validation_only_hint: bool = False,
) -> Fc3VendorDisposition:
    fp = raw.fingerprint
    for expected in EXPECTED_RESIDUAL_FINGERPRINTS:
        if fp == expected:
            if expected[1] in {"lab_embedding_provider_slug", "lab_inmemory_vector_store_slug"}:
                return Fc3VendorDisposition.OPEN_SYNTACTIC_RESIDUAL
            return Fc3VendorDisposition.EXPECTED_RESIDUAL
    if raw.detection_mechanism == "lab_named_constant":
        if validation_only_hint:
            return Fc3VendorDisposition.VALIDATION_ONLY
        return Fc3VendorDisposition.LAB_REFUSED
    if (
        tree is not None
        and vision_bases is not None
        and field_name == "slug"
        and raw.detection_mechanism.startswith("class_field_default")
    ):
        if _fc3_is_provider_identity_slug(
            class_name=raw.enclosing_symbol,
            field_name="slug",
            literal_value=raw.resolved_vendor_value,
            tree=tree,
            vision_bases=vision_bases,
        ):
            return Fc3VendorDisposition.PROVIDER_IDENTITY
    if raw.repo_path.replace("\\", "/") not in FC3_SCOPED_PHYSICAL_PATHS:
        return Fc3VendorDisposition.OUTSIDE_SCOPE_RECORD
    return Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT


def discover_fc3_vendor_defaults_in_source(
    source: str,
    rel: str,
    *,
    module_sources: dict[str, str] | None = None,
) -> tuple[Fc3VendorDefaultFinding, ...]:
    vendor = _fc3_vendor_values_cached()
    tree = ast.parse(source, filename=rel)
    ctx = _fc3_build_scan_context(tree)
    vision_bases = _fc3_vision_adapter_identity_bases(tree, ctx)
    raw_findings = _fc3_scan_module_impl(rel, vendor, source=source, ctx=ctx)
    out: list[Fc3VendorDefaultFinding] = []
    for item in raw_findings:
        semantic_field = _fc3_infer_semantic_field(item)
        normalized = _fc3_normalize_expression(item.expression)
        base = Fc3VendorDefaultFinding(
            repo_path=rel.replace("\\", "/"),
            enclosing_symbol=item.enclosing_symbol,
            semantic_field=semantic_field,
            expression=item.expression,
            normalized_expression=normalized,
            resolved_vendor_value=item.resolved_vendor_value or "",
            detection_mechanism=item.detection_mechanism,
            disposition=Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
            source_line=item.line if hasattr(item, "line") else 0,
            detail=item.reason,
        )
        disposition = _fc3_disposition_for_finding(
            base,
            tree=tree,
            field_name=semantic_field,
            ctx=ctx,
            vision_bases=vision_bases,
            validation_only_hint=getattr(item, "reason", "") == "validation_only_lab_load",
        )
        out.append(
            Fc3VendorDefaultFinding(
                repo_path=base.repo_path,
                enclosing_symbol=base.enclosing_symbol,
                semantic_field=base.semantic_field,
                expression=base.expression,
                normalized_expression=base.normalized_expression,
                resolved_vendor_value=base.resolved_vendor_value,
                detection_mechanism=base.detection_mechanism,
                disposition=disposition,
                source_line=getattr(item, "line", 0),
                detail=base.detail,
            )
        )
    return tuple(
        sorted(out, key=lambda f: (f.repo_path, f.enclosing_symbol, f.semantic_field, f.expression, f.source_line))
    )


def _fc3_infer_semantic_field(item: _Fc3RawFinding | Fc3VendorDefaultFinding) -> str:
    # placeholder — patched after core rename
    return getattr(item, "semantic_field", "")


def discover_fc3_vendor_defaults_for_path(rel: str) -> tuple[Fc3VendorDefaultFinding, ...]:
    path = repo_root() / rel
    if not path.is_file():
        return (
            Fc3VendorDefaultFinding(
                repo_path=rel,
                enclosing_symbol="<missing-file>",
                semantic_field="",
                expression="",
                normalized_expression="",
                resolved_vendor_value="",
                detection_mechanism="missing_file",
                disposition=Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
                source_line=0,
                detail="expected scoped path file is missing",
            ),
        )
    return discover_fc3_vendor_defaults_in_source(path.read_text(encoding="utf-8"), rel)


def discover_fc3_vendor_defaults_inventory() -> tuple[Fc3VendorDefaultFinding, ...]:
    paths = set(_scan_paths_for_inventory_surfaces())
    paths.update(FC3_SCOPED_PHYSICAL_PATHS)
    findings: list[Fc3VendorDefaultFinding] = []
    for rel in sorted(paths):
        if _fc3_is_excluded_scan_path(rel):
            continue
        findings.extend(discover_fc3_vendor_defaults_for_path(rel))
    return tuple(findings)


def discover_fc3_scoped_blocking_candidates() -> tuple[Fc3VendorDefaultFinding, ...]:
    return tuple(
        f
        for f in discover_fc3_vendor_defaults_inventory()
        if f.repo_path in FC3_SCOPED_PHYSICAL_PATHS
        and f.disposition
        in (
            Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
            Fc3VendorDisposition.LAB_REFUSED,
            Fc3VendorDisposition.OPEN_SYNTACTIC_RESIDUAL,
        )
    )


def discover_fc3_outside_scope_vendor_findings() -> tuple[Fc3VendorDefaultFinding, ...]:
    return tuple(
        f
        for f in discover_fc3_vendor_defaults_inventory()
        if f.repo_path not in FC3_SCOPED_PHYSICAL_PATHS
        and f.disposition == Fc3VendorDisposition.OUTSIDE_SCOPE_RECORD
        and f.resolved_vendor_value
    )


def evaluate_fc3_scoped_gate_from_inventory(
    inventory: tuple[Fc3VendorDefaultFinding, ...],
) -> tuple[bool, tuple[Fc3VendorDefaultFinding, ...], tuple[Fc3VendorDefaultFinding, ...]]:
    scoped = tuple(f for f in inventory if f.repo_path in FC3_SCOPED_PHYSICAL_PATHS)
    expected_multiset = Counter(EXPECTED_RESIDUAL_FINGERPRINTS)
    residual_multiset = Counter(f.fingerprint for f in scoped if f.fingerprint in expected_multiset)
    residuals = tuple(f for f in scoped if f.fingerprint in expected_multiset)
    unexpected = tuple(
        f
        for f in scoped
        if f.fingerprint not in expected_multiset
        and f.disposition not in (Fc3VendorDisposition.PROVIDER_IDENTITY, Fc3VendorDisposition.VALIDATION_ONLY)
    )
    ok = len(unexpected) == 0 and residual_multiset == expected_multiset
    return ok, residuals, unexpected


def evaluate_fc3_scoped_gate() -> tuple[bool, tuple[Fc3VendorDefaultFinding, ...], tuple[Fc3VendorDefaultFinding, ...]]:
    return evaluate_fc3_scoped_gate_from_inventory(discover_fc3_vendor_defaults_inventory())


@dataclass(frozen=True, slots=True)
class _Fc3RawFinding:
    repo_path: str
    line: int
    enclosing_symbol: str
    expression: str
    resolved_vendor_value: str | None
    detection_mechanism: str
    lab_exemption: str
    reason: str
    semantic_field: str = ""


def _fc3_module_string_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                out[t.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                out[node.target.id] = node.value.value
    return out


def _module_all_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    names: set[str] = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
                    return names
    return set()


def _module_str_enum_members(tree: ast.Module) -> dict[str, dict[str, str]]:
    members: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases}
        if not any("StrEnum" in b for b in bases):
            continue
        body_members: dict[str, str] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if (
                        isinstance(t, ast.Name)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        body_members[t.id] = stmt.value.value
        if body_members:
            members[node.name] = body_members
    return members


def _enum_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local name -> canonical enum class name."""
    _fc3_bootstrap_known_enums()
    aliases: dict[str, str] = {}
    module_to_enum = {v: k for k, v in _FC3_KNOWN_ENUM_MODULES.items()}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            canonical_mod = node.module
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name in _FC3_KNOWN_ENUMS:
                    aliases[name] = alias.name
                elif canonical_mod in module_to_enum and alias.name == module_to_enum[canonical_mod]:
                    aliases[name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FC3_KNOWN_ENUMS:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _expr_source(expr: ast.expr, source: str) -> str:
    try:
        segment = ast.get_source_segment(source, expr)
        return segment if segment else ast.dump(expr)
    except Exception:
        return ast.dump(expr)


def _line_of(node: ast.AST) -> int:
    return getattr(node, "lineno", 0) or 0


def _resolve_expr_to_vendor(
    expr: ast.expr | None,
    bindings: dict[str, str],
    enum_aliases: dict[str, str],
    vendor: frozenset[str],
    local_enums: dict[str, dict[str, str]] | None = None,
) -> tuple[str | None, str]:
    if local_enums is None:
        local_enums = {}
    if expr is None:
        return None, "none"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        v = expr.value.strip().lower()
        return (v, "string_literal") if v in vendor else (None, "non_vendor_literal")
    if isinstance(expr, ast.Name):
        if expr.id in bindings:
            v = bindings[expr.id].strip().lower()
            return (v, "module_constant") if v in vendor else (None, "non_vendor_constant")
        return None, "unresolved_name"
    if isinstance(expr, ast.Attribute):
        base = expr.value
        if isinstance(base, ast.Name):
            cls_name = enum_aliases.get(base.id, base.id)
            member = expr.attr
            if cls_name in local_enums and member in local_enums[cls_name]:
                v = local_enums[cls_name][member].strip().lower()
                return (v, "local_enum_member") if v in vendor else (None, "enum_non_vendor")
            if base.id in enum_aliases:
                cls = _FC3_KNOWN_ENUMS[enum_aliases[base.id]]
                if member in cls.__members__:
                    v = cls.__members__[member].value.strip().lower()
                    return (v, "enum_member") if v in vendor else (None, "enum_non_vendor")
        if expr.attr == "value" and isinstance(base, ast.Attribute):
            inner = base
            if isinstance(inner.value, ast.Name):
                cls_name = enum_aliases.get(inner.value.id, inner.value.id)
                member = inner.attr
                if cls_name in local_enums and member in local_enums[cls_name]:
                    v = local_enums[cls_name][member].strip().lower()
                    return (v, "enum_member_value_attr") if v in vendor else (None, "enum_non_vendor")
                if inner.value.id in enum_aliases:
                    cls = _FC3_KNOWN_ENUMS[enum_aliases[inner.value.id]]
                    if member in cls.__members__:
                        v = cls.__members__[member].value.strip().lower()
                        return (v, "enum_member_value_attr") if v in vendor else (None, "enum_non_vendor")
    return None, "unresolved_expression"


def _lab_opt_in_param(func: ast.FunctionDef | ast.AsyncFunctionDef, ctx: _Fc3ScanContext) -> bool:
    for arg in list(func.args.args) + list(func.args.kwonlyargs) + list(func.args.posonlyargs):
        if arg.arg != "lab_opt_in":
            continue
        if _fc3_annotation_is_lab_selector(arg.annotation, ctx):
            return True
    return False


def _approved_guard_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "lab_opt_in":
        return True
    if isinstance(test, ast.Compare):
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "lab_opt_in"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = test.operand
        if (
            isinstance(inner, ast.Compare)
            and isinstance(inner.left, ast.Name)
            and inner.left.id == "lab_opt_in"
            and len(inner.ops) == 1
            and isinstance(inner.ops[0], ast.Is)
            and len(inner.comparators) == 1
            and isinstance(inner.comparators[0], ast.Constant)
            and inner.comparators[0].value is None
        ):
            return True
    return False


def _expr_in_lab_guard(
    expr: ast.expr,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _Fc3ScanContext,
) -> bool:
    if not _lab_opt_in_param(func, ctx):
        return False

    guarded: set[int] = set()

    def visit_node(node: ast.AST) -> None:
        if isinstance(node, ast.If):
            if _approved_guard_test(node.test):
                _fc3_mark_guarded_stmt_exprs(node.body, guarded)
            for stmt in node.body + node.orelse:
                visit_node(stmt)
            return
        if isinstance(node, ast.IfExp):
            if _approved_guard_test(node.test):
                _fc3_mark_guarded_stmt_exprs([ast.Expr(value=node.body)], guarded)
            visit_node(node.orelse)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) and node.values:
            if _approved_guard_test(node.values[0]):
                _fc3_mark_guarded_stmt_exprs([ast.Expr(value=node.values[-1])], guarded)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AST):
                visit_node(child)

    visit_node(func)
    for sub in ast.walk(expr):
        if id(sub) in guarded:
            return True
    return False


def _is_os_environ_get(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "get"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "environ"
    )


def _is_os_getenv(call: ast.Call) -> bool:
    f = call.func
    return isinstance(f, ast.Attribute) and f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os"


def _iter_functions(tree: ast.Module) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield (node.name, node)
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield (f"{node.name}.{stmt.name}", stmt)


def _env_vendor_fallbacks_in_expr(
    expr: ast.expr,
    func: ast.FunctionDef | ast.AsyncFunctionDef | None,
    bindings: dict[str, str],
    enum_aliases: dict[str, str],
    vendor: frozenset[str],
    rel: str,
    source: str,
    enclosing: str,
    findings: list[_Fc3RawFinding],
    ctx: _Fc3ScanContext,
) -> None:
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and (_is_os_getenv(node) or _is_os_environ_get(node)):
            default_expr: ast.expr | None = None
            if len(node.args) >= 2:
                default_expr = node.args[1]
            if default_expr is not None:
                resolved, mech = _resolve_expr_to_vendor(default_expr, bindings, enum_aliases, vendor, ctx.local_enums)
                if resolved:
                    lab = "YES" if func is not None and _expr_in_lab_guard(default_expr, func, ctx) else "NO"
                    reason = (
                        "lab_syntactic_guard"
                        if lab == "YES"
                        else "class_and_param_never_lab_exempt"
                        if False
                        else "env_fallback_vendor"
                    )
                    findings.append(
                        _Fc3RawFinding(
                            repo_path=rel,
                            line=_line_of(default_expr),
                            enclosing_symbol=enclosing,
                            expression=_expr_source(default_expr, source),
                            resolved_vendor_value=resolved,
                            detection_mechanism=f"env_get_default:{mech}",
                            lab_exemption=lab if lab == "YES" else "NO",
                            reason=reason if lab == "YES" else "environment_vendor_fallback",
                        )
                    )
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            has_env = any(isinstance(v, ast.Call) and (_is_os_getenv(v) or _is_os_environ_get(v)) for v in node.values)
            if has_env:
                for v in node.values:
                    if isinstance(v, ast.Call) and (_is_os_getenv(v) or _is_os_environ_get(v)):
                        continue
                    resolved, mech = _resolve_expr_to_vendor(v, bindings, enum_aliases, vendor, ctx.local_enums)
                    if resolved:
                        lab = "YES" if func is not None and _expr_in_lab_guard(v, func, ctx) else "NO"
                        findings.append(
                            _Fc3RawFinding(
                                repo_path=rel,
                                line=_line_of(v),
                                enclosing_symbol=enclosing,
                                expression=_expr_source(v, source),
                                resolved_vendor_value=resolved,
                                detection_mechanism=f"env_or_fallback:{mech}",
                                lab_exemption=lab,
                                reason="environment_or_vendor_fallback" if lab == "NO" else "lab_syntactic_guard",
                            )
                        )


def _scan_parameter_defaults(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    enclosing: str,
    rel: str,
    source: str,
    bindings: dict[str, str],
    enum_aliases: dict[str, str],
    vendor: frozenset[str],
    findings: list[_Fc3RawFinding],
    ctx: _Fc3ScanContext,
) -> None:
    posonly = list(func.args.posonlyargs)
    pos = list(func.args.args)
    kwonly = list(func.args.kwonlyargs)
    pos_defaults = list(func.args.defaults)
    kw_defaults = list(func.args.kw_defaults)

    # positional-only + positional-or-keyword (no shared padding with kwonly)
    pos_args = posonly + pos
    if len(pos_defaults) < len(pos_args):
        pos_pad = [None] * (len(pos_args) - len(pos_defaults))
        pos_defaults_full = pos_pad + pos_defaults
    else:
        pos_defaults_full = pos_defaults[-len(pos_args) :] if pos_args else []

    for arg, default in zip(pos_args, pos_defaults_full, strict=False):
        if default is None or arg.arg not in _FC3_SEMANTIC_PARAM_NAMES:
            continue
        resolved, mech = _resolve_expr_to_vendor(default, bindings, enum_aliases, vendor, ctx.local_enums)
        if resolved:
            findings.append(
                _Fc3RawFinding(
                    repo_path=rel,
                    line=_line_of(default),
                    enclosing_symbol=enclosing,
                    semantic_field=arg.arg,
                    expression=_expr_source(default, source),
                    resolved_vendor_value=resolved,
                    detection_mechanism=f"positional_default:{mech}",
                    lab_exemption="NO",
                    reason="parameter_defaults_never_lab_exempt",
                )
            )
        _env_vendor_fallbacks_in_expr(
            default, func, bindings, enum_aliases, vendor, rel, source, enclosing, findings, ctx
        )

    for arg, default in zip(kwonly, kw_defaults, strict=False):
        if default is None or arg.arg not in _FC3_SEMANTIC_PARAM_NAMES:
            continue
        resolved, mech = _resolve_expr_to_vendor(default, bindings, enum_aliases, vendor, ctx.local_enums)
        if resolved:
            findings.append(
                _Fc3RawFinding(
                    repo_path=rel,
                    line=_line_of(default),
                    enclosing_symbol=enclosing,
                    semantic_field=arg.arg,
                    expression=_expr_source(default, source),
                    resolved_vendor_value=resolved,
                    detection_mechanism=f"keyword_only_default:{mech}",
                    lab_exemption="NO",
                    reason="parameter_defaults_never_lab_exempt",
                )
            )
        _env_vendor_fallbacks_in_expr(
            default, func, bindings, enum_aliases, vendor, rel, source, enclosing, findings, ctx
        )


def _scan_class_field_defaults(
    tree: ast.Module,
    rel: str,
    source: str,
    bindings: dict[str, str],
    enum_aliases: dict[str, str],
    vendor: frozenset[str],
    findings: list[_Fc3RawFinding],
    ctx: _Fc3ScanContext,
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            isinstance(base, ast.Name)
            and base.id == "StrEnum"
            or (hasattr(ast, "unparse") and "StrEnum" in ast.unparse(base))
            for base in node.bases
        ):
            continue
        for stmt in node.body:
            target_name: str | None = None
            default_node: ast.expr | None = None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
                default_node = stmt.value
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        target_name = t.id
                        default_node = stmt.value
                        break
            if target_name is None or target_name not in _FC3_SEMANTIC_PARAM_NAMES or default_node is None:
                continue
            resolved, mech = _resolve_expr_to_vendor(default_node, bindings, enum_aliases, vendor, ctx.local_enums)
            if resolved:
                findings.append(
                    _Fc3RawFinding(
                        repo_path=rel,
                        line=_line_of(default_node),
                        enclosing_symbol=node.name,
                        semantic_field=target_name,
                        expression=_expr_source(default_node, source),
                        resolved_vendor_value=resolved,
                        detection_mechanism=f"class_field_default:{mech}",
                        lab_exemption="NO",
                        reason="class_field_never_lab_exempt",
                    )
                )


def _enclosing_function_map(tree: ast.Module) -> dict[ast.AST, ast.FunctionDef | ast.AsyncFunctionDef]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def climb(n: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        cur: ast.AST | None = n
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(cur)
        return None

    return {node: fn for node in ast.walk(tree) if (fn := climb(node)) is not None}


def _scan_lab_constants(
    tree: ast.Module,
    rel: str,
    source: str,
    all_exports: set[str],
    findings: list[_Fc3RawFinding],
    ctx: _Fc3ScanContext,
    parents: dict[ast.AST, ast.AST],
) -> None:
    lab_consts = {k: v for k, v in _fc3_module_string_constants(tree).items() if k.startswith("_LAB_")}
    if not lab_consts:
        return
    enc_map = _enclosing_function_map(tree)
    for name in lab_consts:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
                continue
            if name in all_exports:
                findings.append(
                    _Fc3RawFinding(
                        repo_path=rel,
                        line=_line_of(node),
                        enclosing_symbol=name,
                        semantic_field=name,
                        expression=name,
                        resolved_vendor_value=lab_consts[name].strip().lower(),
                        detection_mechanism="lab_named_constant",
                        lab_exemption="REFUSED",
                        reason=f"{name}_listed_in___all__",
                    )
                )
                continue
            parent = enc_map.get(node)
            if parent is None:
                findings.append(
                    _Fc3RawFinding(
                        repo_path=rel,
                        line=_line_of(node),
                        enclosing_symbol=name,
                        semantic_field=name,
                        expression=name,
                        resolved_vendor_value=lab_consts[name].strip().lower(),
                        detection_mechanism="lab_named_constant",
                        lab_exemption="REFUSED",
                        reason="unguarded_module_level_load",
                    )
                )
            elif _fc3_is_validation_only_lab_load(node, parents=parents, func=parent, ctx=ctx):
                findings.append(
                    _Fc3RawFinding(
                        repo_path=rel,
                        line=_line_of(node),
                        enclosing_symbol=parent.name,
                        semantic_field=name,
                        expression=_expr_source(node, source),
                        resolved_vendor_value=lab_consts[name].strip().lower(),
                        detection_mechanism="lab_named_constant",
                        lab_exemption="VALIDATION_ONLY",
                        reason="validation_only_lab_load",
                    )
                )
            elif not _expr_in_lab_guard(node, parent, ctx):
                findings.append(
                    _Fc3RawFinding(
                        repo_path=rel,
                        line=_line_of(node),
                        enclosing_symbol=parent.name,
                        semantic_field=name,
                        expression=_expr_source(node, source),
                        resolved_vendor_value=lab_consts[name].strip().lower(),
                        detection_mechanism="lab_named_constant",
                        lab_exemption="REFUSED",
                        reason="literal_syntactic_contract_violation",
                    )
                )


def _fc3_scan_module_impl(
    rel: str,
    vendor: frozenset[str],
    *,
    source: str | None = None,
    ctx: _Fc3ScanContext,
) -> list[_Fc3RawFinding]:
    if _fc3_is_excluded_scan_path(rel):
        return []
    path = repo_root() / rel
    if source is None:
        if not path.is_file():
            return []
        source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)
    bindings = _fc3_module_string_constants(tree)
    enum_aliases = ctx.enum_aliases or _enum_aliases(tree)
    all_exports = _module_all_names(tree)
    parents = _fc3_ancestor_map(tree)
    findings: list[_Fc3RawFinding] = []

    _scan_class_field_defaults(tree, rel, source, bindings, enum_aliases, vendor, findings, ctx)
    module_enclosing = "<module>"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign):
            _env_vendor_fallbacks_in_expr(
                node.value, None, bindings, enum_aliases, vendor, rel, source, module_enclosing, findings, ctx
            )
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            _env_vendor_fallbacks_in_expr(
                node.value, None, bindings, enum_aliases, vendor, rel, source, module_enclosing, findings, ctx
            )

    for enc, func in _iter_functions(tree):
        _scan_parameter_defaults(func, enc, rel, source, bindings, enum_aliases, vendor, findings, ctx)
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign):
                _env_vendor_fallbacks_in_expr(
                    stmt.value, func, bindings, enum_aliases, vendor, rel, source, enc, findings, ctx
                )
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                _env_vendor_fallbacks_in_expr(
                    stmt.value, func, bindings, enum_aliases, vendor, rel, source, enc, findings, ctx
                )
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                _env_vendor_fallbacks_in_expr(
                    stmt.value, func, bindings, enum_aliases, vendor, rel, source, enc, findings, ctx
                )

    _scan_lab_constants(tree, rel, source, all_exports, findings, ctx, parents)
    _scan_cross_module_lab_imports(tree, rel, findings)
    return findings
