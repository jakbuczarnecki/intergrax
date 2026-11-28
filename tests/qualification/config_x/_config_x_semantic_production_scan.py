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
from enum import Enum, StrEnum
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

_FC3_KNOWN_ENUMS: dict[str, type[Enum]] = {}

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
    parents: dict[ast.AST, ast.AST] = field(default_factory=dict)
    module_sources: dict[str, str] = field(default_factory=dict)
    scope_events: dict[ast.AST, tuple[ast.AST, ...]] = field(default_factory=dict)


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


def _fc3_bootstrap_known_enums() -> dict[str, type[Enum]]:
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
            source_line=raw.source_line,
        ):
            return Fc3VendorDisposition.PROVIDER_IDENTITY
    if raw.repo_path.replace("\\", "/") not in FC3_SCOPED_PHYSICAL_PATHS:
        return Fc3VendorDisposition.OUTSIDE_SCOPE_RECORD
    return Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT


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


_FC3_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _fc3_ancestor_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _fc3_scope_walk(node: ast.AST) -> Iterator[ast.AST]:
    """Visit one execution scope; nested scopes are leaves."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _FC3_SCOPES):
            yield from _fc3_scope_walk(child)


def _fc3_dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _fc3_dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _fc3_imports_at(node: ast.AST, ctx: _Fc3ScanContext) -> dict[str, str]:
    chain: list[ast.AST] = []
    parent = ctx.parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.Module, *_FC3_SCOPES)):
            chain.append(parent)
        parent = ctx.parents.get(parent)
    bindings: dict[str, str] = {}
    for scope in reversed(chain):
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for arg in (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs):
                bindings.pop(arg.arg, None)
        if scope not in ctx.scope_events:
            ctx.scope_events[scope] = tuple(
                item
                for item in _fc3_scope_walk(scope)
                if isinstance(item, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                or isinstance(item, ast.Name)
                and isinstance(item.ctx, (ast.Store, ast.Del))
            )
        for item in ctx.scope_events[scope]:
            if getattr(item, "lineno", 0) >= getattr(node, "lineno", 0):
                continue
            if isinstance(item, ast.Import):
                for alias in item.names:
                    bindings[alias.asname or alias.name] = alias.name
            elif isinstance(item, ast.ImportFrom) and item.module and not item.level:
                for alias in item.names:
                    bindings[alias.asname or alias.name] = f"{item.module}.{alias.name}"
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.pop(item.name, None)
            elif isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                for name in tuple(bindings):
                    if name.split(".")[0] == item.id:
                        bindings.pop(name, None)
    return bindings


def _fc3_bound_reference(node: ast.AST, ctx: _Fc3ScanContext, *, at: ast.AST | None = None) -> str:
    name = _fc3_dotted(node)
    for local, actual in sorted(_fc3_imports_at(at or node, ctx).items(), key=lambda pair: -len(pair[0])):
        if name == local:
            return actual
        if name.startswith(local + "."):
            suffix = name[len(local) :]
            # A type reference needs its actual module binding, not an arbitrary root package.
            if suffix.count(".") == 1 or actual == "os":
                return actual + suffix
    return ""


def _fc3_annotation_is_lab_selector(
    ann: ast.expr | None,
    ctx: _Fc3ScanContext,
    *,
    at: ast.AST | None = None,
) -> bool:
    return ann is not None and _fc3_bound_reference(ann, ctx, at=at) == _LAB_SELECTOR_OPT_IN_FQ


def _fc3_nullable_lab_annotation(ann: ast.expr | None, ctx: _Fc3ScanContext, at: ast.AST) -> bool:
    def none(n: ast.expr) -> bool:
        return isinstance(n, ast.Constant) and n.value is None

    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return (
            none(ann.left)
            and _fc3_annotation_is_lab_selector(ann.right, ctx, at=at)
            or none(ann.right)
            and _fc3_annotation_is_lab_selector(ann.left, ctx, at=at)
        )
    return (
        isinstance(ann, ast.Subscript)
        and _fc3_bound_reference(ann.value, ctx, at=at) == "typing.Optional"
        and _fc3_annotation_is_lab_selector(ann.slice, ctx, at=at)
    )


def _lab_opt_in_param(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _Fc3ScanContext,
    *,
    nullable: bool = True,
) -> bool:
    for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs):
        if arg.arg == "lab_opt_in":
            return (
                _fc3_nullable_lab_annotation(arg.annotation, ctx, func)
                if nullable
                else _fc3_annotation_is_lab_selector(arg.annotation, ctx, at=func)
            )
    return False


def _approved_guard_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "lab_opt_in"
    if isinstance(test, ast.Compare):
        return (
            isinstance(test.left, ast.Name)
            and test.left.id == "lab_opt_in"
            and len(test.ops) == len(test.comparators) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = test.operand
        return (
            isinstance(inner, ast.Compare)
            and isinstance(inner.left, ast.Name)
            and inner.left.id == "lab_opt_in"
            and len(inner.ops) == len(inner.comparators) == 1
            and isinstance(inner.ops[0], ast.Is)
            and isinstance(inner.comparators[0], ast.Constant)
            and inner.comparators[0].value is None
        )
    return False


def _fc3_reassigned_before(func: ast.AST, name: str, node: ast.AST) -> bool:
    limit = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
    return any(
        isinstance(n, ast.Name)
        and n.id == name
        and isinstance(n.ctx, (ast.Store, ast.Del))
        and (n.lineno, n.col_offset) <= limit
        for n in _fc3_scope_walk(func)
    )


def _expr_in_lab_guard(
    expr: ast.expr,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _Fc3ScanContext,
) -> bool:
    if not _lab_opt_in_param(func, ctx) or _fc3_reassigned_before(func, "lab_opt_in", expr):
        return False
    child: ast.AST = expr
    guarded = False
    parent = ctx.parents.get(child)
    while parent is not None and parent is not func:
        if isinstance(parent, _FC3_SCOPES):
            return False
        if isinstance(parent, ast.If) and child in parent.body:
            guarded |= _approved_guard_test(parent.test)
        elif isinstance(parent, ast.IfExp) and child is parent.body:
            guarded |= _approved_guard_test(parent.test)
        elif isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            for operand in parent.values:
                if operand is child:
                    break
                guarded |= _approved_guard_test(operand)
        child, parent = parent, ctx.parents.get(parent)
    return parent is func and guarded


def _fc3_is_validation_only_lab_load(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _Fc3ScanContext,
) -> bool:
    # Exact three-statement validation: caller selector, mismatch -> raise, caller return.
    if not _lab_opt_in_param(func, ctx, nullable=False):
        return False
    body = list(func.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 3:
        return False
    assign, branch, returned = body
    if not (isinstance(assign, ast.Assign) and len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name)):
        return False
    selector = assign.targets[0].id
    value = assign.value
    while isinstance(value, ast.Call):
        if not (
            isinstance(value.func, ast.Attribute)
            and value.func.attr in {"strip", "lower"}
            and not value.args
            and not value.keywords
        ):
            return False
        value = value.func.value
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "selector"
        and isinstance(value.value, ast.Name)
        and value.value.id == "lab_opt_in"
    ):
        return False
    if not (
        isinstance(branch, ast.If)
        and not branch.orelse
        and len(branch.body) == 1
        and isinstance(branch.body[0], ast.Raise)
    ):
        return False
    test = branch.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == len(test.comparators) == 1
        and isinstance(test.ops[0], ast.NotEq)
    ):
        return False
    left, right = test.left, test.comparators[0]
    if not (
        isinstance(left, ast.Name)
        and left.id == selector
        and isinstance(right, ast.Name)
        and right.id.startswith("_LAB_")
    ):
        return False
    if not (
        isinstance(returned, ast.Return) and isinstance(returned.value, ast.Name) and returned.value.id == selector
    ):
        return False
    raised = branch.body[0]
    if not (isinstance(raised.exc, ast.Call) and _fc3_exception_class(raised.exc.func, ctx)):
        return False
    # Error construction may interpolate values, but cannot call a provider/factory.
    if any(
        isinstance(n, ast.Call)
        for arg in (*raised.exc.args, *(k.value for k in raised.exc.keywords))
        for n in ast.walk(arg)
    ):
        return False
    current: ast.AST | None = node
    while current is not None and current is not func:
        if current is right or current is raised:
            return isinstance(node, ast.Name) and node.id == right.id
        if isinstance(current, _FC3_SCOPES):
            return False
        current = parents.get(current)
    return False


def _fc3_exception_class(node: ast.expr, ctx: _Fc3ScanContext) -> bool:
    builtin = {"ValueError", "RuntimeError", "TypeError"}
    if isinstance(node, ast.Name) and node.id in builtin:
        # Refuse a shadowed builtin constructor.
        return not any(
            isinstance(n, ast.Name) and n.id == node.id and isinstance(n.ctx, ast.Store)
            for n in ctx.parents
            if getattr(n, "lineno", 0) < node.lineno
        )
    actual = _fc3_bound_reference(node, ctx)
    return actual in {
        "intergrax.integrations.contracts.base.IntegrationConfigurationError",
        "intergrax.core.security.security_configuration.SecurityConfigurationError",
    }


def _fc3_vision_adapter_identity_bases(tree: ast.Module, ctx: _Fc3ScanContext) -> frozenset[str]:
    return frozenset(
        f"{node.name}@{node.lineno}"
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(_fc3_bound_reference(base, ctx, at=node) == _VISION_INFERENCE_ADAPTER_FQ for base in node.bases)
    )


def _fc3_is_provider_identity_slug(
    *,
    class_name: str,
    field_name: str,
    literal_value: str,
    tree: ast.Module,
    vision_bases: frozenset[str],
    source_line: int,
) -> bool:
    if field_name != "slug":
        return False
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != class_name:
            continue
        if f"{cls.name}@{cls.lineno}" not in vision_bases:
            continue
        for stmt in cls.body:
            value: ast.expr | None = None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "slug":
                value = stmt.value
            elif isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "slug" for t in stmt.targets):
                value = stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.lineno == source_line:
                return value.value.strip().lower() == literal_value and literal_value in _fc3_vendor_values_cached()
    return False


def _fc3_module_string_constants(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for stmt in tree.body:
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        targets = (
            stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name):
                bindings.pop(target.id, None)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    bindings[target.id] = value.value
    return bindings


def _module_all_names(tree: ast.Module) -> set[str]:
    return {
        elt.value
        for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets)
        and isinstance(stmt.value, (ast.List, ast.Tuple))
        for elt in stmt.value.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def _module_str_enum_members(tree: ast.Module, ctx: _Fc3ScanContext) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            _fc3_bound_reference(base, ctx, at=node) == "enum.StrEnum" for base in node.bases
        ):
            out[node.name] = _fc3_module_string_constants(ast.Module(body=node.body, type_ignores=[]))
    return out


def _enum_aliases(tree: ast.Module, ctx: _Fc3ScanContext) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            actual = _fc3_bound_reference(node, ctx)
            for enum, module in _FC3_KNOWN_ENUM_MODULES.items():
                if actual == f"{module}.{enum}":
                    aliases[_fc3_dotted(node)] = enum
    return aliases


def _fc3_build_scan_context(tree: ast.Module, module_sources: dict[str, str] | None = None) -> _Fc3ScanContext:
    ctx = _Fc3ScanContext(parents=_fc3_ancestor_map(tree), module_sources=module_sources or {})
    ctx.local_enums = _module_str_enum_members(tree, ctx)
    # Canonical enum resolution is performed at each load with its lexical bindings.
    return ctx


def _fc3_resolve_value(
    expr: ast.expr,
    constants: dict[str, str],
    ctx: _Fc3ScanContext,
    vendor: frozenset[str],
) -> tuple[str, str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        value = expr.value.strip().lower()
        return (value, "string_literal") if value in vendor else ("", "")
    if isinstance(expr, ast.Name):
        value = constants.get(expr.id, "").strip().lower()
        return (value, "module_constant") if value in vendor else ("", "")
    if isinstance(expr, ast.Attribute):
        member_expr = expr.value if expr.attr == "value" else expr
        if isinstance(member_expr, ast.Attribute):
            base = _fc3_dotted(member_expr.value)
            member = member_expr.attr
            if base in ctx.local_enums:
                value = ctx.local_enums[base].get(member, "").strip().lower()
                mech = "enum_member_value_attr" if expr.attr == "value" else "local_enum_member"
                return (value, mech) if value in vendor else ("", "")
            actual = _fc3_bound_reference(member_expr.value, ctx)
            for name, module in _FC3_KNOWN_ENUM_MODULES.items():
                if actual == f"{module}.{name}":
                    cls = _fc3_bootstrap_known_enums()[name]
                    if member in cls.__members__:
                        value = cls.__members__[member].value.strip().lower()
                        mech = "enum_member_value_attr" if expr.attr == "value" else "enum_member"
                        return (value, mech) if value in vendor else ("", "")
    return "", ""


def _fc3_enclosing(node: ast.AST, ctx: _Fc3ScanContext) -> str:
    names: list[str] = []
    parent = ctx.parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(parent.name)
        parent = ctx.parents.get(parent)
    return ".".join(reversed(names)) or "<module>"


def _fc3_enclosing_function(node: ast.AST, ctx: _Fc3ScanContext) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parent = ctx.parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.Lambda, ast.ClassDef)):
            return None
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent
        parent = ctx.parents.get(parent)
    return None


def _fc3_environment_read(expr: ast.expr, ctx: _Fc3ScanContext) -> bool:
    if isinstance(expr, ast.Call):
        if _fc3_bound_reference(expr.func, ctx) in {"os.getenv", "os.environ.get"}:
            return True
        if (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr in {"strip", "lower"}
            and not expr.args
            and not expr.keywords
        ):
            return _fc3_environment_read(expr.func.value, ctx)
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        return any(_fc3_environment_read(v, ctx) for v in expr.values)
    return False


def _fc3_emit(
    expr: ast.expr,
    field_name: str,
    mechanism: str,
    value: str,
    rel: str,
    source: str,
    ctx: _Fc3ScanContext,
    findings: list[_Fc3RawFinding],
    *,
    enclosing: str | None = None,
) -> None:
    findings.append(
        _Fc3RawFinding(
            repo_path=rel,
            line=expr.lineno,
            enclosing_symbol=enclosing or _fc3_enclosing(expr, ctx),
            semantic_field=field_name,
            expression=ast.get_source_segment(source, expr) or ast.unparse(expr),
            resolved_vendor_value=value,
            detection_mechanism=mechanism,
            lab_exemption="NO",
            reason="environment_vendor_fallback" if mechanism.startswith("env") else "class_and_param_never_lab_exempt",
        )
    )


def _env_vendor_fallbacks_in_expr(
    expr: ast.expr,
    field_name: str,
    rel: str,
    source: str,
    ctx: _Fc3ScanContext,
    constants: dict[str, str],
    vendor: frozenset[str],
    findings: list[_Fc3RawFinding],
    *,
    enclosing: str | None = None,
    allow_guard: bool = True,
) -> None:
    def emit(value_expr: ast.expr, kind: str) -> None:
        value, mech = _fc3_resolve_value(value_expr, constants, ctx, vendor)
        func = _fc3_enclosing_function(value_expr, ctx)
        if value and not (allow_guard and func is not None and _expr_in_lab_guard(value_expr, func, ctx)):
            _fc3_emit(value_expr, field_name, f"{kind}:{mech}", value, rel, source, ctx, findings, enclosing=enclosing)

    def visit(node: ast.AST) -> None:
        if isinstance(node, _FC3_SCOPES):
            return
        if isinstance(node, ast.Call) and _fc3_bound_reference(node.func, ctx) in {"os.getenv", "os.environ.get"}:
            default = (
                node.args[1]
                if len(node.args) > 1
                else next((k.value for k in node.keywords if k.arg == "default"), None)
            )
            if default is not None:
                emit(default, "env_get_default")
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            prior_environment = False
            for operand in node.values:
                if prior_environment:
                    emit(operand, "env_or_fallback")
                prior_environment |= _fc3_environment_read(operand, ctx)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expr)


def _fc3_source_for_module(module: str, ctx: _Fc3ScanContext) -> str | None:
    if module in ctx.module_sources:
        return ctx.module_sources[module]
    # Resolve source statically even when the defining module is outside the scan inventory.
    if not module or any(not part.isidentifier() for part in module.split(".")):
        return None
    for suffix in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
        path = repo_root() / suffix
        if path.is_file():
            return path.read_text(encoding="utf-8-sig")
    return None


def _scan_cross_module_lab_imports(
    tree: ast.Module,
    rel: str,
    findings: list[_Fc3RawFinding],
    ctx: _Fc3ScanContext,
) -> None:
    def emit(module: str, name: str, node: ast.AST, expression: str) -> None:
        source = _fc3_source_for_module(module, ctx)
        value = ""
        if source is not None:
            try:
                value = _fc3_module_string_constants(ast.parse(source)).get(name, "").strip().lower()
            except SyntaxError:
                pass
        findings.append(
            _Fc3RawFinding(
                repo_path=rel,
                line=getattr(node, "lineno", 0),
                enclosing_symbol=_fc3_enclosing(node, ctx),
                semantic_field=name,
                expression=expression,
                resolved_vendor_value=value,
                detection_mechanism="lab_named_constant",
                lab_exemption="REFUSED",
                reason="cross_module_lab_constant_import:" + ("resolved" if value else "unresolved"),
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = rel.replace("\\", "/").removesuffix(".py").split("/")[:-1]
                module = ".".join(package[: len(package) - node.level + 1] + ([module] if module else []))
            for alias in node.names:
                if alias.name.startswith("_LAB_"):
                    emit(module, alias.name, node, alias.asname or alias.name)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_LAB_"):
            module = _fc3_bound_reference(node.value, ctx)
            if module:
                emit(module, node.attr, node, ast.unparse(node))


def _scan_lab_constants(
    tree: ast.Module,
    rel: str,
    source: str,
    ctx: _Fc3ScanContext,
    findings: list[_Fc3RawFinding],
) -> None:
    exports = _module_all_names(tree)
    constants = _fc3_module_string_constants(tree)
    for name, value in constants.items():
        if not name.startswith("_LAB_"):
            continue
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        if name in exports:
            declaration = next(
                n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
            )
            findings.append(
                _Fc3RawFinding(
                    repo_path=rel,
                    line=declaration.lineno,
                    enclosing_symbol=name,
                    semantic_field=name,
                    expression=name,
                    resolved_vendor_value=value.strip().lower(),
                    detection_mechanism="lab_named_constant",
                    lab_exemption="REFUSED",
                    reason=f"{name}_listed_in___all__",
                )
            )
        for node in loads:
            func = _fc3_enclosing_function(node, ctx)
            validation = func is not None and _fc3_is_validation_only_lab_load(
                node, parents=ctx.parents, func=func, ctx=ctx
            )
            if name not in exports and func is not None and _expr_in_lab_guard(node, func, ctx):
                continue
            findings.append(
                _Fc3RawFinding(
                    repo_path=rel,
                    line=node.lineno,
                    enclosing_symbol=_fc3_enclosing(node, ctx) if func else name,
                    semantic_field=name,
                    expression=name,
                    resolved_vendor_value=value.strip().lower(),
                    detection_mechanism="lab_named_constant",
                    lab_exemption="VALIDATION_ONLY" if validation else "REFUSED",
                    reason="validation_only_lab_load" if validation else "literal_syntactic_contract_violation",
                )
            )


def _fc3_scan_module_impl(
    tree: ast.Module,
    rel: str,
    source: str,
    vendor: frozenset[str],
    ctx: _Fc3ScanContext,
) -> list[_Fc3RawFinding]:
    constants = _fc3_module_string_constants(tree)
    findings: list[_Fc3RawFinding] = []
    # Each expression has exactly one lexical owner. No outer-function ast.walk scan.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*node.args.posonlyargs, *node.args.args]
            defaults: list[ast.expr | None] = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
            pairs = [(a, d, "positional_default") for a, d in zip(args, defaults, strict=True)]
            pairs += [
                (a, d, "keyword_only_default") for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
            ]
            for arg, default, kind in pairs:
                if default is None or arg.arg not in _FC3_SEMANTIC_PARAM_NAMES:
                    continue
                enclosing = ".".join(filter(None, (_fc3_enclosing(node, ctx).replace("<module>", ""), node.name)))
                value, mech = _fc3_resolve_value(default, constants, ctx, vendor)
                if value:
                    _fc3_emit(
                        default, arg.arg, f"{kind}:{mech}", value, rel, source, ctx, findings, enclosing=enclosing
                    )
                _env_vendor_fallbacks_in_expr(
                    default,
                    arg.arg,
                    rel,
                    source,
                    ctx,
                    constants,
                    vendor,
                    findings,
                    enclosing=enclosing,
                    allow_guard=False,
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_expr = node.value
            if value_expr is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            field_name = next((t.id for t in targets if isinstance(t, ast.Name)), "")
            parent = ctx.parents.get(node)
            if (
                isinstance(parent, ast.ClassDef)
                and field_name in _FC3_SEMANTIC_PARAM_NAMES
                and parent.name not in ctx.local_enums
            ):
                value, mech = _fc3_resolve_value(value_expr, constants, ctx, vendor)
                if value:
                    _fc3_emit(value_expr, field_name, f"class_field_default:{mech}", value, rel, source, ctx, findings)
            _env_vendor_fallbacks_in_expr(
                value_expr,
                field_name,
                rel,
                source,
                ctx,
                constants,
                vendor,
                findings,
                allow_guard=not isinstance(parent, ast.ClassDef),
            )
        elif isinstance(node, ast.Return) and node.value is not None:
            _env_vendor_fallbacks_in_expr(
                node.value, f"<return:{node.lineno}>", rel, source, ctx, constants, vendor, findings
            )
    _scan_lab_constants(tree, rel, source, ctx, findings)
    _scan_cross_module_lab_imports(tree, rel, findings, ctx)
    return findings


def discover_fc3_vendor_defaults_in_source(
    source: str,
    rel: str,
    *,
    module_sources: dict[str, str] | None = None,
) -> tuple[Fc3VendorDefaultFinding, ...]:
    tree = ast.parse(source, filename=rel)
    if _is_excluded_path(rel):
        return ()
    ctx = _fc3_build_scan_context(tree, module_sources)
    bases = _fc3_vision_adapter_identity_bases(tree, ctx)
    out: list[Fc3VendorDefaultFinding] = []
    for raw in _fc3_scan_module_impl(tree, rel, source, _fc3_vendor_values_cached(), ctx):
        finding = Fc3VendorDefaultFinding(
            repo_path=rel.replace("\\", "/"),
            enclosing_symbol=raw.enclosing_symbol,
            semantic_field=raw.semantic_field,
            expression=raw.expression,
            normalized_expression=_fc3_normalize_expression(raw.expression),
            resolved_vendor_value=raw.resolved_vendor_value or "",
            detection_mechanism=raw.detection_mechanism,
            disposition=Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
            source_line=raw.line,
            detail=raw.reason,
        )
        disposition = _fc3_disposition_for_finding(
            finding,
            tree=tree,
            field_name=raw.semantic_field,
            ctx=ctx,
            vision_bases=bases,
            validation_only_hint=raw.reason == "validation_only_lab_load",
        )
        out.append(
            Fc3VendorDefaultFinding(
                repo_path=finding.repo_path,
                enclosing_symbol=finding.enclosing_symbol,
                semantic_field=finding.semantic_field,
                expression=finding.expression,
                normalized_expression=finding.normalized_expression,
                resolved_vendor_value=finding.resolved_vendor_value,
                detection_mechanism=finding.detection_mechanism,
                disposition=disposition,
                source_line=finding.source_line,
                detail=finding.detail,
            )
        )
    return tuple(
        sorted(out, key=lambda f: (f.repo_path, f.enclosing_symbol, f.semantic_field, f.expression, f.source_line))
    )


def discover_fc3_vendor_defaults_for_path(
    rel: str,
    *,
    module_sources: dict[str, str] | None = None,
) -> tuple[Fc3VendorDefaultFinding, ...]:
    path = repo_root() / rel
    mechanism = "missing_file"
    try:
        if path.is_file():
            return discover_fc3_vendor_defaults_in_source(
                path.read_text(encoding="utf-8-sig"), rel, module_sources=module_sources
            )
    except (SyntaxError, UnicodeError, OSError):
        mechanism = "invalid_source"
    return (
        Fc3VendorDefaultFinding(
            repo_path=rel,
            enclosing_symbol=f"<{mechanism}>",
            semantic_field="",
            expression="",
            normalized_expression="",
            resolved_vendor_value="",
            detection_mechanism=mechanism,
            disposition=Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
            source_line=0,
            detail="scoped source unavailable or invalid; fail closed",
        ),
    )


def discover_fc3_vendor_defaults_inventory() -> tuple[Fc3VendorDefaultFinding, ...]:
    paths = set(_scan_paths_for_inventory_surfaces()) | set(FC3_SCOPED_PHYSICAL_PATHS)
    sources: dict[str, str] = {}
    for rel in sorted(paths):
        path = repo_root() / rel
        if path.is_file():
            module = rel.replace("/", ".").removesuffix(".py").removesuffix(".__init__")
            sources[module] = path.read_text(encoding="utf-8-sig")
    return tuple(
        finding
        for rel in sorted(paths)
        if not _is_excluded_path(rel)
        for finding in discover_fc3_vendor_defaults_for_path(rel, module_sources=sources)
    )


def discover_fc3_scoped_blocking_candidates() -> tuple[Fc3VendorDefaultFinding, ...]:
    return tuple(
        f
        for f in discover_fc3_vendor_defaults_inventory()
        if f.repo_path in FC3_SCOPED_PHYSICAL_PATHS
        and f.disposition
        in {
            Fc3VendorDisposition.BLOCKING_VENDOR_DEFAULT,
            Fc3VendorDisposition.LAB_REFUSED,
            Fc3VendorDisposition.OPEN_SYNTACTIC_RESIDUAL,
        }
    )


def discover_fc3_outside_scope_vendor_findings() -> tuple[Fc3VendorDefaultFinding, ...]:
    return tuple(f for f in discover_fc3_vendor_defaults_inventory() if f.repo_path not in FC3_SCOPED_PHYSICAL_PATHS)
