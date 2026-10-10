# © Artur Czarnecki. All rights reserved.

"""FRZ-CFG-05 — closed-world semantic production selection discovery."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
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


class SemanticProductionFindingClass(StrEnum):
    APPROVED_TYPED_CONFIGURATION = "approved_typed_configuration"
    SANCTIONED_DEPLOYMENT_DEFAULT = "sanctioned_deployment_default"
    DEPLOYMENT_ENVIRONMENT_CONSTANT = "deployment_environment_constant"
    PROTOCOL_FORMAT_CONSTANT = "protocol_format_constant"
    REFERENCE_LAB_TEST_ONLY = "reference_lab_test_only"
    HARD_CODED_SEMANTIC_PRODUCTION_SELECTION = "hard_coded_semantic_production_selection"


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
    if (
        finding.finding_class
        is not SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION
    ):
        return finding
    rel = finding.repo_path.replace("\\", "/")
    if "api.elevenlabs.io" in finding.detail:
        return SemanticProductionFinding(
            repo_path=finding.repo_path,
            finding_class=SemanticProductionFindingClass.PROTOCOL_FORMAT_CONSTANT,
            detail="vendor REST API base URL — not ambient production endpoint selection",
            concern_id=finding.concern_id,
        )
    if "/observability_backend/" in rel and (
        "localhost" in finding.detail or "base_url" in finding.detail
    ):
        return SemanticProductionFinding(
            repo_path=finding.repo_path,
            finding_class=SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT,
            detail=(
                "local observability stack reference default — explicit profile/env binding "
                "required for activation"
            ),
            concern_id=finding.concern_id,
        )
    return finding


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
) -> list[SemanticProductionFinding]:
    findings: list[SemanticProductionFinding] = []
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
                if isinstance(default_node, ast.Constant) and isinstance(default_node.value, str):
                    value = default_node.value.strip()
                    if not value:
                        continue
                    if value == "default" and target_name == "tenant_id":
                        findings.append(
                            SemanticProductionFinding(
                                repo_path=rel,
                                finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                                detail=(
                                    f"{node.name}.{target_name} defaults to ambient tenant literal "
                                    f"{default_node.value!r}"
                                ),
                                concern_id=_primary_concern_for_path(rel),
                            ),
                        )
                    elif "localhost" in value and target_name in {"base_url", "url", "host"}:
                        findings.append(
                            SemanticProductionFinding(
                                repo_path=rel,
                                finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                                detail=(
                                    f"{node.name}.{target_name} defaults to localhost URL/host "
                                    f"{default_node.value!r}"
                                ),
                                concern_id=_primary_concern_for_path(rel),
                            ),
                        )
    return findings


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
                            detail=(
                                f"{node.name} parameter {arg.arg!r} defaults to string literal "
                                f"{default.value!r}"
                            ),
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
            if marker in text and 'get(' in text and "localhost" in text:
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

    if is_config_surface:
        for item in _class_assign_defaults_on_semantic_fields(tree, rel):
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

    hard_coded = [
        f
        for f in findings
        if f.finding_class is SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION
    ]
    if hard_coded:
        return tuple(hard_coded)

    if is_config_surface:
        return (
            SemanticProductionFinding(
                repo_path=rel,
                finding_class=SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
                detail="provider/configuration surface mechanically inspected",
                concern_id=concern_id,
            ),
        )

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
    return frozenset(
        path for path in inventory if path not in classified and path not in lab_reference
    )


@lru_cache(maxsize=1)
def discover_semantic_i_blocker_paths() -> frozenset[str]:
    paths: set[str] = set()
    for finding in discover_semantic_production_findings():
        if (
            finding.finding_class
            is SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION
        ):
            paths.add(finding.repo_path)
    return frozenset(paths)


def frz_cfg_05_semantic_i_blocker_count() -> int:
    return len(discover_semantic_i_blocker_paths())


def frz_cfg_05_unclassified_provider_surface_count() -> int:
    return len(discover_unclassified_provider_surface_paths())
