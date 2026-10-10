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
    },
)

_FORBIDDEN_TEXT_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ('tenant_id: str = "default"', "ambient default tenant literal"),
    ('PREFIX_TENANT_ID", "default")', "ambient default tenant env fallback"),
)


class SemanticProductionFindingClass(StrEnum):
    APPROVED_TYPED_CONFIGURATION = "approved_typed_configuration"
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


def _scan_paths_for_inventory_surfaces() -> frozenset[str]:
    """Selection chokepoints only — not naive whole-provider-tree literal bans."""
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


def _classify_path_context(rel: str) -> SemanticProductionFindingClass | None:
    if _is_excluded_path(rel):
        return SemanticProductionFindingClass.REFERENCE_LAB_TEST_ONLY
    if "/_shared/config.py" in rel or rel.endswith("integrations/_shared/config.py"):
        return SemanticProductionFindingClass.DEPLOYMENT_ENVIRONMENT_CONSTANT
    if "/contracts/" in rel or "/core/" in rel:
        return SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION
    return None


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

    for marker, detail in _FORBIDDEN_TEXT_MARKERS:
        if marker in text:
            findings.append(
                SemanticProductionFinding(
                    repo_path=rel,
                    finding_class=SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION,
                    detail=detail,
                ),
            )

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return tuple(findings)

    for item in _literal_default_on_semantic_param(tree, rel):
        if context in (
            SemanticProductionFindingClass.DEPLOYMENT_ENVIRONMENT_CONSTANT,
            SemanticProductionFindingClass.APPROVED_TYPED_CONFIGURATION,
        ):
            continue
        findings.append(item)

    return tuple(findings)


@lru_cache(maxsize=1)
def discover_semantic_production_findings() -> tuple[SemanticProductionFinding, ...]:
    surfaces = _scan_paths_for_inventory_surfaces()
    all_findings: list[SemanticProductionFinding] = []
    for rel in sorted(surfaces):
        if _is_excluded_path(rel):
            continue
        all_findings.extend(_scan_file_semantic_findings(rel))
    return tuple(all_findings)


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
