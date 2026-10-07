# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1 qualification support (mechanical discovery)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

TRACE_X_P5_R1_START_HEAD: Final[str] = "98c0d9d7ae9763bce931c60e19b6a91af3a2f4e9"
TRACE_X_P5_R1_AUDITED_HEAD: Final[str] = "65f7e1ef4832d99a19be7953734a42d5bd5cbb4f"

_NEUTRAL_REVISION_REF_PATH = (
    repo_root() / "intergrax" / "contracts" / "effective_profile_revision_provenance_ref.py"
)
_CHILD_RUNNER_FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax.applications._shared.profile_resolution",
    "intergrax.applications.contracts.profile_resolution",
)
_CHILD_RUNNER_SCAN_ROOTS: Final[tuple[Path, ...]] = (
    repo_root() / "intergrax" / "runtime" / "execution" / "child.py",
    repo_root() / "intergrax" / "runtime" / "nexus",
)
_PROFILE_CHILD_ADAPTER_PATH = (
    repo_root()
    / "intergrax"
    / "applications"
    / "_shared"
    / "profile_resolution"
    / "profile_resolution_child_context_inheritance_adapter.py"
)

_RECONSTRUCTION_ROOT = repo_root() / "intergrax" / "runtime" / "observability" / "reconstruction"
_FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax.applications.contracts.profile_resolution",
    "intergrax.applications._shared.profile_resolution",
)

_FALLBACK_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"get_active\s*\("),
    re.compile(r"latest_revision"),
    re.compile(r"active_revision"),
    re.compile(r"fingerprint\s*=="),
    re.compile(r"checkpoint.*revision"),
)


def reconstruction_runtime_import_violations() -> list[str]:
    violations: list[str] = []
    for path in _RECONSTRUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                    if node.module == prefix or node.module.startswith(prefix + "."):
                        violations.append(f"{path.relative_to(repo_root())}:{node.lineno}:{node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            violations.append(
                                f"{path.relative_to(repo_root())}:{node.lineno}:{alias.name}"
                            )
    return violations


def reconstruction_profile_fallback_sentinels() -> list[str]:
    hits: list[str] = []
    for path in _RECONSTRUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _FALLBACK_PATTERNS:
            for match in pattern.finditer(text):
                hits.append(f"{path.relative_to(repo_root())}:{match.group()}")
    return hits


@dataclass(frozen=True, slots=True)
class ProfileAwareHostRoot:
    path: Path
    calls_environment_host: bool
    passes_revision_admission: bool


def neutral_revision_ref_shadow_grammar_violations() -> list[str]:
    text = _NEUTRAL_REVISION_REF_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    if "effprof_rev_" in text:
        violations.append("literal effprof_rev_ prefix in neutral provenance ref contract")
    if re.search(r"\[0-9a-f\]\{32\}", text):
        violations.append("canonical revision suffix regex in neutral provenance ref contract")
    if "profile_resolution.revision_id" in text:
        violations.append("profile resolution revision_id import in neutral contract")
    return violations


def child_runner_profile_resolution_import_violations() -> list[str]:
    violations: list[str] = []
    for root in _CHILD_RUNNER_SCAN_ROOTS:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for prefix in _CHILD_RUNNER_FORBIDDEN_IMPORT_PREFIXES:
                        if node.module == prefix or node.module.startswith(prefix + "."):
                            violations.append(
                                f"{path.relative_to(repo_root())}:{node.lineno}:{node.module}"
                            )
    return violations


def profile_child_inheritance_has_production_caller() -> bool:
    text = _PROFILE_CHILD_ADAPTER_PATH.read_text(encoding="utf-8")
    return "inherit_child_execution_pinned_revision" in text


def profile_aware_orchestration_spec_wiring_gaps() -> list[str]:
    roots = (
        repo_root() / "intergrax" / "applications" / "_shared" / "harness_host_runtime.py",
        repo_root() / "intergrax" / "applications" / "_shared" / "scenario_runtime_baseline.py",
    )
    missing: list[str] = []
    for path in roots:
        text = path.read_text(encoding="utf-8")
        if "build_host_orchestration_loop_init_spec_from_environment" not in text:
            continue
        if "child_context_inheritance" not in text:
            missing.append(str(path.relative_to(repo_root())))
    return missing


def discover_profile_aware_environment_host_roots() -> list[str]:
    """Production composition roots that build environment host execution."""
    roots = (
        repo_root() / "intergrax" / "applications" / "_shared" / "harness_host_runtime.py",
        repo_root() / "intergrax" / "applications" / "_shared" / "scenario_runtime_baseline.py",
        repo_root() / "intergrax" / "runtime" / "execution" / "application_host_orchestration_composition.py",
    )
    missing: list[str] = []
    for path in roots:
        text = path.read_text(encoding="utf-8")
        if "build_environment_host_task_execution" not in text:
            continue
        if "revision_admission" not in text:
            missing.append(str(path.relative_to(repo_root())))
    return missing
