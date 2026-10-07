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
TRACE_X_P5_R1_R1_AUDITED_SHA: Final[str] = "a452de39a721cd357be3ba5ecd0c3a6d41b630bd"
TRACE_X_P5_R1_R1_Q1_START_HEAD: Final[str] = TRACE_X_P5_R1_R1_AUDITED_SHA
TRACE_X_P5_R1_R1_Q2_START_HEAD: Final[str] = "538af9ef51a6ca483f607988481ef2794deb87b9"

_NEUTRAL_REVISION_REF_PATH = (
    repo_root() / "intergrax" / "contracts" / "effective_profile_revision_provenance_ref.py"
)
_CHILD_RUNNER_FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax.applications._shared.profile_resolution",
    "intergrax.applications.contracts.profile_resolution",
)
_CHILD_RUNNER_SCAN_ROOTS: Final[tuple[Path, ...]] = (
    repo_root() / "intergrax" / "runtime" / "execution",
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


def global_profile_child_inheritance_structural_chain_gaps() -> list[str]:
    """Mechanical typed-parameter chain host spec → materialization → NexusLoop → GraphExecutor."""
    checks: list[tuple[Path, str]] = [
        (
            repo_root() / "intergrax" / "runtime" / "execution" / "host_orchestration_loop_init_spec.py",
            "child_context_inheritance",
        ),
        (
            repo_root() / "intergrax" / "runtime" / "execution" / "environment_orchestration_materialization.py",
            "spec.child_context_inheritance",
        ),
        (
            repo_root() / "intergrax" / "runtime" / "nexus" / "nexus_loop.py",
            "child_context_inheritance=child_context_inheritance",
        ),
        (
            repo_root() / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
            "child_context_inheritance=child_context_inheritance",
        ),
    ]
    gaps: list[str] = []
    for path, needle in checks:
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            gaps.append(f"{path.relative_to(repo_root())}: missing {needle}")
    return gaps


def production_child_identity_mint_surface_violations() -> list[str]:
    """No extra production child minters beyond ChildExecutionRunner → default_execution_identity_authority."""
    allowed_implementations = frozenset(
        {
            "intergrax/runtime/execution/child.py",
            "intergrax/runtime/execution/identity_authority.py",
            "intergrax/contracts/execution_identity_authority.py",
        },
    )
    violations: list[str] = []
    for root in (
        repo_root() / "intergrax" / "runtime" / "execution",
        repo_root() / "intergrax" / "runtime" / "nexus",
    ):
        for path in root.rglob("*.py"):
            rel = str(path.relative_to(repo_root())).replace("\\", "/")
            if rel in allowed_implementations:
                continue
            text = path.read_text(encoding="utf-8")
            if "mint_child_execution_identity" in text:
                violations.append(rel)
    child_py = (repo_root() / "intergrax" / "runtime" / "execution" / "child.py").read_text(
        encoding="utf-8",
    )
    if "default_execution_identity_authority.mint_child_execution_identity" not in child_py:
        violations.append("child.py missing canonical default_execution_identity_authority mint path")
    return violations


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
