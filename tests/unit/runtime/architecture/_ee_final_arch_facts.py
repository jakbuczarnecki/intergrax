# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — shared paths, allowlists, and scan helpers."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

ARCH_MODEL = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_FINAL_UNIFIED_ENTRY_PLUGINABILITY_ZERO_BYPASS_MODEL.md"
)
QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_FINAL_ARCH_UNIFIED_ENTRY_PLUGINABILITY_ZERO_BYPASS_CERTIFICATION.md"
)
P0_INVENTORY = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)
OWNERSHIP_MODEL = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_OWNERSHIP_MODEL.md"
)

INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
EXECUTION_ROOT = INTERGRAX_ROOT / "runtime" / "execution"

EXECUTION_CORE_FILE_NAMES = frozenset(
    {
        "runtime.py",
        "boundary.py",
        "host_task.py",
        "child.py",
        "orchestration.py",
        "nexus_host_execution.py",
        "strategy_router.py",
        "facade.py",
    },
)

PRODUCTION_SCAN_ROOTS = (
    INTERGRAX_ROOT / "applications",
    _REPO_ROOT / "applications",
    INTERGRAX_ROOT / "agents",
)

BYPASS_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.external_operations.admission.provider_executor",
)

BYPASS_FORBIDDEN_SYMBOL_SUBSTRINGS = ("ExecutionRuntime(",)

BYPASS_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
        "intergrax/applications/_shared/harness_host_runtime.py",
        "intergrax/applications/_shared/acp_session_host_wiring.py",
        "intergrax/applications/_shared/declarative_tool_wiring.py",
        "intergrax/applications/_shared/nexus_factory.py",
        "intergrax/runtime/kernel/step_kernel.py",
        "intergrax/runtime/nexus/execution/graph_executor.py",
        "intergrax/agents/persistence/tool_invoker_wiring.py",
        "intergrax/agents/persistence/compensation_enqueue.py",
    },
)

VENDOR_IMPORT_FRAGMENTS = (
    "openai",
    "anthropic",
    "google.cloud",
    "azure",
    "boto3",
    "boto",
    "redis",
    "psycopg",
    "postgres",
    "sqlalchemy",
    "qdrant",
    "pinecone",
    "weaviate",
    "milvus",
    "datadog",
    "prometheus_client",
    "opentelemetry",
)

VENDOR_NEUTRALITY_ALLOWLIST_PREFIXES = (
    "intergrax/integrations/",
    "intergrax/runtime/events/stores/",
    "intergrax/runtime/observability/",
)

EE_FINAL_ARCH_GATE_MODULES = (
    "tests/unit/runtime/architecture/test_ee_final_arch_execution_entry_inventory.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_zero_execution_bypass.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_scheduler_ownership.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_tool_side_effect_boundary.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_pluginability.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_persistence_abstraction.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_vendor_neutrality.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_composition_root_convergence.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_legacy_nonproduction_paths.py",
)

REUSED_FROZEN_GATES = (
    "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
    "tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py",
    "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py",
    "tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py",
    "tests/unit/runtime/architecture/test_platform_execution_unification_u4_child_execution_closure.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_security_architecture_gate.py",
    "tests/unit/runtime/architecture/test_npsc5f_final_evidence_plane_qualification.py",
    "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
    "tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py",
    "tests/unit/runtime/architecture/test_eec1_execution_engine_cross_plane_certification.py",
)

ARCH_MODEL_REQUIRED_SECTIONS = (
    "## 1. Canonical execution flow",
    "## 2. Final owner matrix",
    "## 3. Execution entry inventory",
    "## 4. Bypass inventory",
    "## 5. Tool and side-effect inventory",
    "## 6. Scheduler ownership",
    "## 7. Persistence architecture",
    "## 8. Pluginability model",
    "## 9. Provider abstraction model",
    "## 10. Accepted legacy and test-only paths",
    "## 11. Zero-bypass conclusion",
)


def iter_python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        paths.append(path)
    return paths


def collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def rel_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def p0_bypass_count() -> int:
    text = P0_INVENTORY.read_text(encoding="utf-8")
    match = re.search(r"^\| BYPASS \| (\d+) \|", text, flags=re.MULTILINE)
    assert match is not None
    return int(match.group(1))


def _skip_non_authoritative_production_path(path: Path) -> bool:
    parts = path.parts
    if "docker" in parts or "runtime-context" in parts:
        return True
    if "node_modules" in parts:
        return True
    if "tests" in parts:
        return True
    return False


def scan_forbidden_imports_in_production() -> list[str]:
    violations: list[str] = []
    for root in PRODUCTION_SCAN_ROOTS:
        for path in iter_python_files(root):
            if _skip_non_authoritative_production_path(path):
                continue
            rel = rel_posix(path)
            if rel in BYPASS_IMPORT_ALLOWLIST:
                continue
            try:
                modules = collect_import_modules(path)
            except SyntaxError:
                continue
            for module in modules:
                for prefix in BYPASS_FORBIDDEN_IMPORT_PREFIXES:
                    if module == prefix or module.startswith(prefix + "."):
                        violations.append(f"{rel}: import {module}")
            if rel.startswith("intergrax/agents/"):
                source = path.read_text(encoding="utf-8-sig")
                if "intergrax.runtime.execution.runtime" in source:
                    violations.append(f"{rel}: imports execution.runtime")
                for symbol in BYPASS_FORBIDDEN_SYMBOL_SUBSTRINGS:
                    if symbol in source:
                        violations.append(f"{rel}: contains {symbol}")
    return sorted(violations)


def scan_vendor_imports_in_execution_core() -> list[str]:
    violations: list[str] = []
    for name in EXECUTION_CORE_FILE_NAMES:
        path = EXECUTION_ROOT / name
        if not path.is_file():
            continue
        rel = rel_posix(path)
        for module in collect_import_modules(path):
            lowered = module.lower()
            for fragment in VENDOR_IMPORT_FRAGMENTS:
                if fragment in lowered:
                    violations.append(f"{rel}: {module}")
    return violations
