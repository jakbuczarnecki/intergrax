# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C — remaining Application contract ownership gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.semver import SemVer as CanonicalSemVer
from intergrax.contracts.task_metadata_keys import TaskMetadataKey as CanonicalTaskMetadataKey
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.applications.contracts.platform_plugin_evidence import DomainPluginLoadReportView
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.fastapi_core.config import ApiEnvironment as LegacyApiEnvironment
from intergrax.contracts.api_environment import ApiEnvironment as CanonicalApiEnvironment
from intergrax.integrations.contracts.integration_profile import (
    IntegrationProfile as CanonicalIntegrationProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile as LegacyIntegrationProfile
from intergrax.runtime.registry.semver_compat import SemVer as LegacySemVer
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey as LegacyTaskMetadataKey

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EBH_2D_C_MODULES = (
    _REPO_ROOT / "intergrax/applications/contracts/manifest.py",
    _REPO_ROOT / "intergrax/applications/contracts/application_package.py",
    _REPO_ROOT / "intergrax/applications/contracts/application_registry.py",
    _REPO_ROOT / "intergrax/applications/contracts/platform_plugin_evidence.py",
    _REPO_ROOT / "intergrax/applications/contracts/build_context.py",
    _REPO_ROOT / "intergrax/applications/contracts/graph_spec.py",
    _REPO_ROOT / "intergrax/applications/contracts/operational_ownership.py",
    _REPO_ROOT / "intergrax/applications/contracts/application_package.py",
    _REPO_ROOT / "intergrax/applications/contracts/agent_ref.py",
    _REPO_ROOT / "intergrax/applications/contracts/factory.py",
    _REPO_ROOT / "intergrax/applications/contracts/graph_builder.py",
    _REPO_ROOT / "intergrax/applications/contracts/execution_mode.py",
    _REPO_ROOT / "intergrax/applications/contracts/settings.py",
    _REPO_ROOT / "intergrax/applications/contracts/capability_alias.py",
    _REPO_ROOT / "intergrax/applications/contracts/environment_snapshot.py",
    _REPO_ROOT / "intergrax/applications/contracts/capability_dependency/provider.py",
    _REPO_ROOT / "intergrax/applications/contracts/capability_health/provider.py",
    _REPO_ROOT / "intergrax/applications/contracts/profile_resolution/delta.py",
)

_FORBIDDEN_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.applications._shared.",
)
_FORBIDDEN_SUBSTRINGS = (
    ".registry.",
)
_FORBIDDEN_EXACT = (
    "intergrax.agents.agent_contract",
    "intergrax.rag.bootstrap",
    "intergrax.fastapi_core.config",
)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.",
        ):
            hits.append(node.module)
    return hits


def test_ebh_2d_c_target_modules_forbid_runtime_shared_registry_imports() -> None:
    problems: list[str] = []
    seen: set[Path] = set()
    for path in _EBH_2D_C_MODULES:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for imported in _module_imports(path):
            if any(imported.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
                problems.append(f"{path.relative_to(_REPO_ROOT)}: {imported}")
            if any(token in f".{imported}." for token in _FORBIDDEN_SUBSTRINGS):
                if imported.startswith("intergrax.integrations.contracts."):
                    continue
                if imported.startswith("intergrax.llm_adapters.contracts."):
                    continue
                if imported.startswith("intergrax.tools.contracts."):
                    continue
                if imported.startswith("intergrax.skills.contracts."):
                    continue
                problems.append(f"{path.relative_to(_REPO_ROOT)}: {imported}")
            if imported in _FORBIDDEN_EXACT or imported.startswith(
                tuple(f"{x}." for x in _FORBIDDEN_EXACT),
            ):
                problems.append(f"{path.relative_to(_REPO_ROOT)}: {imported}")
    assert not problems, "\n".join(problems)


def test_canonical_type_identity_for_moved_application_contract_types() -> None:
    assert CanonicalSemVer is LegacySemVer
    assert CanonicalTaskMetadataKey is LegacyTaskMetadataKey
    assert CanonicalApiEnvironment is LegacyApiEnvironment
    assert isinstance(DomainPluginLoadReport.empty("memory"), DomainPluginLoadReportView)
    assert CanonicalIntegrationProfile is LegacyIntegrationProfile
    assert issubclass(Agent, Tier2Agent)


def test_platform_plugin_evidence_contract_has_no_builder() -> None:
    path = _REPO_ROOT / "intergrax/applications/contracts/platform_plugin_evidence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("build_"):
            raise AssertionError("builder functions must live in _shared composition")
