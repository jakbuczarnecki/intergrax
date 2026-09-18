# © Artur Czarnecki. All rights reserved.

"""PLUG-04 qualification gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS,
    PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS,
    PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS,
    PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS,
)
from intergrax.core.plugins.discovery import EP_INTEGRATIONS, EP_RAG_CHUNKERS
from tests.qualification.plug_04.catalog import PLUG_04_SURFACE_MATRIX

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_PLUGINS = _REPO_ROOT / "intergrax" / "core" / "plugins"


def _module_imports_nexus(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime.nexus"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("intergrax.runtime.nexus"):
                hits.append(node.module)
    return hits


def test_core_plugins_package_does_not_import_nexus() -> None:
    violations: list[str] = []
    for path in sorted(_CORE_PLUGINS.glob("*.py")):
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {hits}")
    assert violations == []


def test_q4_rag_surfaces_at_least_d3_in_matrix() -> None:
    rag_rows = [row for row in PLUG_04_SURFACE_MATRIX if row.ep_group.startswith("intergrax.rag.")]
    assert len(rag_rows) == 3
    for row in rag_rows:
        assert row.level in {"D3", "D4"}
        assert row.typed_report == "DomainPluginLoadReport"


@pytest.mark.no_ci
def test_application_evidence_includes_integrations_without_lazy_host_rag_runtime() -> None:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )
    from lab_application.host.settings import LabApplicationSettings
    from lab_application.manifest import build_lab_manifest

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="plug04.evidence")
    env = env.model_copy(
        update={
            "context_profile": env.context_profile.model_copy(update={"enable_rag": True}),
        },
    )
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
    )
    evidence = wiring.platform_plugin_evidence

    integrations = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS)
    assert integrations is not None
    assert integrations.group == EP_INTEGRATIONS

    assert evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS) is None
    assert evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS) is None
    assert evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS) is None


@pytest.mark.no_ci
def test_application_evidence_includes_rag_after_tenant_runtime_bootstrap() -> None:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )
    from lab_application.host.settings import LabApplicationSettings
    from lab_application.manifest import build_lab_manifest
    from testing_support.builder import FakeLLMAdapter

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="plug04.evidence.tenant")
    env = env.model_copy(
        update={
            "context_profile": env.context_profile.model_copy(update={"enable_rag": True}),
        },
    )
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="plug04-tenant",
        llm_adapter=FakeLLMAdapter(),
        conformance_check=False,
    )
    evidence = wiring.platform_plugin_evidence

    chunkers = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS)
    retrievers = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS)
    rerankers = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS)
    assert chunkers is not None and chunkers.group == EP_RAG_CHUNKERS
    assert retrievers is not None
    assert rerankers is not None
