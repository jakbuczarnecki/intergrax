# © Artur Czarnecki. All rights reserved.

"""PLATFORM-PLUGIN-DOCS-6 — multi-capability reference package tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.context.bootstrap import (
    bootstrap_context_catalog,
    reset_context_catalog_bootstrap_for_tests,
)
from intergrax.context.registry import clear_context_plugin_catalog
from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugins import iter_entry_point_specs, parse_platform_plugin_pyproject_toml
from intergrax.core.plugins.discovery import (
    EP_CONTEXT,
    EP_SKILLS,
    EP_TOOL_INVOCATION_PATTERNS,
    EP_TOOLS,
)
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.runtime.nexus.tools.tool_invocation_registry import load_tool_invocation_pattern
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gate,
    pytest.mark.usefixtures("reference_enterprise_plugin_installed"),
]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENTERPRISE_PKG = _REPO_ROOT / "examples" / "platform_plugins" / "intergrax_reference_enterprise_plugin"


@pytest.fixture(autouse=True)
def _reset_catalog_state() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    clear_context_plugin_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_context_catalog_bootstrap_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    clear_context_plugin_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_context_catalog_bootstrap_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()


def test_pyproject_declares_four_entry_point_groups() -> None:
    manifest = parse_platform_plugin_pyproject_toml(
        (_ENTERPRISE_PKG / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert manifest.package.name == "intergrax-reference-enterprise-plugin"
    assert len(manifest.capabilities) == 4
    groups = {item.entry_point_group for item in manifest.capabilities}
    assert groups == {
        "intergrax.tools",
        "intergrax.skills",
        "intergrax.context",
        "intergrax.tool_invocation_patterns",
    }


def test_entry_points_resolve_to_domain_contracts() -> None:
    tool_specs = {spec.name for spec in iter_entry_point_specs(EP_TOOLS)}
    skill_specs = {spec.name for spec in iter_entry_point_specs(EP_SKILLS)}
    context_specs = {spec.name for spec in iter_entry_point_specs(EP_CONTEXT)}
    pattern_specs = {spec.name for spec in iter_entry_point_specs(EP_TOOL_INVOCATION_PATTERNS)}
    assert "reference_enterprise_echo" in tool_specs
    assert "reference_enterprise_pack" in skill_specs
    assert "reference_enterprise" in context_specs
    assert "reference_enterprise_single_pass" in pattern_specs

    from intergrax_reference_enterprise_plugin.context import ReferenceEnterpriseContextPlugin
    from intergrax_reference_enterprise_plugin.invocation_pattern import (
        ReferenceEnterpriseSinglePassPattern,
    )
    from intergrax_reference_enterprise_plugin.skill import ReferenceEnterprisePackSkillPlugin
    from intergrax_reference_enterprise_plugin.tool import ReferenceEnterpriseEchoToolPlugin

    assert ReferenceEnterpriseEchoToolPlugin.tool_bundle_manifest().bundle_id == "reference_enterprise_echo"
    assert ReferenceEnterprisePackSkillPlugin.skill_bundle_manifest().bundle_id == "reference_enterprise_pack"
    assert ReferenceEnterpriseContextPlugin.plugin_id() == "reference_enterprise.context"
    assert ReferenceEnterpriseSinglePassPattern().pattern_id == "reference_enterprise_single_pass"


def test_bootstrap_discovers_tool_and_skill_catalog_entries() -> None:
    result = bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    assert result.tool_plugins >= 1
    assert result.skill_plugins >= 1
    from intergrax.core.catalog_snapshot import snapshot_catalogs

    snap = snapshot_catalogs()
    assert "reference_enterprise_echo" in snap.tool_bundle_ids
    assert "reference_enterprise_pack" in snap.skill_bundle_ids


def test_context_catalog_discovers_reference_enterprise_plugin() -> None:
    result = bootstrap_context_catalog(register_shipped=False, discover_entry_points=True)
    assert "reference_enterprise.context" in result.catalog_plugin_ids


def test_tool_invocation_pattern_loads_offline() -> None:
    pattern = load_tool_invocation_pattern("reference_enterprise_single_pass")
    assert pattern is not None
    assert pattern.pattern_id == "reference_enterprise_single_pass"
    result = pattern.execute(
        state=object(),  # type: ignore[arg-type]
        invoker=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        plan=None,
        allowed_tool_ids=None,
        max_iterations=1,
        planner_input="offline",
    )
    assert result.stop_reason == "empty_tool_calls"
