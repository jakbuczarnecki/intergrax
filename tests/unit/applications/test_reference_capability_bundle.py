# © Artur Czarnecki. All rights reserved.

"""Reference host capability bundle least-privilege tests (P0-SAFETY-2A)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.reference_capability_bundle import (
    harness_lab_capability_bundle,
    lab_reference_tool_profile,
)
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring, lab_skill_profile
from intergrax.applications._shared.skill_tool_profile import assert_skill_tool_requirements_for_profile
from intergrax.tools.providers.harness.bundle import HARNESS_BUNDLE_ID
from intergrax.tools.providers.rag.bundle import RAG_BUNDLE_ID
from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID
from intergrax.tools.providers.sandbox.service import SANDBOX_EXEC_TOOL_ID
from intergrax.tools.providers.speech.bundle import SPEECH_BUNDLE_ID
from intergrax.tools.providers.speech.service import SPEECH_SYNTHESIZE_TOOL_ID
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_UNRELATED_PLUGIN_TOOL_ID = "plugin.dangerous_action"
_UNRELATED_PLUGIN_BUNDLE_ID = "plugin_dangerous"


@pytest.fixture
def _bootstrapped_tool_catalog() -> None:
    register_default_tools()
    yield
    reset_default_tools_bootstrap()


@pytest.fixture
def _isolated_tool_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()

    def register_unrelated(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del registry, ctx

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=_UNRELATED_PLUGIN_BUNDLE_ID,
            tool_ids=(_UNRELATED_PLUGIN_TOOL_ID,),
            register=register_unrelated,
        )
    )
    register_default_tools()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_lab_reference_tool_profile_is_not_catalog_wide() -> None:
    profile = lab_reference_tool_profile(harness_tools=True)

    assert profile.register_all_catalog_bundles is False
    assert profile.enabled_bundles


def test_harness_tools_false_excludes_harness_optional_capabilities(
    _bootstrapped_tool_catalog: None,
) -> None:
    profile = lab_reference_tool_profile(harness_tools=False)

    assert profile.is_tool_enabled("rag.retrieve") is True
    assert profile.is_tool_enabled(SANDBOX_EXEC_TOOL_ID) is False
    assert profile.is_tool_enabled(SPEECH_SYNTHESIZE_TOOL_ID) is False
    assert SANDBOX_BUNDLE_ID not in profile.enabled_bundles
    assert SPEECH_BUNDLE_ID not in profile.enabled_bundles


def test_harness_tools_true_includes_harness_optional_capabilities(
    _bootstrapped_tool_catalog: None,
) -> None:
    profile = lab_reference_tool_profile(harness_tools=True)

    assert profile.is_tool_enabled(SANDBOX_EXEC_TOOL_ID) is True
    assert profile.is_tool_enabled(SPEECH_SYNTHESIZE_TOOL_ID) is True
    assert SANDBOX_BUNDLE_ID in profile.enabled_bundles
    assert SPEECH_BUNDLE_ID in profile.enabled_bundles


def test_unrelated_catalog_plugin_is_not_auto_granted(
    _isolated_tool_catalog: None,
) -> None:
    profile = lab_reference_tool_profile(harness_tools=True)

    assert profile.is_tool_enabled(_UNRELATED_PLUGIN_TOOL_ID) is False
    assert _UNRELATED_PLUGIN_BUNDLE_ID not in profile.enabled_bundles


def test_lab_skill_profile_satisfied_by_reference_tool_profile() -> None:
    skill_wiring = build_application_skill_wiring(lab_skill_profile())
    for harness_tools in (False, True):
        tool_profile = lab_reference_tool_profile(harness_tools=harness_tools)
        resolution = assert_skill_tool_requirements_for_profile(
            tool_profile,
            lab_skill_profile(),
            skill_registry=skill_wiring.registry,
        )
        assert resolution.is_satisfied


def test_harness_lab_capability_bundle_preserves_bundle_semantics(
    _bootstrapped_tool_catalog: None,
) -> None:
    base = harness_lab_capability_bundle(harness_tools=False)
    full = harness_lab_capability_bundle(harness_tools=True)

    assert base.tools.register_all_catalog_bundles is False
    assert full.tools.register_all_catalog_bundles is False
    assert base.tools.is_tool_enabled(SANDBOX_EXEC_TOOL_ID) is False
    assert full.tools.is_tool_enabled(SANDBOX_EXEC_TOOL_ID) is True
    assert base.tools.is_tool_enabled("harness.get_run") is True
    assert RAG_BUNDLE_ID in base.tools.enabled_bundles
    assert HARNESS_BUNDLE_ID in base.tools.enabled_bundles
