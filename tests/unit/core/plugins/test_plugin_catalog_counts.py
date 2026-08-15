# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = pytest.mark.unit

MIN_FULL_INTEGRATIONS = 95
CORE_INTEGRATION_SLUGS = frozenset(
    {
        "bing",
        "google_cse",
        "inmemory",
        "log",
        "otel",
        "prometheus",
        "qdrant",
        "redis",
        "slack",
        "sqlite",
        "webhook",
    }
)
MIN_TOOL_BUNDLES = 13
MIN_SKILL_BUNDLES = 3
MIN_SKILL_IDS = 10


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()


def test_full_catalog_counts() -> None:
    bootstrap_catalogs(register_shipped=True, integration_preset="full")
    snap = snapshot_catalogs()
    assert len(snap.integration_slugs) >= MIN_FULL_INTEGRATIONS
    assert len(snap.tool_bundle_ids) >= MIN_TOOL_BUNDLES
    assert len(snap.skill_bundle_ids) >= MIN_SKILL_BUNDLES
    assert len(snap.skill_ids) >= MIN_SKILL_IDS


def test_core_integration_preset_count() -> None:
    bootstrap_catalogs(register_shipped=True, integration_preset="core")
    snap = snapshot_catalogs()
    slug_set = set(snap.integration_slugs)
    assert CORE_INTEGRATION_SLUGS <= slug_set
    assert len(snap.integration_slugs) == len(CORE_INTEGRATION_SLUGS)
    assert len(snap.integration_slugs) < MIN_FULL_INTEGRATIONS
