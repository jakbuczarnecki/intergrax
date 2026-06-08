# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.catalog.manifests import CATALOG_TOOL_INTROSPECT
from intergrax.skills.providers.harness.manifests import HARNESS_RUN_COMPARATOR
from intergrax.skills.providers.platform.manifests import PLATFORM_SECRET_ADMIN
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SK_EXP4_SKILL_IDS = (
    "catalog.tool_introspect",
    "cloud_platform.resolver",
    "code.runner",
    "filesystem.local_io",
    "http.api_client",
    "interaction.session_handler",
    "interaction.input_capture",
    "jira.task_navigator",
    "gitlab.issue_creator",
    "ml.explain_predict",
    "openai.vector_admin",
    "browser.interactive_run",
    "cache.key_admin",
    "knowledge.confluence_navigator",
    "data.sql_mutator",
    "data.records_admin",
    "eval.observation_browser",
    "harness.run_comparator",
    "harness.run_exporter",
    "health.full_stack_probe",
    "ops.log_tail",
    "ops.incident_ack",
    "memory.semantic_search",
    "notify.batch_dispatch",
    "platform.secret_admin",
    "platform.workflow_cancel",
    "storage.object_lifecycle",
    "vector_store.purge",
    "modality.vision_segment",
    "research.web_cache_admin",
)

SK_EXP4_NEW_BUNDLES = frozenset(
    {
        "catalog",
        "cloud_platform",
        "code",
        "filesystem",
        "http",
        "interaction",
        "jira",
        "gitlab",
        "ml",
        "openai",
    }
)


@pytest.fixture(autouse=True)
def _reset_catalogs() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    register_default_skills()
    register_default_tools()
    yield
    reset_default_skills_for_tests()
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_sk_exp4_adds_thirty_skills_and_ten_bundles() -> None:
    catalog_ids = set(list_catalog_skill_ids())
    assert set(SK_EXP4_SKILL_IDS) <= catalog_ids
    assert len(SK_EXP4_SKILL_IDS) == 30
    bundle_ids = {skill_id.split(".", 1)[0] for skill_id in SK_EXP4_SKILL_IDS}
    assert SK_EXP4_NEW_BUNDLES <= bundle_ids


@pytest.mark.parametrize("skill_id", SK_EXP4_SKILL_IDS)
def test_sk_exp4_skill_registered_in_full_catalog(skill_id: str) -> None:
    registry = build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))
    assert registry.has(skill_id)


def test_catalog_tool_introspect_includes_skill_resolve() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["catalog"]))
    pack = SkillResolver(registry).resolve([CATALOG_TOOL_INTROSPECT.skill_id])
    assert "catalog.list_tools" in pack.tool_ids
    assert "skill.resolve" in pack.tool_ids


def test_harness_run_comparator_lists_compare_runs() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))
    pack = SkillResolver(registry).resolve([HARNESS_RUN_COMPARATOR.skill_id])
    assert "harness.compare_runs" in pack.tool_ids
    assert "harness.list_runs" in pack.tool_ids


def test_platform_secret_admin_is_high_risk_destructive() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["platform"]))
    pack = SkillResolver(registry).resolve([PLATFORM_SECRET_ADMIN.skill_id])
    assert "platform.delete_secret" in pack.tool_ids
    assert "platform.put_secret" in pack.tool_ids
