# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.cost.manifests import COST_BUDGET_GUARDIAN
from intergrax.skills.providers.eval.manifests import EVAL_TRAJECTORY_JUDGE
from intergrax.skills.providers.rag.manifests import RAG_RETRIEVAL_TUNER
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SK_EXP3_SKILL_IDS = (
    "cost.budget_guardian",
    "identity.access_checker",
    "health.integration_probe",
    "context.token_planner",
    "memory.ltm_curator",
    "agent.roster_introspect",
    "vector_store.admin",
    "eval.trajectory_judge",
    "eval.release_compare",
    "rag.retrieval_tuner",
    "workspace.snapshot_manager",
    "message_bus.task_admin",
    "ops.workflow_admin",
    "hitl.queue_manager",
    "crm.account_lookup",
    "billing.usage_tracker",
    "metrics.run_observer",
    "dev.issue_updater",
    "collaboration.thread_reply",
    "ops.findings_review",
)

SK_EXP3_NEW_BUNDLES = frozenset(
    {
        "agent",
        "billing",
        "context",
        "cost",
        "crm",
        "health",
        "identity",
        "metrics",
        "vector_store",
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


def test_sk_exp3_adds_twenty_skills_and_nine_bundles() -> None:
    catalog_ids = set(list_catalog_skill_ids())
    assert set(SK_EXP3_SKILL_IDS) <= catalog_ids
    assert len(SK_EXP3_SKILL_IDS) == 20
    bundle_ids = {skill_id.split(".", 1)[0] for skill_id in SK_EXP3_SKILL_IDS}
    assert SK_EXP3_NEW_BUNDLES <= bundle_ids


@pytest.mark.parametrize("skill_id", SK_EXP3_SKILL_IDS)
def test_sk_exp3_skill_registered_in_full_catalog(skill_id: str) -> None:
    registry = build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))
    assert registry.has(skill_id)


def test_cost_budget_guardian_resolves_quota_tools() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["cost"]))
    pack = SkillResolver(registry).resolve([COST_BUDGET_GUARDIAN.skill_id])
    assert pack.tool_ids == frozenset(
        {"cost.check_quota", "cost.get_run_budget", "cost.forecast_spend"}
    )


def test_rag_retrieval_tuner_includes_rerank() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag"]))
    pack = SkillResolver(registry).resolve([RAG_RETRIEVAL_TUNER.skill_id])
    assert "rag.rerank" in pack.tool_ids
    assert "rag.preview_retrieval" in pack.tool_ids


def test_eval_trajectory_judge_includes_judge_tool() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["eval"]))
    pack = SkillResolver(registry).resolve([EVAL_TRAJECTORY_JUDGE.skill_id])
    assert "eval.judge" in pack.tool_ids
    assert "eval.trajectory" in pack.tool_ids
