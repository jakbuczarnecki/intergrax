# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.hitl.manifests import HITL_APPROVAL_GATE
from intergrax.skills.providers.rag.manifests import RAG_COLLECTION_LIFECYCLE, RAG_INDEX_ADMIN
from intergrax.skills.providers.sandbox.manifests import SANDBOX_CODE_EXEC
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SK_EXP2_SKILL_IDS = (
    "rag.index_admin",
    "rag.collection_lifecycle",
    "sandbox.code_exec",
    "hitl.approval_gate",
    "graph.entity_explorer",
    "storage.artifact_sync",
    "message_bus.async_runner",
    "cache.session_cache",
    "eval.score_logger",
    "modality.speech_io",
    "modality.vision_ocr",
    "notify.scheduled_alerts",
    "collaboration.calendar",
    "platform.secrets_flags",
    "platform.cicd_inspector",
    "data.records_query",
    "dev.issue_creator",
    "memory.session_cleanup",
)

SK_EXP2_NEW_BUNDLES = frozenset(
    {
        "sandbox",
        "hitl",
        "graph",
        "storage",
        "message_bus",
        "cache",
        "eval",
        "modality",
        "notify",
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


def test_sk_exp2_adds_eighteen_skills_and_nine_bundles() -> None:
    catalog_ids = set(list_catalog_skill_ids())
    assert set(SK_EXP2_SKILL_IDS) <= catalog_ids
    assert len(SK_EXP2_SKILL_IDS) == 18
    bundle_ids = {skill_id.split(".", 1)[0] for skill_id in SK_EXP2_SKILL_IDS}
    assert SK_EXP2_NEW_BUNDLES <= bundle_ids


@pytest.mark.parametrize("skill_id", SK_EXP2_SKILL_IDS)
def test_sk_exp2_skill_registered_in_full_catalog(skill_id: str) -> None:
    registry = build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))
    assert registry.has(skill_id)


def test_rag_index_admin_is_low_risk_read_only() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag"]))
    pack = SkillResolver(registry).resolve([RAG_INDEX_ADMIN.skill_id])
    assert "rag.purge_collection" not in pack.tool_ids
    assert "rag.list_collections" in pack.tool_ids


def test_rag_collection_lifecycle_includes_destructive_tools() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag"]))
    pack = SkillResolver(registry).resolve([RAG_COLLECTION_LIFECYCLE.skill_id])
    assert pack.tool_ids == frozenset(
        {"rag.search_by_metadata", "rag.delete_documents", "rag.purge_collection"}
    )


def test_sandbox_code_exec_includes_workspace_tools() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["sandbox"]))
    pack = SkillResolver(registry).resolve([SANDBOX_CODE_EXEC.skill_id])
    assert "sandbox.exec" in pack.tool_ids
    assert "workspace.read_file" in pack.tool_ids
    assert "workspace.write_file" in pack.tool_ids


def test_hitl_approval_gate_includes_notify() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["hitl"]))
    pack = SkillResolver(registry).resolve([HITL_APPROVAL_GATE.skill_id])
    assert "hitl.list_pending" in pack.tool_ids
    assert "notify.send" in pack.tool_ids
