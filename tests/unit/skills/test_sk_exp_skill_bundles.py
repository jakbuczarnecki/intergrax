# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.legal.manifests import LEGAL_CLAUSE_COMPARE, LEGAL_CONTRACT_REVIEW
from intergrax.skills.providers.platform.manifests import PLATFORM_CONCIERGE
from intergrax.skills.providers.rag.manifests import RAG_HYBRID_QA
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids
from intergrax.skills.registry.shipped_plugins import SHIPPED_SKILL_BUNDLE_IDS, SHIPPED_SKILL_PLUGINS
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile as build_tool_registry
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SK_EXP_SKILL_IDS = (
    "rag.hybrid_qa",
    "rag.document_ingest",
    "research.web_evidence",
    "workspace.authoring",
    "memory.task_scratchpad",
    "knowledge.wiki_navigator",
    "ops.trace_debug",
    "ops.incident_dispatch",
    "ops.security_audit",
    "ops.workflow_runner",
    "dev.issue_triage",
    "browser.research_fetch",
    "collaboration.outreach",
    "legal.clause_compare",
    "legal.case_research",
    "research.citation_synthesis",
    "data.sql_analyst",
    "platform.concierge",
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


def test_shipped_catalog_has_thirty_one_bundles_and_sixty_nine_skills() -> None:
    assert len(SHIPPED_SKILL_PLUGINS) == 31
    assert SHIPPED_SKILL_BUNDLE_IDS == frozenset(
        {
            "agent",
            "billing",
            "browser",
            "cache",
            "collaboration",
            "context",
            "cost",
            "crm",
            "data",
            "dev",
            "eval",
            "graph",
            "harness",
            "health",
            "hitl",
            "identity",
            "knowledge",
            "legal",
            "memory",
            "message_bus",
            "metrics",
            "modality",
            "notify",
            "ops",
            "platform",
            "rag",
            "research",
            "sandbox",
            "storage",
            "vector_store",
            "workspace",
        }
    )
    assert len(list_catalog_skill_ids()) == 69


@pytest.mark.parametrize("skill_id", SK_EXP_SKILL_IDS)
def test_sk_exp_skill_registered_in_full_catalog(skill_id: str) -> None:
    registry = build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))
    assert registry.has(skill_id)


def test_legal_clause_compare_expands_contract_review_dependency() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["legal"]))
    pack = SkillResolver(registry).resolve([LEGAL_CLAUSE_COMPARE.skill_id])
    assert LEGAL_CONTRACT_REVIEW.skill_id in pack.skill_ids
    assert "rag.retrieve" in pack.tool_ids
    assert "workspace.write_file" in pack.tool_ids


def test_rag_hybrid_qa_resolves_against_tool_registry() -> None:
    skill_registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag"]))
    tool_registry = build_tool_registry(
        ToolProfile(
            enabled=[
                "rag.retrieve",
                "rag.get_document",
                "memory.read",
            ]
        ),
        ctx=None,
    )
    pack = SkillResolver(skill_registry, tool_registry).resolve([RAG_HYBRID_QA.skill_id])
    assert pack.tool_ids == frozenset({"rag.retrieve", "rag.get_document", "memory.read"})


def test_platform_concierge_includes_skill_resolve() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["platform"]))
    pack = SkillResolver(registry).resolve([PLATFORM_CONCIERGE.skill_id])
    assert "skill.resolve" in pack.tool_ids
    assert "platform.concierge.system" in pack.prompt_instruction_ids
