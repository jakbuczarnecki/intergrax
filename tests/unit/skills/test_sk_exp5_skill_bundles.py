# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.legal.manifests import LEGAL_REDLINE_DRAFT
from intergrax.skills.providers.ops.manifests import OPS_ONCALL_RUNBOOK
from intergrax.skills.providers.rag.manifests import RAG_SEMANTIC_QA
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SK_EXP5_SKILL_IDS = (
    "rag.semantic_qa",
    "rag.ingest_pipeline",
    "rag.metadata_search",
    "legal.redline_draft",
    "legal.regulatory_scan",
    "legal.obligation_tracker",
    "research.deep_dive",
    "research.source_validator",
    "research.report_compiler",
    "workspace.draft_reviewer",
    "workspace.artifact_exporter",
    "memory.cross_turn_notes",
    "memory.fact_extractor",
    "ops.oncall_runbook",
    "ops.postmortem_writer",
    "ops.change_approver",
    "ops.capacity_planner",
    "dev.pr_reviewer",
    "dev.release_notes",
    "dev.sprint_planner",
    "platform.runbook_hub",
    "platform.flag_rollout",
    "platform.deploy_inspector",
    "collaboration.meeting_brief",
    "collaboration.stakeholder_ping",
    "data.pipeline_probe",
    "data.schema_documenter",
    "hitl.escalation_router",
    "hitl.decision_auditor",
    "graph.path_finder",
    "graph.knowledge_linker",
    "sandbox.test_runner",
    "sandbox.refactor_loop",
    "storage.backup_sync",
    "storage.presigned_share",
    "message_bus.retry_handler",
    "message_bus.dead_letter",
    "cache.warm_prefetch",
    "eval.baseline_runner",
    "eval.regression_guard",
    "modality.audio_transcript",
    "modality.image_analyst",
    "notify.escalation_ladder",
    "cost.chargeback_report",
    "identity.session_bootstrap",
    "health.identity_probe",
    "filesystem.stat_auditor",
    "harness.cost_analyst",
    "harness.integration_sweep",
    "agent.capability_mapper",
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


def test_sk_exp5_adds_fifty_skills() -> None:
    catalog_ids = set(list_catalog_skill_ids())
    assert set(SK_EXP5_SKILL_IDS) <= catalog_ids
    assert len(SK_EXP5_SKILL_IDS) == 50
    assert len(catalog_ids) == 153


@pytest.mark.parametrize("skill_id", SK_EXP5_SKILL_IDS)
def test_sk_exp5_skill_registered_in_full_catalog(skill_id: str) -> None:
    registry = build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))
    assert registry.has(skill_id)


def test_rag_semantic_qa_includes_memory_search() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag", "memory"]))
    pack = SkillResolver(registry).resolve([RAG_SEMANTIC_QA.skill_id])
    assert "memory.search" in pack.tool_ids
    assert "rag.retrieve" in pack.tool_ids


def test_legal_redline_draft_includes_workspace_io() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["legal", "workspace"]))
    pack = SkillResolver(registry).resolve([LEGAL_REDLINE_DRAFT.skill_id])
    assert "workspace.write_file" in pack.tool_ids
    assert "rag.retrieve" in pack.tool_ids


def test_ops_oncall_runbook_correlates_logs_and_traces() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["ops"]))
    pack = SkillResolver(registry).resolve([OPS_ONCALL_RUNBOOK.skill_id])
    assert "logs.search" in pack.tool_ids
    assert "observability.query_traces" in pack.tool_ids
