# © Artur Czarnecki. All rights reserved.

"""Platform harness skill manifests (Phase S-H.1) — no business domain logic."""

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

HARNESS_TOOL_SMOKE = SkillManifest(
    skill_id="harness.tool_smoke",
    version="1.0.0",
    description="Validate catalog tool wiring: RAG retrieve and web search query (lab smoke).",
    tool_ids=("rag.retrieve", "websearch.query"),
    prompt_instruction_ids=("harness.tool_smoke.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "tools", "smoke"),
)

HARNESS_CONTEXT_DEMO = SkillManifest(
    skill_id="harness.context_demo",
    version="1.0.0",
    description="Context assembly demo: retrieval-only pack for ContextBudgetPolicy exercises.",
    tool_ids=("rag.retrieve",),
    prompt_instruction_ids=("harness.context_demo.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "context"),
)

HARNESS_TRACE_READ = SkillManifest(
    skill_id="harness.trace_read",
    version="1.0.0",
    description="Observability-oriented pack: persisted harness run trace read and event filtering.",
    tool_ids=("harness.get_run", "harness.get_run_events", "observability.query_traces"),
    prompt_instruction_ids=("harness.trace_read.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "observability", "trace"),
)

HARNESS_MODALITY_SMOKE = SkillManifest(
    skill_id="harness.modality_smoke",
    version="1.0.0",
    description="Modality plane smoke: vision detect, ML predict, and batch predict.",
    tool_ids=("vision.detect", "ml.predict", "ml.batch_predict"),
    prompt_instruction_ids=("harness.modality_smoke.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "modality", "smoke"),
)

HARNESS_VISION_QA = SkillManifest(
    skill_id="harness.vision_qa",
    version="1.0.0",
    description="Vision QA smoke: dedicated CV detect plus RAG context retrieval.",
    tool_ids=("vision.detect", "rag.retrieve"),
    prompt_instruction_ids=("harness.vision_qa.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "vision", "rag"),
)

HARNESS_SKILL_REGISTRY = SkillManifest(
    skill_id="harness.skill_registry",
    version="1.0.0",
    description="Skill resolver smoke: single-tool pack for registry merge tests.",
    tool_ids=("rag.retrieve",),
    prompt_instruction_ids=("harness.skill_registry.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "registry", "skills"),
)

HARNESS_INTEGRATION_BRIDGE_SMOKE = SkillManifest(
    skill_id="harness.integration_bridge_smoke",
    version="1.0.0",
    description="Integration bridge smoke: provider-agnostic storage and knowledge tool paths.",
    tool_ids=("storage.get", "knowledge.search"),
    prompt_instruction_ids=("harness.integration_bridge_smoke.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "integrations", "tools"),
)

HARNESS_RELIABILITY_SMOKE = SkillManifest(
    skill_id="harness.reliability_smoke",
    version="1.0.0",
    description="Reliability exercises: idempotent-friendly read paths and observability query.",
    tool_ids=("observability.query_traces", "rag.retrieve", "security.scan", "workflow.trigger"),
    prompt_instruction_ids=("harness.reliability_smoke.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "reliability", "ops"),
)

HARNESS_POLICY_SMOKE = SkillManifest(
    skill_id="harness.policy_smoke",
    version="1.0.0",
    description="Policy and governance smoke: low-risk tools under harness policy bundle.",
    tool_ids=("rag.retrieve", "websearch.query"),
    prompt_instruction_ids=("harness.policy_smoke.system",),
    policy_fragment_id="harness.policy_smoke",
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "policy", "governance"),
)

HARNESS_STACK_DEMO = SkillManifest(
    skill_id="harness.stack_demo",
    version="1.0.0",
    description="Demonstrates requires_skills: merges tools from harness.tool_smoke before this pack.",
    tool_ids=("websearch.read_url",),
    requires_skills=("harness.tool_smoke",),
    prompt_instruction_ids=("harness.stack_demo.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "requires_skills", "demo"),
)

HARNESS_RUN_COMPARATOR = SkillManifest(
    skill_id="harness.run_comparator",
    version="1.0.0",
    description="Harness run comparison: list runs, fetch details, and compare outcomes.",
    tool_ids=("harness.list_runs", "harness.get_run", "harness.compare_runs"),
    prompt_instruction_ids=("harness.run_comparator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "runs", "compare"),
)


HARNESS_RUN_EXPORTER = SkillManifest(
    skill_id="harness.run_exporter",
    version="1.0.0",
    description="Harness run export: bundle export with events and run metadata.",
    tool_ids=("harness.export_run_bundle", "harness.get_run_events", "harness.get_run"),
    prompt_instruction_ids=("harness.run_exporter.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("harness", "runs", "export"),
)

