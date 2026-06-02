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
    description="Observability-oriented pack: sandbox exec for isolated diagnostics (trace-friendly).",
    tool_ids=("sandbox.exec",),
    prompt_instruction_ids=("harness.trace_read.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("harness", "observability", "trace"),
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
