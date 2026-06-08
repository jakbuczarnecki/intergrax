# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

GRAPH_ENTITY_EXPLORER = SkillManifest(
    skill_id="graph.entity_explorer",
    version="1.0.0",
    description="Knowledge graph traversal with RAG grounding for entity-centric research.",
    tool_ids=("graph.run_query", "graph.get_node", "rag.retrieve"),
    prompt_instruction_ids=("graph.entity_explorer.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("graph", "entity", "knowledge"),
)
