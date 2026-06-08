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

GRAPH_PATH_FINDER = SkillManifest(
    skill_id="graph.path_finder",
    version="1.0.0",
    description="Graph path exploration with node fetch and session memory.",
    tool_ids=("graph.run_query", "graph.get_node", "memory.read"),
    prompt_instruction_ids=("graph.path_finder.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("graph", "path", "finder"),
)


GRAPH_KNOWLEDGE_LINKER = SkillManifest(
    skill_id="graph.knowledge_linker",
    version="1.0.0",
    description="Link graph entities to RAG grounding and LTM facts.",
    tool_ids=("graph.run_query", "rag.retrieve", "ltm.write_fact"),
    prompt_instruction_ids=("graph.knowledge_linker.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("graph", "knowledge", "linker"),
)

