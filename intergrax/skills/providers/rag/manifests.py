# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

RAG_HYBRID_QA = SkillManifest(
    skill_id="rag.hybrid_qa",
    version="1.0.0",
    description="Hybrid Q&A over vector index with document fetch and session memory read.",
    tool_ids=("rag.retrieve", "rag.get_document", "memory.read"),
    prompt_instruction_ids=("rag.hybrid_qa.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("rag", "qa", "retrieval", "memory"),
)

RAG_DOCUMENT_INGEST = SkillManifest(
    skill_id="rag.document_ingest",
    version="1.0.0",
    description="Document parse and ingest pipeline with collection status probe.",
    tool_ids=("document.parse", "rag.ingest_document", "rag.describe_collection"),
    prompt_instruction_ids=("rag.document_ingest.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("rag", "ingest", "document"),
)
