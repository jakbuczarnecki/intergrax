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

RAG_INDEX_ADMIN = SkillManifest(
    skill_id="rag.index_admin",
    version="1.0.0",
    description="Vector index introspection: list collections, documents, and readiness probes.",
    tool_ids=(
        "rag.list_collections",
        "rag.describe_collection",
        "rag.check_index_status",
        "rag.list_documents",
    ),
    prompt_instruction_ids=("rag.index_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("rag", "index", "admin"),
)

RAG_COLLECTION_LIFECYCLE = SkillManifest(
    skill_id="rag.collection_lifecycle",
    version="1.0.0",
    description="Controlled index lifecycle: metadata search, document delete, and collection purge.",
    tool_ids=("rag.search_by_metadata", "rag.delete_documents", "rag.purge_collection"),
    prompt_instruction_ids=("rag.collection_lifecycle.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("rag", "lifecycle", "purge"),
)
