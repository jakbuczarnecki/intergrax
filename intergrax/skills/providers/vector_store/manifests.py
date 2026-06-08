# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

VECTOR_STORE_ADMIN = SkillManifest(
    skill_id="vector_store.admin",
    version="1.0.0",
    description="Vector store administration: list collections, count vectors, and health probes.",
    tool_ids=(
        "vector_store.list_collections",
        "vector_store.count",
        "vector_store.health",
    ),
    prompt_instruction_ids=("vector_store.admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("vector_store", "admin", "health"),
)

VECTOR_STORE_PURGE = SkillManifest(
    skill_id="vector_store.purge",
    version="1.0.0",
    description="Vector store purge: delete vectors with count and collection listing.",
    tool_ids=("vector_store.delete", "vector_store.count", "vector_store.list_collections"),
    prompt_instruction_ids=("vector_store.purge.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("vector_store", "purge", "delete"),
)

