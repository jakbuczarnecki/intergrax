# © Artur Czarnecki. All rights reserved.

"""Vector index collection namespace resolution (MEM-VEC namespace isolation)."""

from __future__ import annotations

LTM_INDEX_DOMAIN = "ltm"
EPISODIC_INDEX_DOMAIN = "episodic"


def resolve_memory_index_collection(
    *,
    vector_index_namespace: str | None,
    tenant_id: str,
    domain: str,
) -> str:
    """
    Derive logical collection key for memory vector domains.

    Default pattern: ``{tenant_id}:ltm`` / ``{tenant_id}:episodic`` unless
    ``vector_index_namespace`` overrides the prefix.
    """
    if vector_index_namespace and vector_index_namespace.strip():
        prefix = vector_index_namespace.strip()
    elif tenant_id and str(tenant_id).strip():
        prefix = str(tenant_id).strip()
    else:
        raise ValueError(
            "tenant_id or vector_index_namespace required for memory index collection"
        )
    normalized_domain = domain.strip().lower()
    if normalized_domain in {LTM_INDEX_DOMAIN, EPISODIC_INDEX_DOMAIN}:
        return f"{prefix}:{normalized_domain}"
    return f"{prefix}:{normalized_domain}"
