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
    prefix = (vector_index_namespace or tenant_id or "default").strip()
    normalized_domain = domain.strip().lower()
    if normalized_domain in {LTM_INDEX_DOMAIN, EPISODIC_INDEX_DOMAIN}:
        return f"{prefix}:{normalized_domain}"
    return f"{prefix}:{normalized_domain}"
