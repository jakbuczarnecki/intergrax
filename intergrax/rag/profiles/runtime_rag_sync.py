# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map legacy RuntimeConfig RAG fields onto RagProfile (Phase Q-R.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dataclasses import replace

from intergrax.rag.profiles.rag_profile import RagProfile

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig


def sync_rag_profile_from_runtime_config(
    cfg: RuntimeConfig,
    *,
    base: Optional[RagProfile] = None,
) -> RagProfile:
    profile = base or cfg.rag_profile or RagProfile()
    updates: dict = {"final_top_k": cfg.max_docs_per_query}
    if cfg.rag_score_threshold is not None:
        updates["score_threshold"] = cfg.rag_score_threshold
    if profile.prefetch_top_k < cfg.max_docs_per_query:
        updates["prefetch_top_k"] = max(profile.prefetch_top_k, cfg.max_docs_per_query)
    return replace(profile, **updates)
