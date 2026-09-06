# © Artur Czarnecki. All rights reserved.

"""Session semantic recall provider (CE-VEC-1)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    ContextProviderDescriptor,
    content_hash_for_text,
)
from intergrax.context.contracts import BUILTIN_PROVIDER_VERSION
from intergrax.context.provider_descriptor import build_provider_descriptor


class SessionSemanticRecallProvider:
    """Emits ``SESSION_HISTORY_SEMANTIC`` fragments when vector recall is enabled."""

    provider_id = "builtin.session_history_semantic"
    _PROVIDER_VERSION = BUILTIN_PROVIDER_VERSION

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.SESSION_HISTORY_SEMANTIC})

    @property
    def descriptor(self) -> ContextProviderDescriptor:
        return build_provider_descriptor(
            self.provider_id,
            provider_version=self._PROVIDER_VERSION,
            supported_sources=self.supported_sources,
            origin="builtin",
        )

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        if not ctx.handles.get("enable_session_vector_index"):
            return []
        query = request.objective.strip()
        if not query:
            return []
        hits: list[dict[str, object]] = list(ctx.handles.get("session_vector_hits") or [])
        fragments: list[ContextFragment] = []
        for index, hit in enumerate(hits[:8]):
            text = str(hit.get("text") or "")
            if not text:
                continue
            score = float(hit.get("score") or 0.5)
            fragments.append(
                ContextFragment(
                    fragment_id=f"session-semantic-{index}",
                    source=ContextFragmentSource.SESSION_HISTORY_SEMANTIC,
                    source_id=str(hit.get("message_id") or f"hit-{index}"),
                    content=text,
                    token_estimate=max(1, len(text) // 4),
                    relevance_score=min(1.0, max(0.0, score)),
                    freshness_score=0.6,
                    confidence_score=0.7,
                    mandatory=False,
                    metadata={"recall_mode": "vector"},
                    content_hash=content_hash_for_text(text),
                )
            )
        return fragments
