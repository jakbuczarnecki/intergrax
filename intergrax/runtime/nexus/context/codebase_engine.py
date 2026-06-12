# © Artur Czarnecki. All rights reserved.

"""Codebase context engine preset (CE-7.3)."""

from __future__ import annotations

from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_validator import DefaultContextValidator


class CodebaseContextRanker(DefaultContextRanker):
    """Boost workspace fragments for codebase preset."""

    ranker_id = "codebase"

    def rank(self, fragments, request):  # type: ignore[no-untyped-def]
        ranked = super().rank(fragments, request)
        boosted = []
        from intergrax.context.contracts import ContextFragment, ContextFragmentSource

        for fragment in ranked:
            if fragment.source == ContextFragmentSource.WORKSPACE:
                boosted.append(
                    ContextFragment(
                        fragment_id=fragment.fragment_id,
                        source=fragment.source,
                        source_id=fragment.source_id,
                        content=fragment.content,
                        token_estimate=fragment.token_estimate,
                        relevance_score=min(1.0, fragment.relevance_score + 0.1),
                        freshness_score=fragment.freshness_score,
                        confidence_score=fragment.confidence_score,
                        mandatory=fragment.mandatory,
                        metadata=dict(fragment.metadata),
                        content_hash=fragment.content_hash,
                    )
                )
            else:
                boosted.append(fragment)
        return sorted(boosted, key=lambda item: item.relevance_score, reverse=True)


class CodebaseContextEngine(DefaultNexusContextEngine):
    """Shipped codebase preset engine."""

    def __init__(
        self,
        *,
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
    ) -> None:
        super().__init__(
            engine_id="codebase",
            registry=registry,
            compiler=compiler,
            validator=validator,
            ranker=CodebaseContextRanker(),
        )
