# © Artur Czarnecki. All rights reserved.

"""Reference ContextPlugin surface for the enterprise multi-capability package."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.registry import ContextPluginRegistry


class _ReferenceEnterpriseProvider:
    @property
    def provider_id(self) -> str:
        return "reference_enterprise.stub"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.CUSTOM})

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        _ = request, ctx
        return [
            ContextFragment(
                fragment_id="reference-enterprise-stub-1",
                source=ContextFragmentSource.CUSTOM,
                source_id="reference_enterprise",
                content="Reference enterprise context contribution",
                token_estimate=6,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
            )
        ]


class ReferenceEnterpriseContextPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "reference_enterprise.context"

    @classmethod
    def plugin_version(cls) -> str:
        return "0.1.0"

    @classmethod
    def plugin_description(cls) -> str:
        return "Reference enterprise context source (multi-capability package)."

    @classmethod
    def register(cls, registry: ContextPluginRegistry) -> None:
        registry.add_provider(_ReferenceEnterpriseProvider())
