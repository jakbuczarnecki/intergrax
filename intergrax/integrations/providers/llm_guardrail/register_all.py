# © Artur Czarnecki. All rights reserved.

"""Register all LLM guardrail catalog slugs (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.llm_guardrail._stub_backend import create_stub_guardrail
from intergrax.integrations.registry.plugin_register import register_from_manifest

_GUARD_SLUGS: tuple[str, ...] = (
    "llm_guard",
    "guardrails_ai",
    "nemo_guardrails",
    "openguardrails",
    "presidio",
    "llama_guard",
    "lakera",
    "azure_content_safety",
    "bedrock_guardrails",
)


def _manifest_for(slug: str) -> IntegrationManifest:
    return IntegrationManifest(
        slug=slug,
        categories=(IntegrationCategory.LLM_GUARDRAIL,),
        status=IntegrationStatus.BETA,
        env_prefix=f"INTERGRAX_{slug.upper()}",
        description=f"{slug} LLM guardrail harness adapter (M.12)",
    )


def register_llm_guardrail_integrations(*, override: bool = False) -> None:
    for slug in _GUARD_SLUGS:
        register_from_manifest(
            _manifest_for(slug),
            lambda _slug=slug: create_stub_guardrail(_slug),
            override=override,
        )
