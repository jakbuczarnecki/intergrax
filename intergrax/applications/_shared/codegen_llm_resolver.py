# © Artur Czarnecki. All rights reserved.

"""Resolve separate LLM adapter for Code Craft codegen (ECC-MAINT-02)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.codecraft.codegen_adapter import CodeGenerationAdapter, TemplateCodeGenerationAdapter
from intergrax.codecraft.llm_codegen_adapter import LLMCodeGenerationAdapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile


def resolve_codegen_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: LLMAdapter,
) -> CodeGenerationAdapter:
    """
    Resolve codegen LLM with producer/codegen separation.

    Precedence:
    1. ``CodeCraftProfile.codegen_llm_profile`` when set
    2. ``codegen_llm_profile_ref`` with producer adapter (distinct trace identity)
    3. Template adapter for gate/offline hosts
    """
    cc = env.codecraft_profile
    if cc is None or not cc.generation_allowed():
        return TemplateCodeGenerationAdapter()

    separate: LLMProfile | None = cc.codegen_llm_profile
    if separate is not None:
        return LLMCodeGenerationAdapter(
            separate.create_adapter(),
            profile_ref=cc.codegen_llm_profile_ref,
        )
    if cc.codegen_llm_profile_ref:
        return LLMCodeGenerationAdapter(
            producer_adapter,
            profile_ref=cc.codegen_llm_profile_ref,
        )
    return TemplateCodeGenerationAdapter()
