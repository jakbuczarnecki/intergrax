# © Artur Czarnecki. All rights reserved.

"""CodeCraft profile and tool wiring for Tier-3 hosts (ECC-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.codegen_llm_resolver import resolve_codegen_llm_adapter
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.tools.providers.codecraft.service import CODECRAFT_TOOL_IDS
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True)
class CodeCraftWiring:
    profile: CodeCraftProfile | None
    domain_fragments: dict[str, Any]
    wiring_extras: dict[str, Any]


def resolve_codecraft_profile(env: ApplicationEnvironmentProfile) -> CodeCraftProfile | None:
    return env.codecraft_profile


def tool_profile_with_codecraft(env: ApplicationEnvironmentProfile) -> ToolProfile:
    from intergrax.tools.providers.codecraft.bundle import CODECRAFT_BUNDLE_ID

    profile = env.tool_profile
    cc = env.codecraft_profile
    if cc is None or not cc.generation_allowed():
        return profile
    if all(profile.is_tool_enabled(tool_id) for tool_id in CODECRAFT_TOOL_IDS):
        return profile
    if profile.enabled_bundles and not profile.enabled:
        bundles = list(profile.enabled_bundles)
        if CODECRAFT_BUNDLE_ID not in bundles:
            bundles.append(CODECRAFT_BUNDLE_ID)
        return profile.model_copy(update={"enabled_bundles": bundles})
    enabled = list(profile.enabled)
    for tool_id in CODECRAFT_TOOL_IDS:
        if tool_id not in enabled:
            enabled.append(tool_id)
    return profile.model_copy(update={"enabled": enabled})


def wire_application_codecraft(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: object | None = None,
) -> CodeCraftWiring:
    cc = resolve_codecraft_profile(env)
    extras: dict[str, Any] = {}
    fragments: dict[str, Any] = {}
    if cc is not None:
        extras["codecraft_profile"] = cc
        if producer_adapter is not None:
            from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

            if isinstance(producer_adapter, LLMAdapter):
                extras["codecraft_codegen_adapter"] = resolve_codegen_llm_adapter(
                    env,
                    producer_adapter=producer_adapter,
                )
        fragments["codecraft_governance"] = {
            "mode": cc.mode,
            "isolation_tier": cc.isolation_tier,
            "require_hitl_before_exec": cc.require_hitl_before_exec,
            "network_egress": cc.network_egress,
            "max_iterations": cc.max_iterations,
            "codegen_llm_profile_ref": cc.codegen_llm_profile_ref,
        }
    return CodeCraftWiring(profile=cc, domain_fragments=fragments, wiring_extras=extras)


def apply_codecraft_to_wiring_context(
    ctx: ToolWiringContext,
    wiring: CodeCraftWiring,
) -> ToolWiringContext:
    from dataclasses import replace

    merged_extras = dict(ctx.extras)
    merged_extras.update(wiring.wiring_extras)
    if wiring.profile is not None:
        merged_extras.setdefault("codecraft_profile", wiring.profile)
    return replace(ctx, extras=merged_extras)


def lab_codecraft_profile(*, mode: str = "supervised") -> CodeCraftProfile:
    """Lab preset — supervised craft with local sandbox."""
    return CodeCraftProfile(
        mode=mode,  # type: ignore[arg-type]
        isolation_tier="local",
        require_hitl_before_exec=True,
        max_iterations=8,
    )


def regulated_codecraft_profile(*, sandbox_host_slug: str = "e2b") -> CodeCraftProfile:
    """Regulated preset — cloud isolation + security scan."""
    return CodeCraftProfile(
        mode="supervised",
        isolation_tier="cloud",
        sandbox_host_slug=sandbox_host_slug,
        require_hitl_before_exec=True,
        security_scan_before_exec=True,
        network_egress="deny",
    )
