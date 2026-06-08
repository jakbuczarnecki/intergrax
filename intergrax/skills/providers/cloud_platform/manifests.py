# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CLOUD_PLATFORM_RESOLVER = SkillManifest(
    skill_id="cloud_platform.resolver",
    version="1.0.0",
    description="Cloud platform resolution: health probe, endpoint resolve, and integration check.",
    tool_ids=("cloud_platform.health", "cloud_platform.resolve", "health.check_integration"),
    prompt_instruction_ids=("cloud_platform.resolver.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("cloud_platform", "resolve", "health"),
)

