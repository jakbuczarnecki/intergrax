# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

HEALTH_INTEGRATION_PROBE = SkillManifest(
    skill_id="health.integration_probe",
    version="1.0.0",
    description="Integration health probes for operators: check backends, profiles, and relational stores.",
    tool_ids=(
        "health.check_integration",
        "health.check_profile",
        "health.check_relational_store",
    ),
    prompt_instruction_ids=("health.integration_probe.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("health", "integration", "probe"),
)
