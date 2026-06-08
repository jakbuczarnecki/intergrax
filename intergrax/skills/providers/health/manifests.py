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

HEALTH_FULL_STACK_PROBE = SkillManifest(
    skill_id="health.full_stack_probe",
    version="1.0.0",
    description="Full-stack health probe: graph store, message bus, object storage, search provider.",
    tool_ids=("health.check_graph_store", "health.check_message_bus", "health.check_object_storage", "health.check_search_provider"),
    prompt_instruction_ids=("health.full_stack_probe.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("health", "full_stack", "probe"),
)

