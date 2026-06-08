# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

IDENTITY_ACCESS_CHECKER = SkillManifest(
    skill_id="identity.access_checker",
    version="1.0.0",
    description="Identity and tenancy checks: verify tokens, resolve users, and list tenants.",
    tool_ids=("identity.verify_token", "identity.get_user", "identity.list_tenants"),
    prompt_instruction_ids=("identity.access_checker.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("identity", "access", "tenancy"),
)

IDENTITY_SESSION_BOOTSTRAP = SkillManifest(
    skill_id="identity.session_bootstrap",
    version="1.0.0",
    description="Bootstrap session from verified identity and memory seed.",
    tool_ids=("identity.verify_token", "identity.get_user", "memory.write"),
    prompt_instruction_ids=("identity.session_bootstrap.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("identity", "session", "bootstrap"),
)

