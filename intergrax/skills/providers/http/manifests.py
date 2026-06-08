# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

HTTP_API_CLIENT = SkillManifest(
    skill_id="http.api_client",
    version="1.0.0",
    description="HTTP API client: outbound requests with error capture and log correlation.",
    tool_ids=("http.request", "errors.capture", "logs.search"),
    prompt_instruction_ids=("http.api_client.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("http", "api", "client"),
)

