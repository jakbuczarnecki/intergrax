# © Artur Czarnecki. All rights reserved.

"""Host context passed into direct ACP runs (Tier-3 slices)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding

ACP_HOST_CONTEXT_KEY = "acp.host.v1"


class ACPSessionHostContext(BaseModel):
    """Optional Tier-3 host slices for merge_environment on direct run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    app_profile: ApplicationEnvironmentProfile | None = None
    binding: AgentBinding | None = Field(default=None, exclude=True)
