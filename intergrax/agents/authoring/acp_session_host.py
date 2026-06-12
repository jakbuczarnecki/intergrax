# © Artur Czarnecki. All rights reserved.

"""Host context passed into direct ACP runs (Tier-3 slices)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey

ACP_HOST_CONTEXT_KEY = AcpMetadataKey.HOST_CONTEXT


class ACPSessionHostContext(BaseModel):
    """Optional Tier-3 host slices for merge_environment on direct run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    app_profile: ApplicationEnvironmentProfile | None = None
    binding: AgentBinding | None = Field(default=None, exclude=True)
    declarative_tool_invoker: Any = Field(default=None, exclude=True)
    critic_graph_hooks: Any = Field(default=None, exclude=True)
    notification_adapter: Any = Field(default=None, exclude=True)
    budget_reaction_hook: Any = Field(default=None, exclude=True)
