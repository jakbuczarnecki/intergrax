# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent stable-identity ranking adapter (Stage 4)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.ranking import StableIdentityRanker

AGENT_STABLE_IDENTITY_RANKER_ID: Final = "agent.stable_identity"


class AgentStableIdentityCapabilityRanker(StableIdentityRanker):
    """Agent domain ranker — reuses AC-4 stable identity ordering primitive, not selection."""

    @property
    def ranker_id(self) -> str:
        return AGENT_STABLE_IDENTITY_RANKER_ID
