# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Response governance hook (Tier-2): draft finalize answer → product-shaped client response.

Wired from :class:`~legal_agent.steps.legal_finalize_answer_step.LegalFinalizeAnswerStep`
after the finalize LLM returns and **before** :class:`~intergrax.runtime.nexus.responses.response_schema.RuntimeAnswer`
is stored. Tier-1 Nexus is unchanged.

Ready-made implementations: :mod:`legal_agent.governance.legal_response_governance_impl`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from legal_agent.domain.legal_shaped_client_response import (
    LegalShapedClientResponse,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

if TYPE_CHECKING:
    from legal_agent.config.legal_agent_config import LegalAgentConfig
    from legal_agent.domain.legal_agent_state import LegalAgentState


class LegalResponseGovernancePort(ABC):
    """
    Transform the finalize-step draft into a :class:`LegalShapedClientResponse`.

    Implementations may read ``agent_state`` (decision, uncertainties, violations) and
    ``state.request`` (tenant, metadata). Keep deterministic when possible; optional LLM
    rewrites belong here only if budgeted by the host.
    """

    @abstractmethod
    def shape_legal_client_response(
        self,
        draft_answer: str,
        *,
        state: RuntimeState,
        agent_state: LegalAgentState,
        legal_config: LegalAgentConfig,
    ) -> LegalShapedClientResponse:
        ...
