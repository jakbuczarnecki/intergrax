# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Concrete :class:`LegalResponseGovernancePort` helpers for :class:`LegalAgentConfig`.

* ``None`` on config — finalize draft is used as ``RuntimeAnswer.answer`` (unchanged).
* :class:`PassthroughLegalResponseGovernance` — explicit no-op (same as ``None`` but non-null slot).
* :class:`CallableLegalResponseGovernance` — delegate to ``(draft, state, agent_state, legal_config)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.governance.legal_response_governance_port import (
    LegalResponseGovernancePort,
)
from intergrax.agents_packages.legal_agent.domain.legal_shaped_client_response import (
    LegalShapedClientResponse,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

LegalResponseGovernanceFn = Callable[
    [str, RuntimeState, LegalAgentState, LegalAgentConfig],
    LegalShapedClientResponse,
]


@dataclass(slots=True, frozen=True)
class PassthroughLegalResponseGovernance(LegalResponseGovernancePort):
    """Returns ``LegalShapedClientResponse(body=draft_answer)``."""

    def shape_legal_client_response(
        self,
        draft_answer: str,
        *,
        state: RuntimeState,
        agent_state: LegalAgentState,
        legal_config: LegalAgentConfig,
    ) -> LegalShapedClientResponse:
        return LegalShapedClientResponse(body=draft_answer)


@dataclass(slots=True)
class CallableLegalResponseGovernance(LegalResponseGovernancePort):
    """Delegates to a plain callable (tenant templates, DB copy, etc.)."""

    _fn: LegalResponseGovernanceFn

    def shape_legal_client_response(
        self,
        draft_answer: str,
        *,
        state: RuntimeState,
        agent_state: LegalAgentState,
        legal_config: LegalAgentConfig,
    ) -> LegalShapedClientResponse:
        return self._fn(draft_answer, state, agent_state, legal_config)
