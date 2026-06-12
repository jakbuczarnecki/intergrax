# © Artur Czarnecki. All rights reserved.

"""CE-3.2: ContextCandidate ↔ ContextFragment bridge."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import ContextFragmentSource
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_compiler_models import (
    ContextCandidate,
    ContextCandidateSource,
)
from intergrax.runtime.nexus.context.fragment_bridge import (
    candidate_from_fragment,
    fragment_from_candidate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_fragment_candidate_round_trip() -> None:
    message = ChatMessage(role="user", content="hello")
    candidate = ContextCandidate(
        source=ContextCandidateSource.USER_TURN,
        message_index=0,
        score=1.0,
        token_estimate=2,
        mandatory=True,
    )
    fragment = fragment_from_candidate(candidate, message)
    assert fragment.source == ContextFragmentSource.TASK_MESSAGE
    restored = candidate_from_fragment(fragment, message_index=0)
    assert restored.source == ContextCandidateSource.USER_TURN
    assert restored.mandatory is True
