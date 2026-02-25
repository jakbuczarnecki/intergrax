# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from tests._support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def test_runtime_context_requires_governance_in_production_mode():
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
    )
    config.production_mode = True

    sm = SessionManager(storage=InMemorySessionStorage())

    with pytest.raises(ValueError) as exc:
        RuntimeContext.build(
            config=config,
            session_manager=sm,
            governance_service=None,
        )

    assert "GovernanceService is required" in str(exc.value)
