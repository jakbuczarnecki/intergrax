# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


import pytest

from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from testing_support.builder import build_engine_harness_production_trace, prepare_sqlite_db

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runtime_sqlite_replay_e2e():
    db_path = prepare_sqlite_db("reply_e2e.db")
    
    harness = build_engine_harness_production_trace(
        trace_db_path=db_path
    )

    engine = harness.engine

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent_test",
        user_id="user_test",
        session_id="session_test",
        message="Hello",
    )

    runtime_answer = await engine.run(request)

    assert db_path.exists()

    store = SQLiteRunTraceStore(db_path=db_path)
    persisted = store.read_run(runtime_answer.run_id, request.tenant_id)

    assert persisted.metadata is not None
    assert persisted.metadata.run_id == runtime_answer.run_id
    assert persisted.metadata.stats.duration_ms >= 0
    assert len(persisted.events) > 0
