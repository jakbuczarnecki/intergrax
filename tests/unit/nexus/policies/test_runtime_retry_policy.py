# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.responses.response_schema import RouteInfo, RuntimeAnswer, RuntimeRequest, RuntimeStats
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore

from tests._support.builder import build_engine_harness, build_runtime_config_deterministic


class _FlakyInternalErrorPipeline(RuntimePipeline):
    def __init__(self) -> None:
        self._calls = 0

    async def _inner_run(self, state) -> RuntimeAnswer:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("boom")

        answer = RuntimeAnswer(
            answer="OK",
            run_id=state.run_id,
            citations=[],
            route=RouteInfo(
                used_rag=False,
                used_websearch=False,
                used_tools=False,
                used_user_profile=False,
                used_user_longterm_memory=False,
                strategy="llm_only",
                extra={},
            ),
            tool_calls=[],
            stats=RuntimeStats(
                total_tokens=None,
                input_tokens=None,
                output_tokens=None,
                rag_tokens=None,
                websearch_tokens=None,
                tool_tokens=None,
                duration_ms=None,
                extra={},
            ),
            raw_model_output=None,
        )

        state.runtime_answer = answer
        return answer


@pytest.mark.asyncio
async def test_runtime_retries_on_internal_error(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic()
    cfg.max_run_retries = 1
    cfg.retry_run_on = frozenset({RuntimeErrorCode.INTERNAL_ERROR})

    harness = build_engine_harness(cfg=cfg)
    store = InMemoryRunTraceStore()
    harness.engine.context.trace_writer = store

    pipeline = _FlakyInternalErrorPipeline()

    def _fake_build_pipeline(*, state):
        return pipeline

    monkeypatch.setattr(PipelineFactory, "build_pipeline", _fake_build_pipeline)

    request = RuntimeRequest(
        user_id="user-1",
        session_id="sess-1",
        message="hello",
    )

    answer = await harness.engine.run(request)
    assert answer.answer == "OK"
    assert answer.run_id is not None

    run = store.read_run(answer.run_id)
    assert run.metadata.error is None
