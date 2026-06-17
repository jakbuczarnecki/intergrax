# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import urllib.request
import urllib.error
import numpy as np
import pytest
from numpy.typing import NDArray
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm.messages import ChatMessage
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.providers.inmemory_vectorstore import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.runtime.governance.execution_guard import ExecutionGuard, GovernanceEvaluation
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy import PolicyDecision, PolicyDecisionType
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.tools.core.contracts import ToolContract



class FakeLLMAdapter(LLMAdapter):
    """
    Deterministic LLM adapter for CI-safe tests.

    Goals:
    - no network
    - stable output
    - still exercises CoreLLMStep / finalization path
    """

    provider = "fake"
    model = "fake"

    @property
    def context_window_tokens(self) -> int:
        # Large enough for tests; avoids truncation logic influencing results.
        return 128_000


    def __init__(
        self,
        *,
        fixed_text: str = "OK",
        fake_structured_data: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._fixed_text = fixed_text
        self._fake_structured_data = fake_structured_data

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        # Deterministic response for tests.
        # Keep it simple: do NOT depend on message content.
        call = self.usage.begin_call(run_id=run_id)
        try:
            return build_adapter_response(content=self._fixed_text)
        finally:
            # Tokens are fake here; that's fine for tests.
            self.usage.end_call(
                call,
                input_tokens=0,
                output_tokens=len(self._fixed_text),
                success=True,
            )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        call = self.usage.begin_call(run_id=run_id)

        try:
            if self._fake_structured_data is not None:
                obj = self._fake_structured_data

                if not isinstance(obj, output_model):
                    raise TypeError(
                        "FakeLLMAdapter: fake_structured_data must be instance of output_model"
                    )

                parsed = obj
            else:
                parsed = output_model()

            response = build_adapter_response(content="")
            return LLMStructuredResult(parsed=parsed, response=response)

        finally:
            self.usage.end_call(
                call,
                input_tokens=0,
                output_tokens=0,
                success=True,
            )


class MeteringFakeLLMAdapter(FakeLLMAdapter):
    """Fake adapter with prompt-sized token metering (ACP-TOK smoke / host runs)."""

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

        prompt = messages[-1].content if messages else ""
        word_count = max(len(prompt.split()), 1)
        call = self.usage.begin_call(run_id=run_id)
        try:
            return build_adapter_response(
                content=self._fixed_text,
                usage=LLMTokenUsage(input_tokens=word_count, output_tokens=word_count),
            )
        finally:
            self.usage.end_call(
                call,
                input_tokens=word_count,
                output_tokens=word_count,
                success=True,
            )


class DummyExecutionGuard(ExecutionGuard):
    """
    Minimal execution guard for production-trace tests.

    It does NOT:
    - reconstruct execution
    - compute real metrics
    - evaluate history
    - execute actions

    It only returns a deterministic ALLOW decision.
    """

    def __init__(self) -> None:
        # We intentionally do NOT call super().__init__
        # because we do not need replay/metrics/policy engines.
        pass

    def evaluate_run(
        self,
        run_id: str,
        agent_id: str,
    ) -> GovernanceEvaluation:
        decision = PolicyDecision(
            decision=PolicyDecisionType.ALLOW,
            reasons=["dummy-allow"],
        )

        # Minimal dummy objects (empty but correctly typed)
        metrics = ExecutionMetrics()
        regression = RegressionSignals()

        return GovernanceEvaluation(
            decision=decision,
            metrics=metrics,
            regression=regression,
        )

class FakeEmbeddingProvider(EmbeddingProvider):
    
    def __init__(
        self,
    ) -> None:
        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "fake"

    def _ensure_model(self) -> None:
        pass        

    def _resolve_dim(self) -> None:

        if self._dim is None:
            self._dim = 256
            

    def dimension(self) -> int:

        self._resolve_dim()
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._resolve_dim()

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        self._ensure_model()

        return np.zeros((len(texts), self._dim), dtype=np.float32)


def require_ollama_reachable(
    *,
    base_url: Optional[str] = None,
    timeout_sec: float = 3.0,
) -> None:
    """
    Skip the current test if the Ollama HTTP API is not reachable.

    Resolution order for the server base URL:
    1. explicit ``base_url`` argument
    2. environment variable ``OLLAMA_HOST`` (e.g. ``http://127.0.0.1:11434``)
    3. default ``http://127.0.0.1:11434``
    """

    default_ollama_base_url = "http://127.0.0.1:11434"
    raw = (base_url or os.environ.get("OLLAMA_HOST") or default_ollama_base_url).strip().rstrip("/")
    tags_url = f"{raw}/api/tags"
    try:
        urllib.request.urlopen(tags_url, timeout=timeout_sec)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"Ollama not reachable at {raw}: {e}")


def build_in_memory_session_manager() -> SessionManager:
    storage = InMemorySessionStorage()
    return SessionManager(storage)

def build_in_memory_vectorstore_manager(*, tenant_id: Optional[str] = None)-> BaseVectorstoreManager:
    if tenant_id is None:
        tenant_id = "in_memory_tenant_id"

    manager = VectorstoreManager(store=InMemoryVectorStore(tenant_id=tenant_id))
    return manager


def build_fake_embedding_manager() -> EmbeddingManager:
    registry = EmbeddingProviderRegistry()

    provider = FakeEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id=provider.provider_name(),
    )

    manager = EmbeddingManager(pipeline=pipeline)

    return manager


def build_runtime_config_for_tests(
    *,
    llm_text: str = "OK",
    idempotency_store: Optional[IdempotencyStore] = None,
    production_mode: bool = False,
    trace_db_path: str | None = None,
) -> RuntimeConfig:
    """Minimal deterministic RuntimeConfig for unit tests."""
    cfg = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text=llm_text),
        tenant_id="test-tenant",
        workspace_id="test-workspace",
        enable_rag=False,
        enable_websearch=False,
        enable_org_profile_memory=False,
        tools_mode="off",
        idempotency_store=idempotency_store,
        production_mode=production_mode,
        trace_db_path=trace_db_path,
    )
    cfg.validate()
    return cfg


def build_runtime_state_for_tests(*, run_id: str) -> RuntimeState:
    """
    Minimal RuntimeState builder for unit tests that only need tracing.
    No engine, no pipeline, no planner — just state + trace_event support.
    """

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent_test",
        user_id="test-user",
        session_id="test-session",
        message="test",
    )

    cfg = RuntimeConfig(
        llm_adapter=None,
        embedding_manager=None,
        vectorstore_manager=None,
        tenant_id="test-tenant",
        workspace_id="test-workspace",
        websearch_executor=None,
        websearch_config=None,
        tool_planner=None,
        enable_rag=False,
        enable_websearch=False,
        enable_org_profile_memory=False,
        tools_mode="off",
        production_mode=False,
    )

    sm = SessionManager(storage=InMemorySessionStorage())

    ctx = RuntimeContext.build(
        config=cfg,
        session_manager=sm,
        ingestion_service=None,
        context_builder=None,
        rag_prompt_builder=None,
        user_longterm_memory_prompt_builder=None,
        websearch_prompt_builder=None,
        history_prompt_builder=None,
    )

    return RuntimeState(context=ctx, run_id=run_id, request=request)


class DummyRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunResponse] = {}

    def create(self) -> RunResponse:
        run_id = "r1"
        run = RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]

    def cancel(self, run_id: str) -> RunResponse:
        raise AssertionError("Should not reach store.cancel() if transition invalid")

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        result_payload: dict | None = None,
    ) -> RunResponse:
        current = self._runs[run_id]

        updated = RunResponse(
            run_id=current.run_id,
            status=status,
            error_type=error_type if error_type is not None else current.error_type,
            error_message=error_message if error_message is not None else current.error_message,
            started_at=started_at if started_at is not None else current.started_at,
            finished_at=finished_at if finished_at is not None else current.finished_at,
            duration_ms=duration_ms if duration_ms is not None else current.duration_ms,
            result_payload=result_payload if result_payload is not None else current.result_payload,
        )

        self._runs[run_id] = updated
        return updated


def tools_agent_make_contract(tool_id: str, input_model, output_model):
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=f"{tool_id} description",
        input_schema=input_model,
        output_schema=output_model,
        error_mapping={},
        side_effects=False,
    )

def prepare_sqlite_db(name:str)->Path:
    db_path = Path(f"temp_documents/{name}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    return db_path