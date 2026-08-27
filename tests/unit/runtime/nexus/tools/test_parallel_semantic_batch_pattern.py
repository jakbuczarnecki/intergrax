# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-25 — ParallelSemanticBatchPattern acceptance tests."""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document
from numpy.typing import NDArray
from pydantic import BaseModel

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.patterns.parallel_semantic_batch import ParallelSemanticBatchPattern
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_invocation_pattern import pattern_for_mode
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, canonical_execution_identity_scope, canonical_run_id_for_tests

pytestmark = pytest.mark.unit

_DIM = 32


class _QueryIn(BaseModel):
    query: str = ""


class _Out(BaseModel):
    result: str = ""


class BagOfWordsEmbeddingManager(BaseEmbeddingManager):
    def embed_one(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(_DIM, dtype=np.float32)
        for token in text.lower().split():
            vec[hash(token) % _DIM] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.stack([self.embed_one(text) for text in texts])

    def embed_documents(self, documents: Sequence[Document]) -> EmbeddingResult:
        vectors = [list(self.embed_one(doc.page_content)) for doc in documents]
        return EmbeddingResult(vectors=vectors, model_name="test")


class _ReadHandler:
    def execute(self, request: ToolExecutionRequest[_QueryIn]) -> _Out:
        return _Out(result=request.input.query)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id, description in (
        ("rag.retrieve", "Retrieve hybrid documents from vector store"),
        ("websearch.query", "Run web search query for snippets"),
    ):
        contract = ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=description,
            input_schema=_QueryIn,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
        )
        registry.register(contract, _ReadHandler())
    return registry


def _state(registry: ToolRegistry, embedder: BagOfWordsEmbeddingManager) -> RuntimeState:
    run_id = canonical_run_id_for_tests("run-semantic-batch")
    task_id = TaskId(f"task_{run_id[4:]}")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        embedding_manager=embedder,
        tool_invocation_mode=ToolInvocationMode.PARALLEL_SEMANTIC_BATCH,
        tool_selection_top_k=2,
        max_parallel_tool_calls=2,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="retrieve documents about contracts",
            task_id=task_id,
            run_id=run_id,
        ),
        run_id=run_id,
    )


def test_pattern_for_mode_parallel_semantic_batch() -> None:
    pattern = pattern_for_mode(ToolInvocationMode.PARALLEL_SEMANTIC_BATCH)
    assert pattern.pattern_id == "parallel_semantic_batch"


def test_parallel_semantic_batch_invokes_semantic_top_k() -> None:
    registry = _registry()
    embedder = BagOfWordsEmbeddingManager()
    state = _state(registry, embedder)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))

    with canonical_execution_identity_scope(state.run_id):
        result = ParallelSemanticBatchPattern().execute(
            state=state,
            invoker=invoker,
            planner=MagicMock(),
            plan=None,
            allowed_tool_ids=None,
            max_iterations=1,
            planner_input="retrieve documents about contracts",
        )

    assert result.aggregate is not None
    assert len(result.tool_traces) >= 1
    assert result.tool_traces[0].tool_name == "rag.retrieve"
