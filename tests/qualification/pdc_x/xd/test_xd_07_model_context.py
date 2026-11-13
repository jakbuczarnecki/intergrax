# © Artur Czarnecki. All rights reserved.

"""XD-07 (Model <-> Context): budgets follow the model, mandatory instructions survive, restart reconstructs.

PDC-D2-R013. The model side is a real ``BaseLLMAdapter`` (the external provider is the only fake: it
returns canned text and streamed tool calls); the context side is the real engine, the real compiler
and the durable SQLite optimization repository.

* the context window the adapter declares drives the assembled budget, and a mandatory instruction
  survives the smaller window while optional context does not;
* a tool call streamed by the adapter keeps its (minted) provenance id through compilation under
  pressure, with call and result still linked;
* a summary produced by the model for durable compaction survives a restart of every store and is
  reconstructable back to its source messages.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError
from intergrax.context.contracts import ContextFragmentSource
from intergrax.context.registry import ContextPluginRegistry
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind
from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    finalize_accepted_tool_call_identities,
    validate_tool_call_identities,
)
from intergrax.runtime.context_lifecycle import SQLiteOptimizationArtifactRepository
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.session.context_revision import (
    SessionContextRevisionActivationRequest,
    SessionContextRevisionActivationService,
    SessionContextRevisionActivationStatus,
    SQLiteSessionContextRevisionStore,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidateStatus,
    DurableCompactionCandidateBuilder,
)
from intergrax.runtime.token_optimization.durable_compaction_validation import (
    DurableCompactionValidationCompiler,
    DurableCompactionValidationStatus,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    InternalMessageSequenceModelCall,
    MessageSequenceArtifactExecutor,
)
from tests.qualification.pdc_x.d2._support import (
    FakeAdapter,
    FragmentProvider,
    assembly_request,
    make_fragment,
    message_text,
    provider_context,
    tool_linkage_violations,
)
from tests.unit.runtime.token_optimization import test_durable_compaction_closeout as fx

pytestmark = [pytest.mark.integration]

MANDATORY = "XD07-MANDATORY-POLICY-INSTRUCTION"


def _count(text: str) -> int:
    return len(text)


# ----------------------------------------------------------------------- window drives the budget


class _WindowAdapter(FakeAdapter):
    provider = "fake"

    def __init__(self, window: int, model: str) -> None:
        super().__init__(window=window)
        self.model = model


async def _assemble(adapter: FakeAdapter, *, request_budget: int = 6000):  # type: ignore[no-untyped-def]
    registry = ContextPluginRegistry()
    registry.add_provider(
        FragmentProvider(
            "xd07.context",
            {ContextFragmentSource.SYSTEM_INSTRUCTIONS, ContextFragmentSource.RAG},
            lambda: [
                make_fragment(
                    "mandatory",
                    ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                    MANDATORY,
                    mandatory=True,
                    relevance=1.0,
                ),
                make_fragment(
                    "rag-1",
                    ContextFragmentSource.RAG,
                    "XD07-RAG-FILLER " + ("filler " * 700),
                    relevance=0.95,
                    token_estimate=2500,
                ),
            ],
        )
    )
    return await DefaultNexusContextEngine(registry=registry).assemble(
        assembly_request(budget_tokens=request_budget),
        provider_ctx=provider_context(adapter, "summarise the policy"),
    )


async def test_xd07_the_model_context_window_drives_the_budget_and_the_mandatory_instruction_survives() -> None:
    large = await _assemble(_WindowAdapter(window=32_768, model="fake-large"))
    small = await _assemble(_WindowAdapter(window=2_048, model="fake-small"))

    assert MANDATORY in message_text(large.messages)
    assert "XD07-RAG-FILLER" in message_text(large.messages), "optional context fits the large window"

    assert MANDATORY in message_text(small.messages), "the mandatory instruction is never dropped"
    assert "XD07-RAG-FILLER" not in message_text(small.messages), "optional context is shed for the small window"
    assert small.budget_tokens < large.budget_tokens, "the budget follows the model, not the request default"
    assert small.resolved_model_budget is not None and large.resolved_model_budget is not None
    assert small.resolved_model_budget.model_context_window == 2_048
    assert large.resolved_model_budget.model_context_window == 32_768
    assert small.total_tokens <= small.budget_tokens


async def test_xd07_a_mandatory_instruction_that_cannot_fit_the_model_fails_closed() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(
        FragmentProvider(
            "xd07.too-big",
            {ContextFragmentSource.SYSTEM_INSTRUCTIONS},
            lambda: [
                make_fragment(
                    "mandatory-huge",
                    ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                    "MUST-FOLLOW " * 4000,
                    mandatory=True,
                    relevance=1.0,
                    token_estimate=6000,
                )
            ],
        )
    )
    with pytest.raises(ContextBudgetUnsatisfiableError):
        await DefaultNexusContextEngine(registry=registry).assemble(
            assembly_request(budget_tokens=6000),
            provider_ctx=provider_context(_WindowAdapter(window=1_024, model="fake-tiny"), "q"),
        )


# ----------------------------------------------------------------------- streamed tool-call provenance


class _StreamingToolAdapter(FakeAdapter):
    """Provider fake that streams text deltas and then a FINAL carrying a tool call with no id."""

    model = "fake-streaming-tools"

    def supports_streaming(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def supports_tool_streaming(self) -> bool:
        return True

    def stream_with_tools(self, messages: Sequence[ChatMessage], tools, **_: object) -> Iterable[LLMStreamEvent]:  # type: ignore[no-untyped-def]
        yield LLMStreamEvent(kind=LLMStreamEventKind.PARTIAL, delta_content="looking ")
        yield LLMStreamEvent(kind=LLMStreamEventKind.PARTIAL, delta_content="up")
        yield LLMStreamEvent(
            kind=LLMStreamEventKind.FINAL,
            response=LLMAdapterResponse(
                content="looking up",
                provider=self.provider,
                model=self.model,
                tool_calls=(
                    LLMToolCall(id="", name="lookup", arguments_json='{"q": "policy"}'),
                    LLMToolCall(id="", name="fetch", arguments_json='{"url": "https://example.test"}'),
                ),
            ),
        )


def _streamed_final(adapter: _StreamingToolAdapter):  # type: ignore[no-untyped-def]
    events = list(adapter.stream_with_tools([ChatMessage(role="user", content="q")], []))
    assert [e.kind for e in events] == [LLMStreamEventKind.PARTIAL] * 2 + [LLMStreamEventKind.FINAL]
    final = events[-1].response
    assert final is not None
    return final


def test_xd07_streamed_tool_calls_keep_unique_provenance_ids_through_compilation_under_pressure() -> None:
    adapter = _StreamingToolAdapter(window=4096)
    final = _streamed_final(adapter)
    calls = finalize_accepted_tool_call_identities(final.tool_calls)
    validate_tool_call_identities(calls)
    assert len({call.id for call in calls}) == 2 and all(call.id.startswith("toolcall-") for call in calls)

    history = [
        ChatMessage(role="system", content=MANDATORY, entry_id="sys-1"),
        ChatMessage(role="user", content="old question " * 20, entry_id="user-old"),
        ChatMessage(
            role="assistant",
            content=final.content or "",
            entry_id="asst-1",
            tool_calls=[
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments_json}}
                for call in calls
            ],
        ),
        *[
            ChatMessage(
                role="tool",
                content=f"result of {call.name} " + "R" * 1500,
                entry_id=f"tool-{index}",
                tool_call_id=call.id,
                name=call.name,
            )
            for index, call in enumerate(calls)
        ],
        ChatMessage(role="user", content="final question", entry_id="user-final"),
    ]
    compiler = ContextCompiler(count_tokens=_count)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    for budget in (4000, 900, 300, 120):
        result = compiler.compile(list(history), config, max_output_tokens=64, input_budget_tokens=budget)
        messages = list(result.messages)
        assert tool_linkage_violations(messages) == [], budget
        assert messages[0].entry_id == "sys-1" and messages[0].content == MANDATORY
        assert messages[-1].entry_id == "user-final"
        kept_calls = {c["id"] for m in messages if m.role == "assistant" and m.tool_calls for c in m.tool_calls}
        kept_results = {m.tool_call_id for m in messages if m.role == "tool"}
        assert kept_calls == kept_results, "a call and its result stay together or leave together"
        assert kept_calls <= {call.id for call in calls}, "no id other than the streamed provenance ids appears"


# ----------------------------------------------------------------------- restart reconstruction


class _SummarisingAdapter(FakeAdapter):
    """Provider fake for the internal summarisation call: returns a summary keeping protected regions."""

    model = "fake-summariser"

    def __init__(self) -> None:
        super().__init__(window=8192)
        self.calls = 0

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:  # type: ignore[no-untyped-def]
        self.calls += 1
        return LLMAdapterResponse(
            content=f"preserved {fx._URL} {fx._PATH} {fx._COMMAND} {fx._HASH}",
            provider=self.provider,
            model=self.model,
        )


def test_xd07_model_produced_compaction_survives_restart_and_reconstructs_its_source(tmp_path: Path) -> None:
    db_path = tmp_path / "xd07.sqlite"
    adapter = _SummarisingAdapter()
    request = fx._request()
    originals = fx._messages()

    def invoke_model(call: InternalMessageSequenceModelCall) -> LLMAdapterResponse:
        return adapter.generate_messages(list(call.messages))  # the real adapter contract

    repository = SQLiteOptimizationArtifactRepository(
        str(db_path), clock=lambda: fx._NOW, reservation_id_factory=lambda: "reservation-xd07"
    )
    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _: None,
        invoke_model=invoke_model,
        count_tokens=lambda text: max(1, len(text.split())),
        clock=lambda: fx._NOW,
        operation_id_factory=lambda: "internal-operation-xd07",
        receipt_id_factory=lambda: "message-receipt-xd07",
    )
    built = DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=executor,
        artifact_id_factory=lambda: "artifact-xd07",
        wait_timeout_seconds=0.0,
    ).build(request)
    assert built.candidate is not None
    assert built.candidate.status is CompactionCandidateStatus.CREATED_NEW_ARTIFACT
    assert adapter.calls == 1, "the summary really came from a model call"
    outcome = DurableCompactionValidationCompiler(
        repository=repository,
        clock=lambda: fx._NOW,
        receipt_id_factory=lambda: "durable-receipt-xd07",
    ).compile(fx._validation_request(request, built))
    assert outcome.status is DurableCompactionValidationStatus.PASSED
    repository.close()

    SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW).close()
    fx._seed_active_pointer(db_path)
    repo_a = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: fx._NOW)
    store_a = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW)
    activated = SessionContextRevisionActivationService(
        repository=repo_a, revision_store=store_a, clock=lambda: fx._NOW
    ).activate(
        SessionContextRevisionActivationRequest(
            tenant_id="tenant-1",
            context_scope_id="context-1",
            operation_id="operation-1",
            outcome=outcome,
            expected_active_revision=4,
        )
    )
    assert activated.status is SessionContextRevisionActivationStatus.ACTIVATED
    repo_a.close()
    store_a.close()

    # Restart: every durable store is reopened. The active pointer resolves to the artifact, and the
    # artifact reconstructs to the original messages by id and content hash.
    recovered_repo = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: fx._NOW)
    recovered_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW)
    try:
        pointer = recovered_store.get_active_pointer(tenant_id="tenant-1", context_scope_id="context-1")
        assert pointer.active_artifact_id == outcome.candidate.artifact_reference.artifact_id
        artifact = recovered_repo.resolve(outcome.candidate.artifact_reference)
        assert artifact is not None
        payload = json.loads(artifact.payload.decode("utf-8"))
        assert tuple(payload["source_refs"]) == tuple(m.message_id for m in originals)
        assert payload["source_content_hash"] == request.snapshot.source_identity.source_content_hash
        for region in (fx._URL, fx._PATH, fx._COMMAND, fx._HASH):
            assert region in payload["summary"]
        assert fx._SECRET not in repr((outcome, payload["summary"], pointer))
    finally:
        recovered_repo.close()
        recovered_store.close()
