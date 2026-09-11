# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Callable, Optional, Any, Dict, Union, List, TypeVar
import json
import re
import uuid
import time
import tiktoken
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig, parse_call_config
from intergrax.llm_adapters._shared.resilience import execute_with_resilience
from intergrax.llm_adapters._shared.retry import call_with_retry
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition

if TYPE_CHECKING:
    from intergrax.contracts.dependency_concurrency_admission import (
        DependencyConcurrencyAdmissionRequest,
    )
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationCancellationPort,
        ExternalOperationStatusPort,
    )
    from intergrax.contracts.external_operation_termination import (
        ExternalOperationCapabilities,
        ExternalOperationTerminationPort,
    )
    from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
        ProviderStreamTransportRegistry,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateStore,
    )
    from intergrax.runtime.external_operations.external_operation_ownership import (
        ProcessLocalExternalOperationOwner,
    )
    from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
        DependencyAttemptExecutionBoundary,
    )
    from intergrax.runtime.external_operations.admission.execution_gate import (
        ExternalOperationExecutionGate,
    )
    from intergrax.contracts.external_operations.attempt import ExternalOperationAttempt

T = TypeVar("T")


# ============================================================
# Universal interface (ABC)
# ============================================================

class LLMAdapter(ABC):
    """
    Universal runtime interface for LLM adapters.

    This used to be a Protocol. It is now an ABC to provide:
      - strong runtime guarantees (abstract methods)
      - shared base implementation (token counting)

    Required:
      - generate_messages(...)
      - context_window_tokens

    Optional (default to NotImplemented/False):
      - streaming
      - tools
      - structured output
    """

    provider: LLMProvider
    model: str

    # Hint used by the generic token estimator (e.g. OpenAI model name).
    model_name_for_token_estimation: Optional[str] = None

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.usage = LLMAdapterUsageLog()
        self.call_config = LLMCallConfig()
        self._provider_dependency_boundary: DependencyAttemptExecutionBoundary | None = (
            None
        )
        self._external_operation_store: ExternalOperationStateStore | None = None
        self._external_operation_owner: ProcessLocalExternalOperationOwner | None = None
        self._external_operation_cancellation_port: (
            ExternalOperationCancellationPort | None
        ) = None
        self._external_operation_status_port: ExternalOperationStatusPort | None = None
        self._external_operation_termination_port: (
            ExternalOperationTerminationPort | None
        ) = None
        self._external_operation_stream_registry: (
            ProviderStreamTransportRegistry | None
        ) = None
        self._external_operation_capabilities: ExternalOperationCapabilities | None = (
            None
        )
        self._external_operation_execution_gate: ExternalOperationExecutionGate | None = (
            None
        )

    def bind_external_operation_admission_gate(
        self,
        gate: ExternalOperationExecutionGate | None,
    ) -> None:
        """Inject R1 admission gate — when set, provider calls require ALLOW."""
        from intergrax.runtime.external_operations.admission.execution_gate import (
            ExternalOperationExecutionGate as _Gate,
        )

        if gate is not None and not isinstance(gate, _Gate):
            raise TypeError("gate must be ExternalOperationExecutionGate or None")
        self._external_operation_execution_gate = gate

    def bind_external_operation_ports(
        self,
        *,
        store: ExternalOperationStateStore | None,
        owner: ProcessLocalExternalOperationOwner | None = None,
        cancellation_port: ExternalOperationCancellationPort | None = None,
        status_port: ExternalOperationStatusPort | None = None,
        termination_port: ExternalOperationTerminationPort | None = None,
        stream_registry: ProviderStreamTransportRegistry | None = None,
        capabilities: ExternalOperationCapabilities | None = None,
    ) -> None:
        """Inject W4-C durable external operation tracking for provider calls."""
        from intergrax.runtime.external_operations.external_operation_ownership import (
            ProcessLocalExternalOperationOwner,
        )

        self._external_operation_store = store
        if store is not None and owner is None:
            owner = ProcessLocalExternalOperationOwner.mint()
        self._external_operation_owner = owner
        self._external_operation_cancellation_port = cancellation_port
        self._external_operation_status_port = status_port
        self._external_operation_termination_port = termination_port
        self._external_operation_stream_registry = stream_registry
        if capabilities is not None:
            self._external_operation_capabilities = capabilities
        elif store is not None:
            from intergrax.llm_adapters._shared.provider_external_operation_capabilities import (
                external_operation_capabilities_for_provider,
            )

            self._external_operation_capabilities = (
                external_operation_capabilities_for_provider(self._provider_slug())
            )

    def bind_provider_dependency_boundary(
        self,
        boundary: DependencyAttemptExecutionBoundary | None,
    ) -> None:
        """Inject shared process-local provider dependency boundary (W2-B3)."""
        from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
            DependencyAttemptExecutionBoundary,
        )

        if boundary is not None and not isinstance(
            boundary, DependencyAttemptExecutionBoundary
        ):
            raise TypeError(
                "boundary must be DependencyAttemptExecutionBoundary or None"
            )
        self._provider_dependency_boundary = boundary

    def _apply_defaults_call_config(self, defaults: Dict[str, Any]) -> None:
        """Merge ``LLMCallConfig`` fields from adapter constructor kwargs."""
        self.call_config = parse_call_config(defaults)

    def _provider_slug(self) -> str:
        prov = self.provider
        if isinstance(prov, LLMProvider):
            return prov.value
        return str(prov or "unknown")

    def _adapter_identity(self) -> tuple[str, str]:
        return self._provider_slug(), str(self.model or "")

    def _provider_dependency_admission_request(
        self,
    ) -> DependencyConcurrencyAdmissionRequest:
        from intergrax.contracts.dependency_concurrency_admission import (
            DependencyConcurrencyAdmissionRequest,
            DependencyConcurrencyIdentity,
            DependencyConcurrencyKind,
        )
        from intergrax.llm_adapters.tracking.context import get_llm_tenant_id

        return DependencyConcurrencyAdmissionRequest(
            dependency=DependencyConcurrencyIdentity(
                kind=DependencyConcurrencyKind.LLM_PROVIDER,
                value=self._provider_slug(),
            ),
            tenant_id=get_llm_tenant_id(),
        )

    def _admit_llm_provider_intent(self, *, call_scope: str) -> ExternalOperationAttempt | None:
        gate = self._external_operation_execution_gate
        if gate is None:
            return None
        from datetime import datetime, timezone

        from intergrax.contracts.execution_identity import mint_task_id
        from intergrax.contracts.external_operations.admission import (
            ExternalOperationAdmissionContext,
        )
        from intergrax.contracts.external_operations.intent import (
            ExternalOperationIntent,
            ExternalOperationType,
            mint_external_operation_intent_id,
        )
        from intergrax.llm_adapters.tracking.context import get_llm_tenant_id

        tenant = (get_llm_tenant_id() or "").strip() or "tenant_platform"
        intent = ExternalOperationIntent(
            intent_id=mint_external_operation_intent_id(),
            tenant_id=tenant,
            task_id=mint_task_id(),
            operation_type=ExternalOperationType.LLM_PROVIDER_CALL,
            target_resource=f"{self._provider_slug()}:{self.model}:{call_scope}",
            requested_by="llm_adapter",
            justification="llm provider inference call",
            created_at=datetime.now(timezone.utc),
        )
        return gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(
                tenant_id=tenant,
                provider_id=self._provider_slug(),
            ),
            provider_id=self._provider_slug(),
        )

    def _run_physical_provider_attempt(self, fn: Callable[[], T]) -> T:
        from intergrax.runtime.external_operations.llm_external_operation_attempt import (
            LlmExternalOperationAttempt,
            llm_external_operation_identity,
        )

        admission_attempt = self._admit_llm_provider_intent(call_scope="sync")
        ext_op = LlmExternalOperationAttempt(
            store=self._external_operation_store,
            owner=self._external_operation_owner,
            identity=(
                llm_external_operation_identity(
                    provider_slug=self._provider_slug(),
                    model=str(self.model or ""),
                    call_scope="sync",
                )
                if self._external_operation_store is not None
                else None
            ),
            cancellation_port=self._external_operation_cancellation_port,
            status_port=self._external_operation_status_port,
            termination_port=self._external_operation_termination_port,
            capabilities=self._external_operation_capabilities,
            admission_attempt=admission_attempt,
        )
        ext_op.before_physical_call()
        boundary = self._provider_dependency_boundary
        if boundary is None:
            ext_op.mark_running()
            try:
                return fn()
            except BaseException:
                ext_op.mark_failed()
                raise
            else:
                ext_op.mark_succeeded()
        handle = boundary.acquire(self._provider_dependency_admission_request())
        try:
            ext_op.mark_running()
            result = fn()
        except BaseException:
            ext_op.mark_failed()
            boundary.complete_direct(handle)
            raise
        ext_op.mark_succeeded()
        boundary.complete_direct(handle)
        return result

    def _execute(self, fn: Callable[[], T]) -> T:
        """Run a provider SDK call with rate limit, circuit breaker, and optional retry."""
        from intergrax.llm_adapters.governance.quota import check_llm_tenant_quota
        from intergrax.llm_adapters.tracking.context import get_llm_tenant_id

        check_llm_tenant_quota(get_llm_tenant_id())

        def physical_attempt() -> T:
            return self._run_physical_provider_attempt(fn)

        return execute_with_resilience(
            physical_attempt,
            provider=self._provider_slug(),
            config=self.call_config,
            retry_fn=lambda f: call_with_retry(f, config=self.call_config),
            tenant_id=get_llm_tenant_id(),
        )

    def _execute_streaming(self, factory: Callable[[], Iterable[T]]) -> Iterable[T]:
        """Acquire permit per creation attempt; hold through stream consumption."""
        from intergrax.llm_adapters.governance.quota import check_llm_tenant_quota
        from intergrax.llm_adapters.tracking.context import get_llm_tenant_id
        from intergrax.llm_adapters._shared.provider_stream_admission import (
            stream_factory_with_admission,
            stream_with_external_operation_lifecycle,
        )
        from intergrax.runtime.external_operations.llm_external_operation_attempt import (
            LlmExternalOperationAttempt,
            llm_external_operation_identity,
        )

        check_llm_tenant_quota(get_llm_tenant_id())

        def physical_attempt() -> Iterable[T]:
            admission_attempt = self._admit_llm_provider_intent(call_scope="stream")
            ext_op = LlmExternalOperationAttempt(
                store=self._external_operation_store,
                owner=self._external_operation_owner,
                identity=(
                    llm_external_operation_identity(
                        provider_slug=self._provider_slug(),
                        model=str(self.model or ""),
                        call_scope="stream",
                    )
                    if self._external_operation_store is not None
                    else None
                ),
                cancellation_port=self._external_operation_cancellation_port,
                status_port=self._external_operation_status_port,
                termination_port=self._external_operation_termination_port,
                capabilities=self._external_operation_capabilities,
                admission_attempt=admission_attempt,
            )
            ext_op.before_physical_call()
            boundary = self._provider_dependency_boundary
            if boundary is None:
                ext_op.mark_running()
                return stream_with_external_operation_lifecycle(
                    ext_op=ext_op,
                    factory=factory,
                    stream_registry=self._external_operation_stream_registry,
                )
            handle = boundary.acquire(self._provider_dependency_admission_request())
            ext_op.mark_running()
            return stream_with_external_operation_lifecycle(
                ext_op=ext_op,
                factory=lambda: stream_factory_with_admission(
                    boundary=boundary,
                    handle=handle,
                    factory=factory,
                ),
                stream_registry=self._external_operation_stream_registry,
            )

        return execute_with_resilience(
            physical_attempt,
            provider=self._provider_slug(),
            config=self.call_config,
            retry_fn=lambda f: call_with_retry(f, config=self.call_config),
            tenant_id=get_llm_tenant_id(),
        )

    @contextmanager
    def _provider_dependency_attempt(self) -> Iterator[None]:
        """Hold one provider permit for a multi-step physical attempt (e.g. context-managed stream)."""
        boundary = self._provider_dependency_boundary
        if boundary is None:
            yield
            return
        handle = boundary.acquire(self._provider_dependency_admission_request())
        try:
            yield
        except BaseException:
            boundary.complete_direct(handle)
            raise
        boundary.complete_direct(handle)
    
    
    def validate(self) -> None:
        provider = self.provider
        if isinstance(provider, LLMProvider):
            provider = provider.value
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(
                f"{self.__class__.__name__}.provider must be a non-empty string"
            )

    def supports_streaming(self) -> bool:
        """Whether stream_messages is implemented for this adapter."""
        return False

    def supports_structured_output(self) -> bool:
        """Whether generate_structured is natively supported (not prompt-only)."""
        return False

    def supports_vision(self) -> bool:
        """Whether the adapter can consume image attachments in user messages."""
        return False

    def supports_audio_input(self) -> bool:
        """Whether the adapter can consume audio attachments in user messages."""
        return False

    def supports_audio_output(self) -> bool:
        """Whether the adapter can emit audio responses."""
        return False

    @abstractmethod
    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise NotImplementedError


    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError("Streaming is not supported by this adapter.")


    # ---- Tools (optional) ----
    def supports_tools(self) -> bool:
        return False

    def supports_strict_tool_argument_conformance(self) -> bool:
        """Whether provider-enforced strict tool argument schemas are supported."""
        return False

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise NotImplementedError("Tools are not supported by this adapter.")

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError("Tools streaming is not supported by this adapter.")

    # ---- Structured output (optional) ----
    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        raise NotImplementedError("Structured output is not supported by this adapter.")

    # ---- Token counting (base impl; moved from the removed LLMAdapter) ----
    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int:
        return self.estimate_tokens_for_messages(
            messages,
            model_hint=self.model_name_for_token_estimation,
        )

    @property
    @abstractmethod
    def context_window_tokens(self) -> int:
        raise NotImplementedError
    
    
    def _strip_code_fences(self, text: str) -> str:
        """
        Remove wrappers like ```json ... ``` or ``` ... ``` if present.
        Useful when the model wraps a JSON object in Markdown fences.
        """
        if not text:
            return text
        fence_re = r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$"
        m = re.match(fence_re, text, flags=re.DOTALL)
        return m.group(1) if m else text


    def _extract_json_object(self, text: str) -> str:
        """
        Extract the first balanced top-level {...} JSON object.

        Uses brace depth (not rfind), so concatenated objects like ``{...}{...}``
        do not produce invalid slices that trigger JSON "Extra data" errors.
        """
        if not text:
            return ""
        text = self._strip_code_fences(text).strip()
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return ""


    def _model_json_schema(self, model_cls: type) -> Dict[str, Any]:
        """
        Return JSON Schema for the model class (Pydantic v2/v1).
        If unavailable, return a minimal object schema.
        """
        # pydantic v2
        if hasattr(model_cls, "model_json_schema"):
            try:
                return model_cls.model_json_schema()  # type: ignore[attr-defined]
            except Exception:
                pass

        # pydantic v1
        if hasattr(model_cls, "schema"):
            try:
                return model_cls.schema()  # type: ignore[attr-defined]
            except Exception:
                pass

        # fallback
        return {"type": "object"}


    def _validate_with_model(self, model_cls: type, json_str: str):
        """
        Validate and create a model instance from JSON string.

        Supports:
        - Pydantic v2 (model_validate_json/model_validate)
        - Pydantic v1 (parse_raw/parse_obj)
        - Plain dataclasses/classes via **data
        """
        if not json_str or not json_str.strip():
            raise ValueError("Empty JSON content for structured output.")

        data = json.loads(json_str)

        # pydantic v2
        if hasattr(model_cls, "model_validate_json"):
            try:
                return model_cls.model_validate_json(json_str)  # type: ignore[attr-defined]
            except Exception:
                pass

        if hasattr(model_cls, "model_validate"):
            try:
                return model_cls.model_validate(data)  # type: ignore[attr-defined]
            except Exception:
                pass

        # pydantic v1
        if hasattr(model_cls, "parse_raw"):
            try:
                return model_cls.parse_raw(json_str)  # type: ignore[attr-defined]
            except Exception:
                pass

        if hasattr(model_cls, "parse_obj"):
            try:
                return model_cls.parse_obj(data)  # type: ignore[attr-defined]
            except Exception:
                pass

        # fallback (plain class/dataclass with compatible __init__)
        try:
            return model_cls(**data)
        except Exception as e:
            raise ValueError(f"Cannot validate structured output with {model_cls}: {e}")


    def estimate_tokens_for_messages(
        self,
        messages: Sequence[ChatMessage],
        model_hint: Optional[str] = None,
    ) -> int:
        """
        Estimate token count for a list of ChatMessage objects.

        Strategy:
        - If tiktoken is available:
            * use encoding_for_model(model_hint) when model_hint is provided,
            * otherwise fall back to a generic encoding (e.g. cl100k_base).
        - If tiktoken is not available:
            * use a simple character-based heuristic (approx. 4 chars/token).

        This is a generic, model-agnostic estimator designed to be "good enough"
        for budgeting and trimming, not for billing accuracy.
        """
        # Aggregate all message contents into a single string.
        parts: List[str] = []
        for m in messages:
            content = m.content
            if not isinstance(content, str):
                content = str(content)
            parts.append(content)
        joined = "\n".join(parts)
        if not joined:
            return 0

        try:
            if model_hint:
                enc = tiktoken.encoding_for_model(model_hint)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(joined))
        except Exception:
            return max(1, len(joined) // 4)


    def estimate_tokens_for_text(
        self,
        text: str,
        model_hint: Optional[str] = None,
    ) -> int:
        """
        Estimate token count for a plain text string.

        Uses the same strategy as estimate_tokens_for_messages:
        - tiktoken encoding_for_model(model_hint) if available
        - else cl100k_base
        - fallback heuristic if needed
        """
        if not text:
            return 0

        mh = model_hint or self.model_name_for_token_estimation
        try:
            if mh:
                enc = tiktoken.encoding_for_model(mh)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)


class LLMAdapterUsageLog:

    def __init__(self) -> None:
        self._run_stats: Dict[str, LLMRunStats] = {}


    def begin_call(
        self,
        run_id: Optional[str] = None,
        *,
        adapter: Optional[LLMAdapter] = None,
    ) -> LLMCallStats:
        """
        Begin one LLM call (not the whole runtime.run()).

        Returns a per-call context object, safe for nested/parallel use
        because it is local to the caller.

        When ``adapter`` is passed, provider/model are attached for observability metrics.
        """
        from intergrax.runtime.execution.budget.consumption import consume_llm_call

        consume_llm_call()
        rid = run_id or "general"
        if rid not in self._run_stats:
            self._run_stats[rid] = LLMRunStats()
        call = LLMCallStats(run_id=rid)
        if adapter is not None:
            prov = adapter.provider
            if isinstance(prov, LLMProvider):
                call.provider = prov.value
            elif isinstance(prov, str):
                call.provider = prov
            call.model = str(adapter.model or "")
        return call


    def end_call(
        self,
        call: LLMCallStats,
        *,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Finish one LLM call and aggregate into per-run stats.
        """
        from intergrax.runtime.execution.budget.consumption import consume_llm_token_usage

        dt_ms = int((time.perf_counter() - call.t0) * 1000)

        call.input_tokens = int(input_tokens or 0)
        call.output_tokens = int(output_tokens or 0)
        call.total_tokens = call.input_tokens + call.output_tokens
        call.duration_ms = dt_ms

        call.success = bool(success)
        call.error_type = error_type

        consume_llm_token_usage(
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            total_tokens=call.total_tokens,
        )

        st = self._run_stats.get(call.run_id)
        if st is None:
            st = LLMRunStats()
            self._run_stats[call.run_id] = st

        st.calls += 1
        st.input_tokens += call.input_tokens
        st.output_tokens += call.output_tokens
        st.total_tokens += call.total_tokens
        st.duration_ms += call.duration_ms

        if not call.success:
            st.errors += 1

        if call.provider:
            from intergrax.llm_adapters.tracking.metrics import record_llm_call

            record_llm_call(
                provider=call.provider,
                model=call.model or "",
                run_id=call.run_id,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                duration_ms=call.duration_ms,
                success=call.success,
                error_type=call.error_type,
            )

    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats:
        """
        Get aggregated stats for a given run_id.
        Returns None if no stats exist for that run_id.
        """
        rid = run_id or "general"
        st = self._run_stats.get(rid)

        if st is None:
            return LLMRunStats(
                calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                duration_ms=0,
                errors=0,
            )
        
        return LLMRunStats(
            calls=st.calls,
            input_tokens=st.input_tokens,
            output_tokens=st.output_tokens,
            total_tokens=st.total_tokens,
            duration_ms=st.duration_ms,
            errors=st.errors,
        )


    def get_all_run_stats(self) -> Dict[str, LLMRunStats]:
        """
        Get a shallow copy of all aggregated run stats.
        """
        return dict(self._run_stats)


    def reset_run_stats(self, run_id: Optional[str] = None) -> None:
        """
        Reset stats for a specific run_id (or 'general' if None).
        """
        if run_id is None:
            self._run_stats.clear()
            return
        
        rid = run_id or "general"
        self._run_stats.pop(rid, None)



    def export_run_stats_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export aggregated stats to a JSON-serializable dict.
        Helpful for trace / logging.
        """
        rid = run_id or "general"
        st = self._run_stats.get(rid)
        if st is None:
            return {
                "run_id": rid,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "duration_ms": 0,
                "errors": 0,
            }

        return {
            "run_id": rid,
            "calls": int(st.calls),
            "input_tokens": int(st.input_tokens),
            "output_tokens": int(st.output_tokens),
            "total_tokens": int(st.total_tokens),
            "duration_ms": int(st.duration_ms),
            "errors": int(st.errors),
        }
    
    
@dataclass
class LLMCallStats:
    run_id: str
    t0: float = field(default_factory=time.perf_counter)

    # filled on end
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0

    success: bool = True
    error_type: Optional[str] = None

    # observability (set when begin_call(..., adapter=self))
    provider: str = ""
    model: str = ""


@dataclass
class LLMRunStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0