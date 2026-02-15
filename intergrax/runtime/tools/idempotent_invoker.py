from pydantic import BaseModel

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.tools.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult


class IdempotentToolInvoker:
    def __init__(
        self,
        *,
        base_invoker: RuntimeToolInvoker,
        idempotency_store: IdempotencyStore,
    ) -> None:
        self._base_invoker = base_invoker
        self._store = idempotency_store

    def invoke(
        self,
        *,
        state: RuntimeState,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:

        # side_effects decision requires contract lookup
        registry = self._base_invoker._registry  # read-only usage
        reg = registry.get(request.tool_id)
        contract = reg.contract

        if not contract.side_effects or not request.idempotency_key:
            return self._base_invoker.invoke(
                state=state,
                request=request,
            )

        key = request.idempotency_key

        cached = self._store.check(key)
        if cached is not None:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="idempotency_cache_hit",
                level=TraceLevel.INFO,
                message=f"Tool call deduplicated via idempotency (tool_id={request.tool_id}, key={key}).",
            )
            return cached
        
        result = self._base_invoker.invoke(            
            state=state,
            agent_id=agent_id,
            request=request,
        )

        self._store.save(key, result)

        return result
