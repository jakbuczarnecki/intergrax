# © Artur Czarnecki. All rights reserved.

"""Typed host ports for ACP runtime session hooks (neutral Agent boundary)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from intergrax.agents.authoring.shared_context_access import AcpSharedContextMetadata
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.shared_context import SharedContextView
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing.contracts import RoutingEvaluation
from intergrax.runtime.kernel.step_kernel import StepKernelContext

if TYPE_CHECKING:
    from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext


class ApplyRuntimeProfileKernelWiringPort(Protocol):
    def __call__(
        self,
        *,
        host: ACPSessionHostContext,
        kernel_ctx_holder: list[StepKernelContext],
        merged: EffectiveAgentRunEnvironment,
        request: AgentRunRequest,
    ) -> LLMAdapter | None: ...


class AttachCatalogExecutionContextPort(Protocol):
    def __call__(
        self,
        step_ctx: AgentStepContext,
        *,
        kernel_ctx: StepKernelContext,
        request: AgentRunRequest,
        contract: AgentContract,
    ) -> None: ...


class CloseCatalogExecutionContextPort(Protocol):
    def __call__(self, step_ctx: AgentStepContext) -> None: ...


class RoutingEvaluationObserverPort(Protocol):
    def __call__(
        self,
        kernel_ctx: StepKernelContext,
        evaluation: RoutingEvaluation,
    ) -> None: ...


class SharedContextLoadPort(Protocol):
    def __call__(self, metadata: AcpSharedContextMetadata) -> SharedContextView | None: ...


class SharedContextPersistPort(Protocol):
    def __call__(self, metadata: AcpSharedContextMetadata, view: SharedContextView) -> None: ...


class SharedContextProjectionPort(Protocol):
    def __call__(self, metadata: AcpSharedContextMetadata, *, task_id: str) -> SharedContextView: ...


@dataclass(frozen=True, slots=True)
class AcpRuntimeSessionHooks:
    """Optional host-injected runtime behavior for direct ``Agent.run()`` sessions."""

    apply_runtime_profile_kernel_wiring: ApplyRuntimeProfileKernelWiringPort | None = None
    attach_acp_catalog_exec_ctx: AttachCatalogExecutionContextPort | None = None
    close_acp_catalog_exec_ctx: CloseCatalogExecutionContextPort | None = None
    on_llm_routing_evaluated: RoutingEvaluationObserverPort | None = None
    load_shared_context_view: SharedContextLoadPort | None = None
    persist_shared_context_view: SharedContextPersistPort | None = None
    view_shared_context_for_task: SharedContextProjectionPort | None = None
