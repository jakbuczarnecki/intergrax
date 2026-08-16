# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.artifacts.models import ArtifactRef
from intergrax.runtime.nexus.context.context_assembler import (
    bridge_shared_context_reads,
    collect_dependency_records,
    compose_agent_message,
    prior_outputs_dict,
    provenance_for_shared_reads,
)
from intergrax.contracts.context_assembly import (
    ContextAssemblyMetadataKey,
    ContextSummaryTier,
    TaskContextAssemblyOptions,
)
from intergrax.runtime.nexus.context.context_models import (
    ContextProvenance,
    ContextSourceType,
    PriorOutputRecord,
)
from intergrax.runtime.nexus.context.metadata_keys import AgentContextMetadataKey
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey
from intergrax.runtime.nexus.context.shared_task_context import (
    SharedArtifactEntry,
    SharedContextConflictError,
    SharedTaskContext,
    get_or_create_shared_task_context,
    load_shared_task_context,
    save_shared_task_context,
)
from intergrax.contracts.execution_identity import ActiveExecutionIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.events.context_skill_recording import record_context_assembly
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.context_budget import (
    ContextBudgetPolicy,
    ContextTrimResult,
    trim_message_to_budget,
)
from intergrax.runtime.nexus.context.graph_assembly import (
    build_graph_assembly_request,
    compatibility_text_from_assembled_messages,
    graph_messages_from_text,
)
from intergrax.llm.messages import (
    build_model_input_messages_envelope,
    compute_model_facing_messages_hash,
    MODEL_INPUT_MESSAGES_METADATA_KEY,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.task.task import Task

if TYPE_CHECKING:
    from intergrax.context.orchestrator import ContextOrchestrator
    from intergrax.context.protocols import ContextEngine
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.middleware.pipeline import MiddlewarePipeline


class AgentContextBundle(BaseModel):
    """Bounded context passed to an agent for a graph node (§28, §42.14)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    prior_outputs: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    shared_context: Optional[SharedTaskContext] = None
    shared_reads: Dict[str, Any] = Field(default_factory=dict)
    prior_records: List[PriorOutputRecord] = Field(default_factory=list)
    provenance: List[ContextProvenance] = Field(default_factory=list)
    summary_tier: ContextSummaryTier = ContextSummaryTier.FULL
    schema_version: str = "agent_context_bundle.v2"
    model_input_messages: tuple[ChatMessage, ...] = Field(
        default_factory=tuple,
        exclude=True,
        repr=False,
    )


class ContextManager:
    """
    Builds per-node agent context from task state, shared context, and prior graph outputs.

    v2 adds provenance tracking, summary tiers, and explicit shared-context reads (§28).
    """

    def __init__(
        self,
        *,
        max_prior_chars: int = 4000,
        default_policy: Optional[TaskContextAssemblyOptions] = None,
        budget_policy: Optional[ContextBudgetPolicy] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        context_engine: Optional["ContextEngine"] = None,
        context_orchestrator: Optional["ContextOrchestrator"] = None,
        llm_adapter: Optional["LLMAdapter"] = None,
        middleware: Optional["MiddlewarePipeline"] = None,
        execution_identity: ActiveExecutionIdentity | None = None,
    ) -> None:
        self._default_policy = default_policy or TaskContextAssemblyOptions(
            max_prior_chars=max_prior_chars,
        )
        self._budget_policy = budget_policy or ContextBudgetPolicy(
            max_chars=max(max_prior_chars, 4000),
        )
        self._event_bus = event_bus
        self._context_engine = context_engine
        self._context_orchestrator = context_orchestrator
        self._llm_adapter = llm_adapter
        self._middleware = middleware
        self._execution_identity = execution_identity or ActiveExecutionIdentity()

    def _active_run_id(self) -> str:
        run_id, _ = self._execution_identity.require()
        return run_id

    def bind_middleware(self, middleware: "MiddlewarePipeline") -> None:
        """Attach hook pipeline for graph context assembly (CE-HOOKS-GRAPH)."""
        self._middleware = middleware

    def use_execution_identity(self, execution_identity: ActiveExecutionIdentity) -> None:
        """Share the Nexus loop execution identity for context assembly events."""
        self._execution_identity = execution_identity

    @property
    def context_engine(self) -> Optional["ContextEngine"]:
        return self._context_engine

    @property
    def llm_adapter(self) -> Optional["LLMAdapter"]:
        return self._llm_adapter

    def get_shared_context(self, task: Task) -> Optional[SharedTaskContext]:
        return load_shared_task_context(task)

    def ensure_shared_context(self, task: Task) -> SharedTaskContext:
        shared = get_or_create_shared_task_context(task, task_id=task.task_id)
        save_shared_task_context(task, shared)
        return shared

    def resolve_policy(self, task: Task) -> TaskContextAssemblyOptions:
        policy = task.options.context
        factory_default = TaskContextAssemblyOptions()
        if (
            policy.max_prior_chars == factory_default.max_prior_chars
            and self._default_policy.max_prior_chars != factory_default.max_prior_chars
        ):
            return policy.model_copy(update={"max_prior_chars": self._default_policy.max_prior_chars})
        return policy

    async def build_agent_context_async(
        self,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        policy: Optional[TaskContextAssemblyOptions] = None,
    ) -> AgentContextBundle:
        """Graph context assembly — uses ``ContextEngine.assemble`` when wired (CE-3.7)."""
        engine = self._context_engine
        if (
            engine is not None
            and node.delegation is not None
            and engine.engine_id != "explore_child"
        ):
            from intergrax.runtime.nexus.context.preset_engines import ExploreChildContextEngine

            engine = ExploreChildContextEngine(registry=engine.registry)

        use_engine = engine is not None and self._llm_adapter is not None
        bundle = self._build_agent_context_core(
            task,
            node,
            prior_outputs,
            policy=policy,
            emit_events=not use_engine,
        )
        if not use_engine:
            return bundle

        hook_base = HookContext(
            task_id=task.task_id,
            run_id=self._active_run_id(),
            node_id=node.node_id,
            agent_id=node.agent_id,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            runtime_state={"message": task.message, "capability": node.capability or ""},
        )
        if self._middleware is not None:
            await self._middleware.run_before(HookPoint.BEFORE_CONTEXT_BUILD, hook_base)

        from intergrax.context.contracts import ContextProviderContext
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.nexus.context.provider_handles import build_graph_provider_handles

        resolved_policy = policy or self.resolve_policy(task)
        shared = self.ensure_shared_context(task)
        prior_records, _, _ = collect_dependency_records(
            node,
            prior_outputs,
            policy=resolved_policy,
            shared_version=shared.version,
        )
        shared_reads = bridge_shared_context_reads(shared, node, resolved_policy)
        request = build_graph_assembly_request(
            task,
            node,
            policy=resolved_policy,
            budget_policy=self._budget_policy,
        )
        runtime_config = RuntimeConfig(llm_adapter=self._llm_adapter, production_mode=False)
        provider_ctx = ContextProviderContext(
            engine_id=engine.engine_id,
            handles=build_graph_provider_handles(
                task,
                runtime_config=runtime_config,
                messages=graph_messages_from_text(task.message or ""),
                event_bus=self._event_bus,
                node_id=node.node_id,
                agent_id=node.agent_id,
                engine_id=engine.engine_id,
                prior_output_records=prior_records,
                shared_context_reads=shared_reads,
            ),
        )
        if self._context_orchestrator is not None and engine.engine_id == "codebase":
            assembled = await self._context_orchestrator.assemble_with_hops(
                request,
                provider_ctx=provider_ctx,
            )
        else:
            assembled = await engine.assemble(request, provider_ctx=provider_ctx)
        final_message = compatibility_text_from_assembled_messages(assembled.messages)
        original_chars = len(bundle.message)
        final_chars = len(final_message)
        trim = ContextTrimResult(
            message=final_message,
            trimmed=final_chars < original_chars or bool(assembled.degradation_steps),
            original_chars=original_chars,
            final_chars=final_chars,
        )
        bundle_metadata = dict(bundle.metadata)
        bundle_metadata.update(
            {
                "context_trimmed": trim.trimmed,
                "context_original_chars": trim.original_chars,
                "context_final_chars": trim.final_chars,
                "engine_id": engine.engine_id,
                "degradation_steps": list(assembled.degradation_steps),
                "model_input_message_count": len(assembled.messages),
                "model_input_messages_hash": compute_model_facing_messages_hash(assembled.messages),
            }
        )
        if self._event_bus is not None:
            record_context_assembly(
                self._event_bus,
                task_id=task.task_id,
                run_id=self._active_run_id(),
                node_id=node.node_id,
                agent_id=node.agent_id,
                trim=trim,
                metadata={
                    **bundle_metadata,
                    "tenant_id": task.tenant_id,
                },
                engine_id=engine.engine_id,
                step_kind=node.capability,
                emit_assembled=False,
            )
        if self._middleware is not None:
            await self._middleware.run_after(
                HookPoint.AFTER_CONTEXT_BUILD,
                hook_base.model_copy(update={"phase": ExecutionPhase.CONTEXT_BUILDING}),
            )
        return bundle.model_copy(
            update={
                "message": trim.message,
                "metadata": bundle_metadata,
                "model_input_messages": assembled.messages,
            }
        )

    def build_agent_context(
        self,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        policy: Optional[TaskContextAssemblyOptions] = None,
    ) -> AgentContextBundle:
        """Sync graph context assembly (legacy trim path when engine is not wired)."""
        return self._build_agent_context_core(task, node, prior_outputs, policy=policy)

    def _build_agent_context_core(
        self,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        policy: Optional[TaskContextAssemblyOptions] = None,
        emit_events: bool = True,
    ) -> AgentContextBundle:
        shared = self.ensure_shared_context(task)
        resolved_policy = policy or self.resolve_policy(task)

        records, evidence, provenance = collect_dependency_records(
            node,
            prior_outputs,
            policy=resolved_policy,
            shared_version=shared.version,
        )
        shared_reads = bridge_shared_context_reads(shared, node, resolved_policy)
        provenance.extend(
            provenance_for_shared_reads(shared_reads, shared_version=shared.version)
        )

        if task.message.strip():
            provenance.insert(
                0,
                ContextProvenance(
                    source_type=ContextSourceType.TASK_MESSAGE,
                    source_id=task.task_id,
                ),
            )

        structured = dict(shared.structured_outputs)
        structured.update(prior_outputs_dict(records))

        message = compose_agent_message(
            task,
            node=node,
            records=records,
            evidence=evidence,
            shared_reads=shared_reads,
            policy=resolved_policy,
        )
        trim: ContextTrimResult = trim_message_to_budget(message, self._budget_policy)

        bundle_metadata = {
            "node_id": node.node_id,
            "capability": node.capability,
            "depends_on": list(node.depends_on),
            "shared_context_version": shared.version,
            "summary_tier": resolved_policy.summary_tier.value,
            "shared_read_keys": sorted(shared_reads.keys()),
            "context_trimmed": trim.trimmed,
            "context_original_chars": trim.original_chars,
            "context_final_chars": trim.final_chars,
        }
        if node.delegation is not None:
            bundle_metadata["delegation_memory_namespace"] = node.delegation.resolved_memory_namespace(
                task_id=task.task_id,
                node_id=node.node_id,
            )

        if emit_events and self._event_bus is not None:
            record_context_assembly(
                self._event_bus,
                task_id=task.task_id,
                run_id=self._active_run_id(),
                node_id=node.node_id,
                agent_id=node.agent_id,
                trim=trim,
                metadata={
                    **bundle_metadata,
                    "tenant_id": task.tenant_id,
                },
                step_kind=node.capability,
            )

        return AgentContextBundle(
            message=trim.message,
            prior_outputs=structured,
            evidence=evidence,
            shared_context=shared,
            shared_reads=shared_reads,
            prior_records=records,
            provenance=provenance,
            summary_tier=resolved_policy.summary_tier,
            metadata=bundle_metadata,
        )

    def apply_to_task(self, task: Task, bundle: AgentContextBundle) -> Task:
        """Return task copy with bounded message, provenance, and shared context for agent execution."""
        shared = bundle.shared_context or self.ensure_shared_context(task)
        task_metadata = {
            **task.metadata,
            AgentContextMetadataKey.AGENT_CONTEXT: bundle.metadata,
            AgentContextMetadataKey.AGENT_CONTEXT_BUNDLE: bundle.model_dump(mode="json"),
            AgentContextMetadataKey.CONTEXT_PROVENANCE: [
                entry.model_dump(mode="json") for entry in bundle.provenance
            ],
            ContextAssemblyMetadataKey.SUMMARY_TIER: bundle.summary_tier.value,
            AgentContextMetadataKey.SHARED_CONTEXT_READS: dict(bundle.shared_reads),
            AgentContextMetadataKey.PRIOR_AGENT_OUTPUTS: bundle.prior_outputs,
            TaskMetadataKey.SHARED_TASK_CONTEXT: shared.model_dump(mode="json"),
        }
        if bundle.model_input_messages:
            task_metadata[MODEL_INPUT_MESSAGES_METADATA_KEY] = build_model_input_messages_envelope(
                bundle.model_input_messages,
            )
        return task.model_copy(
            update={
                "message": bundle.message,
                "metadata": task_metadata,
            }
        )

    def record_node_output(
        self,
        task: Task,
        node: ExecutionNode,
        execution: AgentExecutionResult,
    ) -> SharedTaskContext:
        """Merge a completed node result into the task-level shared context."""
        shared = self.ensure_shared_context(task)
        shared.structured_outputs[node.node_id] = {
            "agent_id": execution.agent_id,
            "node_id": node.node_id,
            "capability": node.capability,
            "summary": execution.summary,
            "structured_data": dict(execution.structured_data or {}),
            "status": execution.status.value,
            "provenance": {
                "source_type": ContextSourceType.DEPENDENCY_OUTPUT.value,
                "source_id": node.node_id,
                "agent_id": execution.agent_id,
                "shared_version": shared.version,
            },
        }
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared

    def put_structured_output(
        self,
        task: Task,
        *,
        key: str,
        payload: Dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> SharedTaskContext:
        """Explicit shared-context write (Tier-1 API for orchestrators / handoff)."""
        shared = self.ensure_shared_context(task)
        if expected_version is not None and shared.version != expected_version:
            raise SharedContextConflictError(
                f"shared context version mismatch: expected {expected_version}, got {shared.version}"
            )
        shared.structured_outputs[key.strip()] = dict(payload)
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared

    def put_artifact(
        self,
        task: Task,
        *,
        label: str,
        artifact: ArtifactRef,
        expected_version: Optional[int] = None,
    ) -> SharedTaskContext:
        shared = self.ensure_shared_context(task)
        if expected_version is not None and shared.version != expected_version:
            raise SharedContextConflictError(
                f"shared context version mismatch: expected {expected_version}, got {shared.version}"
            )
        shared.artifacts[label.strip()] = SharedArtifactEntry.from_ref(artifact)
        shared.version += 1
        save_shared_task_context(task, shared)
        return shared
