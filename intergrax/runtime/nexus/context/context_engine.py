# © Artur Czarnecki. All rights reserved.

"""Default Nexus context engine (CE-3.1)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextFragment,
    ContextProviderContext,
)
from intergrax.context.errors import (
    ContextProviderContractViolationError,
    RequiredContextSourceUnavailableError,
)
from intergrax.context.provider_lifecycle import (
    assembly_provenance_for_fragment,
    canonicalize_fragment,
    collection_outcome,
    is_provider_eligible,
    resolve_bound_context_provider_set,
    safe_provider_failure_reason,
    validate_required_sources_fulfilled,
    validate_required_sources_have_eligible_providers,
)
from intergrax.context.planner import ContextPlanner
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    SessionHistorySnapshot,
)
from intergrax.context.policy.pipeline import (
    ContextPolicyStrategies,
    default_context_policy_strategies,
)
from intergrax.context.policy.authority import filter_fragments_by_authority_contract
from intergrax.context.policy.hard_stages import run_hard_policy_pre_stages
from intergrax.context.policy.invariants import (
    build_fragment_invariant_snapshots,
    validate_policy_pipeline_result,
)
from intergrax.context.policy.scope_isolation import isolate_assembly_scope
from intergrax.context.protocols import ContextPolicyPipeline
from intergrax.context.formatter import (
    DefaultContextFormatter,
    merge_fragment_messages,
    merge_iterative_tool_feedback_messages,
)
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.tracking.context_spans import context_span
from intergrax.llm.messages import compute_model_facing_messages_hash
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.observability.context_counters import get_context_counters
from intergrax.runtime.policy.context_assembly_policy import run_pre_context_policy_gate
from intergrax.runtime.nexus.context.compile_service import compile_chat_messages
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_compiler_models import (
    DegradationStepKind,
)
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight
from intergrax.runtime.nexus.context.context_validator import DefaultContextValidator
from intergrax.runtime.wiring.context_runtime_bridge import resolve_context_optimization_policy
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    ContextAssemblyRuntimeDependencies,
    ensure_context_assembly_runtime,
)
from intergrax.runtime.nexus.context.ucl_orchestration import (
    NexusUCLExecutionError,
    NexusUCLExecutionReason,
    NexusUCLRuntimeDependencies,
    resolve_ucl_context_plan,
)
if TYPE_CHECKING:
    from intergrax.llm.messages import ChatMessage
    from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
    from intergrax.runtime.nexus.config import RuntimeConfig

logger = logging.getLogger("intergrax.context.engine")

_NO_MUTATION_DEGRADATION_STEPS = frozenset(
    {
        (),
        (DegradationStepKind.FULL.value,),
    }
)


def _compile_preserved_planned_context(
    *,
    degradation_steps: tuple[str, ...],
    planned_hash: str,
    compiled_hash: str,
    compiled_budget_tokens: int,
    planned_budget_tokens: int,
) -> bool:
    return (
        degradation_steps in _NO_MUTATION_DEGRADATION_STEPS
        and planned_hash == compiled_hash
        and compiled_budget_tokens == planned_budget_tokens
    )


def _resolve_optimization_policy(
    runtime: ContextAssemblyRuntimeDependencies,
) -> ContextOptimizationPolicy | None:
    return resolve_context_optimization_policy(
        runtime.runtime_config,
        direct_policy=runtime.optimization_policy,
    )


class DefaultNexusContextEngine:
    """Shipped CE engine — provider collect + rank + ContextCompiler budget (CE-3)."""

    def __init__(
        self,
        *,
        engine_id: str = "default",
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
        ranker: DefaultContextRanker | None = None,
        formatter: DefaultContextFormatter | None = None,
        policy_pipeline: ContextPolicyPipeline | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._registry = registry or ContextPluginRegistry()
        self._compiler = compiler or ContextCompiler()
        self._validator = validator or DefaultContextValidator()
        self._ranker = ranker or DefaultContextRanker()
        self._formatter = formatter or DefaultContextFormatter()
        if policy_pipeline is None:
            from intergrax.context.policy.pipeline import ContextCrossSourcePolicyPipeline

            policy_pipeline = ContextCrossSourcePolicyPipeline()
        self._policy_pipeline = policy_pipeline

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def registry(self) -> ContextPluginRegistry:
        return self._registry

    def _resolve_policy_strategies(self) -> ContextPolicyStrategies:
        defaults = default_context_policy_strategies()
        return ContextPolicyStrategies(
            score_normalizer=self._registry.score_normalizer or defaults.score_normalizer,
            semantic_deduper=self._registry.semantic_deduper or defaults.semantic_deduper,
            conflict_resolver=self._registry.conflict_resolver or defaults.conflict_resolver,
            ranker=self._registry.ranker or self._ranker,
            budget_allocator=self._registry.allocator or defaults.budget_allocator,
        )

    async def assemble(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext | None = None,
    ) -> AssembledContext:
        counters = get_context_counters()
        counters.record_assemble(self._engine_id)
        logger.info(
            "assemble scope=%s task_id=%s step_kind=%s",
            request.assembly_scope,
            request.task_id,
            request.step_kind,
        )
        with context_span("context.engine.assemble"):
            return await self._assemble_inner(request, provider_ctx=provider_ctx)

    async def _assemble_inner(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext | None = None,
    ) -> AssembledContext:
        ctx = ensure_context_assembly_runtime(
            provider_ctx or ContextProviderContext(engine_id=self._engine_id)
        )
        runtime = ctx.runtime
        if runtime is None:
            raise ValueError(
                "ContextProviderContext.runtime is required for canonical assembly "
                "(runtime_config and base_messages)"
            )
        runtime_config = runtime.runtime_config
        raw_messages: list[ChatMessage] = list(runtime.base_messages)
        max_output_tokens = runtime.max_output_tokens

        event_bus = runtime.event_bus if isinstance(runtime.event_bus, RuntimeEventBus) else None
        event_ctx = _assembly_event_context(request, runtime)

        pre_gate = run_pre_context_policy_gate(request)
        if not pre_gate.allowed:
            _record_validation_failed(event_bus, event_ctx, pre_gate.errors, stage="pre_context_policy")
            raise ValueError("; ".join(pre_gate.errors))

        collected_fragments: list[ContextFragment] = []
        fragments_excluded: list[tuple[ContextFragment, str]] = []
        provider_outcomes: list = []
        bound_set = resolve_bound_context_provider_set(
            registry=self._registry,
            engine_id=self._engine_id,
            request=request,
            ctx=ctx,
        )
        validate_required_sources_have_eligible_providers(bound_set, request)
        with context_span("context.provider.collect"):
            for bound in bound_set.providers:
                descriptor = bound.descriptor
                provider = bound.provider
                if not is_provider_eligible(descriptor, request):
                    provider_outcomes.append(
                        collection_outcome(descriptor=descriptor, status="skipped"),
                    )
                    continue
                try:
                    fragments = await provider.collect(request, ctx)
                except RequiredContextSourceUnavailableError:
                    raise
                except ContextProviderContractViolationError:
                    raise
                except Exception as exc:
                    reason = safe_provider_failure_reason(exc)
                    provider_outcomes.append(
                        collection_outcome(
                            descriptor=descriptor,
                            status="failed",
                            failure_reason=reason,
                            reason_code="provider.collection_failed",
                        ),
                    )
                    if event_bus is not None:
                        from intergrax.runtime.events.context_skill_recording import (
                            record_context_candidate_dropped,
                        )

                        record_context_candidate_dropped(
                            event_bus,
                            provider_id=descriptor.provider_id,
                            provider_version=descriptor.provider_version,
                            drop_reason="provider.collection_failed",
                            engine_id=self._engine_id,
                            **event_ctx,
                        )
                    continue
                canonical_fragments: list[ContextFragment] = []
                contract_violations = 0
                for fragment in fragments or []:
                    try:
                        canonical_fragments.append(
                            canonicalize_fragment(fragment, descriptor=descriptor),
                        )
                    except ContextProviderContractViolationError:
                        contract_violations += 1
                if canonical_fragments:
                    counters = get_context_counters()
                    counters.candidate_collected_total += len(canonical_fragments)
                    collected_fragments.extend(canonical_fragments)
                    provider_outcomes.append(
                        collection_outcome(
                            descriptor=descriptor,
                            status="success",
                            fragment_count=len(canonical_fragments),
                        ),
                    )
                    if event_bus is not None:
                        from intergrax.runtime.events.context_skill_recording import (
                            record_context_candidate_collected,
                        )

                        record_context_candidate_collected(
                            event_bus,
                            provider_id=descriptor.provider_id,
                            provider_version=descriptor.provider_version,
                            fragment_count=len(canonical_fragments),
                            engine_id=self._engine_id,
                            **event_ctx,
                        )
                elif contract_violations:
                    provider_outcomes.append(
                        collection_outcome(
                            descriptor=descriptor,
                            status="failed",
                            failure_reason="ContextProviderContractViolationError",
                            reason_code="provider.contract_violation",
                        ),
                    )
                    if event_bus is not None:
                        from intergrax.runtime.events.context_skill_recording import (
                            record_context_candidate_dropped,
                        )

                        record_context_candidate_dropped(
                            event_bus,
                            provider_id=descriptor.provider_id,
                            provider_version=descriptor.provider_version,
                            drop_reason="provider.contract_violation",
                            engine_id=self._engine_id,
                            **event_ctx,
                        )
                else:
                    provider_outcomes.append(
                        collection_outcome(descriptor=descriptor, status="success", fragment_count=0),
                    )

        validate_required_sources_fulfilled(
            collected_fragments=collected_fragments,
            provider_outcomes=provider_outcomes,
            request=request,
        )

        descriptors_by_id = {
            bound.descriptor.provider_id: bound.descriptor for bound in bound_set.providers
        }
        collected_fragments, scope_excluded = isolate_assembly_scope(collected_fragments, request)
        fragments_excluded.extend(scope_excluded)

        collected_fragments, hard_excluded, hard_decisions = run_hard_policy_pre_stages(
            collected_fragments,
        )
        fragments_excluded.extend(hard_excluded)

        invariant_snapshots = build_fragment_invariant_snapshots(collected_fragments)

        policy_strategies = self._resolve_policy_strategies()
        policy_result = self._policy_pipeline.execute(
            collected_fragments,
            request,
            strategies=policy_strategies,
        )
        validate_policy_pipeline_result(
            invariant_snapshots,
            policy_result,
            pipeline_id=self._policy_pipeline.pipeline_id,
        )
        collected_fragments = list(policy_result.fragments)
        fragments_excluded.extend(policy_result.excluded)
        policy_decisions = (*hard_decisions, *policy_result.decisions)
        collected_fragments, authority_excluded = filter_fragments_by_authority_contract(
            collected_fragments,
            descriptors_by_id=descriptors_by_id,
        )
        fragments_excluded.extend(authority_excluded)
        collected_fragments, post_scope_excluded = isolate_assembly_scope(
            collected_fragments,
            request,
        )
        fragments_excluded.extend(post_scope_excluded)
        if policy_result.excluded:
            counters = get_context_counters()
            counters.candidate_dropped_total += len(policy_result.excluded)

        _record_fragment_exclusion_drop_events(
            event_bus,
            fragments_excluded,
            engine_id=self._engine_id,
            event_ctx=event_ctx,
        )

        post_gate = run_pre_context_policy_gate(request, collected=tuple(collected_fragments))
        if not post_gate.allowed:
            get_context_counters().validation_failed_total += 1
            _record_validation_failed(event_bus, event_ctx, post_gate.errors, stage="post_collect_policy")
            raise ValueError("; ".join(post_gate.errors))

        ranked_fragments: list[ContextFragment] = list(collected_fragments)

        formatter = self._registry.formatter or self._formatter
        fragment_messages = formatter.format(ranked_fragments, request)
        if request.assembly_scope == "acp_step" and request.step_kind == "tool_call":
            messages_for_compile = merge_iterative_tool_feedback_messages(
                raw_messages,
                fragment_messages,
            )
        else:
            messages_for_compile = merge_fragment_messages(raw_messages, fragment_messages)

        resolved_budget = self._compiler.resolve_global_input_budget(
            runtime_config,
            max_output_tokens=max_output_tokens,
        )
        session_history = await _load_session_history_snapshot(request, ctx)
        optimization_policy = _resolve_optimization_policy(runtime)
        planner = ContextPlanner(count_tokens=self._compiler.count_tokens)
        context_plan = planner.plan(
            request,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            ranked_fragments=ranked_fragments,
            session_history=session_history,
            resolved_global_budget_tokens=resolved_budget,
            optimization_policy=optimization_policy,
            model_family=(
                runtime_config.llm_adapter.model
                if runtime_config.llm_adapter is not None
                else None
            ),
        )

        ucl_runtime = runtime.ucl_runtime
        if ucl_runtime is not None and not isinstance(ucl_runtime, NexusUCLRuntimeDependencies):
            raise ValueError("ContextAssemblyRuntimeDependencies.ucl_runtime must be NexusUCLRuntimeDependencies")

        try:
            ucl_resolution = await resolve_ucl_context_plan(
                request=request,
                context_plan=context_plan,
                optimization_policy=optimization_policy,
                session_history=session_history,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                ranked_fragments=ranked_fragments,
                runtime=ucl_runtime,
                count_tokens=self._compiler.count_tokens,
            )
        except NexusUCLExecutionError as exc:
            _record_validation_failed(event_bus, event_ctx, (str(exc),), stage="ucl_resolution")
            raise ValueError(str(exc)) from exc

        planned_hash = compute_model_facing_messages_hash(ucl_resolution.messages)
        compile_result = compile_chat_messages(
            list(ucl_resolution.messages),
            runtime_config,
            compiler=self._compiler,
            max_output_tokens=max_output_tokens,
            run_preflight=False,
        )
        compiled_hash = compute_model_facing_messages_hash(compile_result.messages)
        if not _compile_preserved_planned_context(
            degradation_steps=compile_result.degradation_steps,
            planned_hash=planned_hash,
            compiled_hash=compiled_hash,
            compiled_budget_tokens=compile_result.budget_tokens,
            planned_budget_tokens=context_plan.resolved_global_budget_tokens,
        ):
            raise ValueError(NexusUCLExecutionReason.FINAL_COMPILE_MUTATED_PLAN.value)

        messages = tuple(compile_result.messages)
        fragments_included = ucl_resolution.fragments_included
        fragments_excluded = tuple(fragments_excluded) + ucl_resolution.fragments_excluded
        provenance = tuple(
            assembly_provenance_for_fragment(fragment)
            for fragment in fragments_included
        )

        assembled = AssembledContext(
            messages=messages,
            fragments_included=fragments_included,
            fragments_excluded=fragments_excluded,
            provenance=provenance,
            total_tokens=compile_result.total_tokens,
            budget_tokens=compile_result.budget_tokens,
            degradation_steps=compile_result.degradation_steps,
            context_plan=context_plan,
            provider_outcomes=tuple(provider_outcomes),
            provider_set_snapshot=bound_set.snapshot,
            policy_decisions=policy_decisions,
            policy_semantic_dedup=policy_result.semantic_dedup_decisions,
            policy_conflicts=policy_result.conflict_decisions,
        )

        validation = self._validator.validate(
            assembled,
            request,
            runtime_config=runtime_config,
            max_output_tokens=max_output_tokens,
        )
        if not validation.valid:
            get_context_counters().validation_failed_total += 1
            _record_validation_failed(event_bus, event_ctx, validation.errors, stage="assembled_validation")
            raise ValueError("; ".join(validation.errors))

        verify_context_preflight(
            list(messages),
            runtime_config.llm_adapter,
            max_output_tokens=max_output_tokens,
            count_tokens=self._compiler.count_tokens,
        )

        if event_bus is not None:
            from intergrax.runtime.events.context_skill_recording import (
                record_context_assembled_from_engine,
            )

            record_context_assembled_from_engine(
                event_bus,
                assembled=assembled,
                task_id=request.task_id,
                run_id=request.run_id,
                node_id=str(event_ctx.get("node_id") or request.graph_node_id or ""),
                agent_id=event_ctx.get("agent_id") if isinstance(event_ctx.get("agent_id"), str) else None,
                engine_id=self._engine_id,
                step_kind=request.step_kind,
            )

        return assembled


async def _load_session_history_snapshot(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> SessionHistorySnapshot | None:
    provider = HandleSessionHistoryProvider()
    return await provider.load_snapshot(request, ctx)


def _record_fragment_exclusion_drop_events(
    event_bus: RuntimeEventBus | None,
    exclusions: list[tuple[ContextFragment, str]],
    *,
    engine_id: str,
    event_ctx: dict[str, str | None],
) -> None:
    if event_bus is None or not exclusions:
        return
    from intergrax.runtime.events.context_skill_recording import (
        record_context_candidate_dropped,
    )

    for fragment, drop_reason in exclusions:
        provenance = fragment.provider_provenance
        record_context_candidate_dropped(
            event_bus,
            provider_id=provenance.provider_id if provenance is not None else "unknown",
            provider_version=provenance.provider_version if provenance is not None else "",
            drop_reason=drop_reason,
            engine_id=engine_id,
            **event_ctx,
        )


def _assembly_event_context(
    request: ContextAssemblyRequest,
    runtime: ContextAssemblyRuntimeDependencies,
) -> dict[str, str | None]:
    node_id = runtime.node_id
    agent_id = runtime.agent_id
    return {
        "task_id": request.task_id,
        "run_id": request.run_id,
        "node_id": node_id if isinstance(node_id, str) else (request.graph_node_id or ""),
        "agent_id": agent_id if isinstance(agent_id, str) else None,
        "correlation_id": request.trace_id or request.task_id,
    }


def _record_validation_failed(
    event_bus: RuntimeEventBus | None,
    event_ctx: dict[str, str | None],
    errors: tuple[str, ...] | list[str],
    *,
    stage: str,
) -> None:
    if event_bus is None or not errors:
        return
    from intergrax.runtime.events.context_skill_recording import (
        record_context_validation_failed,
    )

    record_context_validation_failed(
        event_bus,
        errors=tuple(errors),
        stage=stage,
        **event_ctx,
    )
