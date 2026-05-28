# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.context_assembly import (
    ContextAssemblyMetadataKey,
    ContextSummaryTier,
    TaskContextAssemblyOptions,
    context_assembly_options_from_metadata,
)
from intergrax.runtime.nexus.context.context_models import ContextSourceType
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.nexus.context.metadata_keys import (
    AgentContextMetadataKey,
    HANDOFF_STRUCTURED_OUTPUT_PREFIX,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions


def _task(*, context: TaskContextAssemblyOptions | None = None) -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="analyze vendors",
        task_id="task_ctx_v2",
        options=TaskExecutionOptions(
            context=context or TaskContextAssemblyOptions(),
        ),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_v2_records_provenance_on_bundle():
    manager = ContextManager()
    task = _task()
    node = ExecutionNode(node_id="n2", agent_id="agent_b", capability="cap.b", depends_on=["n1"])
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus

    prior = {
        "n1": AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="prior evidence",
        )
    }
    manager.record_node_output(task, ExecutionNode(node_id="n1", agent_id="agent_a"), prior["n1"])

    bundle = manager.build_agent_context(task, node, prior)

    assert bundle.schema_version == "agent_context_bundle.v2"
    assert len(bundle.provenance) >= 2
    assert bundle.prior_records[0].provenance.source_type == ContextSourceType.DEPENDENCY_OUTPUT
    assert bundle.prior_records[0].node_id == "n1"
    assert "n1" in bundle.shared_reads


@pytest.mark.unit
@pytest.mark.gate
def test_context_summary_tier_structured_only_omits_prior_narrative():
    manager = ContextManager()
    task = _task(
        context=TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.STRUCTURED_ONLY),
    )
    node = ExecutionNode(node_id="n2", agent_id="agent_b", depends_on=["n1"])
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus

    prior = {
        "n1": AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="should not appear in message",
        )
    }

    bundle = manager.build_agent_context(task, node, prior)

    assert bundle.summary_tier == ContextSummaryTier.STRUCTURED_ONLY
    assert "should not appear" not in bundle.message
    assert bundle.message == "analyze vendors"
    assert bundle.prior_outputs["n1"]["summary"] == "should not appear in message"


@pytest.mark.unit
@pytest.mark.gate
def test_context_summary_tier_minimal_shows_dependency_refs_only():
    manager = ContextManager()
    task = _task(context=TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.MINIMAL))
    node = ExecutionNode(node_id="n2", agent_id="agent_b", depends_on=["n1"])
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus

    prior = {
        "n1": AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="hidden narrative",
        )
    }

    bundle = manager.build_agent_context(task, node, prior)

    assert "hidden narrative" not in bundle.message
    assert "n1(agent_a)" in bundle.message


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_truncates_prior_text_by_policy():
    manager = ContextManager(default_policy=TaskContextAssemblyOptions(max_prior_chars=64))
    task = _task()
    node = ExecutionNode(node_id="n2", agent_id="agent_b", depends_on=["n1"])
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus

    prior = {
        "n1": AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="x" * 200,
        )
    }

    bundle = manager.build_agent_context(task, node, prior)

    assert "...[truncated]" in bundle.message


@pytest.mark.unit
@pytest.mark.gate
def test_apply_to_task_injects_context_v2_metadata():
    manager = ContextManager()
    task = _task()
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus

    manager.record_node_output(
        task,
        ExecutionNode(node_id="n1", agent_id="agent_a"),
        AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="evidence",
        ),
    )
    node = ExecutionNode(node_id="n2", agent_id="agent_b", depends_on=["n1"])
    prior = {
        "n1": AgentExecutionResult(
            agent_id="agent_a",
            run_id="task_ctx_v2",
            status=AgentExecutionStatus.COMPLETED,
            summary="evidence",
        )
    }
    bundle = manager.build_agent_context(task, node, prior)
    node_task = manager.apply_to_task(task, bundle)

    assert node_task.metadata[ContextAssemblyMetadataKey.SUMMARY_TIER] == ContextSummaryTier.FULL.value
    assert node_task.metadata[AgentContextMetadataKey.SHARED_CONTEXT_READS]["n1"]["summary"] == "evidence"
    assert len(node_task.metadata[AgentContextMetadataKey.CONTEXT_PROVENANCE]) >= 1
    assert (
        node_task.metadata[AgentContextMetadataKey.AGENT_CONTEXT_BUNDLE]["schema_version"]
        == "agent_context_bundle.v2"
    )
    assert node_task.options.context.summary_tier == ContextSummaryTier.FULL


@pytest.mark.unit
@pytest.mark.gate
def test_bridge_shared_handoff_reads():
    manager = ContextManager()
    task = _task()
    handoff_key = f"{HANDOFF_STRUCTURED_OUTPUT_PREFIX}ho_1"
    manager.put_structured_output(
        task,
        key=handoff_key,
        payload={"from_agent_id": "agent_a", "reason": "delegate"},
    )
    node = ExecutionNode(node_id="n2", agent_id="agent_b", depends_on=[])

    bundle = manager.build_agent_context(task, node, {})

    assert handoff_key in bundle.shared_reads
    assert any(p.source_type == ContextSourceType.HANDOFF for p in bundle.provenance)


@pytest.mark.unit
@pytest.mark.gate
def test_context_assembly_options_from_metadata_parses_flat_keys():
    options = context_assembly_options_from_metadata(
        {
            ContextAssemblyMetadataKey.SUMMARY_TIER: "structured_only",
            ContextAssemblyMetadataKey.MAX_PRIOR_CHARS: 800,
        }
    )
    assert options.summary_tier == ContextSummaryTier.STRUCTURED_ONLY
    assert options.max_prior_chars == 800


@pytest.mark.unit
@pytest.mark.gate
def test_task_metadata_bridge_hydrates_context_options():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        metadata={
            "context_summary_tier": "minimal",
            "context_max_prior_chars": 1500,
        },
    )
    assert task.options.context.summary_tier == ContextSummaryTier.MINIMAL
    assert task.options.context.max_prior_chars == 1500

    task.sync_metadata()
    assert task.metadata[ContextAssemblyMetadataKey.POLICY]["summary_tier"] == "minimal"
    assert task.metadata[ContextAssemblyMetadataKey.MAX_PRIOR_CHARS] == 1500
