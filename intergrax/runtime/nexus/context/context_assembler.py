# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded context assembly helpers (ContextManager v2, §28)."""

from __future__ import annotations

from typing import Any, Dict, List

from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.context.context_models import (
    ContextProvenance,
    ContextSourceType,
    PriorOutputRecord,
)
from intergrax.runtime.nexus.context.metadata_keys import HANDOFF_STRUCTURED_OUTPUT_PREFIX
from intergrax.runtime.nexus.context.shared_task_context import SharedTaskContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task


def bridge_shared_context_reads(
    shared: SharedTaskContext,
    node: ExecutionNode,
    policy: TaskContextAssemblyOptions,
) -> Dict[str, Any]:
    """Select relevant ``SharedTaskContext`` entries for the current graph node."""
    reads: Dict[str, Any] = {}
    for dep_id in node.depends_on:
        if dep_id in shared.structured_outputs:
            reads[dep_id] = dict(shared.structured_outputs[dep_id])

    if policy.include_shared_handoffs:
        for key, value in shared.structured_outputs.items():
            if key.startswith(HANDOFF_STRUCTURED_OUTPUT_PREFIX):
                reads[key] = dict(value)

    if policy.include_shared_artifacts and shared.artifacts:
        reads["artifacts"] = {
            label: artifact.model_dump(mode="json")
            for label, artifact in shared.artifacts.items()
        }

    return reads


def collect_dependency_records(
    node: ExecutionNode,
    prior_outputs: Dict[str, AgentExecutionResult],
    *,
    policy: TaskContextAssemblyOptions,
    shared_version: int,
) -> tuple[List[PriorOutputRecord], List[str], List[ContextProvenance]]:
    records: List[PriorOutputRecord] = []
    evidence: List[str] = []
    provenance: List[ContextProvenance] = []

    for dep_id in node.depends_on[: policy.max_prior_entries]:
        prior = prior_outputs.get(dep_id)
        if prior is None:
            continue
        summary = prior.summary or ""
        record = PriorOutputRecord(
            node_id=dep_id,
            agent_id=prior.agent_id,
            summary=summary,
            evidence=summary,
            structured_data=dict(prior.structured_data or {}),
            provenance=ContextProvenance(
                source_type=ContextSourceType.DEPENDENCY_OUTPUT,
                source_id=dep_id,
                agent_id=prior.agent_id,
                shared_version=shared_version,
            ),
        )
        records.append(record)
        if summary:
            evidence.append(summary)
        provenance.append(record.provenance)

    return records, evidence, provenance


def provenance_for_shared_reads(
    shared_reads: Dict[str, Any],
    *,
    shared_version: int,
) -> List[ContextProvenance]:
    entries: List[ContextProvenance] = []
    for key in shared_reads:
        if key == "artifacts":
            entries.append(
                ContextProvenance(
                    source_type=ContextSourceType.ARTIFACT,
                    source_id="artifacts",
                    shared_version=shared_version,
                )
            )
            continue
        source_type = (
            ContextSourceType.HANDOFF
            if key.startswith(HANDOFF_STRUCTURED_OUTPUT_PREFIX)
            else ContextSourceType.SHARED_CONTEXT
        )
        payload = shared_reads[key]
        agent_id = payload.get("agent_id") if isinstance(payload, dict) else None
        entries.append(
            ContextProvenance(
                source_type=source_type,
                source_id=key,
                agent_id=str(agent_id) if agent_id else None,
                shared_version=shared_version,
            )
        )
    return entries


def compose_agent_message(
    task: Task,
    *,
    node: ExecutionNode,
    records: List[PriorOutputRecord],
    evidence: List[str],
    shared_reads: Dict[str, Any],
    policy: TaskContextAssemblyOptions,
) -> str:
    base = task.message or ""

    if policy.summary_tier == ContextSummaryTier.STRUCTURED_ONLY:
        return base

    if policy.summary_tier == ContextSummaryTier.MINIMAL:
        if not records and not shared_reads:
            return base
        refs: List[str] = []
        for record in records:
            refs.append(f"{record.node_id}({record.agent_id})")
        for key in shared_reads:
            if key != "artifacts":
                refs.append(key)
        suffix = "Dependencies: " + ", ".join(refs)
        return f"{base}\n\n--- context ---\n{suffix}" if base.strip() else suffix

    prior_text = "\n\n".join(evidence)
    if len(prior_text) > policy.max_prior_chars:
        prior_text = prior_text[: policy.max_prior_chars] + "\n...[truncated]"

    if policy.summary_tier == ContextSummaryTier.SUMMARY_ONLY and not prior_text:
        prior_text = ""

    if not prior_text:
        return base

    return (
        f"{base}\n\n--- prior agent outputs ---\n{prior_text}"
        if base.strip()
        else prior_text
    )


def prior_outputs_dict(records: List[PriorOutputRecord]) -> Dict[str, Any]:
    return {
        record.node_id: {
            "agent_id": record.agent_id,
            "summary": record.summary,
            "evidence": record.evidence,
            "structured_data": record.structured_data,
            "provenance": record.provenance.model_dump(mode="json"),
        }
        for record in records
    }
