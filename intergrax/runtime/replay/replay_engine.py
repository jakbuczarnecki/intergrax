# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.runtime.replay.contracts.artifact_store import ArtifactStore
from intergrax.runtime.replay.contracts.run_record_store import RunRecordStore
from intergrax.runtime.replay.contracts.trace_event_store import TraceEventStore
from intergrax.runtime.replay.models import (
    ReconstructedRun,
    ReconstructedStep,
    ArtifactRef,
    ToolCallInfo,
    LLMCallInfo,
)


class ReplayEngine:
    """
    Read-only reconstruction engine.

    Reconstructs execution semantics from:
    - RunRecord
    - TraceEvents
    - Artifact metadata

    Does NOT execute runtime logic.
    """

    def __init__(
        self,
        run_store: RunRecordStore,
        trace_store: TraceEventStore,
        artifact_store: ArtifactStore,
    ) -> None:
        self._run_store = run_store
        self._trace_store = trace_store
        self._artifact_store = artifact_store

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def reconstruct(self, run_id: str) -> ReconstructedRun:
        run_record = self._load_run_record(run_id)
        trace_events = self._load_trace(run_id)
        artifacts = self._load_artifacts(run_id)

        steps = self._reconstruct_steps(trace_events, artifacts)
        tool_calls = self._collect_tool_calls(steps)
        llm_calls = self._collect_llm_calls(steps)
        final_answer = self._extract_final_answer(run_record)

        return ReconstructedRun(
            run_id=run_id,
            steps=steps,
            artifacts=artifacts,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            final_answer=final_answer,
        )

    # -------------------------------------------------
    # LOADERS
    # -------------------------------------------------

    def _load_run_record(self, run_id: str):
        return self._run_store.get(run_id)

    def _load_trace(self, run_id: str):
        return list(self._trace_store.get_events(run_id))

    def _load_artifacts(self, run_id: str) -> List[ArtifactRef]:
        artifacts = self._artifact_store.list_for_run(run_id)

        return [
            ArtifactRef(
                artifact_id=a.artifact_id,
                name=a.name,
                type=a.type,
                produced_by_step=a.produced_by_step,
                metadata=a.metadata,
            )
            for a in artifacts
        ]

    # -------------------------------------------------
    # RECONSTRUCTION STAGES
    # -------------------------------------------------

    def _reconstruct_steps(
        self,
        trace_events,
        artifacts: List[ArtifactRef],
    ) -> List[ReconstructedStep]:
        """
        Build step timeline from trace events.
        """
        raise NotImplementedError

    def _collect_tool_calls(
        self, steps: List[ReconstructedStep]
    ) -> List[ToolCallInfo]:
        calls: List[ToolCallInfo] = []
        for s in steps:
            calls.extend(s.tool_calls)
        return calls

    def _collect_llm_calls(
        self, steps: List[ReconstructedStep]
    ) -> List[LLMCallInfo]:
        calls: List[LLMCallInfo] = []
        for s in steps:
            calls.extend(s.llm_calls)
        return calls

    def _extract_final_answer(self, run_record) -> str | None:
        return getattr(run_record, "final_answer", None)
