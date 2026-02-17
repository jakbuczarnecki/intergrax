# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable, List

from intergrax.runtime.replay.contracts.artifact_store import ArtifactStore
from intergrax.runtime.replay.contracts.run_record_dto import RunRecordDTO
from intergrax.runtime.replay.contracts.run_record_store import RunRecordStore
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
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

    def _load_run_record(self, run_id: str) -> RunRecordDTO:
        return self._run_store.get(run_id)
    
    def _load_trace(self, run_id: str)-> List[TraceEventDTO]:
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

    def _extract_final_answer(self, run_record: RunRecordDTO) -> str | None:
        return run_record.final_answer      
    

    def _reconstruct_steps(
        self,
        trace_events: Iterable[TraceEventDTO],
        artifacts: List[ArtifactRef],
    ) -> List[ReconstructedStep]:

        steps: Dict[str, ReconstructedStep] = {}

        # Map artifact → step
        artifacts_by_step: Dict[str, List[ArtifactRef]] = {}
        for a in artifacts:
            artifacts_by_step.setdefault(a.produced_by_step, []).append(a)

        for ev in trace_events:
            step_id = ev.step_id
            if not step_id:
                continue

            # ---------------- STEP START ----------------
            if ev.event_type == "STEP_STARTED":
                steps[step_id] = ReconstructedStep(
                    step_id=step_id,
                    step_type=ev.payload.get("step_type", "unknown"),
                    started_at=ev.timestamp,
                    finished_at=None,
                    status="running",
                    llm_calls=[],
                    tool_calls=[],
                    artifacts=artifacts_by_step.get(step_id, []),
                )

            # ---------------- STEP FINISH ----------------
            elif ev.event_type == "STEP_FINISHED":
                step = steps.get(step_id)
                if step:
                    step.finished_at = ev.timestamp
                    step.status = ev.payload.get("status", "finished")

            # ---------------- TOOL CALL ----------------
            elif ev.event_type == "TOOL_EXECUTED":
                step = steps.get(step_id)
                if step:
                    step.tool_calls.append(
                        ToolCallInfo(
                            step_id=step_id,
                            tool_id=ev.payload.get("tool_id"),
                            input_payload=ev.payload.get("input"),
                            output_payload=ev.payload.get("output"),
                            success=ev.payload.get("success", True),
                            error=ev.payload.get("error"),
                        )
                    )

            # ---------------- LLM CALL ----------------
            elif ev.event_type == "LLM_CALL":
                step = steps.get(step_id)
                if step:
                    step.llm_calls.append(
                        LLMCallInfo(
                            step_id=step_id,
                            model=ev.payload.get("model"),
                            prompt_tokens=ev.payload.get("prompt_tokens", 0),
                            completion_tokens=ev.payload.get("completion_tokens", 0),
                            total_tokens=ev.payload.get("total_tokens", 0),
                            finish_reason=ev.payload.get("finish_reason"),
                            request_payload=ev.payload.get("request"),
                            response_payload=ev.payload.get("response"),
                        )
                    )

        ordered = sorted(steps.values(), key=lambda s: s.started_at)

        return ordered

