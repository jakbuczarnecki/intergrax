# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------- LLM CALL INFO ----------


@dataclass(slots=True)
class LLMCallInfo:
    step_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    request_payload: Any | None = None
    response_payload: Any | None = None


# ---------- TOOL CALL INFO ----------


@dataclass(slots=True)
class ToolCallInfo:
    step_id: str
    tool_id: str
    input_payload: Any
    output_payload: Any
    success: bool
    error: str | None = None


# ---------- REPLAY ARTIFACT REF ----------
# Replay-plane metadata view of an artifact. Not the shared reference contract
# (``intergrax.contracts.artifact_ref``); the former name ``ArtifactRef`` still resolves below.


@dataclass(slots=True)
class ReplayArtifactRef:
    artifact_id: str
    name: str
    type: str
    produced_by_step: str
    metadata: dict | None = None


# ---------- RECONSTRUCTED STEP ----------


@dataclass(slots=True)
class ReconstructedStep:
    step_id: str
    step_type: str
    started_at: float
    finished_at: float | None
    status: str
    llm_calls: list[LLMCallInfo]
    tool_calls: list[ToolCallInfo]
    artifacts: list[ReplayArtifactRef]


# ---------- FINAL RUN STRUCTURE ----------


@dataclass(slots=True)
class ReconstructedRun:
    run_id: str
    steps: list[ReconstructedStep]
    artifacts: list[ReplayArtifactRef]
    tool_calls: list[ToolCallInfo]
    llm_calls: list[LLMCallInfo]
    final_answer: str | None


def __getattr__(name: str) -> Any:
    """COMPAT-X shim: the replay type was named ``ArtifactRef`` (PDC-D8-R018)."""
    if name == "ArtifactRef":
        return ReplayArtifactRef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
