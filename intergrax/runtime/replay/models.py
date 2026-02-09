# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any


# ---------- LLM CALL INFO ----------

@dataclass(slots=True)
class LLMCallInfo:
    step_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Optional[str]
    request_payload: Optional[Any] = None
    response_payload: Optional[Any] = None


# ---------- TOOL CALL INFO ----------

@dataclass(slots=True)
class ToolCallInfo:
    step_id: str
    tool_id: str
    input_payload: Any
    output_payload: Any
    success: bool
    error: Optional[str] = None


# ---------- ARTIFACT REF ----------

@dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    name: str
    type: str
    produced_by_step: str
    metadata: Optional[dict] = None


# ---------- RECONSTRUCTED STEP ----------

@dataclass(slots=True)
class ReconstructedStep:
    step_id: str
    step_type: str
    started_at: float
    finished_at: Optional[float]
    status: str
    llm_calls: List[LLMCallInfo]
    tool_calls: List[ToolCallInfo]
    artifacts: List[ArtifactRef]


# ---------- FINAL RUN STRUCTURE ----------

@dataclass(slots=True)
class ReconstructedRun:
    run_id: str
    steps: List[ReconstructedStep]
    artifacts: List[ArtifactRef]
    tool_calls: List[ToolCallInfo]
    llm_calls: List[LLMCallInfo]
    final_answer: Optional[str]
