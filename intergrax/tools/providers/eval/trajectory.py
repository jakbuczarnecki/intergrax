# © Artur Czarnecki. All rights reserved.

"""Trajectory evaluation catalog tool (Phase CRIT-V-2.2)."""

from __future__ import annotations

from collections import Counter
from uuid import uuid4

from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput, EvalTrajectoryOutput
from intergrax.tools.providers.eval.service import _append_critic_observation
from intergrax.tools.registry.runtime_bindings import RunTraceReaderBinding
from intergrax.tools.registry.wiring import ToolWiringContext

EVAL_TRAJECTORY_TOOL_ID = "eval.trajectory"

_TOOL_START_STEPS = frozenset({"tool_invocation_start", "ToolRuntime", "UAEPToolGateway"})


def _require_trace_reader(ctx: ToolWiringContext) -> RunTraceReaderBinding:
    reader = ctx.trace_reader
    if reader is None:
        raise RuntimeError("trace_reader_not_configured")
    return reader


def _tool_signature(event: dict[str, object]) -> str:
    step = str(event.get("step") or "")
    message = str(event.get("message") or "")
    payload = event.get("payload")
    tool_name = ""
    if isinstance(payload, dict):
        raw_name = payload.get("tool_name") or payload.get("tool_id")
        if isinstance(raw_name, str):
            tool_name = raw_name
    return f"{step}:{tool_name or message}"


def _score_trajectory(
    *,
    tool_call_count: int,
    duplicate_tool_calls: int,
    error_count: int,
    denied_count: int,
    min_score: float,
) -> tuple[float, bool, list[str]]:
    reasons: list[str] = []
    score = 1.0
    if error_count:
        penalty = min(0.5, 0.15 * error_count)
        score -= penalty
        reasons.append(f"{error_count} tool error(s) in trace")
    if denied_count:
        penalty = min(0.4, 0.1 * denied_count)
        score -= penalty
        reasons.append(f"{denied_count} denied tool invocation(s)")
    if duplicate_tool_calls:
        penalty = min(0.4, 0.08 * duplicate_tool_calls)
        score -= penalty
        reasons.append(f"{duplicate_tool_calls} duplicate tool invocation pattern(s)")
    if tool_call_count == 0:
        score -= 0.2
        reasons.append("no tool invocations recorded in trace")
    score = max(0.0, min(1.0, score))
    passed = score >= min_score
    if passed and not reasons:
        reasons.append("trajectory within expected process bounds")
    return score, passed, reasons


def eval_trajectory(ctx: ToolWiringContext, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
    reader = _require_trace_reader(ctx)
    persisted = reader.read_run(params.run_id, params.tenant_id)
    events = list(persisted.events)

    tool_signatures: list[str] = []
    error_count = 0
    denied_count = 0
    for raw in events:
        if not isinstance(raw, dict):
            continue
        step = str(raw.get("step") or "")
        if step == "tool_invocation_error":
            error_count += 1
        elif step == "tool_invocation_denied":
            denied_count += 1
        elif step in _TOOL_START_STEPS:
            tool_signatures.append(_tool_signature(raw))

    counts = Counter(tool_signatures)
    duplicate_tool_calls = sum(max(0, count - 1) for count in counts.values())
    tool_call_count = len(tool_signatures)

    score, passed, reasons = _score_trajectory(
        tool_call_count=tool_call_count,
        duplicate_tool_calls=duplicate_tool_calls,
        error_count=error_count,
        denied_count=denied_count,
        min_score=params.min_score,
    )

    observation_recorded = _append_critic_observation(
        ctx,
        record=params.record_observation,
        observation_id=params.observation_id or f"traj-{uuid4().hex[:12]}",
        run_id=params.run_id,
        agent_id=params.agent_id,
        scenario_id=params.scenario_id or f"critic.trajectory:{params.run_id}",
        mode=params.mode,
        passed=passed,
        score=score,
        candidate_profile_version_id=params.candidate_profile_version_id,
    )

    return EvalTrajectoryOutput(
        run_id=params.run_id,
        score=score,
        passed=passed,
        reasons=reasons,
        tool_call_count=tool_call_count,
        duplicate_tool_calls=duplicate_tool_calls,
        error_count=error_count,
        denied_count=denied_count,
        observation_recorded=observation_recorded,
    )
