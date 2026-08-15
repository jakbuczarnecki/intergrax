---
id: IJ-2026-06-10-012
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-OBS-1
  - ACP-OBS-2
  - ACP-LLM-1
  - ACP-STATE-1
status: completed
commit: pending
adr: none — implements architecture §31–§34 trace and handoff contracts per ADR-AGENT-002
---

# ACP Wave 3 — typed trace, LLM router, shared context, app run summary

## Operator request

Continue Phase ACP with Wave 3: strong typing throughout, no magic strings for control flow or metadata keys, and deliver Plane A/B observability plus per-step LLM routing.

## Summary

Added typed Plane B models (`AgentRunTrace`, `AgentStepRecord`, gateway call records, policy verdicts) in `agent_run_trace.py`; migrated `HarnessKernel` to append typed steps with LLM drain from step context. Implemented `StepLLMRouter`, `SharedContextView` with Nexus bridge, `ApplicationRunSummary` builder hooked into `build_nexus_task_result`, and `AcpMetadataKey` / `AcpStructuredDataKey` enums replacing ad-hoc metadata strings.

## Project impact

Agents can record per-step LLM calls in the execution journal; multi-agent tasks emit a host-facing orchestration summary; graph handoffs use a typed shared context view with optimistic concurrency.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §31–§34 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw Wave 3 |
| ADR | ADR-AGENT-002 (ACP session model) — no new ADR |

## Changed artifacts

- `intergrax/contracts/agent_run_trace.py` — Plane B typed journal
- `intergrax/contracts/application_run_summary.py` — Plane A summary
- `intergrax/contracts/shared_context.py` — author-facing handoff view
- `intergrax/contracts/acp_metadata_keys.py` — typed metadata keys
- `intergrax/agents/authoring/llm_router.py` — per-step LLM routing
- `intergrax/agents/authoring/shared_context_bridge.py` — Nexus persistence bridge
- `intergrax/runtime/kernel/step_kernel.py` — typed trace append
- `intergrax/runtime/nexus/orchestration/application_run_summary_builder.py` — summary builder
- `intergrax/runtime/nexus/orchestration/task_finisher.py` — emit summary on task complete

## Verification

```bash
uv run pytest tests/unit/contracts/test_agent_run_trace.py tests/unit/agents/authoring/test_llm_router.py tests/unit/agents/authoring/test_shared_context_view.py tests/unit/agents/authoring/test_acp_wave3_trace.py tests/unit/runtime/nexus/orchestration/test_application_run_summary_builder.py tests/unit/runtime/kernel/test_step_kernel.py -q
```

Result: 22 passed.

## Risks and follow-ups

- `StepLLMRouter` uses stub adapter until Tier-0 LLM port wiring in a later wave.
- `agent_os` acceptance 02 should assert `TaskResult.metadata` summary once graph stubs emit ACP trace payloads (Wave 4+).
- Wave 4: UAEP bridge (`ACP-STEP-3`) and legacy deprecation.
