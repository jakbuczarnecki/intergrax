---
id: IJ-2026-06-12-026
date: 2026-06-12
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: REASONING_AND_COGNITION
plan_ref:
  - COG-PROD.1
  - COG-PROD.2
  - COG-PROD.3
  - COG-PROD.4
  - COG-PROD.5
  - AUDIT-IDEAL-7.1
status: completed
commit: 218fd286
adr: none — extends existing ReasoningProfile and DECISION_EMITTED contracts; no new Tier-0 mechanism
---

# COG-PROD — Reasoning plane production hardening and doc reconciliation

## Operator request

Close the Reasoning/Cognition layer to production readiness: reconcile documentation with implementation, wire partial COG-DEPTH items (planner LLM separation, parse retries, prompt templates, planning DecisionRecord), and fix schema drift blocking durable event persistence.

## Summary

- Reconciled `REASONING_AND_COGNITION` architecture canon (removed stale §2/§10/§14 gaps vs §21 Done).
- Added Phase **COG-PROD** to plan with five deliverables — all **Done**.
- Implemented `resolve_planner_llm_adapter()`, `resolve_planner_model_id()`, `resolve_engine_planner_prompt_config()`.
- Wired `planner_parse_retries`, `nexus_task_planner` `user_template`, enriched planning `DecisionRecord`.
- Fixed `DECISION_EMITTED` multi-phase schema guard (planning + step_execution).

## Project impact

Nexus task cognition is now honestly L3+ production-ready: typed plans, registry-backed prompts, separable planner LLM, policy deny gate, planning-phase decision audit, and parse retry budget — without doc↔code contradictions.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/REASONING_AND_COGNITION.md` §2, §16, §21 |
| Plan | `docs/plan/REASONING_AND_COGNITION.md` Phase COG-PROD |
| ADR | none — mirror `CriticProfile` planner separation pattern |

## Changed artifacts

- `intergrax/applications/_shared/reasoning_wiring.py` — planner LLM + engine prompt resolvers
- `intergrax/applications/_shared/nexus_factory.py` — producer/planner adapter wiring
- `intergrax/runtime/nexus/planning/nexus_plan_bridge.py` — parse retries
- `intergrax/runtime/nexus/orchestration/planning_runner.py` — enriched DecisionRecord
- `intergrax/runtime/events/phase_coverage.py` — multi-phase DECISION_EMITTED
- `scripts/check_reasoning_gates.py` — stronger CI gate
- Tests: `test_reasoning_wiring.py`, `test_nexus_plan_bridge.py`, `test_planning_decision_record_gate.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_reasoning_wiring.py tests/unit/runtime/nexus/planning/ tests/integration/runtime/test_planning_decision_record_gate.py -q
python scripts/check_reasoning_gates.py
```

## Risks and follow-ups

- `tests/acceptance/agent_os/test_lab_application.py::test_lab_application_runs_research_mock_with_graph_trace` fails on branch with pre-existing `RuntimeRequest.model_copy` AttributeError (out of COG-PROD scope).
- L4 adaptive planner selection remains AHI observe-only scope.
