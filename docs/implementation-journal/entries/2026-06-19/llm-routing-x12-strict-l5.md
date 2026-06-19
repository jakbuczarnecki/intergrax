---
id: IJ-2026-06-19-003
date: 2026-06-19
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.12.1
  - M-LLM-X.12.2
  - M-LLM-X.12.3
  - M-LLM-X.12.4
  - M-LLM-X.12.5
  - M-LLM-X.12.6
  - M-LLM-X.12.7
  - M-LLM-X.12.8
  - M-LLM-X.12.9
  - M-LLM-X.12.10
  - M-LLM-X.12.11
  - M-LLM-X.12.12
  - LLM-AUDIT-19
status: completed
commit: bbd32054
adr: none — tier move and wiring per ADR-LLM-003; no new public contract
---

# M-LLM-X.12 — Strict L5 LLM routing closeout

## Operator request

Close all post-X-11 routing gaps (budget meter drift, narrow Nexus path coverage, Tier-0 import violation) and promote multi-model routing to honest **L5** with production metering proof.

## Summary

Delivered wave **M-LLM-X.12**: Tier-3 `RoutingEvaluatingLLMAdapter` with injected factory; Tier-0 `metering.py` and `runtime_sync.py`; inner-adapter tracker re-registration on swap; Nexus graph and context-engine snapshot sync; per-call refresh via config provider; removed global routing observers; ACP routing diagnostics on `DynamicLLMRouter`; production metering E2E; CI `check_llm_routing_tier_boundary.py`. Closes **LLM-AUDIT-19**.

## Project impact

Declarative LLM routing is now **strict L5** on core UAEP/Nexus/ACP paths: budget rules can fire on real token usage mid-run, context stays fresh across graph/CE/UAEP hot paths, and observability is per-run instance-bound. Secondary LLM surfaces (planner, websearch, critic) are explicitly documented as out of auto-wrap scope until product extends wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/LLM_ADAPTERS.md` § Routing strict enterprise closeout |
| Plan | `docs/plan/LLM_ADAPTERS.md` Wave M-LLM-X-12 |
| ADR | ADR-LLM-003 (routing rules); no new ADR |
| Audit | LLM-AUDIT-19 **Done** |

## Changed artifacts

- `intergrax/applications/_shared/routing_evaluating_adapter.py` — Tier-3 evaluating wrapper
- `intergrax/llm_adapters/routing/metering.py`, `runtime_sync.py` — Tier-0 metering + snapshot refresh
- `intergrax/runtime/nexus/context/routing_snapshot_sync.py` — graph/CE hooks
- `scripts/check_llm_routing_tier_boundary.py` — CI tier gate
- Tests under `tests/unit/llm_adapters/routing/`, `tests/acceptance/llm_routing/`

## Verification

```bash
uv run pytest tests/unit/llm_adapters/routing/ tests/acceptance/llm_routing/ -m "gate and not no_ci" -q
python scripts/check_llm_routing_tier_boundary.py
python scripts/check_llm_routing_context_wiring.py
```

Result: 36 passed; tier boundary and context wiring gates OK.

## Risks and follow-ups

- Secondary LLM surfaces (tool planner, websearch, critic) remain static-profile unless hosts add explicit routing wiring (M-LLM-X.12.12 policy).
- `runtime_state.py` still imports Tier-3 evaluating type for isinstance wiring — acceptable bridge debt; duck-typing refactor optional later.
- **M-LLM-X.8** domain closeout can proceed now that strict L5 routing is closed.
