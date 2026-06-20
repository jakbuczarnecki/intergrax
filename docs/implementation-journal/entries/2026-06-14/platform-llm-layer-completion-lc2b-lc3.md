---
id: IJ-2026-06-14-009
date: 2026-06-14
tiers:
  - tier-0
  - tier-1
  - tier-2
  - tier-3
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.3.4
  - M-LLM-X.4.1
  - M-LLM-X.4.2
  - M-LLM-X.4.3
  - M-LLM-X.5.1
  - M-LLM-X.5.2
  - M-LLM-X.5.3
  - M-LLM-X.5.4
  - M-LLM-X.5.5
  - M-LLM-X.7.2
  - LC-2b
  - LC-3
status: completed
commit: 96b937e2
adr: no ADR needed — implements ADR-LLM-002 and existing M-LLM-X plan without contract change
---

# LLM Layer Completion — LC-2b budget adoption + LC-3 failover/routing/ACP bridge

## Operator request

Complete Intergrax Layer Completion Mode for LLM_ADAPTERS: close remaining P0/P1 gaps after LC-1/LC-2.

## Summary

Closed Nexus context budget adoption (`ContextBudgetPolicy.from_adapter` on RuntimeConfig and ContextManager wiring), added preflight CI guard, implemented `FailoverLLMAdapter` with profile chain fields on `LLMProfile`, expanded `ModelRouter` hints, wired `resolve_llm_adapter` to live routing and failover, and bridged ACP `StepLLMRouter` to Tier-0 `LLMAdapter` via `LLMAdapterCompletePort`.

## Project impact

LLM layer reaches production-ready routing and token accounting consistency; ACP agents share the same adapter spine as Nexus; product hosts apply AHI routing hints at adapter creation time.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/LLM_ADAPTERS.md` — audit register LC-3 closeout |
| Plan | `docs/plan/LLM_ADAPTERS.md` — sprint LC-2b/LC-3 Done |
| ADR | ADR-LLM-002 (ModelCatalog); no new ADR |

## Changed artifacts

- `intergrax/llm_adapters/registry/failover_adapter.py`
- `intergrax/llm_adapters/registry/model_router.py`
- `intergrax/llm_adapters/registry/profile.py`
- `intergrax/applications/_shared/llm_resolver.py`
- `intergrax/applications/_shared/context_wiring.py`
- `intergrax/applications/_shared/runtime_config_bridge.py`
- `intergrax/applications/_shared/nexus_factory.py`
- `intergrax/applications/_shared/llm_routing_wiring.py`
- `intergrax/agents/authoring/llm_router.py`
- `intergrax/agents/authoring/acp_run.py`
- `scripts/check_context_preflight_uses_adapter_tokens.py`
- `docs/architecture/LLM_ADAPTERS.md`
- `docs/plan/LLM_ADAPTERS.md`
- `docs/audit/LLM_ADAPTERS.md`

## Verification

```bash
uv run pytest tests/unit/llm_adapters/test_failover_adapter.py tests/unit/llm_adapters/test_model_router.py tests/unit/llm_adapters/test_context_window_wiring.py tests/unit/agents/authoring/test_llm_router.py -q
python scripts/check_context_preflight_uses_adapter_tokens.py
python scripts/check_docs_domain_pairs.py
```

## Risks and follow-ups

- M-LLM-X.2 dynamic OpenRouter metadata fetch remains P2 backlog.
- M-LLM-X.4.4 trace DTO and M-LLM-X.1.7 catalog-driven capability flags remain P2.
- `intergrax doctor` hook for `validate_runtime()` not wired yet.
