---
id: IJ-2026-06-14-008
date: 2026-06-14
tiers:
  - tier-0
  - tier-1
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.1.1
  - M-LLM-X.1.6
  - M-LLM-X.3.1
  - M-LLM-X.3.3
  - M-LLM-X.3.5
  - LC-1
  - LC-2
status: completed
commit: pending
adr: ADR-LLM-002 — implementation per accepted decision
---

# LLM Layer Completion — LC-1 ModelCatalog + LC-2 preflight alignment

## Operator request

Execute Intergrax Layer Completion Mode for LLM_ADAPTERS: audit, doc sync, sprint LC-1/LC-2 implementation.

## Summary

Implemented Tier-0 `ModelCatalog`, bundled YAML (50+ models), `resolve_context_window_tokens`, wired all provider adapters, optional catalog overlay env. Aligned `verify_context_preflight` with adapter tokenizer and added `ContextBudgetPolicy.from_adapter`. Updated architecture/plan Layer Completion sprint register.

## Project impact

Context budgeting for new model strings no longer depends on stale per-adapter dicts alone; OpenRouter defaults 128k via catalog. Preflight and history layer share adapter token counting path.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/LLM_ADAPTERS.md` — audit register LC-1/LC-2 |
| Plan | `docs/project/maintainers/plans/LLM_ADAPTERS.md` — Layer Completion sprint register |
| ADR | `docs/project/technical/adr/entries/2026-06-14/ADR-LLM-002.md` |

## Changed artifacts

- `intergrax/llm_adapters/registry/model_catalog.py`
- `intergrax/llm_adapters/registry/model_catalog.yaml`
- `intergrax/llm_adapters/registry/context_window.py`
- `intergrax/llm_adapters/providers/*` (context window wiring)
- `intergrax/runtime/nexus/context/context_preflight.py`
- `intergrax/runtime/nexus/context/context_budget.py`
- `docs/project/architecture/LLM_ADAPTERS.md`
- `docs/project/maintainers/plans/LLM_ADAPTERS.md`
- `tests/unit/llm_adapters/test_model_catalog.py`
- `tests/unit/llm_adapters/test_context_window_wiring.py`

## Verification

```bash
uv run pytest tests/unit/llm_adapters/test_model_catalog.py tests/unit/llm_adapters/test_context_window_wiring.py -q
python scripts/audit/check_docs_domain_pairs.py
python scripts/maintenance/check_implementation_journal.py
```

## Risks and follow-ups

- LC-3 (failover, routing, ACP bridge) and LC-4 (OpenRouter fetch, validate_runtime) remain P1/P2.
- LLM-AUDIT-3/11 Partial until CI guard and Nexus compile paths adopt `from_adapter`.
- Pre-existing bedrock converse stream test failures (3) — out of LC-1/LC-2 scope.
