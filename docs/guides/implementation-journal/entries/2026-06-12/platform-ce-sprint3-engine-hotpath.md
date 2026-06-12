---
id: IJ-2026-06-12-008
date: 2026-06-12
tiers:
  - tier-0
  - tier-1
  - tier-2
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-3.1
  - CE-3.2
  - CE-3.9
  - CE-3.10
status: completed
commit: 228ebff0
adr: no ADR needed — wires existing ContextCompiler library to ACP hot path per CE-3 plan
---

# CE Sprint 3 — DefaultNexusContextEngine and compiler hot path

## Operator request

Continue CE sprint workflow: engine skeleton plus ContextCompiler on ACP/UAEP before LLM (S3).

## Summary

Shipped `DefaultNexusContextEngine`, fragment bridge, `DefaultContextValidator`, `compile_service` helpers, `resolve_context_engine_from_environment`, and ACP `llm_router` / `acp_run` prompt compilation before LLM calls.

## Project impact

ContextCompiler and preflight validation are on the ACP production hot path; Tier-0 engine contract is callable with provider collect + budget compile spine.

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/context/ tests/unit/context/ tests/unit/applications/test_context_engine_wiring.py -m gate -q
python scripts/check_context_tier0_import_boundary.py
```

Result: 47 passed.
