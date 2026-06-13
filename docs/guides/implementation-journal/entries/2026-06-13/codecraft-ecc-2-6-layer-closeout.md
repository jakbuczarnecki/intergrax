---
id: IJ-2026-06-13-002
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: CODE_CRAFT
plan_ref:
  - ECC-2
  - ECC-3
  - ECC-4
  - ECC-5
  - ECC-6
status: completed
commit: f6cd2d4d
adr: none — extends ADR-CODECRAFT-001
---

# ECC-2…ECC-6 — Code Craft layer completion closeout

## Operator request

Complete all remaining Code Craft sprints (S2–S6) iteratively to production-ready layer state per Layer Completion Mode.

## Summary

Delivered session orchestrator (`CodeCraftOrchestrator`, `CodeCraftSessionManager`), iteration tools (`start`, `iterate`, `get_state`, `dispose`, `promote`, `list_ephemeral_tools`), codegen adapter, test runner, CVL bridge, Tier-3 `codecraft_profile` wiring, HITL gate, promoter, cloud isolation routing, `harness_codecraft_stack` preset, ephemeral tool registry, graph binding, AHI catalog-miss trigger, and skill `codecraft.ephemeral_builder`.

## Project impact

Harness-owned ephemeral code synthesis loop is end-to-end usable from lab hosts with supervised/autonomous modes, typed promotion, and adaptive suggest path when catalog tools are missing.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CODE_CRAFT.md` |
| Plan | `docs/plan/CODE_CRAFT.md` — ECC-2…ECC-6 |
| ADR | ADR-CODECRAFT-001 |

## Changed artifacts

- `intergrax/runtime/codecraft/` — orchestrator, session manager, cv_bridge, ephemeral_registry, sandbox_resolver, adaptive_trigger
- `intergrax/codecraft/` — codegen_adapter, test_runner, promoter, extended contracts
- `intergrax/tools/providers/codecraft/` — full tool surface
- `intergrax/applications/_shared/codecraft_wiring.py` — Tier-3 wiring
- `intergrax/applications/contracts/environment_profile.py` — `codecraft_profile`
- `intergrax/skills/providers/codecraft/` — ephemeral_builder skill
- `intergrax/integrations/registry/presets.py` — `harness_codecraft_stack`

## Verification

```bash
uv run pytest tests/unit/codecraft/ tests/unit/tools/providers/codecraft/ tests/unit/runtime/codecraft/ tests/unit/applications/test_p6_integration_tool_wiring.py -q
python scripts/check_harness_no_getattr.py
python scripts/check_docs_domain_pairs.py
```

Result: 31 tests pass; harness checks pass.

## Risks and follow-ups

- Container isolation tier remains documented-only until OCI runner lands.
- `health.check_codecraft` probe implemented but not yet registered in health tool bundle.
- Full Nexus graph executor bridge for `CodeCraftGraphBinding` is spec-only; lab uses tools directly.
