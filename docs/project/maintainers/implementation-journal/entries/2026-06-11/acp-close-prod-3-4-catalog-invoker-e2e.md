---
id: IJ-2026-06-11-005
date: 2026-06-11
tiers:
  - tier-0
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PROD-3
  - ACP-CLOSE-PROD-4
status: completed
commit: pending
adr: none — replaces test-only MagicMock with typed stub; no contract change
---

# ACP-CLOSE PROD-3/4 — catalog declarative invoker real context + Nexus E2E

## Operator request

Execute sprint 4 of ACP-CLOSE: remove `MagicMock` shim from `CatalogDeclarativeToolInvoker` and add Nexus + harness host E2E acceptance for catalog declarative mutating resume.

## Summary

Replaced `unittest.mock.MagicMock` in `catalog_declarative_invoker.py` with `_CatalogDispatchLLMStub`, `SessionManager(InMemorySessionStorage)`, and resolved `YamlPromptRegistry`. Avoids `RuntimeContext.build()` because it rebuilds the tool catalog and overwrites the host-bound invoker. Added `test_acceptance_05e_nexus_harness_catalog_declarative_mutating_resume` exercising `build_harness_host_runtime` → `NexusLoop` → `CatalogDeclarativeToolInvoker` with checkpoint resume (single catalog invoke).

## Project impact

Declarative mutating tools on Tier-3 hosts now dispatch through a real minimal Nexus runtime slice, not mocks. E2E evidence closes the gap between unit wiring and production harness path for §40.3 declarative execution.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §32.8 · §40.3 |
| Plan | `ACP-CLOSE-PROD-3` · `ACP-CLOSE-PROD-4` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/persistence/catalog_declarative_invoker.py` — real runtime context slice
- `tests/unit/agents/persistence/test_catalog_declarative_invoker.py` — context wiring assertion
- `tests/acceptance/agent_os/test_acp_nexus_catalog_declarative_resume.py` — Nexus E2E (new)
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — PROD-3/4 Done

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_catalog_declarative_invoker.py tests/acceptance/agent_os/test_acp_nexus_catalog_declarative_resume.py -q
```

Result: 4 passed.

## Risks and follow-ups

- Durable compensation queue (ACP-CLOSE-PROD-5) and scoreboard 100% (PROD-8) remain open.
- Acceptance test patches `build_declarative_invoker_from_tool_wiring` to register probe tool; product hosts rely on manifest tool bundles for real mutating tools.
