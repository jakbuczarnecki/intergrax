---
id: IJ-2026-06-13-013
date: 2026-06-13
tiers:
  - tier-1
  - tier-2
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - EBE-1
  - EBE-2
  - EBE-3
  - EBE-4
  - EBE-5
  - EBE-6
status: completed
commit: pending
adr: docs/project/technical/adr/entries/2026-06-13/ADR-OBS-002.md · applications/attestation_demo/docs/adr/ADR-ATTESTATION_DEMO-001.md
---

# EBE PoC v1 — deployable attestation_demo for AgentReceipt partner handoff

## Operator request

Bring `attestation_demo` iteratively to a deployable state where the external partner (AgentReceipt) can integrate per agreed trust model: unsigned `execution_boundary_event.v1` in API response, `client_observed` receipts, full scaffold compliance, tests, and partner handoff artifacts.

## Summary

Completed PoC v1 partner-ready package: Tier-1 EBE wired through `attestation_demo` host with `POST /v1/attestation_demo/poc/run`, optional harness API key auth, `partner_handoff/` sample JSON and integration guide, platform OBSERVABILITY canon (§18 + Phase EBE), harness ADR-OBS-002, and expanded smoke tests validating full event contract and debug trace comparison path.

## Project impact

External partners can trigger governed `records.put` execution, receive unsigned boundary facts synchronously, map to AgentReceipt receipts without Intergrax fork, and compare against internal HOS trace — while Intergrax documentation honestly states no platform attestation in PoC v1.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §18 · `applications/attestation_demo/docs/ARCHITECTURE.md` |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` Phase EBE |
| ADR | `docs/project/technical/adr/entries/2026-06-13/ADR-OBS-002.md` · `applications/attestation_demo/docs/adr/ADR-ATTESTATION_DEMO-001.md` |
| Partner handoff | `applications/attestation_demo/partner_handoff/README.md` |

## Changed artifacts

- `intergrax/runtime/attestation/` — EBE schema, emitter, buffer (prior iteration)
- `applications/attestation_demo/` — scaffold layout, PoC routes, partner_handoff, deploy triad
- `agents/boundary_demo/` — demo UAEP agent calling `records.put`
- `docs/project/architecture/OBSERVABILITY.md` — §18 EBE
- `docs/project/maintainers/plans/OBSERVABILITY.md` — Phase EBE register
- `docs/project/technical/adr/entries/2026-06-13/ADR-OBS-002.md` — harness ADR

## Verification

```bash
uv run pytest applications/attestation_demo/attestation_demo_tests -q
uv run pytest tests/unit/runtime/attestation/ -q
uv run pytest tests/unit/applications/ -q -k attestation_demo
python scripts/maintenance/check_harness_adr.py
```

Result: pass (4 app tests, 4 runtime attestation tests, 3 gate tests).

## Risks and follow-ups

- Webhook delivery (EBE-7), HarnessKernel step export (EBE-8), host signing (EBE-9) deferred.
- Partner deploy should set `INTERGRAX_HARNESS_API_KEY` before public exposure.
- `agents/boundary_demo/` lacks Tier-2 doc pair (not gate-listed; optional follow-up).
