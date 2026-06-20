---
id: IJ-2026-06-19-001
date: 2026-06-19
tiers:
  - tier-1
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - EBE-8
status: completed
commit: 6c6fef1c
adr: none — partner validation closeout; no schema or trust-model change
---

# EBE-8 — Partner live validation (AgentReceipt / Cullen)

## Operator request

Record partner sign-off after live Docker-backed PoC v2 validation, update OBSERVABILITY plan/architecture status, clarify Docker buildx fallback in the runbook, and draft a reply email acknowledging trace correlation scope.

## Summary

Partner Cullen validated PoC v2 against live Intergrax Docker at commit `106aee776fcc6053e8265b9c3656638d107d351d` (branch `agent_experiment_runtime`). Successful run `run_7bfa10ae1ec9471a8b38eb1a7b4a1011` returned two boundary events; AgentReceipt mapped each to an independent `client_observed` receipt with matching input/output hashes and independent verification. Failed-tool fixture confirmed dual claims (tool `failed`, harness `completed`); missing failed-tool output hash treated as not comparable. AgentReceipt suite: build, 28/28 tests, live example, chain — all passed without core schema changes.

## Project impact

EBE-8 PoC v2 is **partner-validated** end-to-end. Handoff reference commit and branch are frozen for external integration; trace endpoint scope documented (run/task correlation only — per-event `event_id` / `tool_id` from `boundary_events[]` response).

## Traceability

| Link | Target |
|------|--------|
| Prior delivery | `docs/implementation-journal/entries/2026-06-18/observability-ebe-8-poc-v2-harness-step.md` |
| Architecture | `docs/architecture/OBSERVABILITY.md` §18 |
| Plan | `docs/plan/OBSERVABILITY.md` EBE-8 |
| Handoff | `applications/attestation_demo/partner_handoff/README.md` |
| Runbook | `applications/attestation_demo/DOCKER_VERIFY_RUNBOOK.md` |

## Partner evidence (external)

| Item | Value |
|------|-------|
| Intergrax commit | `106aee776fcc6053e8265b9c3656638d107d351d` |
| Live run | `run_7bfa10ae1ec9471a8b38eb1a7b4a1011` |
| Tool event | `event_id` `4d9b7c34-ff54-4451-b6d3-54402c265715`, `event_sequence` 1 |
| Harness event | `event_id` `501811f8-2fc0-486b-a567-477662b83a56`, `event_sequence` 2 |

## Changed artifacts

- `docs/plan/OBSERVABILITY.md` — EBE-8 partner-validated acceptance
- `docs/architecture/OBSERVABILITY.md` — §18 partner validation + trace scope
- `applications/attestation_demo/DOCKER_VERIFY_RUNBOOK.md` — buildx fallback
- `applications/attestation_demo/docker/build-docker.sh` — auto-fallback on `--ignorefile` failure
- `applications/attestation_demo/partner_handoff/README.md` — trace vs boundary correlation

## Verification

Documentation-only closeout; partner ran live Docker + AgentReceipt CI externally.

## Risks and follow-ups

- EBE-7 webhook and EBE-9 host signing remain deferred.
- Optional future: enrich HOS trace with `event_id` / `tool_id` for journal-side per-event correlation (not required for PoC v2).
