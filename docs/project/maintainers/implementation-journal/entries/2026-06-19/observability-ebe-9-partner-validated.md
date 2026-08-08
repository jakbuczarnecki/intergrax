---
id: IJ-2026-06-19-005
date: 2026-06-19
tiers:
  - tier-1
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - EBE-9
status: completed
commit: 55501d76
adr: none — partner validation closeout; no schema or trust-model change
---

# EBE-9 — Partner live validation (BoundaryAttest / Cullen)

## Operator request

Record BoundaryAttest sign-off after EBE-9 host-signed boundary event verification, update OBSERVABILITY plan/architecture and partner handoff pins, and align documentation with end-to-end validated trust model (host claim separate from `client_observed` wrapper).

## Summary

Partner Cullen confirmed BoundaryAttest verifies Intergrax EBE-9 host attestation on `main` @ `61be9918bc8f91fc8f160e0392d2914f38f3d4cb`: golden vector byte-for-byte pass, 39/39 tests, live two-event signed response captured from Intergrax @ `96b7f997` validated, unsigned v2 regression pass, receipt chain intact. Verifier checks event digest vs `signed_payload_hash`, canonical host-attestation statement Ed25519 signature with pinned Intergrax pubkey, and negative hash/signature/algorithm/key-id cases. BoundaryAttest receipts remain `client_observed`; host signature kept as separate runtime claim.

## Project impact

EBE-9 PoC is **partner-validated** end-to-end on both sides. Intergrax host signing contract (`boundaryattest.host-attestation.v1`, golden vector, optional profile flag) is frozen for external integration reference.

## Traceability

| Link | Target |
|------|--------|
| Prior delivery | `docs/project/maintainers/implementation-journal/entries/2026-06-19/observability-ebe-9-host-signing.md` |
| EBE-8 validation | `docs/project/maintainers/implementation-journal/entries/2026-06-19/observability-ebe-8-partner-validated.md` |
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §18 |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` EBE-9 |
| ADR | `docs/project/technical/adr/entries/2026-06-19/ADR-OBS-004.md` |
| Handoff | `applications/attestation_demo/partner_handoff/EBE-9_HOST_SIGNING.md` |

## Partner evidence (external)

| Item | Value |
|------|-------|
| BoundaryAttest commit | `61be9918bc8f91fc8f160e0392d2914f38f3d4cb` |
| Intergrax live commit | `96b7f9974869e484406cbade3160b61c71b2980c` |
| Intergrax handoff branch | `agent_experiment_runtime` @ `13102cfaff1a7a9d212c16cd16587477cc533dc0` (docs sync) |
| Partner CI | 39/39 tests; golden vector; Intergrax example; unsigned v2 regression |

## Changed artifacts

- `docs/project/maintainers/plans/OBSERVABILITY.md` — EBE-9 partner-validated acceptance
- `docs/project/architecture/OBSERVABILITY.md` — §18 EBE-9 validation pins + non-goals cleanup
- `applications/attestation_demo/partner_handoff/README.md` — dual validation pins (EBE-8 + EBE-9)
- `applications/attestation_demo/IMPLEMENTATION_PLAN.md` — partner-validated status
- `applications/attestation_demo/ARCHITECTURE.md` — partner-validated status

## Verification

Documentation-only closeout; partner ran BoundaryAttest CI + golden vector + captured live Intergrax response externally.

## Risks and follow-ups

- EBE-7 webhook remains deferred.
- Partner noted fresh Docker re-run blocked in final review environment; prior live pass @ `96b7f997` + captured response validation suffice for PoC closeout.
- Production key management (KMS/HSM, rotation) out of PoC scope.
