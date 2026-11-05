# IIAC ACT-01 corrective specification

**Policy:** `IIAC-2026-11-05.1`  
**Parent baseline:** `5d6b44b05f90ecf420d45d01659ea64565b52848`  
**Status during this batch:** `NOT_IN_FORCE`  
**Not a master plan.** This file is the ticket plan-digest source for `T-IGX-016-B`.

## In scope

- Fail-closed modes on `verify_iiac_policy_activation.py`: default and `--require-in-force` accept only a fully proven `IN_FORCE` record; `--expect-not-in-force` accepts only an inactive record with no active claim.
- Independent PASS manifest for `IIAC-GOV-1`, digest binding, findings closure for BLOCKER/CRITICAL, role separation, and companion script execution.
- Canonical evidence copies under `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/`.
- Operator decision file checks. Session A approval is not activation authority.
- Two-phase `effective_from_sha`: phase 1 has an empty effective SHA; phase 2 points at that commit. Reject HEAD self-reference and pre-activation blobs.
- Adversarial tests ACT-N01 through ACT-N18 plus one synthetic positive fixture.
- No `IN_FORCE` on the canonical record. No fabricated operator signature. No self-issued PASS.

## Out of scope

- Restart of G0–G3, FRZ mass updates, EVOL-X / ENT-AUDIT-X / ARCH-FREEZE closure, external-human reactivation, Docker, vendor, production config.
