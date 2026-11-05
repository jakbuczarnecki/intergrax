# IIAC operator activation packet — NOT SIGNED

**Decision status: PENDING**  
**Policy version:** `IIAC-2026-11-05.1`  
**Scope:** IIAC policy activation for pre-production B6 authority (`EVOL-X`, `ENT-AUDIT-X`) after independent acceptance of the ACT-01 gate. This packet does not close EVOL-X, ENT-AUDIT-X, or ARCH-FREEZE.  
**Not an activation record.** Do not rename this file to `IIAC_OD_ACTIVATION_*.md` until the decision below is actually authorized.

## Evidence the decision must cite

| Item | Value |
|---|---|
| Governance implementation SHA | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| F01–F03 corrective / re-audit baseline | `5d6b44b05f90ecf420d45d01659ea64565b52848` |
| Re-audit verdict | `PASS — F01–F03 INDEPENDENTLY VERIFIED` |
| Re-audit report digest | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` |
| Historical Session B | `PARTIAL / EXACT BLOCKERS`; manifest `CONDITIONAL`; digest `56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1` |
| ACT-01 implementation | gate logic `1d0722723687fb93d0a538161ca4cce376b3ecef`; byte-correct evidence `2a6545f1d551814a0ba8896be79576c580353771` |
| Independent ACT-01 re-audit | **PASS** — dossier digest `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` @ `iiac_evidence/IIAC-GOV-1/certification/IIAC_ACT01_INDEPENDENT_REAUDIT_DOSSIER.md` |
| Governance certification | manifest @ `iiac_evidence/IIAC-GOV-1/certification/` — policy still `NOT_IN_FORCE` |

Session A line `Approved: operator Session A @ 2026-11-05T13:00:00+01:00 (IIAC-GOVERNANCE-REFORM)` authorizes implementation only. The handoff line remains `Approved handoff: operator @ ________________`.

## Exact decision to record later

Only after the independent re-audit PASSes and the operator authorizes activation, publish a new file `IIAC_OD_ACTIVATION_<decision-id>.md` containing at least:

```text
Decision ID: <id>
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation
Decision status: AUTHORIZED
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <ISO-8601 timestamp>
```

`________________`, `PENDING`, `NOT SIGNED`, and a bare `Approved` do not satisfy the gate.

## Publication procedure

1. Phase 1 commit: set `IIAC_POLICY_ACTIVATION_RECORD.json` to `IN_FORCE` with the bound evidence fields and `effective_from_sha` empty.
2. Phase 2 commit: set `effective_from_sha` and `policy_activation_sha` to the phase-1 commit.
3. Run `uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force` and require exit 0.
4. Do not set those fields in this ACT-01 commit.

## Rollback and supersession

Revert the phase-1 and phase-2 commits to return the record to `NOT_IN_FORCE`. Do not delete `iiac_evidence/IIAC-GOV-1/historical_session_b/` or the reaudit copies. A later policy version supersedes `IIAC-2026-11-05.1` only by a new activation record that names the new version; this packet does not do that.

## Signature

`Approved IIAC policy activation IIAC-2026-11-05.1: operator @ ________________` **(NOT SIGNED — not fabricated)**
