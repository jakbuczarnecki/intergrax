# IIAC governance OD-01 activation terminal report

**Disposition:** `IIAC GOVERNANCE — CERTIFIED — POLICY IN_FORCE — PUBLISHED`

**Policy version:** `IIAC-2026-11-05.1`  
**Decision ID:** `IIAC-OD-01-2026-11-05`  
**Decision class:** `OD-01`

## Operator authorization provenance

Verbatim operator authorization is recorded in [`IIAC_OD_ACTIVATION_IIAC-OD-01-2026-11-05.md`](./IIAC_OD_ACTIVATION_IIAC-OD-01-2026-11-05.md) (supplied via ChatGPT → Cursor session prompt; no ChatGPT transcript in git).

**Recording / publication timestamp (decision file):** `2026-11-05T18:14:38+01:00` (`+01:00`) — repository entry time, not the original ChatGPT message time.

**Activation approval line:**

`Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-05T18:14:38+01:00`

## Publication SHAs

| Item | SHA |
|---|---|
| Certification publication | `a5fb6cb51ecd982284fd53ebc41e284e18b5523d` |
| OD-01 decision + phase 1 activation record | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| Phase 2 SHA binding | `847e9c84a1f2440161cce5a1510b7bb7ff8fcfad` |
| Final `origin/development` (after push commit) | see git footer below |

## Evidence digests

| Audit | Digest |
|---|---|
| F01–F03 re-audit | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` |
| ACT-01 independent re-audit (active report) | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` |

**Active certification manifest:** `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/certification/iiac_audit_session_manifest.IIAC-GOV-1.json`  
**Manifest SHA-256:** `c53e95db143cc6b3525b78e929aa6f42d01984c02fbce704fd2ccf63b57cdc28`

## Activation sequence

1. **Phase 1** (`f8c29fb8…`): `status=IN_FORCE`, `effective_from_sha` / `policy_activation_sha` empty; OD-01 decision published.
2. **Phase 2** (`847e9c84a…): `effective_from_sha` = `policy_activation_sha` = `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`.

## Qualification output

```text
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d
exit 0

verify_iiac_audit_manifest.py (certification manifest): PASS
verify_iiac_role_separation.py --strict-product-diff: PASS
```

**Canonical status:** `IN_FORCE` in `IIAC_POLICY_ACTIVATION_RECORD.json`.

**Product paths** (`intergrax/`, `agents/`, `applications/`) unchanged across activation commits.

## Remaining limitations

- EVOL-X / ENT-AUDIT-X / ARCH-FREEZE are **not** closed. Stage closure preflights remain fail-closed (now include IIAC-specific missing Session C/D artifacts — expected until independent stage audits).
- Windows checkout: activation record is `-text` in `.gitattributes` so worktree bytes match git blobs for `--require-in-force`.
- F03 / certification handoff operator lines may remain `PENDING`; activation used OD-01 decision file, not handoff signature.

## EVOL-X Session C readiness

**Ready to open** (new independent auditor chat). Handoff: [`IIAC_SESSION_C_EVOL_HANDOFF_POST_OD01.md`](./IIAC_SESSION_C_EVOL_HANDOFF_POST_OD01.md) and [`IIAC_SESSION_C_EVOL_INDEPENDENT_AUDITOR_INSTRUCTIONS.md`](./IIAC_SESSION_C_EVOL_INDEPENDENT_AUDITOR_INSTRUCTIONS.md).

Do not certify EVOL-X in the activation session.
