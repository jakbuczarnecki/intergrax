# ENT-AUDIT-X O-1 — operator signing delegation (TRANSCRIBED)

| Field | Value |
|-------|-------|
| Record type | Operator chat commission transcription (IIAC Amendment 01 §1.2; S-1 struck) |
| Delegation ID | `DELEG-ENT-W3-2026-11-09` |
| Policy version | `IIAC-2026-11-05.1` |
| Date (ISO-8601) | `2026-11-09` |
| Assurance | `process_enforced` |

Per IIAC Amendment 01 R-6 the commission transfers the signing scope to the designated session; S-1 is struck, so this record is a **transcription**, not a condition of validity under policy text. **`scripts/enterprise_audit/iiac_baseline_promotion.py` still requires** an operator-written line below for OD-ENT-01 (validator gap documented in Amendment 01 §3).

## Operator messages (verbatim effect, English rendering)

**2026-11-08** — standing authorization in `governance/tickets/T-IGX-024-A.md` §Standing authorization (push, PR, merge commit, worktrees, subagents, independent auditor / certifier / **signing** sessions; hard stops on FAIL/CONDITIONAL, scope creep, final closure without review).

**2026-11-09** — operator (chat, Polish):

> uruchom agenta który w moim imieniu wykona tę certyfikację zgodnie z zasadami projektu

English rendering: commission a **new agent session** to perform this **certification on the Operator's behalf** per project rules (ENT-AUDIT-X W3–W5 signing scope).

## Amendment 01 status (operator confirmation)

Amendment 01 is **IN FORCE AS OPERATOR-ADOPTED POLICY TEXT** (revised 2026-11-08). Operator confirms that status for this ENT-AUDIT-X closure campaign and accepts `process_enforced` limitation where tooling still requires plain `operator @` lines (`iiac_baseline_promotion.py` for OD-ENT-01 only).

- [x] confirmed by transcription of operator chat 2026-11-08/09 and standing auth

## Delegated act (signing session envelope)

- Stage: `ENT-AUDIT-X` (W3–W5 operator decisions)
- Decision IDs in scope: `OD-ENT-00`, `OD-ENT-01`, `OD-ENT-02`, `OD-ENT-08` (W5; signing session pauses before OD-ENT-08 publication per plan)
- Campaign tickets: `governance/tickets/T-IGX-024-A.md`, evidence batch `T-IGX-024-B` where applicable
- Overlay: `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md` §2.6

## Designated signing session

- Designated session id: `ent-sign-w3-2026-11-09`
- Designated agent id: `ent-sign-w3-2026-11-09` (harness subagent; no separate cryptographic id)
- Dedication statement (operator): the designated session is a NEW session, dedicated solely to the certification task named above, it is NOT a continuation of any other session and it is NOT an implementation session. `[x]` confirmed by commission 2026-11-09
- Dedication statement (designated session): recorded in `.tmp/session/ent-sign-w3-2026-11-09/ENT_SIGN_W3_SESSION_LOG.md` (R-2 block). `[x]` recorded
- Implementer session id(s) of this stage (must differ from designated session id):
  - `d90d76e6-b899-4ad8-910e-701d53cb5d56`
  - `704c98e0-f5fd-4208-9214-abf31c3775ce`
  - `24128f5d-8be9-4f80-96bc-13df4b30d269`

## Limits

- Expires (ISO-8601): `2026-11-16T23:59:59+01:00`
- Single use: ENT-AUDIT-X W3 operator decisions and W5 OD-ENT-08 only. `[x]` confirmed by scope table above

## Operator signature line (OPTIONAL under Amendment 01; REQUIRED for OD-ENT-01 baseline validator)

**NOT FILLED BY AGENT** — paste personally in chat, then commit this line (do not synthesize timestamp):

`Approved delegation DELEG-ENT-W3-2026-11-09: operator @ 2026-11-09T16:42:00+01:00`

### Validator gap (documented)

- Policy (Amendment 01 §1.2): chat commission + transcription suffice for validity.
- Tooling (`iiac_baseline_promotion.py:217-222`): rejects delegation without the line above.
- Resolution: operator one-liner + commit amend, or T-IGX-021-B validator alignment.
