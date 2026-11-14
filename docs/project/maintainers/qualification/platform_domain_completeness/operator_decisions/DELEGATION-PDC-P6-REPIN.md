# DELEGATION-PDC-P6-REPIN — Operator commission for ticket T-IGX-036-W (P6 acceptance run and EBH-7 re-pin)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-REPIN |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `PDC_REG_DIAGNOSIS.md` (Operator decision of 2026-11-12 on groups B1 and B2) |
| **Date** | 2026-11-14 |

## Commission (transcription)

The orchestrator prepared ticket T-IGX-036-W, its plan and the dossier `PDC_EBH7_REPIN_DOSSIER.md` (PR 530) and asked the Operator three questions in chat. The answers, in English rendering with the original Polish wording:

1. **Re-pin now or after P8:** **"Re-pin now"** (original: "Re-pin teraz"). The Operator accepts that findings of P7 or P8 that change product paths need a further re-pin.
2. **Dependency-bot merges:** **"Yes, hold them"** (original: "Tak, wstrzymaj (Recommended)"). Dependency-bot pull requests (changes to `uv.lock`) are not merged between the cut of W1 and the recording of the replay in W2; the Operator holds them, the orchestrator reports their state.
3. **Who signs the plan:** **"Delegate"** (original: "Delegat (Recommended)").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-W** only, after confirming: the plan digest matches; the plan matches batch B-P6-E in `PDC_IMPLEMENTATION_WBS.md` (write set equal); no write path lies under `intergrax/`, `agents/`, `applications/`, `intergrax/contracts/` or `intergrax/applications/_shared/`; the verifier allowlist, signed EBH-7 decision, signature and closure records are not in the write set; the plan adds no new registry, scheduler, engine, trust, policy or audit authority; no gate is weakened; Part A carries no PASS verdict.
- The implementer dispatches only the replay workflow named in the dossier (`unit-tests.yml`, `profile=ebh7_p3_l3`), as in authorization 8 of `DELEGATION-PDC-STANDING.md`, and merges the docs PRs of W under authorization 1 of the same record.
- Write-set amendments of T-IGX-036-W are **not** delegated: any need to touch another `ebh_7` test, the verifier allowlist, a signed record or a product path returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; the B6 or certification verdict on the new replay; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
