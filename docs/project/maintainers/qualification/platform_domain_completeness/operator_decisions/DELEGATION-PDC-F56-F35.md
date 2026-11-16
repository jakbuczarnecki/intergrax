# DELEGATION-PDC-F56-F35 — Operator commission for plan-approval signing (tickets T-IGX-040-C and T-IGX-040-D)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-F56-F35 |
| **Stage** | PDC-X (closing wave) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-WAVE2.md`; precedents `DELEGATION-PDC-RC-HANDOFF.md` and `DELEGATION-PDC-P6-ECAPFIX.md`; `DECISION-PDC-F56.md` |
| **Date** | 2026-11-15 |

## Commission (transcription)

1. Two findings of the independent audit were assigned to PDC-X for repair, and the implementer lane produced one ticket per finding, both `plan_pending`: **T-IGX-040-C** (F-56: the scoped production-delta check of `roadmap_replay_x` accepts a net-zero path; the loosening itself was decided by the Operator in `DECISION-PDC-F56.md`, 2026-11-14) and **T-IGX-040-D** (F-35: classification of the evidence-plane security audit trail in the COMPAT-X event-evolution gates). Both plans state that signing is not covered by an earlier delegation and returns to the Operator.
2. The orchestrator asked the Operator who signs the two plans. The Operator answered in English rendering: **"Yes, a delegate will sign"** (original wording in Polish: "Tak, delegat podpisze"). This is the Operator's answer as relayed by the orchestrator session; the rest of this record is the orchestrator's specification of the checks.
3. The earlier Operator decision `DECISION-PDC-F56.md` (gate-semantics loosening limited to the net-zero case) stands; this record does not widen it.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-040-C** and **T-IGX-040-D** only, and only after confirming, per ticket:
  - the plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest in the ticket;
  - the write set of the plan equals the change actually present on the branch and is limited to `tests/qualification/roadmap_replay_x/` and `tests/qualification/compat_x/` (plus the ticket, plan and decision documents of the batch);
  - no write path lies under a product path (`intergrax/`, `agents/`, `applications/`), `scripts/governance/`, `.github/`, `intergrax/contracts/` or `intergrax/applications/_shared/`;
  - no new registry, scheduler, engine, trust, policy or audit authority is introduced;
  - the plan states that it is not a certification verdict;
  - the implementer session and the delegate session are different sessions.
- Additional checks for **T-IGX-040-C**: the loosening is limited exactly to the net-zero case of `DECISION-PDC-F56.md` (a scoped path absent from the prod diff is accepted only when the file exists neither at HEAD nor at the pin); negative tests prove that every other case still fails (path present at HEAD, path existing at the pin, renamed-away path, wrong head, missing disposition, unknown pin); the frozen PRE-ENT list data and all existing tests are byte-identical.
- Additional checks for **T-IGX-040-D**: the change is classification only, bound to exact module, function and callee (no module-wide trust); a negative test proves that unclassified or reconstruction-shaped sites still fail; the two gate test files are unchanged; the diagnosis in the plan finds no real bypass (if the delegate finds a real bypass, it writes nothing and reports); the lint normalization of `_compat_x_event_evolution_r3_r1.py` is formatting, import order and `noqa` only.
- **Merge gate.** The test-only pull request that carries each batch merges under authorization 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes, head equals the verified SHA. The signer states in each ticket which gate governs. The Operator's acceptance that the EBH-7 identity gate stays red until the campaign's final re-pin applies.
- Write-set amendments of either ticket are **not** delegated and return to the Operator.

## Not delegated (unchanged)

OD, FRZ and closure signatures; lifting C10/C15; any write-set amendment; any other gate-meaning change (a further loosening or narrowing of any gate, including a second class of accepted scoped paths or a broader COMPAT-X classification); IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; the EBH-7 re-pin; certification.

## Separation

The implementer session and its implementer subagents never write `Approved:` or `Delegation:` lines. A delegate session does not implement the tickets it signs. The delegate never certifies and never merges outside the merge gate above.
