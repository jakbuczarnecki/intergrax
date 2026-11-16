# DELEGATION-PDC-CLOSE — Operator decisions for the closing wave (P8 and P9)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-CLOSE |
| **Stage** | PDC-X P8, P9 preparation |
| **Basis** | `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-WAVE2.md` (decisions 4, 5, 6, 7, 8, 9); `DELEGATION-PDC-P7-REPAIR.md`; `DELEGATION-PDC-P7-REPIN.md`; `OD-PDC-IIAC-SCOPE.md` (verdict vocabulary) |
| **Date** | 2026-11-15 |

## Commission (transcription)

After the P7 repairs (tickets T-IGX-041-A..I merged) the Operator asked for one wave of decisions covering everything that remains, after which the orchestrator executes to the end. The orchestrator listed three decisions; the Operator answered in English rendering: **"I approve the whole wave (1)-(3)"** (original wording in Polish: "Zatwierdzam całą falę (1)-(3)").

1. **Declared exceptions.** The following stay declared, labelled exceptions with a reopen condition and are not repaired in PDC-X: F-D1-04 (fail-closed rule for a revoked revision without a trusted rollback target), the remainder of F-D5-02 (semantics of the MVP promotion gate), F-D7-03 (graceful worker shutdown with host drain), the "no production composition or caller" family (dossier F-04) and the missing-feature-versus-claim findings of option B (`DELEGATION-PDC-P7-REPAIR.md`), the four `fixture_strict_adapter_parity` tests of D3 (T-IGX-041-C deviation), and the row PDC-D7-R014, which is rated PASS_SCOPED by this authorization. Each appears in the exception register of the P9 closure package with its reopen condition; none is silently upgraded.
2. **P8 and one repair loop.** The global audit (IA-GLOBAL, one new independent auditor session) verifies every repaired HIGH or MEDIUM finding of T-IGX-041-A..H and the declared exceptions at the audited SHA. Findings that need a product repair go through one repair loop: the plan is signed by a delegate under the limits of `DELEGATION-PDC-P7-REPAIR.md`, the orchestrator merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md`, and one additional EBH-7 re-pin follows (`DELEGATION-PDC-WAVE2.md` decision 6). A third loop, a finding of a new kind, a hotspot path outside the named files, a change of the meaning of a gate or a new authority returns to the Operator.
3. **Merges, dependency bot, activation, closure.** The orchestrator merges pull requests covered by authorizations 1 and 3 after green CI (known red classes only: IIAC activation staleness until the P8-04 rebind; the EBH-7 identity gate until the re-pin is recorded) and a head match. The held dependency-bot and `uv.lock` pull requests are merged in one batch after the final re-pin is recorded (`DELEGATION-PDC-WAVE2.md` decision 7). A delegate session writes the IIAC activation record line of P8-04 after confirming that the protocol permits it. The orchestrator prepares the whole P9 package (FRZ promotion proposal, roadmap closure text, exception register, debt register); the Operator signs once at the end.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X closure line (PDC-P9-03, PDC-P9-04); lifting C10/C15; a change of the meaning of a gate; the verdict on B6 or certification; edits of hotspot paths outside files named in a signed plan.

## Separation

The implementer session and its subagents never write `Approved:` or activation lines. A delegate session does not implement or audit what it signs. An auditor never certifies and never fixes what it audits.
