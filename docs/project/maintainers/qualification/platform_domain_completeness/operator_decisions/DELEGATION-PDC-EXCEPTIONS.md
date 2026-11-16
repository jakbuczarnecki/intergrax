# DELEGATION-PDC-EXCEPTIONS — Operator decision on the open P8 items (declared exceptions)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-EXCEPTIONS |
| **Stage** | PDC-X P8, before P8-04 |
| **Basis** | `DELEGATION-PDC-CLOSE.md` (decision 1: declared exceptions; decision 2: a third loop returns to the Operator); `DELEGATION-PDC-LOOP2.md`; `PDC_INDEPENDENT_AUDIT.md` sections 13.4, 13.5, 13.7 and 14 (Addendum 2 at `969cbd964`); `OD-PDC-IIAC-SCOPE.md` (verdict vocabulary) |
| **Date** | 2026-11-16 |

## Commission (transcription)

The global audit rated the stage PARTIAL at `d39a520dd` and again after repair loop 2 at `969cbd964`. The auditor stated that the verdict would be PASS_SCOPED for the scoped rows if the Operator accepted the open items as declared exceptions. The orchestrator asked one question; the Operator answered in English rendering: **"Yes, all as exceptions"** (original wording in Polish: "Tak, wszystkie jako wyjątki").

## The accepted declared exceptions

Each item is a declared, labelled exception with a reopen condition; none is repaired in PDC-X and none is silently upgraded. The affected requirement rows are rated PASS_SCOPED by this Operator authorization (`OD-PDC-IIAC-SCOPE.md` vocabulary). Each appears in the exception register of the P9 closure package.

| Item | What stays open | Source |
| --- | --- | --- |
| G-05 residual | 77 red tests of the incident scenario suite under `platform_proofs/`, 59 of them one signature drift (`IncidentInvestigatorAgent.get_steps()` versus `get_steps(context)`) | audit 13.4 |
| E1–E11 | The product MEDIUM findings excluded by T-IGX-041-M: G-08 durable CAS, G-09 strict host with `tenant_id=None`, G-11 stream resilience claim, G-12 `install_agent` into an occupied slot, F-D3-02 pins, and the others listed in the plan of T-IGX-041-M | audit 13.7 item 3 |
| G-10 residual | The `trace_bridge.v1` mismatch (runtime-event schema guard rejects `step_started` payloads on the finalized-trace path of the scenario runtime) | audit 13.5 |
| G-28 | The Cypher/Neo4j graph store is unchanged by T-IGX-041-P; with a bound coordinator and a scoped publication its evidence stays invisible after a re-ingest (fail closed, no leak); a fix needs a fourth file and Docker qualification, which the C10/C15 freeze excludes | audit 14.2 |

The exceptions already declared by `DELEGATION-PDC-CLOSE.md` decision 1 stay as declared. LOW and INFO findings (G-24 to G-27 and the earlier ones) go to the debt register of the P9 closure package.

## Effect

- The global auditor may state its final verdict at the audited SHA as PASS_SCOPED for the scoped rows, with the items above listed as exceptions and the open residuals named; it may not widen the scope or claim a PASS it cannot support.
- The certifier session may author the binding manifest with verdict `PASS` from the PASS_SCOPED audit as decided in `DELEGATION-PDC-P8-ACT.md` (b), with every item above in the manifest `limitations`. This is not a verdict on B6, not a certification of the whole PDC-X stage and not an FRZ or roadmap action.
- This record does not lift the C10/C15 freeze and does not change the status of the EBH-7 L3 replay (run 43738431866 stays recorded as failed and "not B6 PASS").

## Not delegated (unchanged)

OD signatures other than the activation line already delegated; FRZ promotion and the PDC-X closure line (PDC-P9-03, PDC-P9-04); lifting C10/C15; a change of the meaning of a gate; the verdict on B6 or on certification; a third repair loop.

## Separation

The implementer, the global auditor, the certifier and the activation delegate are different sessions. The orchestrator never edits an auditor or certifier verdict.
