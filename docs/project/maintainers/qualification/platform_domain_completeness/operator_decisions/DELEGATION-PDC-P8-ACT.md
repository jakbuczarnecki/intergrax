# DELEGATION-PDC-P8-ACT — Operator decisions for the P8-04 ACT-N11 rebind

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P8-ACT |
| **Stage** | PDC-X P8-04 |
| **Basis** | `DELEGATION-PDC-STANDING.md` answer 6; `DELEGATION-PDC-P7-ACT.md` (precedent for PDC-P7-00); `DELEGATION-PDC-CLOSE.md` decision 3; Amendment 01 of the IIAC protocol (S-2); `OD-MCP-RC-04` section 3 (MCP-RC precedent); `PDC_P8_ACTIVATION_RUNBOOK.md` section 10 |
| **Date** | 2026-11-15 |

## Commission (transcription)

The activation runbook for PDC-P8-04 found three points that the existing delegation records leave open. The orchestrator asked the Operator one question with three parts; the Operator answered in English rendering: **"Yes, I approve (a)-(c)"** (original wording in Polish: "Tak, zatwierdzam (a)-(c)").

| Part | Decision |
| --- | --- |
| (a) | The orchestrating session merges the P8-04 pull request, which changes IIAC activation files, with a **merge commit**, after green CI (known red classes only) and a match of the head with the verified SHA. This is the same lift that `DELEGATION-PDC-P7-ACT.md` (d) granted for the PDC-P7-00 path. |
| (b) | The certifier session may author the binding manifest with verdict `PASS` from the audits whose verdicts are `PASS_SCOPED` (`OD-MCP-RC-04` section 3, the "P6 pattern"), with every scope limit, declared exception and open item written into the manifest `limitations`. The gate (ACT-N05) accepts only `PASS` in the binding manifest. This is **not** a verdict on B6, not a certification of the whole PDC-X stage and not an FRZ or roadmap action. |
| (c) | The activation delegate writes the activation line in the plain form `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>`, as the commission names it (Amendment 01, S-2). |

## Limits (unchanged)

The activation delegate confirms first, as in `DELEGATION-PDC-STANDING.md` answer 6: no product delta between the audited SHA and the activation HEAD (ACT-N11); `verify_iiac_policy_activation` preconditions and `verify_iiac_role_separation` pass; the global audit is published; the IIAC protocol and Amendment 01 permit delegated activation. If any confirmation fails, the delegate writes nothing and reports.

## Not delegated (unchanged)

OD signatures other than this activation line; FRZ promotion and the PDC-X closure line (PDC-P9-03, PDC-P9-04); lifting C10/C15; a change of the meaning of a gate; the verdict on B6 or on certification; a third repair loop.

## Separation

The implementer, the global auditor, the certifier and the activation delegate are four different sessions. The orchestrator never edits an auditor or certifier verdict.
