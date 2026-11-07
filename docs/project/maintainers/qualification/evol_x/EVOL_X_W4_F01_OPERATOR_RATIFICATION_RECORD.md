# EVOL-X W4-F01 - operator ratification record (option C)

**Status:** `RATIFIED BY DELEGATION - option C` · **Date:** 2026-11-07 · **Ticket:** `governance/tickets/T-IGX-019-A.md`
**Assurance level:** `process_enforced`. This file is not a signature mechanism and not a cryptographic proof.

## 1. Source of the decision

1. On 2026-11-06 the operator replied in the chat of the recovery session `local_7eeb7387-df63-460d-88e9-be7359430173`, to the question "Czy ratyfikujesz W4-F01 (opcja C) przed audytem?", with: `Ratyfikuj C`.
2. On 2026-11-07 the operator wrote in the same chat: "Chcę, abyś kompletnie zamknął to zadanie. [...] Dostajesz moje wszystkie zgody i autoryzacje, dlatego przeprowadź tę operację end-to-end autonomicznie." The recovery session records the ratification below on that instruction (delegated entry). The operator may replace it with a handwritten line or their own commit at any time.

## 2. Decision recorded

| Field | Value |
|---|---|
| Decision ID | `W4-F01-RATIFICATION-2026-11-07` (assigned by the recovery session by delegation; not a register id) |
| Finding | W4-F01 (`EVOL_X_W4_01_INDEPENDENT_AUDIT_FINDINGS.json`) |
| Decision scope | Ratification of **option C**: `.gitattributes` `text eol=crlf` for `docs/project/maintainers/qualification/IIAC_SESSION_HANDOFF_RECORD_2026_11_05_CERTIFICATION.md` (commit `4aa91dcc0e569c602baa782d2bb742b3bd55837d`) as a **non-material** repair of the activation `handoff_digest` checkout dependence |
| Decision status | ACCEPTED |
| Evidence | independent auditor `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (rounds 1-5) judged option C acceptable; certifier `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` and the second certifier `local_e34301ce-67df-40bf-b274-b2e062581541` verified it with `core.autocrlf` `true`, `false` and `input` on fresh clones; code pin `fdb1e7a62` |

## 3. Boundaries (what this does NOT decide)

* No edit to `IIAC_POLICY_ACTIVATION_RECORD.json`, `iiac_activation_gate.py`, `scripts/governance/*`, the certification protocol or `MATERIAL_GOVERNANCE_PATHS`.
* Not IIAC OD-01 (`IIAC-OD-01-2026-11-05`, policy activation) and not a re-activation.
* Does not decide OD-EVOL-12, OD-EVOLX-08, P3, B6 acceptance, FRZ promotion or EVOL-X closure.
* Options A (re-bind the digest to the blob) and B (gate hashes the blob) remain the recommended target of the next activation amendment, which needs its own ticket and a full IIAC re-audit.
