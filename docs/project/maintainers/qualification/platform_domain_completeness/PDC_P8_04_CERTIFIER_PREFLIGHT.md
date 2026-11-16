# PDC-X P8-04 — CERT-PDC preflight

Ticket reference: `governance/tickets/T-IGX-040-C.md` (closing-wave ticket named by the runbook; no ticket file edited).

## Role statements (R-2)

- This is a new dedicated session, not a continuation and not an implementation session: `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791`, agent id `certification-governance-agent`, role certifier, started 2026-11-16 by the orchestrating session `local_80099b72-d308-4d1a-a3cb-99adbd0abe51`.
- It is none of: the implementers of `T-IGX-036-*` and `T-IGX-041-*`, IA-D1..IA-D8, IA-GLOBAL (`PDC-P8-IA-GLOBAL-INDEPENDENT-AUDITOR-6573093f-9178-4c2c-93a3-64b0a5b6c657`), a signing delegate, the activation delegate.
- It writes no `Approved:` line, no activation line, no OD file, no activation-record edit; it merges nothing and edits no auditor verdict, gate, verifier, protocol or ticket.

## Read

`PDC_P8_ACTIVATION_RUNBOOK.md` (section 0 first; 2 to 5, 8, 9), `operator_decisions/DELEGATION-PDC-P8-ACT.md`, `DELEGATION-PDC-EXCEPTIONS.md`, `DELEGATION-PDC-CLOSE.md`, `PDC_INDEPENDENT_AUDIT.md` sections 1 to 3, 5 to 9, 13 to 15, `iiac_evidence/PDC-X/global/` (manifest and validator log), the eight P7 manifests and the verdict lines of the eight P7 reports, `PLAN_T-IGX-041-M.md` (E1 to E11), the gate `iiac_activation_gate.py` (sections on SHA ancestry, bound artifacts and ACT-N11), `IIAC_POLICY_ACTIVATION_RECORD.json`, and the merged MCP-RC precedent (re-pin 2 manifest, handoff, scripts, certifier report).

## Method and results

| Step | Result |
| --- | --- |
| Start state | `origin/development` `859d64c08`; PRs #597, #598, #599 merged; `git diff --name-only 969cbd964 origin/development -- intergrax agents applications` empty |
| Pre-checks (runbook 5.2, pre-`C`) | `logs/p8_04_01_precheck_preC.log`: all checks OK, exit 0 |
| Auditor manifests D1..D8 and global (checklist item 3) | `logs/p8_04_02_auditor_manifests_d1_d8_global.log`: 18 validator calls `PASS`, exit 0 |
| Matrix | `PDC_DOMAIN_CERTIFICATION_MATRIX.md` (this commit) |

Commit `C` carries this preflight, the matrix, the pre-check scripts and logs 01 and 02. The binding manifest, the handoff record, the validator logs and the certifier report follow in the next commit.

## Not claimed

No audit, no test run, no verdict on B6, no certification of the stage, no FRZ promotion, no closure, no production-readiness claim.
