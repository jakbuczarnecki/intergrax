# IIAC session handoff — MCP-RC / PDC-P7-00 integrated baseline, activation binding (ACT-N11 re-bind)

| From | To |
|------|-----|
| `certification-governance-agent` (activation certifier) | Operator · orchestrator · terminal-verification session |

**Verdict:** `PASS` (activation binding manifest, authored by the activation certifier); the audits behind it are `PASS_SCOPED` and their manifests are `CONDITIONAL`
**Policy:** `IIAC-2026-11-05.1`
**implementation_session_id:** `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`
**auditor_session_id:** `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-2`
**certifier_session_id:** `local_8974948e-d8d3-45e3-96ca-fb61dfd5f249`
**Evidence publication SHA:** `29d66ae3fcc7a06a919d7e07541740841e8abfe8`
**Operator activation:** recorded in the signed `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (decision `IIAC-OD-MCP-RC-ACT-N11-2026-11-13`)

## Why these two single ids

The manifest schema holds one implementation session id and one auditor session id. The integrated baseline has several of each. The recorded implementation id is the one the addendum 2 manifest carries (the last and widest implementer designation, covering the PDC-X implementers and the MCP-RC implementer). The recorded auditor id is the PDC-P7-00 baseline auditor in its addendum 2 round: it audited exactly the tree that the binding manifest names as `audited_implementation_sha` (`324d9dba37cb54e1b74a79a05148d2dcc05e6d09`), covers all 158 ACT-N11 paths in its scope, and published the evidence at `29d66ae3fcc7a06a919d7e07541740841e8abfe8`. The other sessions are listed below and are not rewritten.

## All sessions that stand behind this binding

| Role | Designation | Notes |
|------|-------------|-------|
| MCP-RC implementer | `T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label, no UUID recorded |
| PDC-X implementers (first audit, addendum 1) | `T-IGX-036-J-T-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label |
| PDC-X implementers (addendum 2) | `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label; the recorded implementation id |
| MCP-RC first audit auditor | `MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2` | dossier `PARTIAL`, superseded by the re-audit |
| MCP-RC delta re-audit auditor | `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48` | dossier `PASS_SCOPED` |
| PDC-P7-00 baseline auditor | `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27` | dossier `PASS_SCOPED`, 153 paths |
| PDC-P7-00 addendum 1 auditor | `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-1` | `PASS_SCOPED`, 156 paths |
| PDC-P7-00 addendum 2 auditor | `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-2` | `PASS_SCOPED`, 158 paths; the recorded auditor id |
| MCP-RC preflight certifier | `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` | designation, `CERTIFIER_PREFLIGHT_PASS_SCOPED` |
| PDC-P7-00 preflight certifier (and addendum 1) | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` | designation, `CERTIFIER_PREFLIGHT_PASS_SCOPED` |
| Evidence-integration sessions | three rounds, see the two `EVIDENCE_INTEGRATION_RECORD.md` files | pin-only edits, no verdict |
| Designated signing session | `129c7788-d201-43f3-b695-ef48cec3a73f` | wrote the plain activation line under `DELEGATION-MCP-RC-2` |
| Activation certifier (this record) | `local_8974948e-d8d3-45e3-96ca-fb61dfd5f249` | new dedicated session; none of the above |

## Binding

- Previous pin `audit_execution_sha` `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` is preserved in history.
- Audited implementation (product tree) `324d9dba37cb54e1b74a79a05148d2dcc05e6d09`; test execution the same; audit execution and evidence publication `29d66ae3fcc7a06a919d7e07541740841e8abfe8`; certification publication `7801dcb1293e31ee0c483184efd59c01a2dc606b`.
- The four digest artifacts are the `IIAC-GOV-1` report set that the activation record already references (unchanged bytes), kept because the gate requires all five bound artifacts under `iiac_evidence/` and the decision cites the report digest `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e`.
- Pre-checks, validator output and the contract run are in `logs/` of this folder.

## Limits

This record is a `process_enforced` attestation (one git identity signs all commits). The binding manifest `PASS` is the certifier's, authored from `PASS_SCOPED` audits as the operator recorded in `OD-MCP-RC-04` section 3. Open findings (F-41, F-42, F-35, F-09 and the rest named in the manifest `limitations`), the unverified CI contexts and the audit coverage limits stay open. No stage closure, no FRZ promotion (counter 225), no production-readiness claim, no pull request, no merge.
