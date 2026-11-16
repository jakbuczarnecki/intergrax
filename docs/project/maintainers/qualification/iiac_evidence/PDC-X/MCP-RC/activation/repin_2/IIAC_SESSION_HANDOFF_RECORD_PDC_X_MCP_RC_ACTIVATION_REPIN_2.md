# IIAC session handoff — MCP-RC / PDC-X final integrated tree, activation re-pin 2 (ACT-N11 re-bind)

| From | To |
|------|-----|
| `certification-governance-agent` (activation certifier) | Operator · orchestrator · terminal-verification session |

**Verdict:** `PASS` (activation binding manifest, authored by the activation certifier); the audits behind it are `PASS_SCOPED` and their manifests are `CONDITIONAL`
**Policy:** `IIAC-2026-11-05.1`
**implementation_session_id:** `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`
**auditor_session_id:** `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-4`
**certifier_session_id:** `local_8974948e-d8d3-45e3-96ca-fb61dfd5f249`
**Evidence publication SHA:** `a1b3053b49c28124a0abde679d733f21a597797f`
**Operator activation:** the signed line of `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (decision `IIAC-OD-MCP-RC-ACT-N11-2026-11-13`), reused unchanged for this re-pin under `OD-MCP-RC-05`

## Why these two single ids

The manifest schema holds one implementation session id and one auditor session id. The recorded implementation id is the one the addendum 3 and addendum 4 manifests carry. The recorded auditor id is the PDC-P7-00 baseline auditor in its addendum 4 round: it audited exactly the tree that the binding manifest names as `audited_implementation_sha` (`d02c399508c40b4a06d846410334d677f226c18d`), covers the cumulative ACT-N11 set (161 paths) and published the evidence at `a1b3053b49c28124a0abde679d733f21a597797f`. The other sessions are listed below and are not rewritten.

## All sessions that stand behind this binding

| Role | Designation | Notes |
|------|-------------|-------|
| MCP-RC implementer | `T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label, no UUID recorded |
| PDC-X implementers (first audit, addendum 1) | `T-IGX-036-J-T-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label |
| PDC-X implementers (addenda 2 to 4) | `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label; the recorded implementation id |
| MCP-RC first audit auditor | `MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2` | dossier `PARTIAL`, superseded by the re-audit |
| MCP-RC delta re-audit auditor | `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48` | dossier `PASS_SCOPED` |
| PDC-P7-00 baseline auditor | `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27` | dossier `PASS_SCOPED`, 153 paths |
| PDC-P7-00 addenda 1, 2, 3 auditor | the same id with suffix `-ADDENDUM-1`, `-ADDENDUM-2`, `-ADDENDUM-3` | dossiers `PASS_SCOPED`, manifests `CONDITIONAL`; 156 paths, 158 paths, and audited tree `0fc8badd8` |
| PDC-P7-00 addendum 4 auditor | `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27-ADDENDUM-4` | `PASS_SCOPED`, 161 paths; the recorded auditor id |
| MCP-RC preflight certifier | `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` | designation, `CERTIFIER_PREFLIGHT_PASS_SCOPED` |
| PDC-P7-00 preflight certifier (parent, addenda 1 to 3) | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` | designation, `CERTIFIER_PREFLIGHT_PASS_SCOPED`; addendum 3 covers audit addendum 4 |
| Evidence-integration sessions | six rounds, see the two `EVIDENCE_INTEGRATION_RECORD.md` files | pin-only edits, no verdict |
| Designated signing session | `129c7788-d201-43f3-b695-ef48cec3a73f` | wrote the plain activation line under `DELEGATION-MCP-RC-2`; the line is not re-signed (`OD-MCP-RC-05`) |
| Activation certifier (this record and the first binding) | `local_8974948e-d8d3-45e3-96ca-fb61dfd5f249` | new dedicated session for the first binding, re-engaged by the orchestrator for this re-pin; none of the above |

## Binding

- Previous pins preserved in history: `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (P6 activation) and `29d66ae3fcc7a06a919d7e07541740841e8abfe8` (first MCP-RC binding, phase-2 `9949fe2c3dd64064f3b2002565b31c3c80b8cad6`).
- Audited implementation (product tree) `d02c399508c40b4a06d846410334d677f226c18d` (development `83d9390956d93484a7f32eed3a1bfa34cc591e14` merged, including the T-IGX-040-B edit of the certification protocol header); test execution the same; audit execution and evidence publication `a1b3053b49c28124a0abde679d733f21a597797f`, which is after the commit that edited the protocol (a material governance path), so the stale-audit rule is satisfied; certification publication `dc1e66fa140d81b28c2658e88b4e1aa16eb29ad2` (the certifier preflight addendum 3 evidence).
- The four digest artifacts are the `IIAC-GOV-1` report set that the activation record already references (unchanged bytes), kept because the gate requires all five bound artifacts under `iiac_evidence/` and the decision cites the report digest `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e`.
- Pre-checks, validator output and the contract run are in `logs/` of this folder.

## Limits

This record is a `process_enforced` attestation (one git identity signs all commits). The binding manifest `PASS` is the certifier's, authored from `PASS_SCOPED` audits as the operator recorded in `OD-MCP-RC-04` section 3. Open findings (F-56, F-35, F-52/F-51, F-53 and the rest named in the manifest `limitations`), the unverified CI contexts and the audit coverage limits stay open. The signed decision text predates this re-pin; its line is reused as the operator confirmed. No stage closure, no FRZ promotion (counter 225), no production-readiness claim, no pull request, no merge.
