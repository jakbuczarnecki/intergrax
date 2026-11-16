# IIAC session handoff — PDC-X P8-04, ACT-N11 rebind at the global audit SHA

| From | To |
|------|-----|
| `certification-governance-agent` (CERT-PDC certifier) | Operator · orchestrator · activation delegate |

**Verdict:** `PASS` (activation binding manifest, authored by the certifier under `DELEGATION-PDC-P8-ACT.md` (b)); the audit behind it is IA-GLOBAL `PASS_SCOPED` at `969cbd964c679d563b02e55f065b452d46fdf90b` (manifest value `PASS_SCOPED`); the domain audits are `PARTIAL` (D5 `FAIL`), manifests `CONDITIONAL`
**Policy:** `IIAC-2026-11-05.1`
**implementation_session_id:** `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER+T-IGX-041-A-Q-PDC-X-REPAIR-IMPLEMENTERS`
**auditor_session_id:** `PDC-P8-IA-GLOBAL-INDEPENDENT-AUDITOR-6573093f-9178-4c2c-93a3-64b0a5b6c657`
**certifier_session_id:** `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791`
**Evidence publication SHA:** `a73d15ab341b30e9748aad03894f6faa75ad4e09` (merge of PR #597)
**Operator activation:** none written here. The activation decision file and the plain-form line belong to the activation delegate (`DELEGATION-PDC-P8-ACT.md` (c)).

## Why these ids

The manifest schema holds one implementation session id and one auditor session id. The auditor id is the IA-GLOBAL session that audited `969cbd964` (the audited tree named in the binding manifest), wrote Addendum 3 and published the evidence at `a73d15ab3`. The implementation id is a label: the earlier PDC-X and MCP-RC implementer lanes plus the `T-IGX-041-A..Q` repair lanes; no application ids of implementers are recorded in the repository. The certifier id is this session's own id.

## All sessions that stand behind this binding

| Role | Designation | Notes |
|------|-------------|-------|
| PDC-X implementers (P1 to P7) | `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` | label (previous binding) |
| PDC-X repair implementers | `T-IGX-041-A-Q-PDC-X-REPAIR-IMPLEMENTERS` | label; tickets `T-IGX-041-A..Q` |
| IA-D1..IA-D8 domain auditors | see the eight manifests under `iiac_evidence/PDC-X/D1..D8/` | dossiers `PARTIAL` (D5 `FAIL`), manifests `CONDITIONAL` (D5 `FAIL`) |
| IA-GLOBAL global auditor | `PDC-P8-IA-GLOBAL-INDEPENDENT-AUDITOR-6573093f-9178-4c2c-93a3-64b0a5b6c657` | final `PASS_SCOPED` at `969cbd964` (section 15 of `PDC_INDEPENDENT_AUDIT.md`) |
| Certifier (this record) | `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791` | new dedicated session; none of the above |
| Activation delegate | not yet started | must differ from every session above (`DELEGATION-PDC-P8-ACT.md` Separation) |

## Binding

- `governance_implementation_sha` `2ae1014116d353bc7742985e62f2bccc2bfac798` unchanged; audited implementation and test execution `969cbd964c679d563b02e55f065b452d46fdf90b`; audit execution and evidence publication `a73d15ab341b30e9748aad03894f6faa75ad4e09`; certification publication `fd2ad28efde6216be38194b020188d116264adb6` (matrix and preflight).
- The four digest artifacts are the `IIAC-GOV-1` report set the activation record already references (unchanged bytes, under `iiac_evidence/`); the global audit set cannot be bound (the audit report sits outside `iiac_evidence/`, ACT-N14). The auditor's global manifest digests were checked equal to the bytes at `a73d15ab3`.
- Product-path set audited, recomputed with the gate's own helper: 74 paths from `a1b3053b4` to `969cbd964`, all under `intergrax/`; 192 paths from the original pin `2d3eceb6c`.
- Pre-checks and validator output are in `logs/` of this folder.

## Limits

`process_enforced` attestation (one git identity signs all commits). The binding `PASS` is the certifier's, authored from `PASS_SCOPED` audits as the Operator recorded in `DELEGATION-PDC-P8-ACT.md` (b), with every declared exception, the failed EBH-7 L3 replay (run 43738431866, not a B6 PASS) and all open items in the manifest `limitations`. Not a verdict on B6, not a certification of the whole PDC-X stage, not an FRZ or roadmap action; no closure, no production-readiness claim, no pull request merge.
