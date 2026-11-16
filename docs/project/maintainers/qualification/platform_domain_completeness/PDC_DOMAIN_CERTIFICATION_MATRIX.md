# PDC-X — domain certification matrix (CERT-PDC consolidation, PDC-P7-CERT / input of PDC-P8-04)

| Field | Value |
| --- | --- |
| **Record** | `PDC_DOMAIN_CERTIFICATION_MATRIX` |
| **Role** | CERT-PDC, certifier (`OD-PDC-IIAC-SCOPE.md`: "Consolidates D1 to D8 verdicts into `PDC_DOMAIN_CERTIFICATION_MATRIX`; checks manifests and role separation. Not an auditor and not an implementer.") |
| **Certifier session** | `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791` (agent id `certification-governance-agent`), a new dedicated session started 2026-11-16 on branch `p8-04-rebind` cut from `origin/development` `859d64c084c527d122d7d5a8ab86f77fb9489c87`. It is not an implementer (`T-IGX-036-*`, `T-IGX-041-*` lanes), not IA-D1..IA-D8, not IA-GLOBAL, not a signing delegate and not the activation delegate. `process_enforced`: one git identity signs all commits; the separation is an attestation. |
| **Date** | 2026-11-16 |
| **Policy** | `IIAC-2026-11-05.1`, profile `iiac-v1` |
| **Authority** | `operator_decisions/DELEGATION-PDC-P8-ACT.md` (b), `DELEGATION-PDC-EXCEPTIONS.md`, `DELEGATION-PDC-CLOSE.md` decisions 1 to 3, `PDC_P8_ACTIVATION_RUNBOOK.md` section 0 and 8 |
| **Nature** | A consolidation. It **upgrades no verdict**: every verdict below is quoted from the auditor or from an Operator record. It is not an audit, not a verdict on B6, not a certification of the whole PDC-X stage, not an FRZ promotion and not a roadmap or closure action. No `Approved:` or activation line is written here. |

## 1. SHAs used

| Role | SHA | Check |
| --- | --- | --- |
| `G` — corrective / audited / test-execution SHA (IA-GLOBAL delta audit) | `969cbd964c679d563b02e55f065b452d46fdf90b` | merge of PR #593; last product-path change is PR #594 (`7f1859e0f`); `git diff --name-only G origin/development -- intergrax agents applications` empty |
| `E` — `audit_execution_sha` = `evidence_publication_sha` | `a73d15ab341b30e9748aad03894f6faa75ad4e09` | merge of PR #597 (IA-GLOBAL Addendum 3, the final verdict, with the auditor manifest, scope, findings and test matrix at their final bytes); descendant of `G`, ancestor of `origin/development`; product diff `G..E` empty (gate helper) |
| `C` — `certification_publication_sha` | the commit that carries this matrix and the preflight (reported in the PR and in the binding manifest; a file cannot hold its own SHA) | descendant of `E`, different from `G` |
| `governance_implementation_sha` | `2ae1014116d353bc7742985e62f2bccc2bfac798` | unchanged |

Why `E` is the merge of PR #597 and not `G` or the tip: `G` is the audited tree and holds none of the addenda; `E` is the first commit at which the global audit report, its manifest, scope, findings, test matrix and the Operator exception record are all at final bytes (the later PRs #598 and #599 add the failed-replay record and the runbook; neither changes a bound artifact, a product path or a material governance path, checked). The later of two valid commits was not chosen because the gate's `audit_execution_sha` anchors the stale-audit scan and `E` already lies after the last material governance change (`7b1c0a4b7`, protocol header).

## 2. Verdict inputs (quoted, not re-derived)

| Source | Value |
| --- | --- |
| IA-GLOBAL final verdict (`PDC_INDEPENDENT_AUDIT.md` section 15, Addendum 3) | **PASS_SCOPED** at `969cbd964` for the scoped rows of sections 7 and 13, "only" with the declared exceptions named; manifest value `PASS_SCOPED`; "not a PASS", "not a repeat-run PASS", "not a verdict on B6 or on certification". It was PARTIAL at `d39a520dd` and at `969cbd964` before the exception record. |
| Operator exception record (`DELEGATION-PDC-EXCEPTIONS.md`, answer "Yes, all as exceptions") | G-05 residual (77 scenario reds), E1 to E11, G-10 residual (`trace_bridge.v1`), G-28 (Cypher store); LOW and INFO (G-24 to G-27 and earlier) go to the debt register |
| Earlier declared exceptions (`DELEGATION-PDC-CLOSE.md` decision 1) | F-D1-04, remainder of F-D5-02, F-D7-03 (with row `PDC-D7-R014` PASS_SCOPED), "no production composition or caller" family (dossier F-04) and the option-B missing-feature findings, four D3 `fixture_strict_adapter_parity` tests |
| Binding authority (`DELEGATION-PDC-P8-ACT.md` (b)) | The certifier may author the binding manifest `PASS` from `PASS_SCOPED` audits with every limit in `limitations`; not a verdict on B6, not a certification of the stage, not an FRZ or roadmap action |

## 3. Matrix

Per-domain re-verification counts are the rows of the IA-GLOBAL section 7 table (76 in-scope rows) with the changes of section 13.2 applied by the certifier (F-D7-02 and F-D6-12 become VERIFIED_FIXED at `d39a520dd`); `969cbd964` changed no other row (section 14.5). **Discrepancy, not resolved:** the auditor's own net (section 13.2) is "63 VERIFIED_FIXED, 13 PARTIALLY_FIXED"; applying both changes to the section 7 rows gives 64 and 12. The auditor's figure is not edited; the certifier does not know which row the auditor counted differently (**UNVERIFIED**). It does not change any disposition below.

| Domain | Audited SHA (P7) | Audit verdict (IA-Dn, manifest) | Audit PR | Repair ticket / PR | P8 re-verification (IA-GLOBAL, rows fixed / partial) | Declared exceptions and open residuals (labels as recorded) | Certifier disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 AGENT-LIFECYCLE | `c08efe46a` | PARTIAL (`CONDITIONAL`) | #560 | T-IGX-041-A / #574; T-IGX-041-M / #584 | 10 / 2. F-D1-02 admin path, F-D1-03 fixed at service level only; F-D1-02 pipeline path and F-D1-11 durable CAS partial | F-D1-04 (revoked revision without trusted rollback target); "no production composition" family (F-D1-01 HIGH attestation never verified, F-D1-02/03 inert in production); G-08 durable CAS narrowing (named among E1-E11 by IA-GLOBAL 15.2); G-12 `install_agent` into an occupied slot (accepted as E1-E11) | PASS_SCOPED only inside the IA-GLOBAL PASS_SCOPED. The P7 verdict PARTIAL is not rewritten. HIGH items stay open by authorization and must be read as "inert in production", not "fixed". |
| D2 CONTEXT-KNOWLEDGE | `c08efe46a` | PARTIAL (`CONDITIONAL`) | #563 | T-IGX-041-B / #575; M / #584; P / #594 | 7 / 3 (F-D2-08, F-D2-10, F-D2-18). G-01 loader HIGH VERIFIED_FIXED by M; G-23 graph reingest VERIFIED_FIXED for the in-memory store by P | none named by CLOSE; E4 remainder of G-04 (extras-profile reds), E3 (`list_collections` on chroma and qdrant), G-28 Cypher store keeps the unscoped publication key (fail closed, no leak, not reproduced) | PASS_SCOPED inside the global verdict; G-28 is an accepted exception with a Docker qualification need excluded by C10/C15. P7 verdict not rewritten. |
| D3 MODEL-RUNTIME | `c08efe46a` | PARTIAL (`CONDITIONAL`) | #561 | T-IGX-041-C / #569; M / #584 | 5 / 2 (F-D3-02 refusal and finish normalization pins, F-D3-05 stale tests) | four `fixture_strict_adapter_parity` tests; G-05 residual, 77 incident scenario reds (59 `get_steps()` missing `context`; 99 at `cba32d247`, 77 at `d39a520dd`); G-11 stream resilience claim; F-D3-02 pins (E1-E11) | PASS_SCOPED inside the global verdict. The 77 reds stay red. P7 verdict not rewritten. |
| D4 HOST-RUNTIME | `c08efe46a` | PARTIAL (`CONDITIONAL`) | #566 | T-IGX-041-D / #571 | 9 / 0 | "no production composition or caller" family (F-D4-02/-03), option-B missing-feature findings | PASS_SCOPED inside the global verdict. Contract run clean at `cba32d247` (491 passed, 3 skipped). P7 verdict not rewritten. |
| D5 EVALUATION | `c08efe46a` | **FAIL** (`FAIL`; the FAIL comes from one row, `PDC-D5-R015`) | #562 | T-IGX-041-E / #568; M / #584 | 11 / 0. F-D5-01 (HIGH, the cause of the row `PDC-D5-R015` FAIL) VERIFIED_FIXED for the writer, reader and the three host call sites | remainder of F-D5-02 (MVP promotion gate semantics, still yields GRANTED); G-09 strict host with `tenant_id=None` wires the shared untenanted registry (accepted as E1-E11); G-10 residual `trace_bridge.v1` mismatch (unowned) | PASS_SCOPED inside the global verdict. **The row verdict `PDC-D5-R015` FAIL of IA-D5 is not rewritten:** IA-GLOBAL issued no row-level re-grade (it did not re-certify the 172 rows), so whether R015 would now be PASS, PARTIAL or PASS_SCOPED is **UNVERIFIED**. The G-09 case keeps the row from a clean PASS. |
| D6 RUNTIME-SECURITY | `c08efe46a` | PARTIAL (`CONDITIONAL`; 0 HIGH, 6 MEDIUM) | #565 | T-IGX-041-F / #573; M / #584 | 5 / 0 (F-D6-12 VERIFIED_FIXED at `d39a520dd`) | "no production composition" (F-D6-04), F-D6-02 relabel only (no tamper evidence), F-D6-05 digest part (missing-feature) | PASS_SCOPED inside the global verdict. P7 verdict not rewritten. |
| D7 BACKGROUND-SCHEDULING | `c08efe46a` | PARTIAL (`CONDITIONAL`) | #559 | T-IGX-041-G / #570; M / #584 | 10 / 2 (F-D7-07, F-D7-12). F-D7-02 VERIFIED_FIXED at `d39a520dd` after G-02 | F-D7-03 (graceful worker shutdown with host drain; row `PDC-D7-R014` PASS_SCOPED by Operator authorization); F-D7-20 (`DocumentStoreTaskWorker.stop()` abandons a running thread, part of F-D7-03 per E5); G-07 / E2 (`recover_interrupted_running` not CAS) | PASS_SCOPED inside the global verdict. P7 verdict not rewritten. |
| D8 ARTIFACT-LIFECYCLE | `c08efe46a` | PARTIAL (`CONDITIONAL`; the manifest field `dossier_verdict` is empty, the report says PARTIAL) | #564 | T-IGX-041-H / #572; M / #584 | 7 / 3 (F-D8-03 raw storage tools, F-D8-04 workspace scope, F-D8-17 docs) | F-D8-01 (retention not enforced by any production composition, HIGH, open by authorization); E1 tenant or prefix binding of every other object key | PASS_SCOPED inside the global verdict. G-03 (`workspace.*` bypass) VERIFIED_FIXED by M. P7 verdict not rewritten. |

The eight P7 manifests (D1..D8) carry `certifier_session_id` placeholders (`...-PENDING`) and `evidence_publication_sha` of 40 zeros; the certifier does not edit an auditor manifest. The single bound manifest of this binding is the one written under `iiac_evidence/PDC-X/P8/activation/` (`PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md`, section "SHAs").

### 3.1 Cross-cutting rows

| Row | Source | Status at `E` | Certifier disposition |
| --- | --- | --- | --- |
| T-IGX-041-I (TENANT-X class-field carriers, carrier-count re-pin) | PR #576, no product path | merged | not re-verified here; TENANT-X 213 passed at `cba32d247` (audit step 05) |
| T-IGX-041-J (EBH-7 re-pin W1', pin `d39a520dd`) | PR #586; W2' record PR #598 | the L3 replay run **43738431866 failed**; recorded as "not B6 PASS"; not re-run by the Operator condition of `DELEGATION-PDC-LOOP2.md`; the two test defects behind it were repaired by T-IGX-041-Q (PR #593) | **not a B6 PASS**; the EBH-7 identity gate stays a known red class; no claim about B6 |
| T-IGX-041-K, -L, -N, -O | PRs #581, #582, #583, #585 | merged; K and O classifier gates exercised by one mutation run in Addendum 2 (8/19 and 8/20 fail on a removed entry) | as stated by IA-GLOBAL; COMPAT-X counters 14 order-dependent reds (G-24, debt) |
| T-IGX-041-M (P8 repair loop 1) | PR #584 | merged; closes G-01, G-02, G-03, G-06, F-D6-12; E1..E11 excluded by the ticket | E1..E11 are accepted exceptions (`DELEGATION-PDC-EXCEPTIONS.md`) |
| T-IGX-041-P, -Q (repair loop 2) | PRs #594, #593 | merged; P closes G-23 for the in-memory graph store; Q de-duplicates appended classifier paths | G-28 stays (Cypher store); a third repair loop returns to the Operator (`DELEGATION-PDC-CLOSE.md` decision 2) |
| MCP-RC carry-overs | MCP-RC re-pin 2 report | owner of the strict xfail (AF-05, AF-16), RA-02, RA-03, RA-04 acceptance not recorded in any record the certifier read | **open**, stated in the manifest limitations |
| F-35 (COMPAT-X), F-56 (frozen PRE-ENT list) | earlier dossiers | not re-verified by the certifier; COMPAT-X counters re-pinned by #582 | **UNVERIFIED** |
| LOW/INFO (G-13..G-22, G-24..G-27) | IA-GLOBAL sections 9, 13.5 | debt register of the P9 package | not a binding input |

## 4. Mechanical checks of the certifier (logs under `iiac_evidence/PDC-X/P8/activation/logs/`)

1. `p8_04_01_precheck_preC.log`: ancestry (`2ae101411 -> G -> E -> HEAD`), product diffs `G..E`, `E..HEAD`, `G..HEAD` empty (gate helper `product_paths_changed_between`), no `MATERIAL_GOVERNANCE_PATHS` change after `E` or `G`, the four bound `IIAC-GOV-1` artifacts unchanged and byte-equal at `E`, findings register passes ACT-N15, the auditor's global manifest digests equal the bytes at `E`. Exit 0.
2. `p8_04_02_auditor_manifests_d1_d8_global.log`: `verify_iiac_audit_manifest --require-handoff-template` and `verify_iiac_role_separation --strict-product-diff` on the nine auditor manifests (D1..D8 and global): 18 calls, all `PASS`, exit 0.

The product-path set audited, recomputed with the gate's own helper: **74** paths from `a1b3053b4` (evidence SHA of the binding in force) to `G`, all under `intergrax/` (none under `agents/` or `applications/`); **192** paths from the original pin `2d3eceb6c` to `G`. The earlier count of 62 is not used.

## 5. Not done, not claimed

Not a row-by-row recertification of the 172 requirement rows, not a re-audit, no test run beyond the two validators and the pre-check. Not stage closure, not an FRZ promotion, not a verdict on B6 or certification, no production-readiness claim, no `Approved:` or activation line, no edit of an auditor file, the gate, a verifier, the protocol or the activation record.
