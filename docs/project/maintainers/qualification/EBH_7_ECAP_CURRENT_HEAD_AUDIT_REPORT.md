# Enterprise Architecture Audit Report — EBH-7 B6 re-audit preparation

**Platform scope banner:** This report covers the **Intergrax Agent OS / Harness AI** four-tier platform (`intergrax/` · `intergrax/runtime/` · `agents/` · `applications/`) at EBH-7 closure scope. It is **pre-audit agent output** for independent B6 re-review — **not** independent B6 PASS, **not** OD-EBH7-08, **not** global FRZ PASS.

Normative ECAP proof bindings: [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md).

## A. HEAD and baseline

- **Audit SHA (dossier / F10 gates):** current `git rev-parse HEAD` @ publication
- **TEST_EXECUTION_SHA (certified L3 code):** `4eb7eb1a89f82215d5356aa7d000198e432777b0`
- **EVIDENCE_PUBLICATION_SHA (MANIFEST role):** `96069c09c2ca5a7a3a9586db336716c9430ca1d7`
- **Accepted baseline (C4):** `81679e67225f42dfbc1c508516e19488b6fbf453` (AUTHORIZED_BASELINE_SHA)
- **B5 canonical GHA replay:** run `41214574614` @ checkout `210accb224aad276a472307c8fefe3798bd44f10`
- **Audit mode:** FULL-HEAD (EBH-7) with DELTA vs accepted baseline + TEST_EXECUTION_SHA
- **Scope statement:** EBH-7 Protocol 03 independent replay dossier; F01–F09 retained; F10/F11 corrective wave

## B. Roadmap progress

- **Current parent stage:** EBH-7 — B5 verified; B6 independent re-audit pending
- **Applicable stages touched:** EBH-5/6 catalogs, DUP-X subset, ROADMAP-REPLAY-X parent closure (entry gate doc)

## C. Enterprise compliance coverage

- **FRZ families in scope:** 26 themes via EBH-6 family catalog + EBH-7 enterprise family matrix (closed-world row count gate)
- **Labels:** EXECUTION PROVEN @ TEST_EXECUTION_SHA for MANIFEST L3 profile; STATIC+EXECUTION for F10 matrix rows @ replay HEAD

| ECAP | Gate | Primary executable evidence |
| --- | --- | --- |
| ECAP-C1 | Closed-world inventory Tier-0..3 + families | `test_ebh6_family_row_count_closed_world`; `test_ebh7_structural_exit_traceability_and_mechanisms`; `EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md` |
| ECAP-C2 | Duplicate authority / cross-domain scan | `test_ebh_7_w4_gov_dup_gates.py` (DUP-X subset executed) |
| ECAP-C3 | VBN taxonomy + discoveries | `test_ebh6_prohibited_unresolved_zero`; `test_ebh6_findings_unclassified_zero`; CONFIG-X reconciliation; INT2E bundles |
| ECAP-C4 | Regression vs accepted baseline | `test_ebh6_config_x_head_reconciliation`; P03 manifest roles; B5 immutable GHA |
| ECAP-C5 | This §14 report + platform banner | `EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md` + F10 matrix |

## D. Architecture findings

| Finding ID | Class | FRZ/SYS-INV | Severity | Label |
| --- | --- | --- | --- | --- |
| B6-F10 | Evidence gap (remediated) | ECAP C1–C5 | BLOCKER→READY | Normative matrix + §14 report |
| B6-F11 | Documentation drift (remediated) | AUD | BLOCKER→READY | Dossier/traceability sync |

Historical corrective findings F01–F09: see `_gate_findings_register.py` — resolutions @ G09.

## E. Vendor/backend dependency changes

- **Inventory delta:** CONFIG-X 54/54 reconciled @ HEAD vs historical pass SHA (EBH-6 audit doc)
- **PROHIBITED unresolved:** 0 @ EBH-6 register scope (`test_ebh6_prohibited_unresolved_zero`)
- **Normative RPL:** appendix + convergence matrix — not 56× replaceability demo

## F. Cross-layer ownership and authority

- DUP-X L2 subset + NPSC boundary gates executed via W4 EBH-7 gates
- Tier boundary: `test_hardening_3_layer_boundary_gate` (family matrix owner)

## G. Runtime reliability and failure semantics

- Scoped to qualification gates and SSOT registers — no new production semantic change @ TEST_EXECUTION_SHA for F10/F11 wave

## H. Impact on historical certification

- Prior B5 run `41212701986` superseded (pre-verify digest reconcile)
- TEST_EXECUTION_SHA unchanged; F10/F11 evidence @ replay HEAD (allowlisted qualification infra)

## I. Scenario readiness / architecture-change risk

- SCENARIO-READINESS-X: not in scope for this corrective wave

## J. Blockers vs missing evidence

- **Unresolved for independent B6:** external auditor must re-run Protocol 03 worksheet and issue PASS/FAIL
- **Agent disposition:** F10/F11 **READY FOR INDEPENDENT B6 RE-AUDIT**

## K. Closure verdict and next action

- **Agent recommendation:** Independent B6 re-audit with MANIFEST L3 profile + F10 matrix + B5 GHA `41214574614`
- **Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · EVOL-X · global FRZ PASS
