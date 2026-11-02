# EVOL-X W1 — Consolidated operator decision register

**Record status:** **Operator authorization package 2026-11-01** — see [`EVOL_X_OPERATOR_AUTHORIZATION_2026-11-01.md`](EVOL_X_OPERATOR_AUTHORIZATION_2026-11-01.md).

## W0 signature status

| ID | Status | Signed date | Notes |
|----|--------|-------------|-------|
| **OD-EVOLX-01** | SIGNED / APPROVED | 2026-11-01 | Batch B–F execution entry |
| **OD-EVOL-01** | SIGNED / APPROVED | 2026-11-01 | Option C @ baseline `884885774…` |
| **OD-EVOL-02** | SIGNED / APPROVED | 2026-11-01 | Policy (i) |
| **OD-EVOL-05a** | SIGNED / APPROVED | 2026-11-01 | Full assessor scope |
| **OD-EVOL-05b** | SIGNED / APPROVED | 2026-11-01 | **Option 1** qual-only; **no** W4-04 port |
| **OD-EVOL-05c** | SIGNED / APPROVED | 2026-11-01 | NA-01..04 |
| **OD-EVOL-06** | SIGNED / APPROVED | 2026-11-01 | Sidecar revision facts |
| **OD-EVOL-09** | SIGNED / APPROVED | 2026-11-01 | X-01,02,04,05; X-03,06,07 proofs in W2 |
| **OD-EVOL-10** | SIGNED / APPROVED | 2026-11-01 | Before W2-09 / W3-05 |
| **N-04 (OD) / N4-A** | SIGNED / APPROVED | 2026-11-01 | Migration-doc semver only |
| **OD-EVOL-03** | SIGNED / APPROVED | 2026-11-01 | W0-07: M thresholds ratified as canonical (no lowering) → W6-01 |
| **OD-EVOL-07** | SIGNED / APPROVED | 2026-11-01 | W0-07: P-H historical-commit promotion ratified |
| **OD-EVOL-08** | SIGNED / APPROVED | 2026-11-01 | W0-07: M-09 SLA ratified |
| **OD-EVOL-08-M1** | **OPERATOR APPROVED** | 2026-11-01 | M-09H measurement policy (human attribution clock). Supplements OD-EVOL-08; thresholds unchanged. See decision text below. |
| **OD-EVOL-13** | **OPERATOR APPROVED** | 2026-11-01 | Automation-first **scoped** EVOL-X acceptance (`EVOL-M-09A`, `AUTOMATION_FIRST_SCOPED` FINAL). **M-09H human remains NOT_MEASURED.** See [EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md](./EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md). |
| **OD-EVOL-14** | **OPERATOR APPROVED — DIRECTION AND IMPLEMENTATION ONLY** | 2026-11-02 | Automation-first organizational B6 **direction** for EVOL-X `AUTOMATION_FIRST_SCOPED` only. Does **not** put `ECAP-P4-EVOL-SCOPE-AUTO-B6` in force, name an auditor, authorize self-approval, promote `FRZ-*`, or declare EVOL-X CLOSED. Activation remains `PENDING_REQUIRED_APPROVALS`. See the decision section below. |
| **OD-EVOL-15** | **OPERATOR APPROVED — FULL AUTO-B6 EXECUTION WITHIN OPERATOR AUTHORITY** | 2026-11-02 | Authorizes completing every remaining **implementer/operator** action on the AUTO-B6 path (technical qualification replay, ECAP amendment **preparation**, independent attestation **deployment specification**, adversarial automation, Mode A/B orchestration, E4 **preparation**). Does **not** authorize ECAP maintainer publication, independent security **signature**, external key custody, formal B6 PASS, `B6_ACCEPTANCE_PUBLICATION_SHA`, FRZ promotion, OD-EVOL-12/OD-EVOLX-08 signing, or EVOL-X CLOSED. Supplements OD-EVOL-14. |
| **OD-EVOL-16** | **OPERATOR APPROVED — PATH_B_SELECTED** | 2026-11-02 | Selects **Path B** (ECAP-P4 AUTO-B6 + external independent attestor) as the certification mechanism for EVOL-X formal B6. Authorizes implementer/operator preparation and technical replay only. Does **not** activate ECAP (`normative_effect` stays `NOT_IN_FORCE`), provision KMS/HSM, impersonate maintainer/security reviewer/attestor, or declare EVOL-X CLOSED. Recorded in governance as `certification_mechanism: PATH_B_SELECTED`. |
| **OD-EVOL-11** | SIGNED / APPROVED (appointment authorized; **auditor identity NOT NAMED**) | 2026-11-01 | W0-08: operator-appointed independent auditor (must differ from implementer/agent). Operator message and auth package carry no name; **operator must record the auditor id in this row before W7-01**. Agent does not self-appoint. |
| **OD-EVOL-12** | **PENDING** | — | W7-03 (Batch G not in 2026-11-01 scope) |
| **OD-EVOLX-08** | **PENDING** | — | W7-04 (Batch G not in 2026-11-01 scope) |

## Decision text (ratified where signed)

| ID | Proposed decision text (operator signs or edits) | Technical basis | Alternatives | W2–W6 impact | Authorizes code? | Before W2? | Before later wave |
|----|---------------------------------------------------|-----------------|--------------|--------------|------------------|------------|-------------------|
| **OD-EVOLX-01** | «Authorize EVOL-X stage entry at START_HEAD `5a4e922…`.» | Roadmap §3.0.9 mandatory stage; EBH-7 closed | Defer stage | Enables all waves | No | **Yes** | — |
| **OD-EVOL-01** | «Adopt placement **Option C**: `testing_support/change_safety/` + `tests/qualification/evol_x/adapters/`.» | Packaging + import boundaries | A/B/D per ADR | W2 paths | W2+ in Q/T only | **Yes** | — |
| **OD-EVOL-02** | «Workflow policy **(i)**: VERSIONED_DEPLOY + pin/checkpoint; at-rest profile MIGRATE; in-flight definition change **EXPLICITLY_UNSUPPORTED**.» | Pinning verified; no in-flight executor | (ii) build in-flight migration (Class C) | W4-02 matrix | No (policy) | **Yes** (for W4) | W4-02 |
| **OD-EVOL-05a** | «Require full EVOL deploy **assessor** scope (DM-01..07) in qualification harness.» | N-03 gap documented | Narrow FRZ-07 scope | W4-03, S7 | No (Q tooling) | No | W4 |
| **OD-EVOL-05b** | «**Option 1:** qualification-only FRZ-07 scope — **no** W4-04 production precondition port.» (Option 2 text retained for reference only) | S7 assessor ≠ production protection | Option 2: AD port default off | W4-04 **not authorized** | No (2026-11-01) | No | W4-04 blocked |
| **OD-EVOL-05c** | «Accept NA-01..04 out-of-scope register with evidence N-06/N-10.» | PROD-Q/DIST-SCALE live OPEN | Expand scope | S7 NA arms | No | No | W4/W6 |
| **OD-EVOL-06** | «Revision state facts via sidecar `schema_snapshot@revision` manifest (not RuntimeRevision fields).» | Avoid Class B unless needed | Add RuntimeRevision fields | W4-03 facts | Optional data files | No | W4-03 |
| **OD-EVOL-09** | «Accept exclusions X-01,02,04,05; require W2 proofs for X-03,06,07.» | W1 exclusion register | Shrink/expand universe | W2 CW-04 | No | **Yes** (proofs W2) | W2-05 |
| **OD-EVOL-10** | «Authorize: (1) `evol_x_p3_l3` profile in `unit-tests.yml`; (2) CONSERVATIVE_FULL = CI `dev-ci` manifest node list, not default local `-m gate` collect; (3) push workflow to `main` before GHA dispatch; (4) optional collect-only reruns under `dev-ci`.» | N-18 seven root causes; RC-7 baseline defect | Use broken local marker set | W2-09, W6-04 | Workflow YAML only | **Yes** (W2-09) | W6 |
| **OD-EVOL-12** | «At W7 promote FRZ-EVOL-08/10 to PASS with scoped footnotes (Option A).» | FRZ shared owners | Leave OPEN to ENT-AUDIT-X | W7-05 checklist | Docs only | No | W7 (**PENDING**) |
| **OD-EVOL-08-M1** | «**EVOL-M-09H measurement policy (OPERATOR APPROVED).** (1) **`attribution_seconds`** is the human clock: it starts when the evaluator receives the **complete diagnostic packet** for the case and stops when the evaluator **submits the diagnosis, before** the expected/canonical answer is shown. (2) **`tool_seconds`** (qualified dossier/drill tool runtime on the seed path, as EVOL-M-09T) is measured **separately** and is **not** part of `attribution_seconds`. (3) **Thresholds unchanged:** `tool_seconds` ≤ **120 s per case**; **median `attribution_seconds` ≤ 900 s** (equality allowed). The 900 s median MUST NOT be applied to tool timing. (4) Drill artifact: `schema_version` exactly `evol_m09_human_drill.v2`, `measurement_policy_id` exactly `OD-EVOL-08-M1`; v1 is rejected fail-closed. (5) Human measurements are produced only by the human evaluator (no agent/synthetic timings); Mode A verifies arithmetic and bindings, not the human attestation.» | Finding `F-EVOLX-P0-M09H-ATTRIBUTION-SLA-FALSE-PASS`: v1 applied 900 s to tool median | Keep v1 (rejected: false PASS) | Drill v2 schema, Mode A dual SLA, N-M09-17..28 | Yes (Q tooling: `artifacts.py`, `p03.py`, tests) | No | M-09H FINAL evidence |
| **OD-EVOL-13** | «**OPERATOR APPROVED — EVOL-X AUTOMATION-FIRST SCOPED ACCEPTANCE** — see [EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md](./EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md).» | Distinct **EVOL-M-09A** + P03 profile `AUTOMATION_FIRST_SCOPED`; **M-09H NOT_MEASURED** | Human-only FINAL (still valid under **HUMAN_FULL**) | Scoped FINAL verifier + evidence | Yes | No | Scoped FINAL + independent B6 |
| **OD-EVOL-14** | «**OPERATOR APPROVED — DIRECTION AND IMPLEMENTATION ONLY.** Approve the EVOL-X-only automation-first B6 direction under `AUTOMATION_FIRST_SCOPED`: separate evidence producer from acceptance authority; require security review, independent key custody, and a signed machine attestation; forbid implementer self-approval, unrelated `FRZ-*` promotion, and automatic EVOL-X CLOSED. Normative ECAP activation stays `PENDING_REQUIRED_APPROVALS` until maintainer publication, security review, and independent custody exist. `B6_ACCEPTANCE_PUBLICATION_SHA` stays unset.» | ECAP-§1.6 / ECAP-P4; amendment `2026-11-02.1-pending` | Human OD-EVOL-11 sign-off (still required while the amendment is not in force) | Attestation contract + verifier gates | Yes (Q tooling only) | No | Does not satisfy W7-01..W7-06 |
| **OD-EVOL-15** | «**OPERATOR APPROVED — FULL AUTO-B6 EXECUTION (SCOPED).** Execute the full authorized implementer/operator workstream for EVOL-X AUTO-B6: consolidated qualification, verifier Mode A/B where TES/FINAL unchanged, adversarial contract tests, unsigned security-review **technical evidence**, ECAP amendment dossier maintenance, independent attestation deployment specification, and E4 handoff **preparation**. External gates remain mandatory: ECAP maintainer act, signed independent security review, independent attestor + trust root + replay ledger, named OD-EVOL-11 auditor (or AUTO-B6 custodian record), OD-EVOL-12, OD-EVOLX-08. No invented signatures or production attestation keys.» | This execution directive + OD-EVOL-14 | Defer formal B6 until external authorities | Wave report + verifier JSON (session) | Yes (Q tooling + docs) | No | Does not close EVOL-X |
| **N-04 (OD)** | «Adopt **N4-A**: graph semver in migration docs only until optional N4-B approved.» | [EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) | N4-B Class B field | W4 WM rows | N4-B only | **Yes** (wording) | W4 if N4-B |

**OD-EVOL-11** (auditor appointment): **APPROVED, identity not named** (see W0 table) — name required before W7 independent B6; recommended before W6 evidence pack. OD-EVOL-14 does not fill this identity.

## OD-EVOL-14 — automated B6 direction (2026-11-02)

**Status:** OPERATOR APPROVED — DIRECTION AND IMPLEMENTATION ONLY.  
**Normative ECAP effect:** none. Amendment activation `PENDING_REQUIRED_APPROVALS`.

Recorded constraints:

- Scope is EVOL-X `AUTOMATION_FIRST_SCOPED` only.
- Evidence producer and acceptance authority must be separated.
- Security review is required before normative activation.
- Signing keys require independent custody. Keys minted in an implementer workspace are `NON_PRODUCTION_TEST_ONLY` and cannot authorize formal B6.
- Formal acceptance requires a signed machine attestation under the versioned contract.
- The implementer must not self-approve.
- Unrelated `FRZ-*` criteria must not be promoted. FRZ-EVOL-07 stays OPEN.
- EVOL-X CLOSED is not automatic. OD-EVOL-12 and OD-EVOLX-08 remain PENDING.
- `B6_ACCEPTANCE_PUBLICATION_SHA` remains unset.

Machine status: [EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json).  
Amendment: [EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md](./EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md).

## OD-EVOL-15 — full AUTO-B6 execution within operator authority (2026-11-02)

**Status:** OPERATOR APPROVED — FULL AUTO-B6 EXECUTION WITHIN OPERATOR AUTHORITY.  
**Normative ECAP effect:** none (unchanged). **B6 acceptance SHA:** unset. **EVOL-X:** OPEN.

Authorized in this wave: technical verifier replay, attestation unit tests, Mode A @ FINAL (orchestrated), adversarial fail-closed controls, unsigned security-review technical package, deployment dossier maintenance, E4 preparation artifacts, git publication of valid evidence.

Not authorized and not performed: ECAP maintainer publication, independent security signature, production attestation signing, `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`, FRZ-EVOL promotions, OD-EVOL-12 / OD-EVOLX-08 signing, EVOL-X CLOSED.

Consolidated report: [b6_technical_qualification/EVOL_X_AUTO_B6_FULL_WAVE_EXECUTION_REPORT.md](./b6_technical_qualification/EVOL_X_AUTO_B6_FULL_WAVE_EXECUTION_REPORT.md).

## OD-EVOL-16 — Path B certification mechanism (2026-11-02)

**Status:** OPERATOR APPROVED — **PATH_B_SELECTED**.  
**Normative ECAP effect:** none (`NOT_IN_FORCE` unchanged). **B6 acceptance SHA:** unset. **EVOL-X parent:** OPEN.

Path B is the chosen formal B6 route: maintainer publication of `ECAP-P4-EVOL-SCOPE-AUTO-B6`, signed independent security review, production key custody + trust root, external attestor runtime, signed attestation, evaluator `ELIGIBLE`, then W7-01 Protocol 03 PASS publication. This decision does not substitute any external act.

Consolidated Path B acceptance execution: [EVOL_X_PATH_B_FINAL_ACCEPTANCE_EXECUTION_RECORD.md](./EVOL_X_PATH_B_FINAL_ACCEPTANCE_EXECUTION_RECORD.md).
