# PRE-ENT Wave 6 — independent cross-audit report (republication)

**Attribution:** Findings below are **republication** of the independent auditor session. The evidence publisher **did not** re-execute W6-G0..G3 and **does not** adopt these results as publisher-authored audit conclusions.

| Field | Value |
| --- | --- |
| **Campaign** | `pre_ent_enterprise_certification` |
| **Role (source session)** | `independent-quality-auditor` (IIAC) |
| **AUDITOR_SESSION_ID** | `W6-INDEP-AUDIT-2026-11-10-f4ca1c864` |
| **Worktree** | `D:\Projekty\_wt\intergrax\w6-cross-audit` @ detached `f4ca1c864b14c917ea032aabcda1dc169654ace2` |
| **DEVELOPMENT_HEAD / W6 tree pin** | `f4ca1c864b14c917ea032aabcda1dc169654ace2` |
| **Date (audit)** | 2026-11-10 |
| **Parent ticket** | `governance/tickets/T-IGX-034-A.md` |

**Ceiling (auditor-declared):** Terminal disposition **B** — **not** full Enterprise certification, **not** `ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED`, **not** campaign lifecycle **CLOSED**, **not** FRZ promotion, **not** §3 stage-row edits.

---

## 1. Baseline and scope delta

| Check | Auditor finding |
| --- | --- |
| `origin/development` @ audit | `f4ca1c864b14c917ea032aabcda1dc169654ace2` |
| Post–Wave 5 merge pin `8d078cc9…` | **Ancestor** of HEAD; delta `8d078cc9..f4ca1c864` = handoff doc commit only |
| Product diff `1e83ade3..f4ca1c864` | **Empty** (`intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`) — W6 audit tree = **AUDIT_HEAD** product + post-audit **docs-only** pins |
| Independence | Isolated worktree; auditor ≠ Wave 4 certifier / Wave 5 orchestrator |

Canonical inputs read @ pin: [`PRE_ENT_WAVE6_CROSS_AUDIT_HANDOFF.md`](PRE_ENT_WAVE6_CROSS_AUDIT_HANDOFF.md), final matrix, closure path, Wave 3/4 registers, `T-IGX-034-A`, `T-IGX-034-I`, roadmap §3 citation notes (§3 body **not edited**), FRZ checklist (**not edited**).

---

## 2. W6-G0 — Cross-audit consistency

**Disposition (auditor):** **PASS** — published SSOT is internally consistent for PRE-ENT scope; no unauthorized campaign **CLOSED** or full-certification promotion detected.

| Lane | Evidence | Consistency |
| --- | --- | --- |
| Wave 3 | [`wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md`](wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md) | **5/5** ticket-scope **PASS** @ `AUDIT_HEAD` `1e83ade3f…`; per-ticket reports present; ceilings state **not** G6 / **not** FRZ |
| Wave 4 | [`wave4_certs/WAVE4_CERT_CONSOLIDATED_REGISTER.md`](wave4_certs/WAVE4_CERT_CONSOLIDATED_REGISTER.md) | **5/5** **ACCEPT_SCOPED** @ same `AUDIT_HEAD`; matrix rows match certifier reports |
| Wave 5 | [`PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md`](PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md) | Open rows = **EXACT_BLOCKER** or **N/A (PRE-ENT)**; predecessors **ACCEPT_SCOPED** only — **no** silent OPEN |
| Roadmap §3 | Cited in matrix §4 only | **Unchanged** by Waves 5–6 publisher scope; §3 labels (e.g. SKILL-X, TOOL-LIFE-X) **not** reconciled to matrix — documented as blockers, **not** edited |
| Roadmap §5 | Wave 5 ledger row | Non-closing; defers terminal label to W6-G3 |
| FRZ | Checklist | **Not** imported campaign labels; **new global FRZ PASS = 0** preserved |
| Tickets | `T-IGX-034-A` | `quality_gate` — **not** CLOSED |
| Tickets | `T-IGX-034-I` | `plan_pending`; chat authorization recorded; **no** forged `Approved:` line (**separate lifecycle gap**) |

**Wave 3 session evidence (local, not committed):** pytest logs under `.tmp/session/pre-ent-wave3-audit/<slug>/` — slugs `bnd-03b`, `replay`, `030-b`, `030-c`, `032-a` (worktrees `w3-*` per register).

**Wave 4 session evidence (local, not committed):** `.tmp/session/pre-ent-wave4-cert/<slug>/` — slugs per register (`w4-*` @ `1e83ade3`).

**Promotions checked:** No register or matrix row claims **ACCEPT_SCOPED →** full independent certification or FRZ promotion.

---

## 3. W6-G1 — SHA integrity

**Disposition (auditor):** **PASS** — ladder complete on `development` @ `f4ca1c864`; reports remain tied to `AUDIT_HEAD` product evidence.

| Pin | SHA | On-tree @ `f4ca1c864` |
| --- | --- | --- |
| **AUDIT_HEAD** | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` | ancestor |
| **CERT_DOCS_HEAD** | `d00df592037404cd6c2b1bc95c00757544d551d8` | ancestor |
| **WAVE4_PUBLICATION_SHA** | `c4a3b751bf7d6716bb0003bed23ae20b1e8f910f` | ancestor |
| **WAVE5_PUBLICATION_SHA** | `0f3b4976726dcbd3046cd888aa91703214f27997` | ancestor |
| **IIAC effective_from_sha** | `661978df191d9356058a1312a712fcfa3c890e9e` | ancestor |
| **Wave 5 merge** | `8d078cc9be92438c214ebf907f86cf3c3e208335` | ancestor |
| **Wave 6 handoff tip** | `f4ca1c864b14c917ea032aabcda1dc169654ace2` | **HEAD** |

Unpublished local logs (Wave 3/4/W6 regression scratch) are **not** treated as formally published evidence pins.

---

## 4. W6-G2 — Parent regression

**Disposition (auditor):** **BOUNDED PASS only — full parent regression NOT RUN.** W6-G2 is **not** a full protocol PASS.

| Tier | Requirement (campaign plan / closure path) | Status | Evidence |
| --- | --- | --- | --- |
| **Bounded blocking bundle** | Replay of five Wave 3 blocking gates @ W6 HEAD | **Executed** — **41 passed**, exit **0** | Scratch log `D:\Projekty\_wt\intergrax\_scratch\w6-cross-audit\parent_regression_5gate.log` (local; **not committed**) |
| **IIAC** | `verify_iiac_policy_activation.py --require-in-force` | **PASS** exit **0** | Scratch `…\iiac_verify.log`; record `IIAC-2026-11-05.1` @ `661978df…` |
| **Full parent regression** | `igx_gov ticket` / push-fast CI profile per enterprise gate | **NOT RUN** (attempt exit **1** — `FileNotFoundError` in `run_regression_gate_ci.py` subprocess on Windows worktree) | Scratch `…\igx_gov_ticket.log`; ticket-ref + policy registry **OK** before failure |
| **L1 / L2 / L3 campaign suites** | Per [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md) / plan | **NOT RUN** in audit session | — |

**Pytest command (auditor, bounded):**

```text
uv run --extra dev-ci python -m pytest -p no:xdist \
  tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py \
  tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py \
  tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py \
  tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q
```

**41 passed** in ~107s @ `f4ca1c864` (auditor log).

---

## 5. W6-G3 — Terminal campaign disposition

**Disposition (auditor):** **`PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`** (plan option **B**).

**Not selected:** option **A** (`ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED`) — matrix open rows remain **EXACT_BLOCKER**; option **C** (`INCOMPLETE / BLOCKED`) — protocol allows honest terminal settlement with documented blockers when Waves 0–5 evidence chain is complete.

**Formal certification limits (auditor):** Variant **B** is terminal **campaign accounting with exact blockers**, **not** Enterprise-wide B6/G6 PASS, **not** predecessor stage §3 promotion, **not** Population E closure.

### OPEN_EXACT_BLOCKERS (from Wave 5 matrix — auditor confirmed unchanged)

| ID | Disposition | Owner |
| --- | --- | --- |
| **G-SKILL** | **EXACT_BLOCKER** | Operator |
| **F10 / T-IGX-034-F** | **EXACT_BLOCKER** | Operator + EBH-7 |
| **OD-HIST04-A** | **EXACT_BLOCKER** | Operator |
| **OD-HIST04-B** | **EXACT_BLOCKER** | Governance |
| **OD-HIST04-C** | **EXACT_BLOCKER** | Operator |
| **T-IGX-034-C** / **HIST-F-DUR-01** | **EXACT_BLOCKER** | Operator |
| **T-IGX-034-D** | **EXACT_BLOCKER** | Campaign |
| **T-IGX-034-G** | **EXACT_BLOCKER** | Campaign |
| **T-IGX-034-H** | **EXACT_BLOCKER** | Campaign |
| **Population E** | **N/A (PRE-ENT)** | ENT-AUDIT-X |

Five Wave 4 predecessors: **ACCEPT_SCOPED** only (034-B, 034-E, 030-B, 030-C, 032-A).

---

## 6. W6-G4 — Formal closure eligibility

| Item | Auditor finding |
| --- | --- |
| Dossier completeness | Sufficient for **publication** of terminal label **B** and Wave 6 SSOT |
| `T-IGX-034-A` lifecycle **CLOSED** | **Not eligible** without **operator sign-off** after published W6-G3 |
| Campaign **CLOSED** | **Not claimed** |
| Wave 6 evidence publication | Deferred to **separate evidence-publisher session** (this report is audit input; publication SHA recorded in campaign progress) |

**Operator decisions still required (auditor):**

1. Accept terminal settlement **`PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`** (not full certification).
2. Authorize `T-IGX-034-A` lifecycle transition **CLOSED** (or alternate disposition) with signed checkpoint line.
3. Resolve `T-IGX-034-I` missing `Approved:` (mechanical `verify_plan_approval` gap) if Wave 5 ticket must advance.
4. Optional: operator budget for **full** W6-G2 parent regression (`igx_gov` / L1–L3) before treating regression gate as CI-parity PASS.

---

## 7. IIAC and governance (audit replay)

| Check | @ `f4ca1c864` | Exit |
| --- | --- | --- |
| `verify_iiac_policy_activation.py --require-in-force` | **IN_FORCE** @ `661978df191d9356058a1312a712fcfa3c890e9e` | **0** |
| `igx_gov ticket` (push-fast) | Failed mid-job (environment `FileNotFoundError`) | **1** |

Publisher re-verified IIAC **IN_FORCE** exit **0** on main checkout @ publication prep (same pin).

---

## 8. Evidence locations

| Artifact | Location | Committed? |
| --- | --- | --- |
| W6 IIAC log | `D:\Projekty\_wt\intergrax\_scratch\w6-cross-audit\iiac_verify.log` | **No** |
| W6 bounded pytest log | `…\parent_regression_5gate.log` | **No** |
| W6 igx_gov log | `…\igx_gov_ticket.log` | **No** |
| Wave 3 auditor logs | `.tmp/session/pre-ent-wave3-audit/` | **No** |
| Wave 4 certifier logs | `.tmp/session/pre-ent-wave4-cert/` | **No** |
| Published SSOT | This file + registers + matrix @ **WAVE6_EVIDENCE_PUBLICATION_SHA** `2e0b708fffb419de7f5de07e5e903218dc96bb3f` | **Yes** |

---

## 9. Auditor session summary fields (handoff to publisher)

| Field | Value |
| --- | --- |
| `DEVELOPMENT_HEAD` | `f4ca1c864b14c917ea032aabcda1dc169654ace2` |
| `AUDITOR_SESSION_ID` | `W6-INDEP-AUDIT-2026-11-10-f4ca1c864` |
| `W6_G0_CROSS_AUDIT` | **PASS** |
| `W6_G1_SHA_INTEGRITY` | **PASS** |
| `W6_G2_REGRESSION` | **BOUNDED PASS (41 tests); FULL NOT RUN** |
| `W6_G3_TERMINAL_VERDICT` | **`PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`** |
| `W6_G4_FORMAL_CLOSURE_ELIGIBILITY` | Publication **yes**; `T-IGX-034-A` **CLOSED** **no** (operator) |
| `FINAL_RECOMMENDATION` | Publish dossier; assign terminal label **B**; **STOP** before unauthorized campaign **CLOSED** |

---

## SSOT cross-links

- Handoff (pre-audit): [`PRE_ENT_WAVE6_CROSS_AUDIT_HANDOFF.md`](PRE_ENT_WAVE6_CROSS_AUDIT_HANDOFF.md)  
- Final matrix: [`PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md`](PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md)  
- Closure path: [`PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md`](PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md)  
- Campaign progress: [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md)  
- Parent: `governance/tickets/T-IGX-034-A.md`

---

Note (EQO-A remediation, as of 2026-11-17): evidence-custody statement and decision cross-reference. The record text above is unchanged (append-only note). This note supplies none of the missing facts, adds no acceptance stamp, audit SHA, closure, PASS, verification or independence claim and changes no status; the roadmap remains the status owner. Lines of this record and roadmap rows are cited as they read at `origin/development` `44d1e5d105beb9b62d0a12e693066b0b26c125eb`. Quotations omit the markdown emphasis of the source.

- Log: L79 reports the bounded blocking bundle as "41 passed", exit 0, with the scratch log `D:\Projekty\_wt\intergrax\_scratch\w6-cross-audit\parent_regression_5gate.log` ("local; not committed"); L69 states that unpublished local logs are not treated as formally published evidence pins, and L160 lists this log as not committed. Those lines describe the state when the report was written and are unchanged.
- Published by this change: a byte copy of that file at `docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/pre-ent-w6-cross-audit/parent_regression_5gate.log` (1646 bytes, sha256 `7c4ab15cff100f393a73dc5a123af89700eca8ed79fd41a033ff1c4454426f6d`, last line `41 passed in 107.13s (0:01:47)`; scan CLEAN, see `docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/EVIDENCE_MANIFEST.md`). The file names no test file, no commit and no exit code; the exit code 0 and the "five Wave 3 blocking gates @ W6 HEAD" description rest on L79 only. The tracked copy is a custody anchor for the file as found; it does not change the tier or the disposition of W6-G2 ("BOUNDED PASS only — full parent regression NOT RUN", L75, as the report words it) and does not make the log an evidence pin.
- G-SKILL: the report lists `G-SKILL` as `EXACT_BLOCKER` with owner Operator (L111; the matrix `PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md` L47 carries the same row). Decision session 4 (2026-11-16) of `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md` (the orchestrator's transcription of the Operator's chat decisions, not an Operator signature) records as DP-44: "SKILL-X: publish AUD-SKILLX byte-preserving with digest and custody statement and settle G-SKILL as a scoped limitation (to be reconciled with DP-42: publication alone may not be independent exact-SHA evidence)".
- SKILL-X artefact: the AUD-SKILLX files are published under `docs/project/maintainers/qualification/iiac_evidence/SKILL-X/audit/AUD-SKILLX/`, with their digests and custody statement in `docs/project/maintainers/qualification/iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md`. Roadmap row L499 (`ARCH-FREEZE`) carries the wording "`SKILL-X` = CLOSED (independent exact-SHA evidence; scope per §3.0.4)".
- What this note does and does not do with DP-44: it records the decision as published and leaves the report rows (L111 and the matrix row) as they are. It does not state the scope of the limitation (the published decision does not define it), does not settle `G-SKILL`, does not treat the publication of AUD-SKILLX as independent exact-SHA evidence, and implies no independence for the report, the log or the artefact.
- Related finding: EQOA-A2-040 (EQO-A Campaign A dossier, PR #589), register row `EDER-F-0069` in `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl` (as the register reads at the head named above: owner_ref `OPERATOR`, owner_status `OPERATOR_ONLY`; the owner vocabulary is confirmed, the row-level owner is not, and no person is assigned).
- This note does not establish that the published log is the run reported at L79, at which commit it was made, or that W6-G2 or any gate passes. It adds no closure or independence claim.
