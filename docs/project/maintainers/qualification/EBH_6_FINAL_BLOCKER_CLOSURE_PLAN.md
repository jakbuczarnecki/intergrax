# EBH-6 — Definitive Final Blocker Closure Plan

> **FINAL COMPLETE CLOSURE PLAN — READY FOR INDEPENDENT PLAN AUDIT**

| Field | Value |
| --- | --- |
| **Mode** | PLAN ONLY — no implementation authorized by this document |
| **EXPECTED_START_HEAD** | `4a6ad89f3954605968f7d4459fbaf49b8781550a` |
| **PLAN_PUBLICATION_SHA** | `aaa19255bb5560d4fabd3898b788eb654fc616c3` |

### Summary metrics (plan baseline @ EXPECTED_START_HEAD)

| Metric | Value |
| --- | ---: |
| Original gap groups (`EVIDENCE_REQUIRED`) | **109** |
| Applicable obligation surface (L9 layer × class) | **133** |
| Mechanism cell rollup (informational L3 annex) | **480** |
| Mandatory EBH-6 group closures (target) | **109 → 0** `EVIDENCE_REQUIRED` |
| Layer attestation rows (`EBH6-LAYER-*`) | **16** (must reach RESOLVED or registered finding) |
| L9 BLK auditor disposition | **8** (PENDING → decided) |
| Family independent auditor disposition | **23** (PENDING → decided) |
| Deferred to EBH-7 (matrix behavioral, NYQ) | **51** replaceability rows (not 109 groups) |
| Deferred to DUP-X | Platform duplicate-mechanism census (class D spot only @ EBH-6) |
| Deferred PROD-Q | PRD family live qualification (Path B) |
| Remediation blockers (conditional) | **0–N** PROHIBITED findings + **0–1** Q2-D2 (operator) |
| Unique TASK-IDs (plan WBS) | **~35** |
| Execution batches | **7** (B1–B7) |
| Operator decisions | **OD-EBH6-Q2D2**, **OD-EBH6-01**, **OD-EBH6-08** |
| Protocol 03 | OPEN until Batch 6 |

**Authorities:** [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.8 / §5 · FRZ checklist · ECAP · VBN spec · [EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md](EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md) · [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) · [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md).

**Explicit non-goals for this document:** No production code changes · no `_ADJUDICATION` promotion until authorized batches · no Protocol 03 PASS · no OD-EBH6-08 · no EBH-6 stage CLOSED · no live PROD-Q · no full DUP-X program · no EBH-7 behavioral matrix closure.

---

## Preface — Why the prior closure plan was rejected (PL-01..PL-05)

The superseded [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) revision (work-package bundling without normative crosswalk) failed independent plan audit for five structural reasons. Each maps to a corrective principle enforced in this document.

| ID | Root cause in rejected plan | Corrective principle |
| --- | --- | --- |
| **PL-01** | Treated **109 gap groups** as uniformly mandatory EBH-6 without normative crosswalk to [EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md](EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md) L9-A matrix (**133** applicable pairs) vs roadmap **§3.0.8 EBH-6 mechanical exit** (four bullets only) vs VBN artifact duties | Every group receives an explicit **normative bucket** (EBH-6 mandatory / deferred with citation / N/A fix) before work |
| **PL-02** | Layer closure described as WP bundles without **closed-world inventory + L3 + register row** per L9 plan PL-02 | Each of **16 layers** (`EBH6-LAYER-01..16`) must reach **RESOLVED** or a registered finding — not only `_ADJUDICATION` promotion |
| **PL-03** | **Qualification gap** (no gate yet) conflated with **architecture defect**; register mechanical PASS treated as progress toward closure | RC-03 taxonomy drives tasks: replay vs new gate vs OD-EBH6-01 vs ADR |
| **PL-04** | Protocol 03 batch “0” before evidence completeness; **“0–30 may remain”** closure arithmetic | Batches 1–5 must leave **zero `EVIDENCE_REQUIRED` groups** at implementer handoff; Protocol 03 only in **Batch 6** |
| **PL-05** | Ignored SHA drift ([EBH_6_EVIDENCE_FRESHNESS_LEDGER.md](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md) `0549a12…` vs [EBH_6_PARENT_CERTIFICATION_DOSSIER.md](EBH_6_PARENT_CERTIFICATION_DOSSIER.md) `1f3fb2ae3…` vs [EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md](EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md) stale headers) | **Single SHA reconciliation** in Batch 1 (one commit), then monotonic ladder |

**SSOT confusion (current → target):** **133** applicable layer×class pairs roll up to **109** default `EVIDENCE_REQUIRED` groups in [`_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py), which in turn relate to **480** mechanism cells as an L3 annex display — not as the Protocol 03 counting unit. Closure target: **133** auditor-visible dispositions, **16** layer rows **RESOLVED**, zero gap groups at handoff.

---

## Section 1 — Root Cause Analysis (RC-01..RC-05)

### RC-01 — Scope inflation vs roadmap

**Roadmap §3.0.8 EBH-6 mechanical exit** ([PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md), approximately lines 1516–1521) requires: findings register complete, **`CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT` accepted**, PROHIBITED unresolved count zero, and CONFIG-X reconciled.

**Extended EBH-6 obligation** (same §3.0.8 plus VBN §2.7) adds a current-HEAD **universal hardcoding re-audit across all layers**, producing updates to [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) and L9 layer/family evidence.

**The 109 groups are not 109 roadmap bullets.** They are the default **`EVIDENCE_REQUIRED`** disposition in `_l3_aj_group_reconciliation.py` wherever no class-specific gate is bound. Semantically this means “register mechanical exit ≠ behavioral proof,” not “109 distinct production defects awaiting 109 fixes.”

**Normative triage rule (per group):**

| Bucket | Criteria | Count @ EXPECTED_START_HEAD |
| --- | --- | ---: |
| **M-EBH6** | L9-A **Y** + no later-stage owner for the proof class | **109** (all current gaps — close with evidence or FINDING) |
| **M-EBH6-LAYER** | Layer row `EBH6-LAYER-*` OPEN until inventory + L3 + register | **16** |
| **DEF-EBH7** | RPL **behavioral** proof for **51 NYQ** matrix rows (`REPLACEABLE — NOT YET QUALIFIED`) | **51 rows** (not 14 J cells alone) |
| **DEF-DUPX** | Duplicate-authority **platform** census (class D deep) | DUP-X program |
| **DEF-PRODQ** | Live provider qualification | PRD family / Path B |
| **DEF-REPLAY** | ROADMAP-REPLAY-X / ENT-AUDIT-X | Post-EBH-6 |
| **N/A** | L9-A **N** with audited rationale in `_layer_evidence` | **27** pairs (already **N/A — WITH EVIDENCE** in cells) |

**Forbidden:** reclassifying **M-EBH6** → EBH-7 without VBN/RPL row plus L9-A **J** applicability proof for that layer.

### RC-02 — Matrix model

- **Certification unit for Protocol 03:** **133** applicable (layer × class) pairs + **16** layer attestations + **23** families + **8** L9 BLK items — not 480 cells.
- **480 cells:** L3 annex display (48 mechanisms × applicable classes); retain for reachability narrative, **not** for closure counting.
- **Redundancy:** Multiple classes on the same layer often share one **closed-world scan + one negative gate** (same production risk). Address via **shared TASK** with an explicit covered group list.

**Model correction (Batch 1, qualification-only):** Document in the scope dossier that closure census uses **133** groups; optional integrity test: `len(EVIDENCE groups) + resolved groups == applicable pairs`.

### RC-03 — Architecture vs qualification (taxonomy)

| Class | Description | @ EXPECTED_START_HEAD |
| --- | --- | ---: |
| (1) Mechanism OK, replay only | Historical stage gates suffice | ~24 groups already PARTIAL/PASS |
| (2) Mechanism OK, missing gate | Need new or extended qual gate | **109** groups |
| (3) Not EBH-6 scope | PRD live, 51 NYQ behavioral, DUP-X platform | Deferred with cite |
| (4) Prod violation | Register-tracked findings | **Q2-D2** (RATIFIED FINDING); **BND-COMPAT-R3** (RATIFIED, non-prod) |
| (5) ADR | Design decision required | Only if scan finds REQUIRES_ADR |
| (6) Skip/stale test | Test path drift | **RAG graph** → [`graph_store_bootstrap.py`](../../../intergrax/rag/graph/bootstrap/graph_store_bootstrap.py) |

### RC-04 — Why tests failed to close EBH-6 alone

| Layer | Issue |
| --- | --- |
| Register integrity | Proves **counts**, not class behavior |
| `_ADJUDICATION` PARTIAL | Often **inventory/reachability index**, not closed-world scan |
| EE-A1 / factory tests | **Slice** only (execution, integrations) |
| Skipped reachability | [`test_ebh5_reach_no_second_global_provider_registry_in_rag_graph`](../../../tests/qualification/ebh_5/test_ebh_5_reachability_gates.py) — path drift vs `integration_graph_store.py` |
| Historical PASS | CONFIG-X, GOV-X2, etc. — **scope-bound**; ECAP-P2 forbids banner-only upgrades |

### RC-05 — SHA reconciliation (Batch 1 deliverable)

After Batch 1, maintain a **single authoritative ladder**:

| Field | Source of truth |
| --- | --- |
| `IMPLEMENTATION_SHA` | Last qualification + documentation evidence commit |
| `TEST_EXECUTION_SHA` | Must equal `IMPLEMENTATION_SHA` at pytest |
| `CERTIFICATION_HANDOFF_SHA` | Final publication commit before Protocol 03 |
| `PLAN_PUBLICATION_SHA` | Git commit publishing this plan (§10) |

**Demote to HISTORICAL sections only** (no banner-only commits): `0549a12…`, `716efd67…`, `d993f49…`, `475c202…` in [EBH_6_EVIDENCE_FRESHNESS_LEDGER.md](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md), [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md), [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) headers — reconcile in **T-SHA-01**.

---

## Section 2 — Canonical Scope and Exclusions

**In scope for EBH-6 closure:** Roadmap EBH-6 row + §3.0.8 layer re-audit + VBN artifacts @ HEAD + ECAP §2.B parent closure + Protocol 03 + **23** family **independent** disposition + **8** L9 BLK replay.

**Explicit exclusions (normative cite required in handoff):**

- Live **PROD-Q** (FRZ-PRD Path B).
- **51 NYQ behavioral qualification** (EBH-7; matrix rows, not EBH-6 group count).
- **DUP-X** duplicate-mechanism program (EBH-6 classifies + spot-gates class D only).
- **ENT-AUDIT-X** runner and **ARCH-FREEZE** global FRZ PASS (separate stage gates).

**J class @ EBH-6:** Each applicable layer requires **RPL-H classification** for inventoried concerns plus RPL-G/regression — **not** full alternate-backend behavioral proof for every NYQ row.

---

## Section 3 — Definitive Acceptance Contract

Independent closure requires **all four** contract blocks (A–D). Partial mechanical PASS without B–D is not EBH-6 handoff.

### A. Mechanical exit (roadmap §3.0.8)

| Criterion | Verification |
| --- | --- |
| Register complete | `test_ebh6_findings_unclassified_zero` |
| PROHIBITED unresolved = 0 | `test_ebh6_prohibited_unresolved_zero` |
| CONFIG-X reconciled | `test_ebh6_config_x_all_concerns_mapped_at_head` |
| CURRENT_HEAD audit **accepted** | Independent Protocol 03 sign-off on [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) |

### B. Architecture recertification

| Criterion | Verification |
| --- | --- |
| 16 layers | Each `EBH6-LAYER-*` → **RESOLVED** or PROHIBITED/ALLOWED with evidence in `_findings_register.py` |
| 133 applicable A–J | No `EVIDENCE_REQUIRED` in `aj_group_by_layer_class()` at handoff |
| 48 L3 records | Each production-effective path has L3 gate + group disposition ≠ EVIDENCE |
| 23 families | Independent disposition ≠ PENDING (auditor) |
| 8 L9 BLK | Auditor replay of cited gates @ SHA |
| EBH-6 vs later stages | Deferred items listed in §2 with FRZ/VBN cite |

### C. Evidence freshness

| Criterion | Verification |
| --- | --- |
| Exact SHA pytest | One consolidated command; log under `.tmp/session/ebh-6-final-closure/` |
| Ledger = dossier = handoff | Single reconciliation commit (T-SHA-01) |
| No historical PASS upgrade | ECAP-P2 checklist in auditor report |

### D. Independent acceptance

| Criterion | Verification |
| --- | --- |
| ECAP §2.B B1–B6 | [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md) updated @ handoff SHA |
| 23 families | `_family_catalog.INDEPENDENT_AUDITOR_DISPOSITION` |
| Outcome | PASS or **mandatory findings list** — not implementer PASS |
| OD-EBH6-08 | Operator stage closure **only** after Protocol 03 PASS |

---

## Section 4 — Mandatory Blocker Inventory (closure target)

At implementer handoff (end of Batch 5), the following inventory must read as the “must become” column — no acceptable residual “unknown 0–30” gap groups.

| Blocker class | Count @ EXPECTED_START_HEAD | Must become @ handoff |
| --- | ---: | --- |
| `EVIDENCE_REQUIRED` groups | **109** | **0** |
| `EBH6-LAYER-*` OPEN | **16** | **0 OPEN** (RESOLVED or registered finding) |
| L9 BLK auditor PENDING | **8** | **8 decided** |
| Family auditor PENDING | **23** | **23 decided** |
| Protocol 03 | OPEN | **PASS or REJECT with findings** |
| OD-EBH6-01 | OPEN | OPEN unless remediation batch executed |
| OD-EBH6-08 | OPEN | CLOSED only post-audit (Batch 7) |

---

## Section 5 — Full Task Breakdown (TASK-IDs)

**Convention:** Each TASK lists covered **group IDs** (`layer×class`). **DONE** = gate green + `_ADJUDICATION` update (when batch authorized) + register/layer row if needed.

### Normative and SHA (Batch 1)

| TASK-ID | Scope | Closes | OD-EBH6-01 |
| --- | --- | --- | --- |
| **T-NORM-01** | Audit 109 groups vs L9-A table in L9 plan; fix any `_layer_evidence` vs L9-A mismatch | Misclassified applicability | No |
| **T-SHA-01** | Reconcile ledger, handoff, parent/scope/VBN/semantic/production dossiers to one ladder | RC-05 | No |
| **T-DOC-01** | Supersede rejected closure plan with this document §1–10 | Plan audit | No |

### Layer closed-world + L3 (Batch 2) — PL-02

For each **EBH6-LAYER-NN** (16 layers; implementation may merge to four quadrant tasks):

| TASK-ID | Layers | Actions | Evidence | Reuse tests |
| --- | --- | --- | --- | --- |
| **T-LAY-CTR-SMC** | contracts, semantic_core | Closed-world paths from `_closed_world_discovery`; map findings; L3 samples | Layer gate module | hardening-3, EE-A1, Q2-D2 register |
| **T-LAY-APP-CMP** | application_services, composition_roots | TRACE-X registry parity per sample | L3 replay | trace_x, sanctioned roots |
| **T-LAY-GOV-EXE** | governance, execution | NPSC42 + EE-A1 on inventoried paths | L3 | npsc42, execution identity |
| **T-LAY-INT-STA** | integrations, state_persistence | Factory + sqlite annex + bundles | L3 | ebh5 reachability, sqlite gate |
| **T-LAY-REC-OBS** | reliability_recovery, observability_diagnostics | Recovery boundary + W5 | L3 | npsc5e, w5-b2 |
| **T-LAY-TRC-WRK** | events_traceability, background_workers | Event spine + worker census | L3 | trace_x, ebh5 |
| **T-LAY-AGD-RAG** | agent_distribution, rag_memory | Lifecycle + RAG bootstrap path | L3 | agent lifecycle, **T-RAG-01** |
| **T-LAY-HST-TLC** | runtime_hosting, tools_skills_capabilities | CONFIG-X + tool SPI | L3 | config_x, plugin gates |

**DONE per layer:** `EBH6-LAYER-*` → RESOLVED in register narrative + ≥3 L3 samples or documented `production_effective=no`.

### Class gate bundles (Batch 3) — closes 109 groups

| TASK-ID | Classes | Mechanism | New qual (minimal) | Groups |
| --- | --- | --- | --- | ---: |
| **T-CLS-A** | A | Per-layer vendor authorship scan on `_layer_evidence.closed_world_paths` | One parametrized AST gate | 13 |
| **T-CLS-B** | B | Hardcoding/bypass census | Extend EE-A1 pattern | 9 |
| **T-CLS-C** | C | Fail-closed defaults on sanctioned roots | EBH-5 behavioral negatives | 12 |
| **T-CLS-D** | D | Duplicate selector spot-check (not full DUP-X) | Static gate + register row | 12 |
| **T-CLS-E** | E | Shadow store / sqlite authority | Annex touchpoint subset | 6 |
| **T-CLS-F** | F | Public seam typing / register | **Operator branch** (§7) | 13 |
| **T-CLS-G** | G | OBS non-authority | W5 + CTRL-X R2 replay | 9 |
| **T-CLS-H** | H | Reachability from sanctioned roots | TRACE-X parametrized | 8 |
| **T-CLS-I** | I | Env/install coupling | CONFIG-X env contract extension | 13 |
| **T-CLS-J** | J | RPL-H row + RPL-G per concern | Matrix citation gate (no NYQ behavioral) | 14 |

**T-RAG-01** (within T-LAY-AGD-RAG + T-CLS-A): Update EBH-5 test path to [`graph_store_bootstrap.py`](../../../intergrax/rag/graph/bootstrap/graph_store_bootstrap.py); assert no shadow `graph_store_backend`; rerun reachability suite.

### Families and L9 BLK (Batch 4)

| TASK-ID | Scope | DONE |
| --- | --- | --- |
| **T-FAM-01** | Package auditor replay bundle: 23 rows from `_family_l9_evidence.py` + gate IDs | Handoff doc complete |
| **T-L9BLK-01** | Map 8 blockers to executed pytest @ SHA | Auditor checklist |

### Production remediation (Batch 5 — conditional)

| TASK-ID | Trigger | Scope | OD-EBH6-01 |
| --- | --- | --- | --- |
| **T-REM-Q2D2** | Operator rejects RATIFIED-only closure for TYP/EBH-6 | `nexus_loop.py` ADE typing | **Yes** |
| **T-REM-PROH** | T-CLS-* finds PROHIBITED | Scoped prod fix | **Yes** |

### Final publication (Batch 5)

| TASK-ID | Actions |
| --- | --- |
| **T-PUB-01** | Single pytest @ IMPLEMENTATION_SHA; update all CURRENT SHA fields; census shows **0 EVIDENCE groups** |
| **T-PUB-02** | Update VBN audit + scope dossier final banner **candidate handoff** (not CLOSED) |

### Protocol 03 and operator (Batch 6–7)

| TASK-ID | Actions |
| --- | --- |
| **T-P03-01** | Independent ECAP B1–B6; update dispositions |
| **T-OD-08** | Operator OD-EBH6-08 only if P03 PASS |

---

## Section 6 — Batch Execution Plan (B1–B7)

| Batch | Name | Tasks | Exit gate |
| --- | --- | --- | --- |
| **B1** | Normative scope + SHA reconciliation | T-NORM-01, T-SHA-01, T-DOC-01 | L9-A ↔ `_layer_evidence` consistent; one SHA ladder |
| **B2** | Production closed-world + L3 (16 layers) | T-LAY-* | All `EBH6-LAYER-*` RESOLVED |
| **B3** | Class gates (109 groups → 0 EVIDENCE) | T-CLS-*, T-RAG-01 | `EV_GROUPS == 0` |
| **B4** | Family + L9 BLK handoff pack | T-FAM-01, T-L9BLK-01 | Handoff ready @ SHA |
| **B5** | Final qual publication (+ optional T-REM-*) | T-PUB-* | Consolidated pytest PASS |
| **B6** | Protocol 03 | T-P03-01 | Independent PASS or findings |
| **B7** | OD-EBH6-08 | T-OD-08 | Roadmap EBH-6 CLOSED |

**Hard sequencing rule:** **No Batch 6 before Batch 5** census shows **0** gap groups and **16** resolved layer rows.

---

## Section 7 — Operator Decisions

| Decision | Question | Default if silent |
| --- | --- | --- |
| **OD-EBH6-Q2D2** | Accept **RATIFIED** Q2-D2 for EBH-6 Protocol 03 (FINDING row + no prod change)? | Align with GOV-X2 / FRZ-TYP notes: RATIFIED acceptable for **stage**; ARCH-FREEZE may require **T-REM-Q2D2** |
| **OD-EBH6-01** | Open implementation wave for Q2-D2 or PROHIBITED fixes? | Stay OPEN |
| **OD-EBH6-08** | Publish stage closure on roadmap | After P03 PASS only (Batch 7) |

---

## Section 8 — Closure Simulation (must pass before publish)

Pre-publication dry-run: each row must be satisfiable at handoff SHA without residual unknown gaps.

| Acceptance criterion | Task(s) | Evidence | Test | Independent | Residual risk |
| --- | --- | --- | --- | --- | --- |
| Register mechanical | T-PUB-01 | register gates | ebh6 integrity | Auditor spot-check | None if green |
| 0 PROHIBITED | T-CLS-B, scans | register | prohibited=0 | Auditor | **T-REM-PROH** if scan hits |
| CONFIG-X | T-CLS-I, CFG family | 54/54 | config_x gate | Auditor | None |
| VBN audit accepted | T-PUB-02, T-P03-01 | VBN md | consolidated pytest | **Protocol 03** | Reject → B3 gap |
| 133 A–J dispositions | T-CLS-* | `_ADJUDICATION` | parametrized class gates | Auditor | **0 EVIDENCE** required |
| 16 layers | T-LAY-* | LAYER rows | L3 integrity | Auditor | None |
| 23 families | T-FAM-01, T-P03-01 | catalog | family gates | Auditor | PENDING forbidden at closure |
| 8 L9 BLK | T-L9BLK-01, T-P03-01 | l9 evidence | cited pytest | Auditor | None |
| Q2-D2 | T-CLS-F + OD decision | FINDING/RATIFIED or REM | typing gate / register | Auditor | Prod only if OD-EBH6-01 |
| 51 NYQ | **Out of scope** | matrix doc | RPL-G only | EBH-7 | Documented deferral |
| SHA freshness | T-SHA-01, T-PUB-01 | ledger | pytest @ SHA | Auditor | None |

**Simulation result:** This plan is **COMPLETE** as a closure blueprint only if implementer executes B1–B5 to **0 EVIDENCE groups** and **16 RESOLVED layers**. Unknown defects discovered in scans → **T-REM-PROH** + re-audit (repeat B3), not stage CLOSED.

---

## Section 9 — Independent Audit Handoff

Update [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md) at **`CERTIFICATION_HANDOFF_SHA`** (Batch 5 publication, not this plan commit) with:

- Consolidated pytest command (ebh_6 + ebh_5 reachability + hardening-3 + rpl-g).
- Explicit **133-group** disposition export from `aj_group_by_layer_class()`.
- Family disposition export from `_family_catalog`.
- Link to this plan §8 simulation table and §3 acceptance contract A–D.

---

## Section 10 — Git Publication Record

**This deliverable (plan publication only):**

1. Replaced prior rejected [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) with sections 1–10 above.
2. Status banner: **FINAL COMPLETE CLOSURE PLAN — READY FOR INDEPENDENT PLAN AUDIT**.
3. Published on branch **`development`** from baseline **`4a6ad89f3954605968f7d4459fbaf49b8781550a`**.

| Record | Value |
| --- | --- |
| **PLAN_PUBLICATION_SHA** | `aaa19255bb5560d4fabd3898b788eb654fc616c3` |
| **Branch** | `development` |
| **Post-approval implementation** | B1–B7 only after independent plan audit PASS; no production code or `_ADJUDICATION` changes from this commit alone |

---

*End of definitive EBH-6 closure plan document.*
