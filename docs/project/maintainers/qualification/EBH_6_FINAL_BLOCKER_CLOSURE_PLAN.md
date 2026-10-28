# EBH-6 — Definitive Final Blocker Closure Plan

> **FINAL COMPLETE CLOSURE PLAN — CORRECTED / READY FOR INDEPENDENT ACCEPTANCE**

| Field | Value |
| --- | --- |
| **Mode** | PLAN ONLY — no implementation authorized by this document |
| **EXPECTED_START_HEAD** | `12a34692e0b5a8113a264d71ce5185dd609da5b9` |
| **PLAN_PUBLICATION_SHA** | *(set in §10 after publication commit)* |

### Summary metrics (plan baseline @ EXPECTED_START_HEAD)

| Metric | Value |
| --- | ---: |
| Original gap groups (`EVIDENCE_REQUIRED`) | **109** |
| Applicable obligation surface (L9 layer × class) | **133** |
| Mechanism cell rollup (informational L3 annex) | **480** |
| Mandatory EBH-6 group closures (target) | **109 → 0** `EVIDENCE_REQUIRED` |
| Layer attestation rows (`EBH6-LAYER-*`) | **16** (B2: inventory+L3; B3: CANDIDATE-RESOLVED; B6: accepted RESOLVED or finding) |
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
| **PL-02** | Layer closure described as WP bundles without **closed-world inventory + L3 + register row** per L9 plan PL-02 | B2 completes verified inventory + L3 per layer; **RESOLVED** is **CANDIDATE** disposition only after B3 reconciles mandatory controls; final **RESOLVED** acceptance only via independent Protocol 03 (B6) — not automatic for all 16 layers at B2 exit |
| **PL-03** | **Qualification gap** (no gate yet) conflated with **architecture defect**; register mechanical PASS treated as progress toward closure | RC-03 taxonomy drives tasks: replay vs new gate vs OD-EBH6-01 vs ADR |
| **PL-04** | Protocol 03 batch “0” before evidence completeness; **“0–30 may remain”** closure arithmetic | Batches 1–5 must leave **zero `EVIDENCE_REQUIRED` groups** at implementer handoff; Protocol 03 only in **Batch 6** |
| **PL-05** | Ignored SHA drift ([EBH_6_EVIDENCE_FRESHNESS_LEDGER.md](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md) `0549a12…` vs [EBH_6_PARENT_CERTIFICATION_DOSSIER.md](EBH_6_PARENT_CERTIFICATION_DOSSIER.md) `1f3fb2ae3…` vs [EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md](EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md) stale headers) | **Single SHA reconciliation** in Batch 1 (one commit), then monotonic ladder |

**SSOT confusion (current → target):** **133** applicable layer×class pairs roll up to **109** default `EVIDENCE_REQUIRED` groups in [`_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py), which in turn relate to **480** mechanism cells as an L3 annex display — not as the Protocol 03 counting unit. Closure target: **133** auditor-visible dispositions with B3 acceptance criteria (§3.B); **16** layer rows with verified closed-world inventory + L3 in B2, **CANDIDATE-RESOLVED** only after B3 layer-control reconciliation, **accepted RESOLVED** only via Protocol 03 (B6); zero gap groups at Batch 5 handoff.

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
| 16 layers — B2 | Each `EBH6-LAYER-*`: verified **closed-world inventory** + **L3 evidence** complete (not automatic RESOLVED) |
| 16 layers — B3 | Mandatory A–J controls for that layer reconciled → register/layer narrative may mark **CANDIDATE-RESOLVED** |
| 16 layers — B6 | Independent Protocol 03 **accepts** RESOLVED disposition or registered PROHIBITED/ALLOWED with evidence in [`_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) |
| **133 applicable A–J — B3 gate** | For **every** applicable group: qual test **tied to the invariant**; **verified production scope**; **concrete positive/negative outcome**; disposition **FINDING** or **N/A** (with evidence) — not `EVIDENCE_REQUIRED`; **no implementer auto-PASS**; EBH-6 scope preserved. **`EV_GROUPS == 0` alone is insufficient.** B3 **fails** if any mandatory blocker remains unresolved or is hidden as PARTIAL/RATIFIED without audited basis |
| 48 L3 records | Each production-effective path has L3 gate + group disposition ≠ EVIDENCE |
| 23 families — B6 | **Independent auditor acceptance** of disposition in EBH-6 scope for **all 23** families — not merely `PENDING == 0`. REJECT/BLOCKED/insufficient evidence → documented remediation + re-audit path |
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
| 23 families | `_family_catalog.INDEPENDENT_AUDITOR_DISPOSITION` — **all 23 accepted** in EBH-6 scope (B6); PENDING=0 without acceptance is not PASS |
| Outcome | PASS or **mandatory findings list** — not implementer PASS |
| OD-EBH6-08 | Operator stage closure **only** after Protocol 03 PASS |

---

## Section 4 — Mandatory Blocker Inventory (closure target)

At implementer handoff (end of Batch 5), the following inventory must read as the “must become” column — no acceptable residual “unknown 0–30” gap groups.

| Blocker class | Count @ EXPECTED_START_HEAD | Must become @ handoff |
| --- | ---: | --- |
| `EVIDENCE_REQUIRED` groups | **109** | **0** @ Batch 5 handoff (B3 must satisfy full §3.B gate, not count alone) |
| `EBH6-LAYER-*` OPEN / incomplete | **16** | **0** missing inventory+L3 @ B2 exit; **CANDIDATE-RESOLVED** @ B3 when layer controls reconciled; **accepted RESOLVED** @ B6 only |
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

**DONE per layer (B2):** Verified closed-world inventory + L3 samples (≥3 or documented `production_effective=no`) + register row updated — **not** final RESOLVED.

**DONE per layer (B3 add-on):** All mandatory A–J controls for that layer reconciled; layer narrative may mark **CANDIDATE-RESOLVED** pending Protocol 03.

**Accepted RESOLVED (B6 only):** Independent auditor confirms disposition per §3.B layer row.

### Class gate bundles (Batch 3) — closes 109 groups

| TASK-ID | Classes | Mechanism | New qual (minimal) | Groups |
| --- | --- | --- | --- | ---: |
| **T-CLS-A** | A | Per-layer vendor authorship scan on `_layer_evidence.closed_world_paths` | One parametrized AST gate | 13 |
| **T-CLS-B** | B | Hardcoding/bypass census | Extend EE-A1 pattern | 9 |
| **T-CLS-C** | C | Fail-closed defaults on sanctioned roots | EBH-5 behavioral negatives | 12 |
| **T-CLS-D** | D | Duplicate selector spot-check (not full DUP-X) | Static gate + register row | 12 |
| **T-CLS-E** | E | Shadow store / sqlite authority | Annex touchpoint subset | 6 |
| **T-CLS-F** | F | Public seam typing / register | **Operator branch** (§7); no RATIFIED-only without OD + evidence pack | 13 |
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
| **T-REM-Q2D2** | Operator declines RATIFIED-only closure or historical ratification does not cover current EBH-6 obligation (§7) | `nexus_loop.py` ADE typing (`EBH6-DEBT-Q2-D2`, FRZ-TYP-03) | **Yes** |
| **T-REM-PROH** | T-CLS-* finds PROHIBITED | Scoped prod fix | **Yes** |

### Final publication (Batch 5)

| TASK-ID | Actions |
| --- | --- |
| **T-PUB-01** | Single pytest @ IMPLEMENTATION_SHA; update all CURRENT SHA fields; census shows **0 EVIDENCE groups** |
| **T-PUB-02** | Update VBN audit + scope dossier final banner **candidate handoff** (not CLOSED) |

### Protocol 03 and operator (Batch 6–7)

| TASK-ID | Actions |
| --- | --- |
| **T-P03-01** | Independent ECAP B1–B6; accept **16** layer RESOLVED + **23** family dispositions + **133** group dispositions |
| **T-OD-08** | Operator OD-EBH6-08 only if P03 PASS |

---

## Section 6 — Batch Execution Plan (B1–B7)

| Batch | Name | Tasks | Exit gate | On FAIL |
| --- | --- | --- | --- | --- |
| **B1** | Normative scope + SHA reconciliation | T-NORM-01, T-SHA-01, T-DOC-01 | L9-A ↔ `_layer_evidence` consistent; one SHA ladder | Fix scope dossier / SHA drift; repeat B1 |
| **B2** | Production closed-world + L3 (16 layers) | T-LAY-* | **Verified** closed-world inventory + L3 evidence **per layer** — **not** automatic RESOLVED for 16 layers | Incomplete inventory/L3 → repeat affected T-LAY-*; no B3 |
| **B3** | Class gates (109 groups → 0 EVIDENCE) | T-CLS-*, T-RAG-01 | **All 133** applicable groups satisfy §3.B B3 gate (test↔invariant, production scope, outcome, FINDING/N/A, no auto-PASS); census **`EV_GROUPS == 0`**; layer rows **CANDIDATE-RESOLVED** when layer controls reconciled | Any unresolved mandatory blocker or unbasis PARTIAL/RATIFIED → fix gates/findings; **repeat B3** (and B2 layers if scope wrong) |
| **B4** | Family + L9 BLK handoff pack | T-FAM-01, T-L9BLK-01 | Handoff ready @ SHA (23 family rows + 8 BLK map — auditor replay bundle, not acceptance) | Complete handoff doc; repeat B4 |
| **B5** | Final qual publication (+ optional T-REM-*) | T-PUB-* | Consolidated pytest PASS @ IMPLEMENTATION_SHA; census 0 EVIDENCE groups | Fix qual/doc; repeat B5; **no B6** |
| **B6** | Protocol 03 | T-P03-01 | Independent **PASS** only when **all 23** families **accepted** in EBH-6 scope + **16** layer RESOLVED accepted + §3 A–D — not merely PENDING=0 | REJECT/BLOCKED → precise remediation list + re-audit (typically B3/B2/B5); **no OD-EBH6-08** |
| **B7** | OD-EBH6-08 | T-OD-08 | Roadmap EBH-6 CLOSED | N/A — only after B6 PASS |

**Hard sequencing rule:** **No Batch 6** before Batch 5 census shows **0** gap groups **and** B3 §3.B gate satisfied **and** B2 inventory+L3 complete for **16** layers. **No final layer RESOLVED** before B6 independent acceptance.

---

## Section 7 — Operator Decisions

### OD-EBH6-Q2D2 — Q2-D2 / EBH6-DEBT-Q2-D2 (no silent default)

**There is no “default if silent” auto-accept of RATIFIED for Protocol 03.** Before B6, the operator and auditor must reconcile **historical ratification** against the **current EBH-6 obligation**:

| Check | Source |
| --- | --- |
| Finding row | [`_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) — `finding_id="EBH6-DEBT-Q2-D2"`, layer `semantic_core`, class F |
| Production seam | `intergrax/runtime/nexus/nexus_loop.py` — AuthoritativeDecisionExposure typing |
| Violated invariant | **FRZ-TYP-03** (`violated_invariant` on register record) |
| Current status | `FindingStatus.RATIFIED` with notes — ratified for EBH-6 wave; not substitute for independent acceptance |

**Decision tree:**

1. **If historical ratification + evidence pack demonstrably covers the current EBH-6 obligation** (invariant, production scope, auditor-visible negative/positive outcome): publish **evidence pack** for independent Protocol 03 confirmation; proceed to B6 with FINDING disposition documented (not prod change required).
2. **If not covered:** require **explicit operator decision** on OD-EBH6-Q2D2 **or** authorize **T-REM-Q2D2** under **OD-EBH6-01** (production remediation). Silence does **not** ratify closure.

| Decision | Question | Default if silent |
| --- | --- | --- |
| **OD-EBH6-Q2D2** | Does ratified **EBH6-DEBT-Q2-D2** satisfy EBH-6 Protocol 03 with independent evidence, or is remediation required? | **No auto-accept** — remain OPEN until explicit operator answer or T-REM-Q2D2 |
| **OD-EBH6-01** | Open implementation wave for Q2-D2 or PROHIBITED fixes? | Stay OPEN |
| **OD-EBH6-08** | Publish stage closure on roadmap | After P03 PASS only (Batch 7) |

---

## Section 8 — Closure Simulation (must pass before publish)

Pre-publication **static** dry-run of B1–B7: each batch exit gate must be **satisfiable** at handoff SHA with **documented FAIL remediation** — **no auto-PASS promise** from mechanical counts alone.

### Batch simulation (B1–B7)

| Batch | Scope | Pass condition (static) | FAIL → remediation |
| --- | --- | --- | --- |
| **B1** | Normative + SHA | 133 vs 109 crosswalk documented; single SHA ladder | Re-run T-NORM-01 / T-SHA-01 |
| **B2** | **16** layers | Per-layer closed-world inventory + L3 verified | Repeat T-LAY-*; block B3 |
| **B3** | **133** groups + layer controls | §3.B B3 gate per group; `EV_GROUPS==0`; **16** CANDIDATE-RESOLVED | Repeat B3; B2 if scope error; no PARTIAL/RATIFIED without basis |
| **B4** | **23** families + **8** L9 BLK | Handoff pack @ SHA (not auditor acceptance) | T-FAM-01 / T-L9BLK-01 |
| **B5** | Publication | T-PUB-* pytest @ SHA; 0 EVIDENCE | Fix qual; block B6 |
| **B6** | Protocol 03 | **All 23** families independently **accepted**; **16** layers RESOLVED **accepted**; mandatory findings reconciled; §3 A–D | REJECT → remediation + re-audit path (B2/B3/B5) |
| **B7** | OD-EBH6-08 | B6 PASS only | Do not close stage |

### Cross-cutting acceptance rows

| Acceptance criterion | Task(s) | Evidence | Test / audit | Independent | FAIL remediation |
| --- | --- | --- | --- | --- | --- |
| Register mechanical | T-PUB-01 | register gates | ebh6 integrity | Auditor spot-check | Fix register; repeat B5 |
| 0 PROHIBITED | T-CLS-B, scans | register | prohibited=0 | Auditor | **T-REM-PROH** + B3 |
| CONFIG-X | T-CLS-I, CFG family | 54/54 | config_x gate | Auditor | B3 class I |
| VBN audit accepted | T-PUB-02, T-P03-01 | VBN md | consolidated pytest | **Protocol 03** | Reject → B3/B5 |
| **133** A–J dispositions | T-CLS-* | `_ADJUDICATION` + §3.B | parametrized class gates | Auditor | **B3** — not EV count alone |
| **16** layers | T-LAY-* + B3 + B6 | inventory, L3, CANDIDATE → accepted RESOLVED | L3 integrity | **Protocol 03** | B2/B3/B6 loop |
| **23** families | T-FAM-01, T-P03-01 | catalog | family gates | **All accepted @ B6** | Remediation + re-audit |
| **8** L9 BLK | T-L9BLK-01, T-P03-01 | l9 evidence | cited pytest | Auditor replay | B4/B6 |
| Q2-D2 / **EBH6-DEBT-Q2-D2** | T-CLS-F, §7 | register + evidence pack | typing gate | Auditor + **OD-EBH6-Q2D2** | **T-REM-Q2D2** if OD-EBH6-01 |
| Mandatory findings | register | `_findings_register.py` | integrity tests | P03 | FINDING or remediate |
| 51 NYQ | **Out of scope** | matrix doc | RPL-G only | EBH-7 | Documented deferral |
| SHA freshness | T-SHA-01, T-PUB-01 | ledger | pytest @ SHA | Auditor | T-SHA-01 |
| **OD-EBH6-08** | T-OD-08 | roadmap | operator record | Post-B6 PASS only | Block B7 |

**Simulation result:** This plan is **COMPLETE** as a closure blueprint only if B1–B5 achieve **0 EVIDENCE groups**, **B2 inventory+L3 for 16 layers**, **B3 full §3.B gate**, and B6 independently accepts **133** groups, **16** layers, and **23** families. Unknown defects → **T-REM-PROH** + re-audit — **not** stage CLOSED and **not** OD-EBH6-08 without B6 PASS.

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

1. Replaced prior rejected [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) with sections 1–10 above; **F01–F03 correction** (B2/B3 layer order, B3 acceptance gate, OD-EBH6-Q2D2 explicit reconciliation).
2. Status banner: **FINAL COMPLETE CLOSURE PLAN — CORRECTED / READY FOR INDEPENDENT ACCEPTANCE**.
3. Published on branch **`development`** from baseline **`12a34692e0b5a8113a264d71ce5185dd609da5b9`**.

| Record | Value |
| --- | --- |
| **PLAN_PUBLICATION_SHA** | *(updated in follow-up publication commit)* |
| **Branch** | `development` |
| **Post-approval implementation** | B1–B7 only after independent plan audit PASS; no production code or `_ADJUDICATION` changes from this commit alone |

---

*End of definitive EBH-6 closure plan document.*
