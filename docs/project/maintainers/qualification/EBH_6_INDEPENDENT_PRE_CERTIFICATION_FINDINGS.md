# EBH-6 — Independent Pre-Certification Findings (Protocol 03 prep)

| Field | Value |
| --- | --- |
| **CLOSURE_BASELINE_SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **CERTIFICATION_HANDOFF_SHA** | @ final enterprise closure wave (see ledger) |
| **Disposition** | **PRE-CERT PACKAGE READY — INDEPENDENT VERDICT PENDING (E01-PRE-01)** — **not CLOSED** |

Agent reconciliation batch for **EBH6-AUD-01..04**. Does **not** issue AUDIT PASS, **OD-EBH6-08**, or **EBH-6 CLOSED**.

## AUD reconciliation

| ID | Finding | Reconciliation |
| --- | --- | --- |
| **EBH6-AUD-01** | Stale evidence provenance | **REMEDIATED (docs + ledger).** EBH-5 parent **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)** under **OD-EBH5-08 APPROVED 2026-10-28** ([`EBH_5_OD_EBH5_08_CLOSURE_SIGNATURE_RECORD.md`](EBH_5_OD_EBH5_08_CLOSURE_SIGNATURE_RECORD.md)). Removed stale `badb7a086…` as current HEAD; SHA ladder separates **HISTORICAL** (EBH-5 audit `33ecf19…`, publication tree `50d3b778…`) vs **CURRENT** @ `716efd67…`. |
| **EBH6-AUD-02** | Synthetic layer coverage | **REMEDIATED (SSOT).** `EBH6-LAYER-*` rows downgraded from false RESOLVED sentinels to **DISCOVERY REQUIRED — L9 LAYER SAMPLE** / **OPEN**; mechanical `unclassified=0` preserved for inventory-style rows only. SQLite **38** paths replay EBH-5 annex @ HEAD (unchanged count; scope verified via closed-world registry). |
| **EBH6-AUD-03** | Architecture-family evidence | **PARTIAL.** Catalog + gate paths verified @ HEAD; **16/23** families remain **PENDING — L9 Protocol 03** (catalog test alone insufficient). Matrix updated with per-family proof ref or blocker — no global FRZ PASS. |
| **EBH6-AUD-04** | Operator authority | **ACKNOWLEDGED.** **OD-EBH6-01**, **OD-EBH6-02** remain **OPEN**. No operator signatures forged. Production code correction beyond doc/qualification reconciliation requires operator **implementation wave** phrase. |

## A–J completeness truth table (honest)

Detection classes **A–J** (VBN §2.2) vs mandatory §3.0.8 layer audit. **Layer sentinels are not proof of zero violations.**

| Class | Scope (normative) | Sampled finding rows @ HEAD | Layer obligations (16) | Honest gap |
| --- | --- | --- | --- | --- |
| **A** | Core vendor construction | Via SQLite + inventory replay | contracts, application_services (partial) | **L9 sample required** on semantic_core paths |
| **B** | Shadow config/selection | Inventory replay; no new PROHIBITED mechanical | semantic_core, application_services | **L9 reachability narrative required** |
| **C** | Silent fallback | RPL-G replay @ HEAD (EBH-5) | — | **L9 spot-check** beyond mechanical gates |
| **D** | Duplicate selectors | Deferred **DUP-X** | composition_roots | **BLOCKED — DUP-X not started** |
| **E** | SQLite / local store | **38** `EBH6-SQLITE-*` @ annex | state_persistence | Annex scope matches HEAD registry |
| **F** | Typing / pseudo-contract seams | **EBH6-DEBT-Q2-D2** RATIFIED | semantic_core | ADR track — not PROHIBITED |
| **G** | Observability vendor lock-in | EBH-5 qualified row replay (partial) | observability_diagnostics | **L9 class-G production sample required** |
| **H** | Bypass / parallel ingress | **EBH6-DEBT-BND-COMPAT-R3** lab-only | composition_roots, background_workers | **L9 production path census required** |
| **I** | Inventory / concern mapping | **EBH6-VBN-INVENTORY-REPLAY** | integrations | Mechanical only |
| **J** | Extensibility claims vs proof | **51 NYQ** EBH-5 matrix rows | tools_skills_capabilities, rag_memory | **EBH-7 scope — explicit limitation** |

**16 layer rows:** `EBH6-LAYER-01..16` — **RESOLVED @ B-L9-2** with [`_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py) closed-world inventory; independent Protocol 03 still required.

## 23 architecture-family dispositions

| family_id | independent_disposition | Current proof @ HEAD or disposition |
| --- | --- | --- |
| BND | REPLAY PASS | `test_hardening_3_layer_boundary_gate.py::test_tier0_must_not_import_agents_or_applications` |
| OWN | PENDING | Catalog only — **EBH6-BLK-L9-OWN-01** (no dedicated ownership rescan @ HEAD) |
| CTR | PENDING | `test_ebh6_no_prohibited_unresolved_in_scope` — register integrity, not full CTR rescan |
| TYP | PENDING | `test_ebh6_tracked_debt_findings_ratified` — Q2-D2 debt only |
| PLG | REPLAY PASS | `test_ebh5_exit_inventory_complete` @ EBH-5 gates |
| RPL | REPLAY PASS | `test_ebh5_rpl_g_config_x_active_blockers_remain_empty` + 51 NYQ documented |
| EXE | PENDING | **EBH6-BLK-L9-EXE-01** — family_row_count gate only |
| GOV | PENDING | **EBH6-BLK-L9-GOV-01** — family_row_count gate only |
| STA | PENDING | `test_ebh6_exit_sqlite_annex_accountability` — SQLite slice only |
| REC | PENDING | **EBH6-BLK-L9-REC-01** |
| CMP | N/A — WITH EVIDENCE | COMPAT-X CLOSED historical — drift-only; no EBH-6 consumer change |
| CFG | REPLAY PASS | `test_ebh6_config_x_all_concerns_mapped_at_head` — 54/54 |
| SEC | PENDING | CTRL-X spot gate @ historical SHA — **EBH6-BLK-L9-SEC-01** current-HEAD gap |
| REL | PENDING | **EBH6-BLK-L9-REL-01** |
| CTL | REPLAY PASS | `test_r2_mw_05_fail_open_rejected_before_attach` @ CTRL-X historical |
| OBS | PENDING | `test_ebh5_exit_qualified_rows_have_behavioral_refs` — partial OTLP replay |
| TRC | PENDING | `test_ebh6_exit_sanctioned_composition_roots_non_empty` — wiring only |
| PRD | N/A — WITH EVIDENCE | Live PROD-Q slice explicitly OUT OF SCOPE |
| REG | PENDING | `test_ebh6_exit_mandatory_docs_present` — EBH-6 docs only, not QUAL-X |
| HRN | PENDING | **EBH6-BLK-L9-HRN-01** |
| TEN | PENDING | **EBH6-BLK-L9-TEN-01** — no global TEN promotion |
| DOC | PENDING | Mandatory doc presence gate — content freshness = L9 |
| VBN | PENDING | Mechanical `unclassified=0` / prohibited=0 — **not** L9 A–J completion |

## Mandatory blockers

| Blocker | Status |
| --- | --- |
| **OD-EBH6-01** | OPEN — `production delta = 0`; wave phrase not recorded |
| **OD-EBH6-02** | OPEN — independent plan audit |
| **OD-EBH6-08** | OPEN — post-L9 only |
| **L9 Protocol 03** | OPEN — no `EBH_6_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` |
| **DUP-X** | Not started — class D / duplicate mechanism closure |
| **51 EBH-5 NYQ rows** | Explicit limitation — EBH-7 |
| **EBH6-BLK-L9-* ** | Sixteen families lack current-HEAD independent proof beyond mechanical catalog gates |

## Required implementation scope

| Scope | Requires OD-EBH6-01? |
| --- | --- |
| Doc/qualification SHA + honesty reconciliation (this batch) | **Published** @ **PRE-CERT PUBLICATION_SHA** `a7d133a91…` on `origin/development` |
| Production remediation of PROHIBITED findings | **Yes** — none mechanical unresolved in frozen scope |
| DUP-X duplicate mechanism closure | **Yes** — separate stage |
| Full A–J manual production sampling | **L9 auditor work** — not agent closure |

## Operator decisions needed

1. **OD-EBH6-02** — accept EBH-6 plan / pre-cert pack for Protocol 03 scheduling (@ **PRE-CERT PUBLICATION_SHA**).
2. **OD-EBH6-01** — only if implementation wave / production delta required before L9.
3. **OD-EBH6-08** — only after independent L9 PASS (forbidden now).
## Protocol 03 readiness

**READY FOR INDEPENDENT AUDIT (CANDIDATE)** @ **PRE-CERT PUBLICATION_SHA** `a7d133a91cebc4c7cd8c5f265587168f05df5ee4` (implementation evidence @ `716efd67…`) — handoff pack + honest gaps documented.

**BLOCKED** for: parent CLOSED, global FRZ PASS, AUDIT PASS self-issue, wave closure without **OD-EBH6-01**.

## References

- [`EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md`](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md)
- [`EBH_6_OPERATOR_DECISION_REGISTER.md`](EBH_6_OPERATOR_DECISION_REGISTER.md)
- [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) §2.B
