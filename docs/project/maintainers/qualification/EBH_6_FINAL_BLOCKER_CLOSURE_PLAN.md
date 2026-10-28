# EBH-6 — Final Blocker Closure Plan

| Field | Value |
| --- | --- |
| **Status** | **READY FOR INDEPENDENT PLAN AUDIT** |
| **Mode** | PLAN ONLY — no implementation authorized by this document |
| **EXPECTED_START_HEAD** | `37ad12e766e2ffb14f89b0053a8e021b92ef167f` |
| **Baseline census @ HEAD** | **109** A–J gap groups · **327** EVIDENCE cells · **8** L9 BLK · **23** families PENDING · Protocol 03 OPEN |

**Authorities:** Roadmap §3.0.8 / §5 · FRZ checklist · ECAP · VBN spec · SYSTEM_INVARIANTS · Architecture Principles · [EBH_6_FINAL_PRODUCTION_CODE_AUDIT_DOSSIER.md](EBH_6_FINAL_PRODUCTION_CODE_AUDIT_DOSSIER.md) · [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md).

**Explicit non-goals:** No Protocol 03 PASS · no OD-EBH6-08 · no EBH-6 CLOSED · no DUP-X / EBH-7 / live PROD-Q · no new audit framework · no production delta without **OD-EBH6-01**.

---

## Executive summary

Closing EBH-6 requires **independent Protocol 03 acceptance** plus closing **109 grouped obligations** through a **small number of mechanism-scoped evidence waves** (mostly qualification replay and targeted gates), not 109 isolated tests. **51 NYQ** replaceability rows remain **EBH-7**. This plan maps blockers → invariants → owners → reusable proofs → batches.

---

## Zadanie 1 — 109 grup A–J (grupowanie mechanizmów)

### 1.1 SSOT and counting

| Item | Source |
| --- | --- |
| Gap groups | `_l3_aj_group_reconciliation.py` → `EBH6-P03-AJ-{layer}-{class}` |
| Applicability | `_layer_evidence.py` L9-A matrix (N/A = 81 cells with rationale) |
| L3 anchor | 48 records in `_l3_reachability_evidence.py` |

**Already PARTIAL/PASS (do not re-audit generically):** 3 PASS + 67 PARTIAL groups including integrations factory slice, execution×B, governance×H, CONFIG-X hosting slice, etc. (see production code audit dossier).

### 1.2 Work packages (109 groups → **9 invariant bundles**)

Each bundle lists **classes × layers** covered among the **109** gap groups. One bundle = one qualification wave + one replay gate family where possible.

| WP-ID | Invariant (FRZ / VBN) | Semantic owner | Production path (representative) | Mandatory @ EBH-6? | Existing evidence to reuse | Missing evidence (concrete) | Bundled groups (count) | Prod code change? |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| **WP-A-VENDOR** | FRZ-PRD / VBN class **A** — production-effective vendor **authorship** of selection | Integrations + per-layer composition | `integrations/registry/factory.py`; layer-specific wiring under `applications/_shared/*_bridge.py` | **Yes** (13 layers × A) | EBH-5 inventory; integrations×A PARTIAL; factory fail-closed tests | **Per-layer closed-world scan** or TRACE-X parity proving no shadow selector outside factory/profile (not integration layer only) | **13** | **No** if static+reachability gates suffice; **Yes** if scan finds PROHIBITED literal |
| **WP-B-HARD** | VBN class **B** — hardcoding / bypass on production paths | Layer owners per roadmap | EE-A1 surface (execution); other layers via `_layer_evidence` path lists | **Yes** (9 layers × B) | EE-A1 P0 (execution); hardening-3 boundary | **Layer-scoped static census** tied to closed-world paths @ HEAD (one gate module parametrized by layer prefix) | **9** | **Only if** census flags PROHIBITED → OD-EBH6-01 |
| **WP-C-DEFAULT** | Class **C** — implicit defaults / shadow config | Runtime hosting + composition | `runtime_config_bridge.materialize_runtime_config`; `context_bootstrap` | **Yes** (12 layers × C) | CONFIG-X 54/54 mapping; integrations factory fail-closed | **Behavioral proof**: unset mandatory profile/category → fail-closed on **sanctioned production roots** (extend EBH-5 behavioral negatives pattern) | **12** | Unlikely; fail-closed already in factory |
| **WP-D-DUP** | Class **D** — duplicate authority / parallel registries | Governance, recovery, tools | Governance middleware; recovery boundary; tool registry | **Yes** (12 layers × D) | NPSC42/NPSC5e samples | **Single “duplicate selector” static gate** per layer closed-world (grep AST for second `resolve_*` registry) | **12** | No unless finding |
| **WP-E-SHADOW** | Class **E** — shadow persistence authority | State / integrations edge | SQLite bundles; annex touchpoints | **Yes** (6 layers × E) | SQLite annex accountability | **Behavioral**: canonical store path cannot open sqlite except via integration bundle (subset of annex touchpoints) | **6** | No unless annex violation |
| **WP-F-TYP** | FRZ-TYP-03 class **F** | Contracts + semantic core + … | Public seams in `_layer_evidence` path lists | **Yes** (13 layers × F) | Register ratified rows; Q2-D2 FINDING on L3-SMC-03 | **Pyright/AST gate** on inventoried public surfaces OR accept FINDING+RATIFIED for closure policy (see §4) | **13** | **Q2-D2 only** via OD-EBH6-01 if operator requires remediation before ARCH-FREEZE |
| **WP-G-OBS** | FRZ-OBS class **G** — diagnostics/OBS must not mint truth | Observability plane | W5 wiring; `runtime_event_bus`; diagnostic orchestrator | **Yes** (9 layers × G) | W5-B2 partial | **Replay W5 + CTRL-X R2** on L3-OBS mechanisms (one parametrized qual module) | **9** | No |
| **WP-H-REACH** | Class **H** — reachability from sanctioned roots | TRACE-X + layer L3 | Registered composition sites; per-layer entrypoints in L3 annex | **Yes** (8 layers × H) | Sanctioned roots non-empty; partial H on several layers | **TRACE-X closed-world replay @ HEAD** for each gap layer’s `_layer_evidence.production_effective_samples` | **8** | No unless new bypass |
| **WP-I-INSTALL** | Class **I** — install-time / env coupling | Hosting + integrations | Env slugs; host profile loaders | **Yes** (13 layers × I) | CONFIG-X + env slug merge test | **Documented env contract test** per concern class (extend CONFIG-X reconciliation, not 13 tests) | **13** | No |
| **WP-J-RPL** | FRZ-RPL class **J** — replaceability behavioral proof | Integrations + matrix | RPL-G; integration factory | **Yes** (14 layers × J) | RPL-G empty blockers; integrations×J PARTIAL | **EBH-7 / 51 NYQ** for full matrix; EBH-6 minimum = **RPL-G + qualified row replay** for layers claiming extensibility in L9 annex | **14** | No @ EBH-6 (EBH-7 scope) |

**Crosswalk check:** 13+9+12+12+6+13+9+8+13+14 = **109** gap groups (classes overlap same layer once per class letter).

**Bundling rule:** Implement **one qual gate file per WP-ID** (parametrized by `EBH6_AUDIT_LAYERS` + class), updating `_ADJUDICATION` only when the gate’s `invariant_proves` string matches the WP invariant — **never** cell-by-cell PASS without gate.

### 1.3 Layers with highest density (prioritize within batches)

| Layer | Gap groups (of 10 classes) | Lead WP-IDs |
| --- | ---: | --- |
| `events_traceability` | 10 | H, G, A, J, I, F, D, C, B |
| `background_workers_infra_adapters` | 10 | same pattern |
| `rag_memory` | 10 | A, WP-RAG (§3), J, I, … |
| `application_services` | 7 | H, I, J, F, D, C, A |
| `integrations` | 5 | D, E, F, G (+ J/A/H/C/I/B already partial) |

---

## Zadanie 2 — 8 blockerów L9 i 23 rodziny

**Separation:** `_family_l9_evidence.py` = implementer proposal; `_family_catalog.INDEPENDENT_AUDITOR_DISPOSITION` = **unchanged PENDING** until Protocol 03.

### 2.1 L9 blockers → families

| L9 blocker | Family | Mandatory invariant | Existing evidence | Missing for independent acceptance | Acceptance gate |
| --- | --- | --- | --- | --- | --- |
| EBH6-BLK-L9-OWN-01 | OWN | Single semantic owner on EE surfaces | `test_ee_final_arch_owner_matrix_documented` | Live duplicate-owner scan @ HEAD (optional EBH-7) | Independent replay @ SHA |
| EBH6-BLK-L9-EXE-01 | EXE | No production execution bypass | EE-A1 P0 inventory | L3 execution mechanism cross-check (already indexed) | Protocol 03 + EE-A1 |
| EBH6-BLK-L9-GOV-01 | GOV | Governance ≠ execution identity | NPSC42 H1 tests | Full GOV-X2 replay **not** required for EBH-6 if plan WP accepts subset | NPSC42 H1 @ SHA |
| EBH6-BLK-L9-REC-01 | REC | No recovery god-object | NPSC5e + W3-C4 | DIST-SCALE-X full replay deferred | NPSC5e + W3 on L3-REC |
| EBH6-BLK-L9-SEC-01 | SEC | Security composition fail-closed | CTRL-X R1 | Full SEC family matrix | CTRL-X R1 @ SHA |
| EBH6-BLK-L9-REL-01 | REL | Capacity ≠ execution owner | EE-B1-2 gate | Full REL/backpressure matrix | EE-B1-2 @ SHA |
| EBH6-BLK-L9-HRN-01 | HRN | Harness boundary | HRN qual bundle | Reference harness drift replay | HRN bundle @ SHA |
| EBH6-BLK-L9-TEN-01 | TEN | Tenant scope | `test_tenant_scope` | Cross-scenario blast radius (EBH-7) | Tenant gate @ SHA |

### 2.2 All 23 families (summary map)

| Family | Mandatory invariant (implementer bundle) | Existing evidence | Missing @ EBH-6 closure | Historical PASS vs replay |
| --- | --- | --- | --- | --- |
| BND | Tier import directions | Hardening-3 gate | Full BND reachability census | EBH-2I scoped PASS — **replay tier gate @ HEAD** |
| OWN | Owner uniqueness | EE owner matrix test | Duplicate-owner scan | Partial — **L9 BLK** |
| CTR | No PROHIBITED leakage | Register prohibited=0 | Runtime leakage sweep | INT-CONFIG historical — **register only** |
| TYP | Typing debt disposition | Register ratified Q2-D2 | Public seam pyright (policy §4) | GOV-X2 historical — **debt RATIFIED** |
| PLG | Plugin SPI / no RAG-local registry | EBH-5 inventory | SPI behavioral matrix | EBH-5 scoped |
| RPL | RPL-G + matrix | RPL-G gate | **51 NYQ rows** | CONFIG-X preserved — **EBH-7** |
| EXE | Single ExecutionRuntime | EE-A1 | L3 cross-ref | EE-A1 historical — **replay @ HEAD** |
| GOV | Governance boundary | NPSC42 H1 | GOV-X2 full | GOV-X2 CLOSED — **subset replay** |
| STA | SQLite annex | Annex gate | Canonical store behavior | STATE-X scoped |
| REC | Recovery plane | NPSC5e/W3 | NPSC5e full replay | Partial — **L9 BLK** |
| CMP | COMPAT policy unchanged | Scope dossier attestation | Consumer blast-radius | COMPAT-X CLOSED — **no HEAD behavioral claim** |
| CFG | CONFIG-X reconciliation | 54/54 gate | Live materialization audit | FRZ-CFG PASS @ fac9a43f — **reconcile only** |
| SEC | Fail-closed security | CTRL-X R1 | Full SEC | CTRL-X scoped — **L9 BLK** |
| REL | Capacity architecture | EE-B1-2 | Full REL matrix | W4 scoped — **L9 BLK** |
| CTL | Control-plane attach | CTRL-X R2 | Twelve-plane drift | CTRL-X historical |
| OBS | OBS non-authoritative | W5-B2 | TRACE-X global | HARNESS-W5 scoped |
| TRC | Trace spine | TRACE-X P4 samples | Full terminal-state completeness | TRACE-X partial |
| PRD | Provider qualification | Register + EBH-5 | **Live PROD-Q** | Path B OUT OF SCOPE |
| REG | Regression bundle | EBH-6 pytest bundle | QUAL-X global | Not FRZ-REG PASS |
| HRN | Harness boundary | HRN bundle | Drift replay | **L9 BLK** |
| TEN | Tenant isolation | Tenant scope test | EVOL blast radius | TENANT-X CLOSED — **replay gate** |
| DOC | Mandatory docs | Scope dossier content gate | FRZ-DOC global | Scoped candidate |
| VBN | VBN artifacts @ HEAD | Register + VBN audit md | Independent VBN PASS | EBH-6 parent stage |

**Acceptance gate (all families):** Independent auditor sets disposition @ **CERTIFICATION_HANDOFF_SHA** after ECAP §2.B/§2.C replay — implementer cannot set PASS.

---

## Zadanie 3 — RAG skipped (EBH-5 reachability)

### 3.1 Root cause

| Fact | Detail |
| --- | --- |
| Test | `test_ebh5_reach_no_second_global_provider_registry_in_rag_graph` |
| Expected path | `intergrax/rag/graph/integration_graph_store.py` |
| @ HEAD | **File absent** — pytest **skips** with `"graph integration module absent"` |
| Actual layout | `intergrax/rag/graph/bootstrap/graph_store_bootstrap.py` adapts `IntegrationGraphStore` → `RagGraphStore`; contracts under `rag/graph/contracts/` |

### 3.2 Disposition (plan)

| Option | Recommendation |
| --- | --- |
| Mechanism absent | **Rejected** — RAG graph materialization exists via bootstrap adapter |
| Test stale | **Accepted** — path renamed/refactored; test targets removed module name |
| Missing dependency | **Rejected** — skip is file existence check, not import failure |
| Missing qualification | **Partial** — invariant “no `graph_store_backend` shadow field” should apply to **current** module(s) |

**Planned closure (qual-only, Batch B-RAG):**

1. Repoint gate to `graph_store_bootstrap.py` (+ any graph provider modules in closed-world list) — **qual test edit only** (no prod).
2. Re-run reachability gate; if pass, promote `rag_memory×A` to **PARTIAL_CANDIDATE** with explicit `scope_limitation` (graph bootstrap slice).
3. Do **not** install packages; static read + existing EBH-2G historical evidence cited as supplementary, not global PASS.

**Operator approval:** Doc-only + qual test path fix — **no OD-EBH6-01**.

---

## Zadanie 4 — Q2-D2 (EBH6-DEBT-Q2-D2)

| Field | Value |
| --- | --- |
| Register status | **RATIFIED** (`_findings_register.py`) |
| Classification | **REQUIRES_ADR** · remediation_class **ADR** |
| Location | `intergrax/runtime/nexus/nexus_loop.py` — ADE typing seam |
| Invariant | **FRZ-TYP-03** (generic object semantic seam) |
| Production-effective | **YES** |
| L3 disposition | **FINDING** on mechanism L3-SMC-03 (not auto-PASS) |

### 4.1 EBH-6 closure policy (plan)

| Statement | Plan |
| --- | --- |
| RATIFIED ≠ mandatory prod fix in EBH-6 | **Correct** — EBH-6 may close with **FINDING + RATIFIED** recorded and independent auditor acceptance of debt disposition |
| When prod fix required | **ARCH-FREEZE / EBH-7** or operator mandate — triggers **OD-EBH6-01** |
| OD-EBH6-01 scope (if opened) | Narrow typing fix on ADE projection in `nexus_loop.py` to `AuthoritativeDecisionExposure[...]` without widening public ABI; pyright on nexus public surface; update register row to RESOLVED with evidence SHA |

**WP-F-TYP** must document auditor choice: **accept RATIFIED** for Protocol 03 vs **require remediation** before EBH-6 CLOSED (operator decision register).

---

## Zadanie 5 — Execution batches (minimal)

### Batch 0 — Independent Protocol 03 (mandatory, no implementer PASS)

| Field | Content |
| --- | --- |
| Scope | Replay @ `CERTIFICATION_HANDOFF_SHA`: 109 groups, 48 L3, 23 families, 8 L9 BLK |
| Files | Existing qual SSOT + dossiers only |
| Closes | **None** (auditor disposition only) |
| Tests | Consolidated pytest command in VBN audit md |
| Cost/risk | Auditor time; risk of REJECT if WP bundles insufficient |
| Acceptance | Auditor attestation; FRZ updates **forbidden** to implementer |
| Dependencies | None |
| Operator | Schedule independent auditor |

### Batch 1 — WP-H-REACH + TRACE-X (qual)

| Scope | Closed-world reachability for **8** gap H groups |
| Areas | `tests/qualification/trace_x/`, `_l3_reachability_evidence.py` refs, `_ADJUDICATION` updates |
| Closes | **WP-H-REACH** (8 groups) |
| Tests | Existing TRACE-X P4/P5 gates + one parametrized “sanctioned root reaches layer sample” qual module |
| Prod change | No |
| Operator | None (qual-only) |

### Batch 2 — WP-A-VENDOR + WP-B-HARD (qual static)

| Scope | Layer-parametrized static gates using `_layer_evidence.closed_world_paths` |
| Closes | Up to **22** groups (13 A + 9 B) — **PARTIAL** unless scan clean → PASS_CANDIDATE per layer |
| Tests | Extend pattern from EE-A1 / EBH-6 mechanical exit — **one module, parametrized** |
| Prod change | Only if PROHIBITED finding → stop → OD-EBH6-01 |
| Operator | Approve if PROHIBITED remediation scope opened |

### Batch 3 — WP-C-DEFAULT + WP-I-INSTALL (qual behavioral)

| Scope | Fail-closed configuration negatives on sanctioned composition registry |
| Closes | Up to **25** groups (12 C + 13 I) — likely **PARTIAL** |
| Tests | Reuse EBH-5 behavioral negative patterns |
| Prod change | No |
| Operator | None |

### Batch 4 — WP-G-OBS + WP-D-DUP + WP-E-SHADOW (qual)

| Scope | W5/CTRL-X replay + static duplicate-selector scan + sqlite annex subset |
| Closes | Up to **27** groups (9+12+6) |
| Tests | Existing W5 + new parametrized static scans (qual) |
| Prod change | No |
| Operator | None |

### Batch 5 — B-RAG (qual test repair)

| Scope | Fix EBH-5 reachability path; re-adjudicate `rag_memory×A` |
| Closes | **1** group (A on rag_memory) → PARTIAL expected |
| Tests | `test_ebh5_reach_no_second_global_provider_registry_in_rag_graph` → current paths |
| Prod change | No |
| Operator | None |

### Batch 6 — WP-F-TYP + WP-J-RPL (policy + EBH-7 boundary)

| Scope | Document auditor policy on Q2-D2; link **14** J groups to RPL matrix NYQ deferral |
| Closes | **13** F groups as PARTIAL/FINDING per policy; **J** groups remain PARTIAL with EBH-7 pointer |
| Tests | Optional pyright gate (qual) — **operator chooses** vs RATIFIED acceptance |
| Prod change | Q2-D2 only via Batch O1 |
| Operator | **Decision:** accept RATIFIED for EBH-6 vs require OD-EBH6-01 before stage CLOSED |

### Batch O1 — OD-EBH6-01 (production, optional)

| Scope | Q2-D2 remediation + any PROHIBITED from Batch 2 |
| Closes | Register debt RESOLVED; may upgrade semantic_core×F |
| Operator | **Required OD-EBH6-01** |

### Batch EBH-7 (out of EBH-6)

| Scope | 51 NYQ replaceability rows |
| Closes | RPL **global** — not EBH-6 |

---

## Closure arithmetic (realistic)

| Milestone | Blockers remaining |
| --- | --- |
| After Batches 1–5 (implementer qual) | **0–30** gap groups (J + policy F + auditor disagreement) |
| After Protocol 03 accept | Auditor sets family dispositions + L9 BLK |
| After OD-EBH6-08 (operator) | Stage CLOSED on roadmap — **not in this plan** |
| EBH-7 | NYQ + full RPL behavioral |

---

## Verification (plan audit)

Independent plan auditor checks:

1. Bundle count **≤ 6 implementer batches** (+ Protocol 03 + optional O1).
2. No claim of 109 separate tests.
3. RAG skip explained and qual-only fix path.
4. Q2-D2 not forced into prod without OD-EBH6-01.
5. 51 NYQ explicitly deferred.

---

## Publication

| Field | Value |
| --- | --- |
| **Document** | This file |
| **IMPLEMENTATION_SHA** | `37ad12e766e2ffb14f89b0053a8e021b92ef167f` (plan baseline HEAD) |
| **PLAN_PUBLICATION_SHA** | *(set @ plan commit)* |
| **Status** | **READY FOR INDEPENDENT PLAN AUDIT** |

**Related SHA @ plan baseline:** IMPLEMENTATION `1f3fb2ae3…` · CERTIFICATION_HANDOFF `3f0a25439…` (production audit) · HEAD @ plan start `37ad12e766e2ffb14f89b0053a8e021b92ef167f`.
