# Enterprise Continuous Architecture Assurance Protocol (ECAP)

> **Role:** Operational protocol for *how* to plan, execute, document, and escalate enterprise architecture assurance across Integrax.  
> **Not:** a second roadmap, freeze checklist, semantic authority, or FRZ status ledger.

**Document ID:** `ECAP-1`  
**Language:** Technical English  
**Branch policy:** `development`  
**Companion authorities (read-only during audits):**

| Artifact | Role |
| --- | --- |
| [Platform Enterprise Completion Roadmap](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) | Program sequence, stage status, closure SSOT |
| [Platform Enterprise Freeze Acceptance Checklist](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) | Global `FRZ-*` acceptance criteria and permitted statuses |
| [Vendor / Backend Neutrality Evidence](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) | VBN artifact schemas (not PASS evidence) |
| [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) | Cross-layer `SYS-INV-*` MUST/MUST NOT index |
| [INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md](INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md) | Post-freeze guard families → gates |
| [INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md](INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) | Post-freeze change classes A/B/C |
| [PLATFORM_WIDE_ENTERPRISE_AUDIT.md](../audits/PLATFORM_WIDE_ENTERPRISE_AUDIT.md) | Historical enterprise audit (non-authoritative for current HEAD) |
| [INTERGRAX_ARCHITECTURE_PRINCIPLES.md](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) | Capability ownership and evolution principles |

**Baseline inventory (recompute at every audit):** **225** `FRZ-*` criteria in **26** families at checklist HEAD synchronized with repository checkout.

---

## ECAP-§1 — Purpose, Scope and Authority

### ECAP-§1.1 Purpose

This protocol MUST ensure that enterprise compliance is **evidence-driven**, **scope-explicit**, and **independently verifiable** at an exact Git SHA. It exists to prevent:

- Undiscovered structural defects reaching `ARCH-FREEZE`;
- Unauthorized vendor dependencies and non-replaceable infrastructure;
- Duplicated mechanisms and shadow execution paths;
- Unsafe distributed semantics and tenant isolation failures;
- Scenario development forcing fundamental platform redesign.

### ECAP-§1.2 Non-negotiable principles

| ID | Principle |
| --- | --- |
| **ECAP-P1** | Passing functional tests or a small suite MUST NOT imply enterprise compliance. |
| **ECAP-P2** | Historical stage `CLOSED` MUST NOT imply current-HEAD `FRZ-*` PASS. |
| **ECAP-P3** | A change-delta audit MUST NOT claim full-platform certification. |
| **ECAP-P4** | Implementation agents MUST NOT self-accept `FRZ-*` PASS. |
| **ECAP-P5** | Replaceability MUST NOT be claimed without RPL-A..H evidence where extensibility is promised. |
| **ECAP-P6** | Corrective work MUST be opened only for proven invariant violations or missing mandatory evidence—not hypothetical perfection. |

### ECAP-§1.3 Mandatory application

This protocol MUST be applied to:

- Every production-relevant implementation and architecture-relevant change;
- Parent stage closure and `ARCH-FREEZE` admission;
- Current-HEAD enterprise recertification (`ROADMAP-REPLAY-X`, `EBH-7`, `ENT-AUDIT-X`);
- Post-freeze evolution (Classes A/B/C per evolution governance);
- Scenario and application readiness waves that touch frozen surfaces.

### ECAP-§1.4 Authority hierarchy

1. **Roadmap** — program order and stage closure authority only.  
2. **Freeze checklist** — `FRZ-*` criteria and permitted statuses (`OPEN`, `PASS`, `BLOCKED`, `N/A — WITH EVIDENCE`).  
3. **Domain architecture / ADRs** — semantic meaning of mechanisms.  
4. **SYSTEM_INVARIANTS** — cross-layer MUST/MUST NOT rules (`SYS-INV-*`).  
5. **VBN evidence specification** — mandatory replaceability artifact shapes.  
6. **Post-freeze guard / evolution docs** — ongoing controls after core freeze.  
7. **This protocol (ECAP)** — procedures, evidence levels, escalation, and report templates only.

Conflicts with ratified domain semantics: **STOP — ARCHITECTURE DECISION REQUIRED** (do not encode a new rule here).

### ECAP-§1.5 Audit events covered

| Event | ECAP mode (§2) |
| --- | --- |
| Implementation PR / agent session | Change Delta Audit |
| Parent stage closure | Parent Closure Audit |
| `EBH-7`, `ENT-AUDIT-X`, `ARCH-FREEZE` prep | Full Platform Current-HEAD Audit |
| Post-freeze merged change | Post-Freeze Change Assurance |

### ECAP-§1.6 Roles

| Role | Responsibilities |
| --- | --- |
| **Implementation agent** | Produce scoped changes; attach *candidate* evidence; never final PASS. |
| **Independent auditor** | Exact-SHA verification; adversarial review; findings classification (§13). |
| **Operator / program owner** | Stage closure on roadmap; FRZ status updates only after independent acceptance. |

### ECAP-§1.7 Evidence definition

**Constitutes evidence:**

- Artifacts at explicit Git SHA (code, contracts, qualification markdown, gate logs);
- Executed tests/gates whose paths are recorded in qualification records;
- Independent auditor attestation tied to SHA and scope;
- VBN inventory/matrix/findings registers per [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md).

**Does NOT constitute evidence:**

- Cursor/ChatGPT narrative without SHA-bound artifacts;
- Import-only analysis without production-effectiveness classification;
- Monkeypatch-only replaceability demos;
- Historical PASS on a different SHA without current-HEAD replay;
- Checklist status columns updated without independent audit.

### ECAP-§1.8 Separation of concerns

| Phase | Owner | Output |
| --- | --- | --- |
| Implementation | Agent/engineer | Code + candidate evidence pack |
| Verification | Auditor + gates | Findings + executed proofs |
| Qualification | Domain qualification stages | Stage certification records |
| Acceptance | Operator per roadmap | Stage CLOSED + FRZ updates |

---

## ECAP-§2 — Enterprise Assurance Operating Model

### ECAP-§2.A Change Delta Audit

**Trigger:** Every production-relevant change before merge or before declaring *READY FOR AUDIT*.

**Mandatory inputs:** `git diff` / changed file list; symbol-level impact; dependency graph slice; composition owners touched.

| Step | Action |
| --- | --- |
| **A1** | Enumerate changed files and public symbols. |
| **A2** | Map direct and indirect dependencies (imports, composition wiring, config keys). |
| **A3** | Identify callers and downstream consumers on production paths. |
| **A4** | Record semantic owner and sanctioned composition owner (lookup domain canon). |
| **A5** | Classify production-effective reachability (`yes` / `no` / `unknown — FAIL` until resolved). |
| **A6** | List applicable `FRZ-*`, `SYS-INV-*`, and guard families (post-freeze). |
| **A7** | Assess impact on historical certifications (scoped vs global). |
| **A8** | Select regression gates — only existing tests confirmed at SHA. |

**Prohibited claim:** “Enterprise certified” or “freeze-ready” from delta audit alone.

### ECAP-§2.B Parent Closure Audit

**Trigger:** Closing a mandatory roadmap parent (e.g. `EBH-5`, `GOV-X2`, `TENANT-X`).

| Step | Action |
| --- | --- |
| **B1** | Closed-world inventory of in-scope surfaces. |
| **B2** | List applicable `FRZ-*` for the parent (roadmap §2.0.2 linkage). |
| **B3** | Positive proofs per stage qualification record. |
| **B4** | Adversarial negatives (fail-closed, bypass attempts). |
| **B5** | Execution evidence bundle at exact SHA. |
| **B6** | Independent exact-SHA acceptance recorded in roadmap §5 ledger. |

### ECAP-§2.C Full Platform Current-HEAD Audit

**Trigger:** `EBH-7`, `ENT-AUDIT-X` FULL/FREEZE modes, `ARCH-FREEZE` admission, `ROADMAP-REPLAY-X`.

| Step | Action |
| --- | --- |
| **C1** | Closed-world inventory across Tier-0..3 and all architecture families (roadmap §3.1 minimum coverage). |
| **C2** | Cross-domain dependency and duplicate authority scan. |
| **C3** | Full VBN taxonomy pass (roadmap §3.0.8 domains 1–31 + discoveries). |
| **C4** | Enterprise regression / drift vs last accepted baseline SHA. |
| **C5** | Produce ECAP report template (§14) with platform scope banner. |

### ECAP-§2.D Post-Freeze Change Assurance

**Trigger:** Any post-freeze change to frozen surfaces per [INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md](INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md).

| Step | Action |
| --- | --- |
| **D1** | Classify change A / B / C (default C if ambiguous). |
| **D2** | Contract compatibility analysis. |
| **D3** | Impact analysis → guard matrix rows ([INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md](INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md)). |
| **D4** | Regression selection (minimum bundles per class). |
| **D5** | Reopen evaluation (Architecture Reopen Record if Class C). |
| **D6** | Evidence continuity — link to frozen baseline SHA and new evidence SHA. |

---

## ECAP-§3 — Canonical Enterprise Invariant Coverage

### ECAP-§3.0 Coverage mechanics

At verified checklist baseline: **225** criteria, **26** families. Each family subsection below maps to checklist rows by explicit ID list. 

**Mechanical coverage statement:** Let `C` be the set of IDs matching `FRZ-[A-Z]+-\d+` in `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` at audit SHA. ECAP requires an audit procedure for every `id ∈ C` via its family subsection. If `|C| ≠ 225` at HEAD, recompute inventories before audit. If any `id ∈ C` lacks a family mapping in this document → **COVERAGE GAP — DOCUMENT NOT READY FOR AUDIT**.

### ECAP-§3.0.1 Theme index (26 assurance themes)

| # | Theme | FRZ family |
| --- | --- | --- |
| 1 | Hard layer boundaries | BND |
| 2 | Exactly-one ownership | OWN |
| 3 | Canonical contracts | CTR |
| 4 | Strong typing | TYP |
| 5 | Pluginability | PLG |
| 6 | Vendor/backend replaceability | RPL |
| 7 | Execution authority | EXE |
| 8 | Governance and authorization | GOV |
| 9 | Observability | OBS |
| 10 | Traceability and evidence | TRC |
| 11 | State and persistence | STA |
| 12 | Recovery | REC |
| 13 | Compatibility | CMP |
| 14 | Configuration-driven activation | CFG |
| 15 | Security | SEC |
| 16 | Reliability | REL |
| 17 | Control planes | CTL |
| 18 | Production qualification | PRD |
| 19 | Regression enforcement | REG |
| 20 | Evolvability | EVOL |
| 21 | Harness architecture | HRN |
| 22 | Documentation consistency | DOC |
| 23 | Architecture debt | DEBT |
| 24 | Freeze readiness | FRZ |
| 25 | Audit completeness | AUD |
| 26 | Tenant isolation | TEN |


#### FRZ-AUD — Audit completeness

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-AUD-01, FRZ-AUD-02, FRZ-AUD-03, FRZ-AUD-04, FRZ-AUD-05, FRZ-AUD-06, FRZ-AUD-07, FRZ-AUD-08, FRZ-AUD-09, FRZ-AUD-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | ENT-AUDIT-X |
| **Protocol sections** | §3.25, §2, §16 |
| **SYS-INV / principles** | Executable enterprise audit — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: ENT-AUDIT-X |
| **Default violation severity** | BLOCKER for continuous enforcement |
| **Regression gates (representative)** | ENT-AUDIT-X runner exit codes — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-AUD-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-BND — Hard layer boundaries

| Attribute | Value |
| --- | --- |
| **FRZ IDs (6)** | FRZ-BND-01, FRZ-BND-02, FRZ-BND-03, FRZ-BND-04, FRZ-BND-05, FRZ-BND-06 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-2*, EBH-3, EBH-4 |
| **Protocol sections** | §3.1, §5 |
| **SYS-INV / principles** | Tier boundary invariants; SYS-INV-00 — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Domain architecture pairs; runtime hub classification register |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | Layer boundary / import direction gates (EBH-2*, EBH-3, EBH-4 scoped) — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-BND-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-CFG — Configuration-driven activation

| Attribute | Value |
| --- | --- |
| **FRZ IDs (8)** | FRZ-CFG-01, FRZ-CFG-02, FRZ-CFG-03, FRZ-CFG-04, FRZ-CFG-05, FRZ-CFG-06, FRZ-CFG-07, FRZ-CFG-08 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | INT-CONFIG-REAL-X, CONFIG-X; current-HEAD: EBH-5, EBH-6, ROADMAP-REPLAY-X |
| **Protocol sections** | §3.14, §4 |
| **SYS-INV / principles** | Configured vs effective vs authorized — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: CONFIG-X / INT-CONFIG-REAL-X |
| **Default violation severity** | BLOCKER for shadow config |
| **Regression gates (representative)** | CONFIG-X historical + current-HEAD replay — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-CFG-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-CMP — Compatibility

| Attribute | Value |
| --- | --- |
| **FRZ IDs (8)** | FRZ-CMP-01, FRZ-CMP-02, FRZ-CMP-03, FRZ-CMP-04, FRZ-CMP-05, FRZ-CMP-06, FRZ-CMP-07, FRZ-CMP-08 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | COMPAT-X, INT-EXTCOMP-X |
| **Protocol sections** | §3.13, §11 |
| **SYS-INV / principles** | No second permanent mechanism — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: COMPAT-X domain |
| **Default violation severity** | ERROR |
| **Regression gates (representative)** | COMPAT-X gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-CMP-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-CTL — Control planes

| Attribute | Value |
| --- | --- |
| **FRZ IDs (12)** | FRZ-CTL-01, FRZ-CTL-02, FRZ-CTL-03, FRZ-CTL-04, FRZ-CTL-05, FRZ-CTL-06, FRZ-CTL-07, FRZ-CTL-08, FRZ-CTL-09, FRZ-CTL-10, FRZ-CTL-11, FRZ-CTL-12 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | CTRL-X |
| **Protocol sections** | §3.17, §7 |
| **SYS-INV / principles** | Control plane mutation governed — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: CTRL-X domain |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | CTRL-X qualification — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-CTL-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-CTR — Canonical contracts

| Attribute | Value |
| --- | --- |
| **FRZ IDs (6)** | FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-03, FRZ-CTR-04, FRZ-CTR-05, FRZ-CTR-06 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-2*, EBH-3, EBH-4, INT-EXTCOMP-X, INT-CONFIG-REAL-X, AW-7C |
| **Protocol sections** | §3.3, §5 |
| **SYS-INV / principles** | Contracts over implementations — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Per-responsibility domain architecture |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | Contract architecture gates; public-contract purity scans — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-CTR-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-DEBT — Architecture debt

| Attribute | Value |
| --- | --- |
| **FRZ IDs (6)** | FRZ-DEBT-01, FRZ-DEBT-02, FRZ-DEBT-03, FRZ-DEBT-04, FRZ-DEBT-05, FRZ-DEBT-06 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | ARCH-FREEZE |
| **Protocol sections** | §3.23, §13 |
| **SYS-INV / principles** | Tracked debt cannot masquerade as PASS — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: ARCH-FREEZE debt register |
| **Default violation severity** | TRACKED_DEBT until disposition |
| **Regression gates (representative)** | Roadmap replay; debt expiry gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-DEBT-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-DOC — Documentation consistency

| Attribute | Value |
| --- | --- |
| **FRZ IDs (6)** | FRZ-DOC-01, FRZ-DOC-02, FRZ-DOC-03, FRZ-DOC-04, FRZ-DOC-05, FRZ-DOC-06 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-6, ARCH-FREEZE |
| **Protocol sections** | §3.22 |
| **SYS-INV / principles** | Canon vs code alignment — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: EBH-6 doc drift checks |
| **Default violation severity** | WARNING (BLOCKER if invariant only in stale doc) |
| **Regression gates (representative)** | ENT-AUDIT-X documentation drift module — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-DOC-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-EVOL — Evolvability

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-EVOL-01, FRZ-EVOL-02, FRZ-EVOL-03, FRZ-EVOL-04, FRZ-EVOL-05, FRZ-EVOL-06, FRZ-EVOL-07, FRZ-EVOL-08, FRZ-EVOL-09, FRZ-EVOL-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EVOL-X, ENT-AUDIT-X, SCENARIO-READINESS-X |
| **Protocol sections** | §3.20, §11 |
| **SYS-INV / principles** | Change safety; extension stability — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: EVOL-X stage |
| **Default violation severity** | ERROR pre-freeze; BLOCKER at freeze |
| **Regression gates (representative)** | EVOL-X qualification — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-EVOL-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-EXE — Execution authority

| Attribute | Value |
| --- | --- |
| **FRZ IDs (7)** | FRZ-EXE-01, FRZ-EXE-02, FRZ-EXE-03, FRZ-EXE-04, FRZ-EXE-05, FRZ-EXE-06, FRZ-EXE-07 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | HARNESS-*, EBH-4, GOV-X*, AW-7C |
| **Protocol sections** | §3.7, §7 |
| **SYS-INV / principles** | Single execution authority; Nexus private impl — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Execution domain architecture |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | EE-A1, U5 zero-bypass, GOV-X2 matrix — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-EXE-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-FRZ — Freeze readiness

| Attribute | Value |
| --- | --- |
| **FRZ IDs (13)** | FRZ-FRZ-01, FRZ-FRZ-02, FRZ-FRZ-03, FRZ-FRZ-04, FRZ-FRZ-05, FRZ-FRZ-06, FRZ-FRZ-07, FRZ-FRZ-08, FRZ-FRZ-09, FRZ-FRZ-10, FRZ-FRZ-11, FRZ-FRZ-12, FRZ-FRZ-13 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | ARCH-FREEZE |
| **Protocol sections** | §3.24, §15 |
| **SYS-INV / principles** | ARCH-FREEZE gate criteria — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: ARCH-FREEZE stage |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | Full-platform FREEZE audit mode — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-FRZ-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-GOV — Governance and authorization

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-GOV-01, FRZ-GOV-02, FRZ-GOV-03, FRZ-GOV-04, FRZ-GOV-05, FRZ-GOV-06, FRZ-GOV-07, FRZ-GOV-08, FRZ-GOV-09, FRZ-GOV-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | HARNESS-*, GOV-X*, INT-CONFIG-REAL-X, AW-7C |
| **Protocol sections** | §3.8, §7 |
| **SYS-INV / principles** | Governance ≠ Execution — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Governance / control-plane domain |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | Governance boundary gates; GR-10/11/12 — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-GOV-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-HRN — Harness architecture

| Attribute | Value |
| --- | --- |
| **FRZ IDs (8)** | FRZ-HRN-01, FRZ-HRN-02, FRZ-HRN-03, FRZ-HRN-04, FRZ-HRN-05, FRZ-HRN-06, FRZ-HRN-07, FRZ-HRN-08 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | HARNESS-FINAL |
| **Protocol sections** | §3.21 |
| **SYS-INV / principles** | SYS-INV-00 Harness is the product — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: HARNESS-FINAL stage |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | HARNESS-* wave gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-HRN-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-OBS — Observability

| Attribute | Value |
| --- | --- |
| **FRZ IDs (7)** | FRZ-OBS-01, FRZ-OBS-02, FRZ-OBS-03, FRZ-OBS-04, FRZ-OBS-05, FRZ-OBS-06, FRZ-OBS-07 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | HARNESS-*, TRACE-X, CTRL-X, EBH-5, EBH-6 |
| **Protocol sections** | §3.9, §10 |
| **SYS-INV / principles** | Observability ≠ execution truth — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Observability / runtime events domain |
| **Default violation severity** | ERROR |
| **Regression gates (representative)** | HARNESS-W5, TRACE-X, CTRL-X scoped — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-OBS-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-OWN — Exactly-one ownership

| Attribute | Value |
| --- | --- |
| **FRZ IDs (5)** | FRZ-OWN-01, FRZ-OWN-02, FRZ-OWN-03, FRZ-OWN-04, FRZ-OWN-05 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-2*, EBH-3 |
| **Protocol sections** | §3.2, §5 |
| **SYS-INV / principles** | Exactly-one semantic/composition owner (cross-layer) — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: intergrax_runtime_architecture.md classification register |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | Ownership / duplicate mechanism gates per domain stage — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-OWN-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-PLG — Pluginability

| Attribute | Value |
| --- | --- |
| **FRZ IDs (8)** | FRZ-PLG-01, FRZ-PLG-02, FRZ-PLG-03, FRZ-PLG-04, FRZ-PLG-05, FRZ-PLG-06, FRZ-PLG-07, FRZ-PLG-08 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-2*, EBH-5, INT-EXTCOMP-X, INT-CONFIG-REAL-X, AW-7C |
| **Protocol sections** | §3.5, §6 |
| **SYS-INV / principles** | Plugin boundaries; no self-expanding authority — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: core/plugins + domain catalog owners |
| **Default violation severity** | BLOCKER when extensibility claimed |
| **Regression gates (representative)** | DS-PLUGIN, HARDENING-5, EP scanner gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-PLG-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-PRD — Production qualification

| Attribute | Value |
| --- | --- |
| **FRZ IDs (12)** | FRZ-PRD-01, FRZ-PRD-02, FRZ-PRD-03, FRZ-PRD-04, FRZ-PRD-05, FRZ-PRD-06, FRZ-PRD-07, FRZ-PRD-08, FRZ-PRD-09, FRZ-PRD-10, FRZ-PRD-11, FRZ-PRD-12 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | PROD-Q |
| **Protocol sections** | §3.18, §12 |
| **SYS-INV / principles** | Production-effective reachability — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: PROD-Q inventory |
| **Default violation severity** | BLOCKER for prod paths |
| **Regression gates (representative)** | PROD-Q gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-PRD-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-REC — Recovery

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-REC-01, FRZ-REC-02, FRZ-REC-03, FRZ-REC-04, FRZ-REC-05, FRZ-REC-06, FRZ-REC-07, FRZ-REC-08, FRZ-REC-09, FRZ-REC-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | STATE-X |
| **Protocol sections** | §3.12, §8 |
| **SYS-INV / principles** | Recovery plane ownership — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Reliability / recovery domain |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | NPSC-5E recovery qualification — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-REC-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-REG — Regression enforcement

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-REG-01, FRZ-REG-02, FRZ-REG-03, FRZ-REG-04, FRZ-REG-05, FRZ-REG-06, FRZ-REG-07, FRZ-REG-08, FRZ-REG-09, FRZ-REG-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | QUAL-X |
| **Protocol sections** | §3.19, §12, §16 |
| **SYS-INV / principles** | Invariants protected by gates not docs alone — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: QUAL-X |
| **Default violation severity** | ERROR |
| **Regression gates (representative)** | QUAL-X certification — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-REG-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-REL — Reliability

| Attribute | Value |
| --- | --- |
| **FRZ IDs (11)** | FRZ-REL-01, FRZ-REL-02, FRZ-REL-03, FRZ-REL-04, FRZ-REL-05, FRZ-REL-06, FRZ-REL-07, FRZ-REL-08, FRZ-REL-09, FRZ-REL-10, FRZ-REL-11 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | HARNESS-W4, CTRL-X, PROD-Q |
| **Protocol sections** | §3.16, §8 |
| **SYS-INV / principles** | REL-* reliability contracts — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Reliability domain architecture |
| **Default violation severity** | BLOCKER on invariant paths |
| **Regression gates (representative)** | EE-B1.1, HARNESS-W4 — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-REL-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-RPL — Vendor/backend replaceability

| Attribute | Value |
| --- | --- |
| **FRZ IDs (4)** | FRZ-RPL-01, FRZ-RPL-02, FRZ-RPL-03, FRZ-RPL-04 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-5, EBH-7, EBH-6, INT-CONFIG-REAL-X, AW-7C |
| **Protocol sections** | §3.6, §4 |
| **SYS-INV / principles** | Provider neutrality at contract seams — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Roadmap §3.0.8; VBN evidence spec |
| **Default violation severity** | BLOCKER for promised extensibility |
| **Regression gates (representative)** | RPL-G; EBH-5/6 qualification gates — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-RPL-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-SEC — Security

| Attribute | Value |
| --- | --- |
| **FRZ IDs (10)** | FRZ-SEC-01, FRZ-SEC-02, FRZ-SEC-03, FRZ-SEC-04, FRZ-SEC-05, FRZ-SEC-06, FRZ-SEC-07, FRZ-SEC-08, FRZ-SEC-09, FRZ-SEC-10 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | CTRL-X, PROD-Q |
| **Protocol sections** | §3.15, §9 |
| **SYS-INV / principles** | Secrets, least privilege — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: CTRL-X, PROD-Q |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | CTRL-X / security qualification — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-SEC-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-STA — State and persistence

| Attribute | Value |
| --- | --- |
| **FRZ IDs (8)** | FRZ-STA-01, FRZ-STA-02, FRZ-STA-03, FRZ-STA-04, FRZ-STA-05, FRZ-STA-06, FRZ-STA-07, FRZ-STA-08 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | STATE-X |
| **Protocol sections** | §3.11, §8 |
| **SYS-INV / principles** | Persistence contract chain — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: STATE-X / persistence domain |
| **Default violation severity** | BLOCKER |
| **Regression gates (representative)** | STATE-X, NPSC-5F persistence boundary — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-STA-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-TEN — Tenant isolation

| Attribute | Value |
| --- | --- |
| **FRZ IDs (12)** | FRZ-TEN-01, FRZ-TEN-02, FRZ-TEN-03, FRZ-TEN-04, FRZ-TEN-05, FRZ-TEN-06, FRZ-TEN-07, FRZ-TEN-08, FRZ-TEN-09, FRZ-TEN-10, FRZ-TEN-11, FRZ-TEN-12 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | TENANT-X, EBH-7, ENT-AUDIT-X, ARCH-FREEZE |
| **Protocol sections** | §3.26, §9 |
| **SYS-INV / principles** | Tenant scope fail-closed — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: TENANT-X; roadmap §2.0.1 |
| **Default violation severity** | BLOCKER (ARCH-FREEZE gate) |
| **Regression gates (representative)** | TEN-* executable modules (ENT-AUDIT-X) — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-TEN-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-TRC — Traceability and evidence

| Attribute | Value |
| --- | --- |
| **FRZ IDs (12)** | FRZ-TRC-01, FRZ-TRC-02, FRZ-TRC-03, FRZ-TRC-04, FRZ-TRC-05, FRZ-TRC-06, FRZ-TRC-07, FRZ-TRC-08, FRZ-TRC-09, FRZ-TRC-10, FRZ-TRC-11, FRZ-TRC-12 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | TRACE-X, GOV-X*, INT-EXTCOMP-X |
| **Protocol sections** | §3.10, §10, §12 |
| **SYS-INV / principles** | Causal reconstruction ownership — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: TRACE-X qualification records |
| **Default violation severity** | BLOCKER for attribution gaps |
| **Regression gates (representative)** | TRACE-X qualification suites — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-TRC-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.

#### FRZ-TYP — Strong typing

| Attribute | Value |
| --- | --- |
| **FRZ IDs (6)** | FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-04, FRZ-TYP-05, FRZ-TYP-06 |
| **Canonical criterion text** | [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) rows for each ID above |
| **Primary closing stage(s)** | EBH-2*, INT-EXTCOMP-X |
| **Protocol sections** | §3.4, §5 |
| **SYS-INV / principles** | Typed semantic boundaries — full index: [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) |
| **Semantic / composition owner** | Lookup: Domain owner + pyright scope per stage |
| **Default violation severity** | ERROR (BLOCKER at freeze) |
| **Regression gates (representative)** | AST gates, pyright scoped suites — confirm file exists at audited SHA before citing in report |

| Audit dimension | Procedure |
| --- | --- |
| **Audit questions** | For each ID: (1) What invariant does the checklist row require? (2) Which production-effective paths could violate it? (3) Is historical stage CLOSED evidence scoped or global? (4) Does current HEAD introduce new reachability? |
| **Static checks** | Import/layer scans; contract-surface grep; AST gates listed in stage qualification; VBN classes A–J where backend-related; ownership inventory diff vs baseline. |
| **Execution checks** | Run only gates/tests explicitly recorded in qualification artifacts at the audit SHA; adversarial negatives per stage record; do not invent suites. |
| **Failure / adversarial checks** | Prove fail-closed behavior where criterion requires it; attempt bypass paths (shadow registry, silent fallback, cross-tenant access). |
| **Mandatory evidence** | Exact SHA; criterion ID mapping; static artifact; executed gate output; independent auditor sign-off for PASS promotion. |
| **Evidence type** | Per ID: static + execution + independent audit per checklist PASS rule. |

**Per-ID traceability:** Each `FRZ-TYP-*` criterion MUST be verified individually. The checklist row is authoritative for acceptance wording; this protocol defines *how* to verify, not *what* PASS means.


---

## ECAP-§4 — Universal Vendor and Backend Independence

### ECAP-§4.1 Objective

The platform MUST NOT become unintentionally dependent on a specific database, storage engine, model provider, messaging system, vector store, graph store, logging system, tracing backend, or cloud provider. Vendor adapters and **explicitly ratified fixed substrates** are permitted only within authorized boundaries.

**Normative taxonomy:** Roadmap [§3.0.8](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) domains **1–31** (relational, NoSQL, KV, cache, vector, graph, object storage, event brokers, queues/scheduling, distributed coordination, transaction/idempotency stores, checkpoints/execution state, agent memory, RAG retrieval/reranking, LLM/embeddings, agent runtimes, hosting substrates, external integrations/MCP, logging, metrics, tracing, diagnostics/error reporting, alerting, audit/evidence archives, secrets, identity, configuration/profile storage, network/API, telemetry/OTLP, evaluation/analytics, plus `DISCOVERED-*`).

### ECAP-§4.2 Hardcoding detection classes (mandatory)

| Class | Focus |
| --- | --- |
| **A** | Direct vendor construction in unrelated semantic layers |
| **B** | Hardcoded backend selection |
| **C** | Implicit defaults and silent fallback |
| **D** | Shadow configuration authority |
| **E** | Shadow persistence/state |
| **F** | Vendor leakage across contracts |
| **G** | Operational coupling (health/metrics/tracing/errors) |
| **H** | Import/execution reachability |
| **I** | Configuration bypass |
| **J** | Scenario coupling |

### ECAP-§4.3 Replaceability proof classes (mandatory)

| Id | Requirement |
| --- | --- |
| **RPL-A** | Contract integrity |
| **RPL-B** | Alternate implementation via sanctioned composition |
| **RPL-C** | No core/scenario consumer changes for compatible switch |
| **RPL-D** | Configuration-driven selection |
| **RPL-E** | Behavioral guarantees preserved |
| **RPL-F** | Failure and recovery without canonical bypass |
| **RPL-G** | Regression protection |
| **RPL-H** | Extensibility classification enum |

### ECAP-§4.4 Per-dependency verification checklist (17 points)

For each **production-effective** vendor/backend dependency, auditors MUST verify:

1. Semantic owner.  
2. Canonical contract.  
3. Sanctioned selection/composition owner.  
4. Concrete adapter location.  
5. Configuration authority.  
6. Runtime activation authority.  
7. Absence of unsanctioned hardcoded construction (classes A–C).  
8. Absence of shadow configuration (class D).  
9. Absence of silent fallback (class C).  
10. Absence of vendor types in neutral contracts (class F).  
11. Alternate compatible implementation proof when extensibility is claimed (RPL-B).  
12. No core/scenario code changes during qualified substitution (RPL-C).  
13. Equivalent applicable behavioral guarantees (RPL-E).  
14. Failure and recovery behavior (RPL-F).  
15. Multi-process / distributed compatibility per claimed scope.  
16. Regression protection (RPL-G).  
17. Explicit extensibility classification (RPL-H).

### ECAP-§4.5 SQLite special rule

Every production-effective SQLite use (logging, tracing, diagnostics, execution state, checkpointing, evidence, memory, configuration, internal persistence) MUST be inventoried. `Protocol` alone does not prove replaceability. SQLite is acceptable only with correct RPL-H classification and evidence per [VBN spec §3](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md).

### ECAP-§4.6 Mandatory VBN artifacts

Auditors MUST cross-check: `VENDOR_BACKEND_DEPENDENCY_INVENTORY`, `CORE_HARDCODING_FINDINGS_REGISTER`, `CROSS_LAYER_REPLACEABILITY_MATRIX`, `ALLOWED_FIXED_SUBSTRATE_DECISIONS`, `VENDOR_SELECTION_REACHABILITY_EVIDENCE`, `ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF`, `CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT`.

---

## ECAP-§5 — Hard Boundaries, Ownership and Composition

Auditors MUST verify:

- Exactly one semantic owner per responsibility (`FRZ-OWN-*`).  
- Exactly one sanctioned composition owner (`FRZ-OWN-03`).  
- No private implementation leakage (`FRZ-BND-04`).  
- No reverse dependencies (`FRZ-BND-03`).  
- No bypass of canonical interfaces (zero-bypass gates).  
- No unauthorized domain construction (`FRZ-BND-05`).  
- No duplicate registry/resolver (`FRZ-OWN-05`, `DUP-X`).  
- No shadow dispatcher or parallel execution mechanism (`FRZ-EXE-*`, `DUP-X`).  
- No parallel configuration or persistence authority (`FRZ-CFG-*`, `FRZ-OWN-04`).  
- No reflection/dynamic probing substituting typed contracts (`FRZ-TYP-04`).  
- No unauthorized responsibility transfer between platform and applications.  
- No authority expansion via delegated work without governance proof.

**Production path investigation:** Trace from host composition → runtime admission → effect egress. Static import graphs alone are insufficient when class H reachability is disputed.

---

## ECAP-§6 — Pluginability and Modularity

Verify end-to-end extensibility for: Agents; Agent Skills; Tools; Integrations; Providers; Models; Backends; Retrieval strategies; Memory providers; Infrastructure adapters; Runtime/hosting extensions; Qualification/evaluation strategies.

| Distinction | Meaning |
| --- | --- |
| **Installed** | Package present on disk |
| **Configured** | Profile/registry references implementation |
| **Effective** | Production path materializes implementation |
| **Authorized** | Governance/admission permits use |

**Requirements:** Representative third-party proofs where extensibility is claimed (`FRZ-PLG-02`); no hidden core patches; no monkeypatch-only proofs (`FRZ-RPL-03`); no unapproved registries.

---

## ECAP-§7 — Execution, Governance and Effect Safety

Verify singular execution authority; canonical admission; fresh effect authorization; proposal ≠ permission ≠ execution; child authority subset; no shadow execution; idempotency/retry ownership; external side-effect truth; partial-finalization; unknown outcome semantics; cancellation/compensation; HITL authority; fail-closed recovery.

**Negative scenarios (sample):** export success must not mint execution; plugin transport must not approve mutations; stale authorization must not admit effects; missing tenant context must fail closed.

---

## ECAP-§8 — Distributed Systems and Resilience

Verify scope **only where the platform claims support** (document scope classification with evidence):

Concurrent races; horizontal scaling; multi-worker/process; multi-node; distributed state consistency; locking/leasing/fencing; retry/dedup; message ordering; queue boundedness; backpressure; resource exhaustion; partial failures; network partitions; durability/restart; backup/restore ownership; safe degradation; deterministic operational failure semantics.

**Linkage:** `DIST-SCALE-X`, `FRZ-REL-*`, `FRZ-STA-*`, `FRZ-REC-*`.

---

## ECAP-§9 — Security, Privacy and Multi-Tenancy

Verify tenant/principal/workspace/execution identity continuity across async and provider boundaries. Cover authorization, authentication, secrets, least privilege, isolation, sensitive data exposure, retention/deletion, provider data sharing, tenant-scoped storage/queries, cross-tenant adversarial tests, missing identity fail-closed, external infrastructure responsibilities.

**ARCH-FREEZE gate:** Applicable `FRZ-TEN-*` MUST be `PASS` or `N/A — WITH EVIDENCE` only (checklist policy).

---

## ECAP-§10 — Observability and Evidence Independence

Verify observability records truth but does not create execution truth; logs are not authoritative state; trace storage replaceability where promised; metrics exporters do not shadow execution; evidence causally bound and durable where required; export failure does not silently corrupt execution semantics; providers not hardcoded in semantic core; evidence identity and tenant isolation intact (`FRZ-OBS-*`, `FRZ-TRC-*`).

---

## ECAP-§11 — Evolvability and Scenario Independence

Verify stable public contracts; versioning/migrations; compatibility; deprecation policy; modularity; impact-aware regression selection; safe deploy/rollback; extension-point stability; scenario independence from private core; no scenario-specific platform mechanism without architecture ownership (`FRZ-EVOL-*`, `SCENARIO-READINESS-X`).

**Representative scenario proof:** Add or extend a Tier-3 application using public contracts and composition only; document that platform core semantics required no unauthorized change.

---

## ECAP-§12 — Audit Evidence and Qualification Requirements

| Level | Description |
| --- | --- |
| **L1 Static inspection** | Imports, AST, architecture gates without execution |
| **L2 Contract evidence** | Public contract files, ADRs, ownership registers |
| **L3 Production reachability** | Traced wiring from host/profile to effect |
| **L4 Executed deterministic tests** | Pytest/gate logs at SHA |
| **L5 Negative/adversarial proof** | Fail-closed and bypass attempts |
| **L6 Integration/E2E proof** | Cross-subsystem qualification |
| **L7 Restart/recovery proof** | Durability and recovery semantics |
| **L8 Alternate backend proof** | RPL-B/E/F behavioral proofs |
| **L9 Independent exact-SHA audit** | Non-implementer attestation |
| **L10 Continuous regression enforcement** | QUAL-X / ENT-AUDIT-X / RPL-G |

**Cursor-provided evidence** is at best L1–L4 candidate material until L9 is satisfied.

---

## ECAP-§13 — Findings Classification and Escalation

| Class | Meaning |
| --- | --- |
| **PROVEN_ARCHITECTURAL_DEFECT** | Invariant violated on production-effective path |
| **IN_SCOPE_BLOCKER** | Prevents parent/ARCH-FREEZE closure |
| **MISSING_EXECUTION_EVIDENCE** | Criterion not provable without running gates |
| **UNQUALIFIED_REPLACEABILITY** | Extensibility claimed without RPL proof |
| **TRACKED_FREEZE_DEBT** | Documented debt with disposition |
| **NONBLOCKING_FUTURE_IMPROVEMENT** | Not tied to current criterion |
| **ENVIRONMENT_TEST_ISSUE** | Qualification infra not representative |
| **ARCHITECTURE_DECISION_REQUIRED** | Conflict with canon — stop implementation |
| **N/A_WITH_EVIDENCE** | Criterion not applicable with recorded rationale |

**Mandatory finding fields:** unique ID; exact SHA; file/symbol; applicable invariant/`FRZ-*`; production path; impact; severity; parent relevance; minimal corrective action; closure criterion; independent verification requirement.

Do **not** open ADRs when local correction suffices. Do **not** open implementation tasks for unsupported theoretical risks.

---

## ECAP-§14 — Mandatory Audit Report Template

```markdown
# Enterprise Architecture Audit Report

## A. HEAD and baseline
- Audit SHA: 
- Baseline SHA: 
- Audit mode: DELTA | PARENT | FULL-HEAD | POST-FREEZE
- Scope statement: 

## B. Roadmap progress
- Current parent stage: 
- Applicable stages touched: 

## C. Enterprise compliance coverage
- FRZ families in scope: 
- Evidence labels used (report-level): VERIFIED AT CURRENT HEAD | STATIC ONLY | EXECUTION PROVEN | NOT PROVEN | NOT ASSESSED

## D. Architecture findings
| Finding ID | Class | FRZ/SYS-INV | Severity | Label |

## E. Vendor/backend dependency changes
- Inventory delta: 
- New/changed concern_ids: 

## F. Cross-layer ownership and authority
- Duplicates/shadows: 

## G. Runtime reliability and failure semantics
- Scope claims vs evidence: 

## H. Impact on historical certification
- Scoped CLOSED stages affected: 

## I. Scenario readiness / architecture-change risk
- SCENARIO-READINESS-X relevance: 

## J. Blockers vs missing evidence
- IN-SCOPE BLOCKER count: 
- MISSING_EXECUTION_EVIDENCE count: 

## K. Closure verdict and next action
- Verdict: READY FOR INDEPENDENT AUDIT | NOT READY | STOP — ADR
- One next best action: 
```

**Label rule:** Report labels MUST NOT replace checklist permitted statuses.

---

## ECAP-§15 — Architecture Freeze Admission

Before `ARCH-FREEZE`, auditors MUST confirm:

- All mandatory stages independently closed per roadmap.  
- All applicable `FRZ-*` are `PASS` or `N/A — WITH EVIDENCE`.  
- No unresolved prohibited core/vendor hardcoding in frozen scope.  
- No unresolved architecture debt inside frozen scope without disposition.  
- All claimed extensibility qualified (RPL-H).  
- No unapproved shadow authorities.  
- Successful current-HEAD full platform audit (§2.C).  
- Successful roadmap replay (`ROADMAP-REPLAY-X`).  
- Evidence continuity from historical PASS records.  
- Qualified evolution controls (`EVOL-X`, ECAP §2.D).  
- Production and distributed semantics qualified per claimed scope.  
- Scenario readiness remains post-freeze gate (`SCENARIO-READINESS-X`).

---

## ECAP-§16 — Maintenance and Continuous Enforcement

Update this protocol when:

- New platform domain or responsibility;  
- New public contract or extension mechanism;  
- New infrastructure/provider class;  
- New enterprise invariant or `FRZ-*` criterion;  
- New architecture-reopen trigger;  
- New regression class;  
- Canonical evidence requirement changes.

**Compatibility check:** On each ECAP revision, re-verify companion documents at current HEAD and re-run FRZ inventory count.

**Continuous enforcement:** After `ENT-AUDIT-X` implementation, `FULL`, `FREEZE`, and `CHANGED-SINCE` modes (roadmap §3.1) MUST be used per scenario-wave policy.

**Forbidden:** Second status ledger; independent roadmap; autonomous FRZ PASS promotion.

---

## ECAP-§17 — FRZ Family Traceability Matrix

| FRZ family | Canonical source | Protocol section | Audit procedure | Evidence type | Closing stage |
| --- | --- | --- | --- | --- | --- |
| FRZ-AUD | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.25, §2, §16 | §3 family procedure `FRZ-AUD`; mode-specific §2 | Static + execution + independent | ENT-AUDIT-X |
| FRZ-BND | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.1, §5 | §3 family procedure `FRZ-BND`; mode-specific §2 | Static + execution + independent | EBH-2*, EBH-3, EBH-4 |
| FRZ-CFG | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.14, §4 | §3 family procedure `FRZ-CFG`; mode-specific §2 | Static + execution + independent | INT-CONFIG-REAL-X, CONFIG-X; current-HEAD: EBH-5, EBH-6, ROADMAP-REPLAY-X |
| FRZ-CMP | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.13, §11 | §3 family procedure `FRZ-CMP`; mode-specific §2 | Static + execution + independent | COMPAT-X, INT-EXTCOMP-X |
| FRZ-CTL | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.17, §7 | §3 family procedure `FRZ-CTL`; mode-specific §2 | Static + execution + independent | CTRL-X |
| FRZ-CTR | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.3, §5 | §3 family procedure `FRZ-CTR`; mode-specific §2 | Static + execution + independent | EBH-2*, EBH-3, EBH-4, INT-EXTCOMP-X, INT-CONFIG-REAL-X, AW-7C |
| FRZ-DEBT | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.23, §13 | §3 family procedure `FRZ-DEBT`; mode-specific §2 | Static + execution + independent | ARCH-FREEZE |
| FRZ-DOC | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.22 | §3 family procedure `FRZ-DOC`; mode-specific §2 | Static + execution + independent | EBH-6, ARCH-FREEZE |
| FRZ-EVOL | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.20, §11 | §3 family procedure `FRZ-EVOL`; mode-specific §2 | Static + execution + independent | EVOL-X, ENT-AUDIT-X, SCENARIO-READINESS-X |
| FRZ-EXE | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.7, §7 | §3 family procedure `FRZ-EXE`; mode-specific §2 | Static + execution + independent | HARNESS-*, EBH-4, GOV-X*, AW-7C |
| FRZ-FRZ | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.24, §15 | §3 family procedure `FRZ-FRZ`; mode-specific §2 | Static + execution + independent | ARCH-FREEZE |
| FRZ-GOV | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.8, §7 | §3 family procedure `FRZ-GOV`; mode-specific §2 | Static + execution + independent | HARNESS-*, GOV-X*, INT-CONFIG-REAL-X, AW-7C |
| FRZ-HRN | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.21 | §3 family procedure `FRZ-HRN`; mode-specific §2 | Static + execution + independent | HARNESS-FINAL |
| FRZ-OBS | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.9, §10 | §3 family procedure `FRZ-OBS`; mode-specific §2 | Static + execution + independent | HARNESS-*, TRACE-X, CTRL-X, EBH-5, EBH-6 |
| FRZ-OWN | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.2, §5 | §3 family procedure `FRZ-OWN`; mode-specific §2 | Static + execution + independent | EBH-2*, EBH-3 |
| FRZ-PLG | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.5, §6 | §3 family procedure `FRZ-PLG`; mode-specific §2 | Static + execution + independent | EBH-2*, EBH-5, INT-EXTCOMP-X, INT-CONFIG-REAL-X, AW-7C |
| FRZ-PRD | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.18, §12 | §3 family procedure `FRZ-PRD`; mode-specific §2 | Static + execution + independent | PROD-Q |
| FRZ-REC | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.12, §8 | §3 family procedure `FRZ-REC`; mode-specific §2 | Static + execution + independent | STATE-X |
| FRZ-REG | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.19, §12, §16 | §3 family procedure `FRZ-REG`; mode-specific §2 | Static + execution + independent | QUAL-X |
| FRZ-REL | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.16, §8 | §3 family procedure `FRZ-REL`; mode-specific §2 | Static + execution + independent | HARNESS-W4, CTRL-X, PROD-Q |
| FRZ-RPL | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.6, §4 | §3 family procedure `FRZ-RPL`; mode-specific §2 | Static + execution + independent | EBH-5, EBH-7, EBH-6, INT-CONFIG-REAL-X, AW-7C |
| FRZ-SEC | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.15, §9 | §3 family procedure `FRZ-SEC`; mode-specific §2 | Static + execution + independent | CTRL-X, PROD-Q |
| FRZ-STA | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.11, §8 | §3 family procedure `FRZ-STA`; mode-specific §2 | Static + execution + independent | STATE-X |
| FRZ-TEN | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.26, §9 | §3 family procedure `FRZ-TEN`; mode-specific §2 | Static + execution + independent | TENANT-X, EBH-7, ENT-AUDIT-X, ARCH-FREEZE |
| FRZ-TRC | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.10, §10, §12 | §3 family procedure `FRZ-TRC`; mode-specific §2 | Static + execution + independent | TRACE-X, GOV-X*, INT-EXTCOMP-X |
| FRZ-TYP | PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md | §3.4, §5 | §3 family procedure `FRZ-TYP`; mode-specific §2 | Static + execution + independent | EBH-2*, INT-EXTCOMP-X |

### ECAP-§17.1 Mechanical FRZ ID coverage

| Metric | Value |
| --- | --- |
| Total `FRZ-*` IDs at generation | 225 |
| Total families | 26 |
| Coverage rule | Every ID listed in §3 family subsections |
| Gap condition | Any checklist ID missing from §3 → **COVERAGE GAP — DOCUMENT NOT READY FOR AUDIT** |

---

## ECAP-§18 — Validation record (documentation)

Auditors maintaining ECAP SHOULD run at SHA:

```powershell
# FRZ inventory count
uv run python -c "import re, pathlib; t=pathlib.Path('docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md').read_text(); print(len(re.findall(r'\| (FRZ-[A-Z]+-\d+) \|', t)))"

# Confirm companion paths exist (8 mandatory sources)
@(
  'docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md',
  'docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md',
  'docs/project/maintainers/qualification/VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md',
  'docs/project/technical/guides/SYSTEM_INVARIANTS.md',
  'docs/project/maintainers/qualification/INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md',
  'docs/project/maintainers/qualification/INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md',
  'docs/project/maintainers/audits/PLATFORM_WIDE_ENTERPRISE_AUDIT.md',
  'docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md'
) | ForEach-Object { if (-not (Test-Path $_)) { Write-Error "MISSING $_" } }
```

---

## Document history

| Version | Date | Notes |
| --- | --- | --- |
| 1.0 | 2026-10-18 | Initial ECAP — 225 FRZ criteria, 26 families, VBN taxonomy §4 aligned to roadmap §3.0.8 |
