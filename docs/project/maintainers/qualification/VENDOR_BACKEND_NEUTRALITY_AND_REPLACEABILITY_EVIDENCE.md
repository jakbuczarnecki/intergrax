# Vendor / Backend Neutrality & Cross-Layer Replaceability — Mandatory Evidence Specification

> **Role:** qualification planning and evidence-schema authority for universal vendor/backend hardcoding detection and replaceability certification. **Not** a PASS artifact. **Not** a second roadmap or freeze system.

**Companion (program SSOT):**

- [Platform Enterprise Completion Roadmap](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — §3.0.8
- [Platform Enterprise Freeze Acceptance Checklist](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — vendor/backend exit obligations section

**CONFIG-X historical evidence (preserved, not revoked):** **CONFIG-X** = **CLOSED / independently accepted** @ `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` — **FRZ-CFG-01..08** = **PASS** on that SHA and closed-world scope (`CONFIG_X_CONCERN_INVENTORY` = 54). Historical PASS does **not** automatically prove current-HEAD absence of structural vendor hardcoding, shadow selection, or unqualified replaceability. Later stages (**EBH-5**, **EBH-6**, **DUP-X**, **ROADMAP-REPLAY-X**, **EBH-7**, **ENT-AUDIT-X**) must revalidate on current HEAD without silently rewriting CONFIG-X status.

**Implementation boundary:** this document defines **future** mandatory artifacts and acceptance mechanics only. No placeholder PASS records. No production code changes by docs alone.

---

## 1. Program objective

Every infrastructure/vendor/backend dependency on the platform must be **classified**, **audited**, and **traceable** to:

- a semantic owner;
- a canonical contract or port (where applicable);
- a sanctioned composition/selection owner;
- explicit extensibility classification (`REPLACEABLE — QUALIFIED` | `REPLACEABLE — NOT YET QUALIFIED` | `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` | `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` | `ARCHITECTURE BLOCKER`).

Where extensibility is claimed, concrete implementations must be selected only through the **existing** authorized configuration/profile/composition boundary — not through hidden core construction, shadow registries, implicit defaults, or silent fallbacks.

---

## 2. Mandatory evidence artifacts (no PASS until independently accepted)

Each artifact is produced at the stated **producing stage**, owned by the **responsible owner**, and must record **evidence SHA** at acceptance. Independent exact-SHA audit is mandatory for stage closure that depends on the artifact.

### 2.1 `VENDOR_BACKEND_DEPENDENCY_INVENTORY`

| Field | Requirement |
| --- | --- |
| **Purpose** | Closed-world inventory of every production-effective vendor/backend dependency relative to current HEAD platform surfaces. |
| **Producing stage** | **EBH-5** (authoritative baseline); reconciled **EBH-6**; drift-checked **ROADMAP-REPLAY-X**; continuously executable via **ENT-AUDIT-X**. |
| **Responsible owner** | Enterprise program / qualification lead (execution: EBH-5 wave owners per domain). |
| **Evidence SHA** | Git commit containing inventory SSOT + qualification runner output. |
| **Acceptance criteria** | `concern_records` = closed-world count; `unclassified` = 0; every record has all required fields below; domains 1–31 (roadmap §3.0.8) covered plus any additional mechanism discovered; inventory reconciles with CONFIG-X historical 54 concerns without assuming equivalence. |
| **FRZ linkage** | **FRZ-PLG-03..05**, **FRZ-RPL-01..04**, **FRZ-CTR-02**, **FRZ-CTR-06**, **FRZ-CFG-04** (current-HEAD requalification only — does not alter CONFIG-X PASS @ `fac9a43f…`). |
| **Independent audit** | Mandatory for **EBH-5** closure. |

**Record schema (`VendorBackendConcernRecord`):**

| Field | Type / notes |
| --- | --- |
| `concern_id` | Stable string; unique in inventory. |
| `domain` | One of roadmap §3.0.8 domain taxonomy (+ `DISCOVERED-<slug>` for closed-world additions). |
| `semantic_owner` | Exactly-one owner id / documented authority. |
| `canonical_contract_or_port` | Contract path or `N/A — WITH EVIDENCE` with reason. |
| `composition_selection_owner` | Sanctioned resolver/composition root id. |
| `concrete_implementation` | Effective implementation class/module or ratified fixed substrate id. |
| `extensibility_promised` | `yes` \| `no` \| `partial — WITH EVIDENCE`. |
| `configuration_profile_source` | Typed config/profile field(s) or `FIXED` / `INTERNAL`. |
| `lifecycle_activation_authority` | Who may activate; fail-closed behavior summary. |
| `vendor_code_location` | Adapter/boundary paths (not an automatic violation list). |
| `core_consumers` | Bounded list of semantic consumers (contracts only where claimed). |
| `core_change_required_to_switch` | `no` \| `yes — BLOCKER` \| `adapter-only` \| `config-only`. |
| `fallback_behavior` | Explicit description; `silent` forbidden unless ratified. |
| `structural_replaceability_evidence_id` | Pointer to matrix row / proof id or `PENDING`. |
| `behavioral_qualification_evidence_id` | Pointer to behavioral proof or `PENDING` / `N/A`. |
| `frz_criteria` | List of applicable `FRZ-*` ids. |
| `extensibility_classification` | RPL-H enum (§4). |
| `conclusion` | `ACCEPTED` \| `DEBT` \| `BLOCKER` \| `REQUIRES ADR`. |
| `outstanding_debt` | Free text or empty. |

### 2.2 `CORE_HARDCODING_FINDINGS_REGISTER`

| Field | Requirement |
| --- | --- |
| **Purpose** | Register of every current-HEAD hardcoding / shadow-selection / leakage finding from universal audit (static + behavioral reachability). |
| **Producing stage** | **EBH-6** (primary audit); remediated before **EBH-7**; regression-protected **QUAL-X** / **EBH-5** gates. |
| **Responsible owner** | EBH-6 certification wave. |
| **Evidence SHA** | Commit + qualification report SHA. |
| **Acceptance criteria** | `unclassified_findings` = 0; every `PROHIBITED-*` in frozen scope = `RESOLVED` or accepted `ALLOWED-*` / `REQUIRES ARCHITECTURE DECISION` with ADR; `EVIDENCE REQUIRED` cannot remain at **ARCH-FREEZE**. |
| **FRZ linkage** | **FRZ-CTR-02**, **FRZ-CTR-06**, **FRZ-CFG-05..06** (current-HEAD), **FRZ-BND-05**, **FRZ-OWN-04**, **FRZ-PLG-04**, **FRZ-REL-07**, **FRZ-OBS-*** (operational coupling rows). |
| **Independent audit** | Mandatory for **EBH-6** and replay validation. |

**Finding schema (`HardcodingFindingRecord`):**

| Field | Type / notes |
| --- | --- |
| `finding_id` | Stable id. |
| `layer` | Roadmap §3.0.8 audit layer. |
| `detection_class` | `A`..`J` (roadmap §3.0.8). |
| `classification` | Allowed/prohibited taxonomy (roadmap §3.0.8). |
| `location` | Path / symbol / production-path trace id. |
| `production_effective` | `yes` \| `no` \| `unknown — FAIL` (unknown forbidden at closure). |
| `concern_ids` | Link to inventory record(s). |
| `violated_invariant` | Short invariant reference. |
| `remediation_class` | `merge` \| `move-to-adapter` \| `fail-closed` \| `ratify-fixed` \| `ADR` |
| `status` | `OPEN` \| `RESOLVED` \| `RATIFIED` \| `REPLAY-REOPENED` |
| `evidence_sha` | Resolution proof SHA. |

### 2.3 `CROSS_LAYER_REPLACEABILITY_MATRIX`

| Field | Requirement |
| --- | --- |
| **Purpose** | Per-concern replaceability certification status and proof pointers (RPL-A..H). |
| **Producing stage** | **EBH-5** (primary); **EBH-7** (whole-platform convergence). |
| **Responsible owner** | EBH-5 wave. |
| **Evidence SHA** | Qualification closure SHA. |
| **Acceptance criteria** | Every `extensibility_promised=yes` concern has RPL-A..G satisfied or explicit `ARCHITECTURE BLOCKER`; no false blanket `REPLACEABLE — QUALIFIED`. |
| **FRZ linkage** | **FRZ-RPL-01..04**, **FRZ-PLG-01..02**, **FRZ-REG-02**, **FRZ-REG-09**. |
| **Independent audit** | Mandatory for **EBH-5** closure. |

**Matrix row schema:** `concern_id`, `rpl_a`..`rpl_h` (`PASS` \| `FAIL` \| `N/A — WITH EVIDENCE`), `alternate_implementation_ref`, `behavioral_proof_ref`, `regression_gate_ids[]`.

### 2.4 `ALLOWED_FIXED_SUBSTRATE_DECISIONS`

| Field | Requirement |
| --- | --- |
| **Purpose** | Explicit ratification of non-replaceable or fixed infrastructure substrates (including intentional SQLite or in-process stores). |
| **Producing stage** | **EBH-5** / **EBH-6**; reviewed **EBH-7**; drift-checked **ENT-AUDIT-X**. |
| **Responsible owner** | Architecture / domain semantic authority per decision. |
| **Evidence SHA** | ADR or qualification record SHA. |
| **Acceptance criteria** | Each fixed substrate has scope boundary, semantic guarantees, multi-instance/DIST-SCALE implications, and expiration/review policy where applicable. |
| **FRZ linkage** | **FRZ-CTR-06**, **FRZ-RPL-01..04**, **FRZ-STA-***, **FRZ-PRD-03**. |
| **Independent audit** | Mandatory when classification = `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE`. |

**Decision schema:** `substrate_id`, `technology`, `scope_paths[]`, `contract_if_any`, `ratification_authority`, `adr_ref`, `multi_instance_constraints`, `review_date`.

### 2.5 `VENDOR_SELECTION_REACHABILITY_EVIDENCE`

| Field | Requirement |
| --- | --- |
| **Purpose** | Prove effective vendor/backend activation paths are reachable only through sanctioned configuration/composition (including negative reachability). |
| **Producing stage** | **EBH-5**; re-audited **EBH-6**; replayed **ROADMAP-REPLAY-X**. |
| **Responsible owner** | EBH-5 / EBH-6. |
| **Evidence SHA** | Qualification SHA. |
| **Acceptance criteria** | Configuration bypass classes **I** (import/install), default construction, and shadow env lookups proven non-effective on production paths or classified and ratified. |
| **FRZ linkage** | **FRZ-CFG-06**, **FRZ-CFG-08**, **FRZ-PLG-05**, **FRZ-PRD-08**. |
| **Independent audit** | Mandatory. |

### 2.6 `ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF`

| Field | Requirement |
| --- | --- |
| **Purpose** | Representative alternate implementation exercises applicable semantic guarantees (RPL-E, RPL-F). |
| **Producing stage** | **EBH-5** (per qualified concern); spot-checked **EBH-7**. |
| **Responsible owner** | Domain qualification owner per concern. |
| **Evidence SHA** | Test/qualification SHA. |
| **Acceptance criteria** | Not monkeypatch-only; alternate injected through sanctioned composition; failures exercised without bypassing governance/execution/reliability contracts. |
| **FRZ linkage** | **FRZ-RPL-02..04**, **FRZ-REL-03..07**, **FRZ-OBS-04..07**, **FRZ-TEN-*** (tenant scope where applicable). |
| **Independent audit** | Mandatory for **EBH-5** closure sample + full matrix at **EBH-7**. |

### 2.7 `CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT`

| Field | Requirement |
| --- | --- |
| **Purpose** | Single current-HEAD rollup: inventory + findings + matrix + SQLite annex + replay delta vs CONFIG-X baseline. |
| **Producing stage** | **EBH-6** (rollup); **ROADMAP-REPLAY-X** (delta); **ENT-AUDIT-X** (executable module). |
| **Responsible owner** | EBH-6 / ENT-AUDIT-X. |
| **Evidence SHA** | Report SHA. |
| **Acceptance criteria** | `prohibited_unresolved` = 0 in frozen scope; `promised_unqualified` = 0; CONFIG-X historical PASS preserved with explicit `baseline_sha` vs `current_sha` comparison. |
| **FRZ linkage** | Cross-cutting; **ARCH-FREEZE** entry gate. |
| **Independent audit** | Mandatory before **ARCH-FREEZE**. |

---

## 3. SQLite mandatory annex (subset of inventory + findings)

For every production-effective SQLite touchpoint, inventory/findings must answer:

1. Intentional provider adapter?
2. Canonical persistence contract exists?
3. Alternate compliant backend via sanctioned composition without core/scenario change?
4. Core or business application change required to switch?
5. Transaction/isolation/idempotency/recovery guarantees expressed without SQLite-only semantics?
6. Configuration-driven?
7. Explicitly ratified and documented if fixed?
8. Valid under **DIST-SCALE-X** multi-process/multi-worker claims?

**Working SQLite alone is not replaceability evidence.**

---

## 4. RPL-A..H proof classes (EBH-5)

| Id | Requirement |
| --- | --- |
| **RPL-A** | Contract integrity — core consumers depend on canonical contracts, not concrete vendor types. |
| **RPL-B** | Alternate implementation — representative contract-conforming implementation via sanctioned composition (not trivial mock/monkeypatch as sole proof). |
| **RPL-C** | No core changes — compatible provider switch does not require platform core or business-scenario code changes (adapters/config/deployment may change). |
| **RPL-D** | Configuration-driven selection — authorized composition owner; no shadow registry or implicit default. |
| **RPL-E** | Behavioral contract — applicable guarantees preserved (transactions, tenancy, retries, durability, ordering, observability, etc.). |
| **RPL-F** | Failure and recovery — representative provider failures without bypassing canonical execution/reliability/governance. |
| **RPL-G** | Regression protection — mandatory tests/gates preventing reintroduction of prohibited coupling. |
| **RPL-H** | Extensibility classification — per concern: `REPLACEABLE — QUALIFIED` \| `REPLACEABLE — NOT YET QUALIFIED` \| `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` \| `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` \| `ARCHITECTURE BLOCKER`. |

---

## 5. Hardcoding finding classification (normative)

| Classification | Meaning |
| --- | --- |
| `ALLOWED — PROVIDER ADAPTER BOUNDARY` | Concrete vendor code only inside sanctioned adapter/composition leaf. |
| `ALLOWED — EXPLICITLY RATIFIED FIXED SUBSTRATE` | Documented in `ALLOWED_FIXED_SUBSTRATE_DECISIONS`. |
| `ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM` | Extensibility not claimed; no consumer contract violation. |
| `PROHIBITED — CORE VENDOR HARDCODING` | Direct construction/selection in unrelated semantic layer. |
| `PROHIBITED — SHADOW CONFIGURATION/SELECTION` | Bypass of canonical typed configuration/profile resolution. |
| `PROHIBITED — SILENT FALLBACK` | Missing/invalid config selects hidden default backend. |
| `PROHIBITED — VENDOR-SPECIFIC CONTRACT LEAKAGE` | Neutral contract exposes vendor SDK/types/settings. |
| `REQUIRES ARCHITECTURE DECISION` | Conflict with ratified architecture — **STOP**. |
| `EVIDENCE REQUIRED` | Cannot close stage until resolved or reclassified. |

Vendor import alone is not a violation (**H — import reachability**). Production-effective unsanctioned behavior is.

---

## 6. Stage responsibility split (non-duplicated)

| Stage | Vendor/backend responsibility |
| --- | --- |
| **EBH-5** | Primary structural + behavioral replaceability certification; produce inventory, matrix, reachability, behavioral proofs, RPL-G gates. |
| **EBH-6** | Current-HEAD universal hardcoding re-audit all layers; reconcile CONFIG-X evidence; produce/update findings register and neutrality rollup. |
| **DUP-X** | Shadow provider registries, duplicate DB selection, parallel exporters, alternate execution stores. |
| **ROADMAP-REPLAY-X** | Revalidate vendor-neutrality evidence after later changes; CONFIG-X PASS ≠ automatic current-HEAD PASS. |
| **EBH-7** | Zero unresolved prohibited hardcoding; all promised extensibility qualified or blocked with explicit finding. |
| **ENT-AUDIT-X** | Executable continuous checks for hardcoding, shadow selection, unauthorized fallback regressions. |
| **ARCH-FREEZE** | Block if prohibited hardcoding, unqualified extensibility, or mandatory artifact missing/stale. |
| **SCENARIO-READINESS-X** | Reference scenario uses contracts only; representative alternate infra config; no scenario-local vendor selection. |

---

## 7. Detection classes A..J (reference)

Aligned with roadmap §3.0.8: **A** direct vendor construction; **B** hardcoded backend selection; **C** implicit defaults; **D** shadow configuration authority; **E** shadow persistence/state; **F** vendor leakage across contracts; **G** operational coupling; **H** import/execution reachability; **I** configuration bypass; **J** scenario coupling.

---

**Disposition:** specification only — **READY FOR AUDIT** when implemented in qualification code on a future SHA; this docs commit does not claim artifact PASS.
