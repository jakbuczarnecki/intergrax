# E2E Scenario Framework Audit

**Status:** Audit (analysis only)  
**Audience:** Architects preparing ERL enterprise qualification scenarios  
**Scope:** Existing Platform Proof **SCENARIO** framework under `platform_proofs/scenarios/`  
**Branch audited:** `development` (filesystem state at audit time)  
**Out of scope:** Runtime redesign, new scenarios, code changes, ERL implementation

**Normative references (not duplicated here):**

- [`platform_proofs/PLATFORM_PROOF_PROTOCOL.md`](../../platform_proofs/PLATFORM_PROOF_PROTOCOL.md)
- [`platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md`](../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md)
- [`platform_proofs/README.md`](../../platform_proofs/README.md)

---

## 1. Executive Summary

The Intergrax **E2E scenario framework** is the **Platform Proof Library SCENARIO class**: problem-first, production-capable application packages under `platform_proofs/scenarios/<scenario_slug>/`, plus shared execution infrastructure in `scripts/proof/`. Scenarios are **executable falsification attempts** against bounded real-world claims—not product demos, not unit tests.

Construction is **two-stage and command-gated**:

1. **Design stage** — `scripts/proof/create_scenario_proof.py` emits only `README.md` + `SCENARIO_SPEC.md` (YAML lifecycle frontmatter in the spec).
2. **Implementation stage** — after human **Scenario Quality Gate** and implementation-preparation gates, `scripts/proof/init_scenario_implementation.py` emits the platform-native skeleton (`application/`, `proof/`, `fixtures/`, `proof.json`, `run_proof.py`, `.env.example`).

Execution uses the shared **scenario runtime baseline** (`intergrax.applications._shared.scenario_runtime_baseline.execute_scenario_task`, composed via `build_scenario_lab_runtime` in generated `application/runtime_composition.py`). Proof entrypoints delegate to `run_proof.py`; suite runs use `scripts/proof/run-intergrax-proof-suite.py` with descriptor discovery from package `proof.json` (`intergrax.platform_proof_descriptor.v3`).

**Related but separate:** `testing_support/decision_e2e/` hosts **behavioral / model-matrix qualification** (e.g. DS-E2E-15J tracks, local AI incident qualification). That layer consumes scenario application code paths but is **not** the authoring scaffold for new Platform Proof scenarios.

**ERL-QUAL-004** (“ERL Payment Uncertainty Recovery Scenario”) does **not** appear in the repository under that identifier. ERL Phase 6 describes a **maintainer qualification showcase** in [`docs/project/maintainers/plans/ENTERPRISE_RELIABILITY_LAYER_IMPLEMENTATION_PLAN.md`](../maintainers/plans/ENTERPRISE_RELIABILITY_LAYER_IMPLEMENTATION_PLAN.md); the closest existing **design-only** payment-adjacent slug is `payment_exception_recovery` (template scaffold, `lifecycle: DESIGN`).

---

## 2. Existing Scenario Inventory

All scenario packages live under `platform_proofs/scenarios/`. Existence is **filesystem-based**; there is no central scenario registry file in `platform_proofs/`.

| `scenario_slug` | Lifecycle (from `SCENARIO_SPEC.md` frontmatter) | `proof.json` / `run_proof.py` | Primary purpose (from package title / README) |
| --- | --- | --- | --- |
| `ai_incident_investigation` | `EXECUTABLE` | Yes | Operational incident investigation; RESOLVED / UNRESOLVED paths; correlation trap |
| `indirect_prompt_injection` | `EXECUTABLE` | Yes | Hostile retrieved data vs governed tool write prevention |
| `verified_product_identification` | `IMPLEMENTATION_INITIALIZED` | Yes | Catalog-scale verified product identity (large dataset / retrieval stack) |
| `agentic_cost_runaway` | `DESIGN` | No | Design package only |
| `autonomous_production_remediation` | `DESIGN` | No | Design package only |
| `governed_data_erasure` | `DESIGN` | No | Design package only |
| `multi_agent_cascade_containment` | `DESIGN` | No | Design package only |
| `payment_exception_recovery` | `DESIGN` | No | Autonomous payment exception resolution under partial failure (template) |
| `persistent_memory_poisoning` | `DESIGN` | No | Design package only |
| `privileged_infrastructure_change` | `DESIGN` | No | Design package only |
| `software_supply_chain_compromise` | `DESIGN` | No | Design package only |
| `strategic_decision_council` | `DESIGN` | No | Design package only |
| `vendor_payment_fraud` | `DESIGN` | No | Vendor payment fraud investigation with governed authorization (template) |

**Maturity tiers observed:**

- **Reference executable:** `ai_incident_investigation` — full `application/` + `proof/` + fixtures, dual-variant evaluator, HTML reports under `output/`, `public_evidence_eligible: false` in `proof.json`.
- **Executable application, proof evolution in progress:** `indirect_prompt_investigation` — same skeleton pattern; README/spec state business path executable, canonical real-model proof verification pending.
- **Initialized, extended domain layout:** `verified_product_identification` — skeleton plus large scenario-specific subtrees (`dataset/`, `storage_bootstrap/`, `qualification/`, `scripts/`, etc.); not yet treated as publicly accepted proof.

**Public catalog (in development only):** [`docs/project/proofs/PROOF_LIBRARY.md`](../proofs/PROOF_LIBRARY.md) lists three scenarios (AI incident, indirect prompt injection, VPI). Status text states **no accepted Scenario Proofs published yet**.

---

## 3. Scenario Structure Standard

### 3.1 Design stage (required for any new scenario)

Created **only** by `create_scenario_proof.py`:

```text
platform_proofs/scenarios/<scenario_slug>/
├── README.md              # public gateway (~3–5 min read)
└── SCENARIO_SPEC.md       # deep contract + YAML frontmatter
```

Optional after Scenario Quality Gate:

```text
└── assets/                # scenario-owned visuals (not auto-generated in scaffold)
```

**Forbidden at design stage** (enforced in `scripts/proof/create_scenario_proof.py` via `DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES`): `proof.json`, `run_proof.py`, `application/`, `proof/`, `fixtures/`, `.env.example`, root-level evaluator/evidence modules, and other implementation artifacts.

**Slug rules:** lowercase `[a-z][a-z0-9_]*`, no path separators; must resolve under `platform_proofs/scenarios/`.

### 3.2 `SCENARIO_SPEC.md` contract

**YAML frontmatter** (machine-readable lifecycle — canonical source per `scripts/proof/scenario_lifecycle.py`):

| Field | Role |
| --- | --- |
| `scenario_slug` | Package directory name |
| `lifecycle` | `DESIGN` → `ACCEPTED_FOR_IMPLEMENTATION` → `IMPLEMENTATION_INITIALIZED` → `EXECUTABLE` → `VERIFIED` |
| `implementation_status` | `NOT_INITIALIZED` / `INITIALIZED` |
| `intergrax_fit` | `NOT_COMPLETED` / `COMPLETED` |
| `gap_decision` | `NOT_COMPLETED` / `RESOLVED` |
| `observability_contract` | `NOT_COMPLETED` / `COMPLETED` |
| `application_vs_proof_ownership` | `NOT_COMPLETED` / `COMPLETED` |

**Body sections (A–E)** — scaffolded by `create_scenario_proof.py`, filled during design:

| Section | Content |
| --- | --- |
| **A. SCENARIO** | Problem, stakes, WOW, Skeptic Challenge, adversarial conditions, Application Survival / Observability tests, observability contract, Scenario Quality Gate record |
| **B. SOLUTION** | APPLICATION vs PROOF table, claim, PASS/FAIL, limitations, excluded claims |
| **C. INTERGRAX FIT** | Mechanism mapping (completed in implementation preparation) |
| **D. GAP DECISION** | Platform gap pause/resume discipline |
| **E. PROOF BUILD** | Cases, artifacts, falsification plan |

### 3.3 `README.md` contract

Required sections are encoded in `DESIGN_STAGE_README_REQUIRED_SECTIONS` in `create_scenario_proof.py`, including: **Abstract**, **At a glance**, **Visual proof story**, problem/risk/trap/adversarial summaries, claim and PASS/FAIL summary, outcomes, **Latest verified run** (placeholder until execution), **Run / report / evidence / source**, **Limitations**, **Go deeper** → `SCENARIO_SPEC.md`.

Human-readable status must align with frontmatter; use precise wording (`DESIGN / NOT YET ACCEPTED`, `ACCEPTED FOR IMPLEMENTATION`, etc.) per Authoring Guide § Lifecycle status wording.

### 3.4 Implementation stage (after gates)

Emitted by `init_scenario_implementation.py` (authors must not hand-copy this layout):

```text
platform_proofs/scenarios/<scenario_slug>/
├── README.md
├── SCENARIO_SPEC.md
├── application/
│   ├── runtime_composition.py   # build_scenario_lab_runtime
│   ├── scenario.py              # execute_scenario → execute_scenario_task
│   ├── agent.py
│   ├── observability.py
│   └── tools.py
├── proof/
│   ├── evaluator.py
│   └── evidence_builder.py
├── fixtures/
├── proof.json                   # descriptor v3
├── run_proof.py                 # configure → run app → evaluate → artifacts
└── .env.example
```

**Separation rule:** `application/` must not import `proof/` or own falsification logic; `proof/` must not own business workflow decisions (`scenario_architecture_conformance.py` enforces import and symbol rules).

**Proof identity:** `proof_id` form `SCENARIO-<SLUG-UPPERCASE-WITH-HYPHENS>` (see Protocol § D).

Scenarios may grow additional directories when justified (e.g. VPI: `dataset/`, `docker-compose.yml`, `output/`). The initializer list is the minimum platform-native shape, not the maximum.

### 3.5 What must exist for a new scenario to be “accepted”

| Stage | Minimum artifacts |
| --- | --- |
| **Design package** | `README.md` + `SCENARIO_SPEC.md` with valid frontmatter and completed A–E intent |
| **Permission to implement** | Scenario Quality Gate → `lifecycle: ACCEPTED_FOR_IMPLEMENTATION` + `observability_contract` / `application_vs_proof_ownership: COMPLETED` |
| **Permission to run init** | Above + `intergrax_fit: COMPLETED`, `gap_decision: RESOLVED`, `implementation_status: NOT_INITIALIZED` |
| **Executable package** | Initialized skeleton + implemented `application/` and `proof/` + working `run_proof.py` + `proof.json` discovered by `scripts/proof/` |
| **Public Library acceptance** | All gates in Authoring Guide § Public Library acceptance gate (evidence, report, reproduction, scenario-specific YES/NO table); update public docs only when accepted |

---

## 4. Scenario Creation Lifecycle

Discovered flow (normative in Protocol § Scenario-first lifecycle and Authoring Guide § Canonical Scenario Lifecycle):

```text
IDEA
↓
uv run python scripts/proof/create_scenario_proof.py --slug <slug> --title "<title>"
↓
DESIGN / NOT YET ACCEPTED  (README.md + SCENARIO_SPEC.md only)
↓
PART A — Design (Stages 1–6)
  REAL PROBLEM → WOW QUALIFICATION → SOLUTION ARCHITECTURE → PROOF DESIGN
  → SCENARIO QUALITY GATE → ACCEPTED FOR IMPLEMENTATION
↓
Implementation preparation (verify Intergrax fit; resolve platform gaps; update frontmatter)
↓
uv run python scripts/proof/init_scenario_implementation.py --slug <slug>
↓
IMPLEMENTATION_INITIALIZED
↓
PART B — Build (implement application/ + proof/; pause scenario if reusable platform gap found)
↓
TARGETED TESTING → REAL PROOF EXECUTION → EVIDENCE / REPORT VERIFICATION
↓
LIBRARY ACCEPTANCE → public catalog / PROOFS updates (when gates pass)
```

**Hard rules (repository-enforced or documented as normative):**

- `create_scenario_proof.py` ≠ `init_scenario_implementation.py`.
- Init **must not** run while `lifecycle` is still `DESIGN` (gate error in `validate_implementation_init_preconditions`).
- Scenario Quality Gate **must not** reject designs solely because Intergrax lacks a capability yet.
- Missing reusable platform capability → pause scenario → implement platform capability → verify → resume (Protocol § Platform gap resolution).

---

## 5. Scaffold Components

| Component | Location | Role |
| --- | --- | --- |
| **Design scaffold** | `scripts/proof/create_scenario_proof.py` | README + SCENARIO_SPEC templates, slug validation, forbidden artifact list |
| **Lifecycle metadata** | `scripts/proof/scenario_lifecycle.py` | Frontmatter parse/render; init preconditions |
| **Implementation scaffold** | `scripts/proof/init_scenario_implementation.py` | Skeleton files, `proof.json` stub, updates frontmatter to `IMPLEMENTATION_INITIALIZED` |
| **Architecture gate** | `scripts/proof/scenario_architecture_conformance.py` | AST/import rules for initialized scenarios (no app→proof/fixtures imports, baseline runtime symbols, agent lifecycle bypass detection) |
| **Descriptor schema** | `scripts/proof/intergrax_platform_proof_descriptor.py` + loader | `proof.json` v3 validation and discovery |
| **Proof runner / suite** | `scripts/proof/run-intergrax-proof-suite.py`, manifest/discovery modules | Suite execution, `SuiteReceipt`, profiles |
| **Evidence / report I/O** | `scripts/proof/intergrax_platform_proof_evidence_io.py`, `intergrax_platform_proof_html_renderer.py` | Typed evidence v3, HTML reports |
| **Runtime baseline** | `intergrax/applications/_shared/scenario_runtime_baseline.py`, `scenario_runtime_profiles.py` | Canonical `execute_scenario_task` entry for scenario applications |
| **Runtime profiles helper** | `intergrax/applications/_shared/scenario_runtime_profiles.py` | `build_scenario_lab_runtime` for generated composition |

**Per-scenario customization** (after init): `proof/evaluator.py`, `proof/evidence_builder.py`, `application/scenario.py`, fixtures, optional `proof/report_sections.py`, scenario-specific scripts under `scripts/` (VPI pattern).

**Reporting:** Proof packages write artifacts declared in `proof.json` `expected_artifacts` (e.g. AI incident: `evidence-resolved.json`, `evidence-unresolved.json`, optional HTML reports). Canonical published output directory convention: `output/` when used.

---

## 6. Documentation Standard

### 6.1 Two-layer documentation

| Layer | File | Audience / depth |
| --- | --- | --- |
| **Public gateway** | `README.md` | Operators, external readers; problem story, at-a-glance table, summaries linking to spec |
| **Canonical contract** | `SCENARIO_SPEC.md` | Architects, proof authors; full A–E, gate decisions, fit/gap, proof build |

Cross-links: README → SCENARIO_SPEC; SCENARIO_SPEC → README (relative paths).

### 6.2 Writing style

- Problem-first, plain language in README; platform mechanism names deferred to spec § C where possible.
- Truthful lifecycle status—no PASS badges or fake run metrics at design stage.
- **Abstract:** 4–8 sentences, no Intergrax internals (Authoring Guide § Abstract contract).
- **At a glance** table: Problem, Observed impact, Trap, Decision risk, Scenario outcome (RESOLVED/UNRESOLVED), Status, Proof class SCENARIO.
- Conditional sections (HITL, recovery, governance, temporal semantics) only when relevant—see spec scaffold “Conditional authoring prompts”.

### 6.3 Validation results in docs

- **Latest verified run** and **Run / report / evidence / source** in README: “Not yet available” until real execution and report acceptance; then verdict, SHA, timestamps, invariant summary, links to `output/` artifacts.
- `SCENARIO_SPEC.md` records Quality Gate decision text under § A Scenario Quality Gate (example: `ai_incident_investigation`, `verified_product_identification`).

### 6.4 Maintainer / architecture docs

- Protocol and Authoring Guide under `platform_proofs/`.
- Public narrative: `docs/project/proofs/PROOF_LIBRARY.md`, `docs/project/proofs/PROOFS.md` (updated only on library acceptance per Authoring Guide).
- ERL capability canon is separate (`docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md`); Phase 6 qualification is planned in maintainer implementation plan, not yet a fourth public scenario entry.

---

## 7. Visual Asset Standard

| Rule | Source |
| --- | --- |
| Location | `platform_proofs/scenarios/<slug>/assets/` |
| Preferred format | Light/dark **SVG** pairs per [`docs/project/technical/guides/DOCUMENTATION_DESIGN_SYSTEM.md`](../technical/guides/DOCUMENTATION_DESIGN_SYSTEM.md) |
| Scaffold behavior | HTML comment placeholder in README until after Scenario Quality Gate—no auto-generated decorative SVG |
| Content | Explanatory proof-story diagrams; no fake dashboards or screenshots of nonexistent runs |
| Target | One strong proof-story diagram; optional second supporting diagram |
| README embedding | `<picture>` / linked full-size SVG pattern (see `ai_incident_investigation`, `indirect_prompt_injection` READMEs) |

**Observed assets (implemented scenarios):**

- `ai_incident_investigation/assets/`: `proof-story-light.svg`, `proof-story-dark.svg`, `correlation-trap-light.svg`, `correlation-trap-dark.svg`
- `indirect_prompt_injection/assets/`: `proof-story-*.svg`, `trusted-policy-boundary-*.svg`, plus `scenario-overview.png` referenced from public catalog

**Public catalog previews:** `PROOF_LIBRARY.md` uses thumbnail links to scenario `assets/` (180px width). Root README may use duplicated marketing diagrams under `docs/project/proofs/assets/public/readme/` for featured scenarios (AI incident).

**Note:** `verified_product_identification` README and `PROOF_LIBRARY.md` reference `assets/scenario-overview.png`; that file was **not** present under `assets/` in the audited tree—only a documentation/catalog reference at audit time.

---

## 8. Publication Process

Scenarios become **visible** through layered steps:

| Step | Mechanism |
| --- | --- |
| **Existence** | Directory under `platform_proofs/scenarios/<slug>/` |
| **Discovery for execution** | Package `proof.json` scanned by `scripts/proof/` (no manual manifest entry for descriptor-backed proofs) |
| **In-development publicity** | Manual curation in [`docs/project/proofs/PROOF_LIBRARY.md`](../proofs/PROOF_LIBRARY.md) (catalog table + featured sections)—currently three scenarios |
| **Claims / evidence dashboard** | [`docs/project/proofs/PROOFS.md`](../proofs/PROOFS.md) when accepted public evidence boundaries change |
| **Full acceptance** | Authoring Guide § Public Library acceptance gate; descriptor `public_evidence_eligible` may remain `false` until publication (all three initialized scenarios audited with `false`) |

Contract tests guard public docs: `tests/unit/docs/test_proof_library_public_contract.py` (problem-first framing, catalog truth, featured scenario wording).

**Community intake:** `PROOF_LIBRARY.md` references `scenario_proposal.yml` for “Challenge Intergrax” proposals—separate from filesystem package creation.

There is **no** automatic registration when `create_scenario_proof.py` runs; ten design-only slugs exist without catalog entries.

---

## 9. Quality Gates

### 9.1 Lifecycle / scaffold (unit tests)

| Test module | What it proves |
| --- | --- |
| `tests/unit/scripts/proof/test_create_scenario_proof.py` | Design package sections, forbidden artifacts |
| `tests/unit/scripts/proof/test_init_scenario_implementation.py` | Init preconditions, generated tree, frontmatter transitions |
| `tests/unit/scripts/proof/test_scenario_scaffold_conformance_proof.py` | Scaffold conformance |
| `tests/unit/scripts/proof/test_all_initialized_scenario_architecture.py` | Repo-wide architecture gate for all initialized slugs |
| `tests/unit/scripts/proof/test_scenario_architecture_conformance.py` | Rule-level architecture violations |

### 9.2 Scenario application tests

- Per-scenario tests under `tests/unit/platform_proofs/scenarios/<slug>/` and `tests/integration/platform_proofs/scenarios/<slug>/` (volume largest for VPI).
- Example gate: `tests/unit/platform_proofs/scenarios/ai_incident_investigation/test_architecture_scaffold_gate.py` (SCENARIO-PLATFORM-5/6A).

### 9.3 Execution / qualification commands

```bash
# Standalone (repository root)
uv run python platform_proofs/scenarios/<scenario_slug>/run_proof.py

# Suite
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
```

Proof lifecycle phases documented for README authors: CONFIGURE → PREFLIGHT → … → RUN → EVALUATE → REPORT → CLEANUP.

### 9.4 Adjacent E2E qualification (not scenario package scaffold)

- `testing_support/decision_e2e/` — local AI incident qualification (`local_ai_incident_qualification.py`), controlled/natural alignment, model matrix (DS-E2E-15J-*), with CLIs under `scripts/proof/ds_e2e_*.py`.
- Uses frozen source lists pointing at `platform_proofs/scenarios/ai_incident_investigation/application/*` for behavioral evidence—not a substitute for Platform Proof library acceptance.

### 9.5 CI

No dedicated GitHub workflow filename was required for this audit; scenario gates run as **pytest unit tests** with `@pytest.mark.gate` (e.g. architecture conformance). Suite proof runs are maintainer/qualification operations per Authoring Guide.

---

## 10. Recommendations for ERL-QUAL-004

These are **process recommendations only** for “ERL Payment Uncertainty Recovery Scenario” (fourth enterprise-oriented showcase), aligned with existing discipline:

1. **Use Platform Proof SCENARIO lifecycle**, not a parallel E2E folder: either qualify and expand an existing slug (`payment_exception_recovery` is design-only and title-aligned) or create a new slug via `create_scenario_proof.py` if the architect requires a distinct identity—do not hand-create implementation directories.

2. **Complete PART A before any init:** ERL UNKNOWN / reconcile / recovery semantics belong in § A–B (Application Observability Test and observability contract are mandatory for payment uncertainty). Map ERL boundaries from [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md) in § C INTERGRAX FIT during implementation preparation, not as proof-local substitutes.

3. **Respect APPLICATION vs PROOF split:** payment truth, reconciliation, and lifecycle handoff live in `application/` consuming platform ports; `proof/` owns adversarial provider fixtures (timeouts, ambiguous gateway responses) and falsification of double-charge / resume invariants.

4. **Plan visuals after Scenario Quality Gate:** light/dark SVG proof-story under `assets/`; hero image per architect remains outside this framework’s auto-scaffold.

5. **Publication expectations:** ERL implementation plan Phase 6 explicitly allows **maintainer qualification notes** without a dedicated **public** proof route until architecture gates pass—mirror that in README status until Public Library acceptance; keep `public_evidence_eligible: false` until acceptance.

6. **Validation path:** reuse `execute_scenario_task` baseline; add targeted unit gates under `tests/unit/platform_proofs/scenarios/<slug>/`; prove via `run_proof.py` + optional suite profile; do not conflate with `testing_support/decision_e2e` unless a separate behavioral qualification track is explicitly requested.

7. **Catalog:** after library acceptance, update `PROOF_LIBRARY.md` (and `PROOFS.md` if claims change)—fourth catalog row is manual, not discovery-driven.

8. **Reference implementations to study (in repo order of maturity):** `ai_incident_investigation` (evaluator + dual RESOLVED/UNRESOLVED), `indirect_prompt_injection` (governance boundary), `verified_product_identification` (large scenario extension pattern only where ERL needs data/infra depth).

---

## Audit metadata

| Item | Value |
| --- | --- |
| Document | `docs/project/architecture/E2E_SCENARIO_FRAMEWORK_AUDIT.md` |
| Method | Read-only inspection of `platform_proofs/`, `scripts/proof/`, tests, and public proof docs |
| Assumptions avoided | ERL-QUAL-004 naming and payment scenario content are **not** treated as implemented facts |
