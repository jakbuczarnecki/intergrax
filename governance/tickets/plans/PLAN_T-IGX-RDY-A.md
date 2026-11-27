# PLAN_T-IGX-RDY-A — `SCENARIO-READINESS-X` P0–P4 Execution Plan

| Field | Value |
|---|---|
| **Stage** | `SCENARIO-READINESS-X` — Scenario Delivery Velocity & Boundary Integration Certification |
| **Canonical Roadmap Authority** | [`docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../../docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.7 (lines 1268–1469) |
| **Bootstrap Ticket** | [`governance/tickets/T-IGX-RDY-A.md`](../T-IGX-RDY-A.md) (`Lifecycle state: plan_pending` — awaits post-freeze Operator `Approved:`) |
| **Stage Status** | **`PLANNED / MANDATORY`** — not started; not CLOSED; `production delta = 0` at P0 |
| **Hard Predecessor** | **`ARCH-FREEZE = CLOSED`** (`FREEZE_CERTIFICATION_SHA` pinned) **ONLY** — NOT blocked on `SEF-X` or `SCENARIO-FACTORY-X` (OD-SEF-02 Option 3) |
| **Next Mandatory Gate** | **`SCENARIO-GATE`** (requires `SCENARIO-READINESS-X = CLOSED` via independent exact-SHA closure) |

---

## 1. Mission, Primary Platform Question & Non-Claims

`SCENARIO-READINESS-X` is a **reference scenario-readiness test** of the formally frozen Intergrax platform. It is **not** a customer production rollout and **not** the start of scenario-driven platform development.

**Primary platform question (Roadmap §3.0.7):**

> Can a real business process be built and run very quickly on formally frozen Intergrax as a thin, separately deployable component between existing customer systems — with **zero core platform changes**, **zero scenario-local bypasses**, and **zero new platform mechanisms**?

### Hard Invariants & Targets

- **Frozen Core Delta Target:** `0` changes under `intergrax/` (Tier-0 / Tier-1).
- **New Scenario-Local Platform Mechanisms Target:** `0` (no scenario-local execution, governance, provider, persistence, or configuration authority).
- **Frozen Architecture Rule:** If the reference scenario requires modifying frozen core architecture, execution MUST immediately halt with:
  `STOP — FROZEN ARCHITECTURE VIOLATION`
  and escalate through the formal freeze-exception / ADR / recertification process. Never patch frozen core inside scenario validation.
- **Deterministic-First Invariant:** `deterministic business logic must remain deterministic`. LLM/agent invocation is forbidden for unambiguous rule-based reconciliation and permitted only where genuine ambiguity exists (e.g., unstructured description classification or exception-resolution recommendation) with documented justification.
- **`EVOL-X` S8 Continuity (Mandatory):** Scenario delivery and onboarding MUST use `EVOL-X`-accepted change-impact and conservative regression-selection contracts via `ENT-AUDIT-X` baseline/drift runs, satisfying experiment-class **S8** expectations (sanctioned extension surfaces only).
- **Environment Freeze (C10/C15):** Preserve the original Windows environment; no mandatory Docker migration, no new vendor integrations, no secrets in source/argv/logs.
- **Non-Claims:** This P0 plan declares NO `CLOSED`, NO `SCENARIO-GATE PASS`, NO `IIAC PASS`, NO `FRZ PASS`, and NO `G6 PASS`.

---

## 2. Reference Business Boundary Scenario Specification

### 2.1 Boundary Architecture

Intergrax operates strictly as a thin component between external customer systems — external systems remain external and isolated behind boundary adapters:

```text
CUSTOMER SYSTEM A (external file / API)
        │
        ▼
  Input Boundary Adapter
        │
        ▼
  Typed Scenario Input Contract
        │
        ▼
  INTERGRAX SCENARIO COMPONENT (Tier-3 package, separately deployable)
    ├── Deterministic normalization & reconciliation rules
    ├── Exception classification (7 deterministic classes)
    ├── Optional HITL (existing platform human-decision mechanism)
    └── Full trace / evidence (existing platform evidence plane)
        │
        ▼
  Typed Scenario Output Contract
        │
        ▼
  Output Boundary Adapter
        │
        ▼
CUSTOMER SYSTEM B (downstream ledger / review queue)
```

### 2.2 Reference Business Flow (Business Settlement / Reconciliation — Validation Only)

Synthetic domain flow (illustrative non-canonical domain names; strictly non-overlapping with `SCENARIO-FACTORY-X` `REF-2` API-to-API order sync per `C-RDY-01`):

1. Ingest settlement / hours / delivery execution records from Customer System A via the input boundary adapter.
2. Ingest invoice or cost document records via typed input contract.
3. Normalize inputs deterministically.
4. Resolve contractor, billing period, contract, and rate deterministically from typed fixture/adapter inputs.
5. Compute `expected_amount` deterministically (integer minor units / exact arithmetic).
6. Compare `expected_amount` vs `submitted_amount`.
7. Classify each record into one of the 7 deterministic outcomes:
   - `MATCH`
   - `HOURS_MISMATCH`
   - `AMOUNT_MISMATCH`
   - `UNKNOWN_CONTRACTOR`
   - `MISSING_RATE`
   - `MISSING_DOCUMENT`
   - `DUPLICATE`
8. Route `MATCH` records to downstream typed output via the output boundary adapter.
9. Route exception records to optional HITL / review strictly through existing platform HITL/decision mechanisms.
10. Emit typed output contract.
11. Persist complete trace and execution evidence through existing platform evidence/observability ports.

---

## 3. Phased Execution Plan (`P0` – `P4`)

### Phase P0 — Pre-Freeze Bootstrap & Operator Decision Lock (`production delta = 0`)

**Executed pre-freeze (`T-IGX-090-D`) + completed at `T = 0` upon `ARCH-FREEZE = CLOSED`:**

- Publish `PLAN_T-IGX-RDY-A.md` and draft `T-IGX-RDY-A.md` (`plan_pending`).
- Prepare Operator Decision Register (`OD-RDY-01..04`) to be signed alongside `T-IGX-RDY-A.md` approval once `FREEZE_CERTIFICATION_SHA` is pinned:
  - **`OD-RDY-01` (Reference Settlement Scope & `C-RDY-01` Guard):** Ratify exact synthetic settlement/reconciliation inputs, deterministic rule matrix, and non-overlap boundary with SFX `REF-2`.
  - **`OD-RDY-02` (Representative Alternate Infrastructure / Provider Concern — EC #23–25):** Select the concrete platform-claimed replaceable concern to prove (e.g., configuration-switched persistence/storage substrate or adapter/queue backend supported by frozen composition root without touching scenario code, compliant with Windows C10/C15).
  - **`OD-RDY-03` (Windows-Compatible Reproducible Local Deployment Shape — EC #15):** Select the clean reproducible local deployment package mechanism runnable on the frozen Windows environment without requiring a Docker migration (roadmap §3.0.7 explicitly notes `docker compose up` is illustrative, not frozen).
  - **`OD-RDY-04` (Scenario Delivery Kit Analysis Template):** Confirm analysis output structure under `docs/project/maintainers/qualification/scenario_readiness_x/` so `SEF-X` (`T-P13-005`) and `SCENARIO-FACTORY-X` (`SFX-P8-01`) can consume standardization findings post-closure.
- Pin `FREEZE_CERTIFICATION_SHA` and baseline `ENT-AUDIT-X` / `EVOL-X` impact manifest before writing any P1 code.

### Phase P1 — Reference Settlement/Reconciliation Component Implementation (Post-Freeze, Core Delta = 0)

**Precondition:** `ARCH-FREEZE = CLOSED` + Operator `Approved:` on `T-IGX-RDY-A.md` + `OD-RDY-01..04` signed.

- Scaffold the separately deployable Tier-3 reference scenario package (e.g., `contracts/`, `rules/`, `adapters/`, `configuration/`, `tests/`, `fixtures/`, local deployment runner) using sanctioned frozen platform contracts and composition only.
- Implement:
  - Typed input contract and typed output contract (`RDY-EC-03`, `RDY-EC-04`).
  - Input and output boundary adapters isolating Customer System A and Customer System B (`RDY-EC-05`).
  - Pure deterministic settlement/reconciliation rules covering all 7 exception classifications (`RDY-EC-07`).
  - Optional HITL routing via existing platform human-decision mechanism (`RDY-EC-08`).
  - Trace and evidence emission via existing platform `EvidencePersistencePort` / observability spine (`RDY-EC-09`).
  - Strict enforcement of Governance/Execution separation (`RDY-EC-10`), tenant isolation (`RDY-EC-11`), zero scenario-local authority (`RDY-EC-12`), zero duplicated mechanisms (`RDY-EC-13`), zero bypasses (`RDY-EC-14`), and zero vendor/backend selection logic in scenario code (`RDY-EC-22`).
- Record exact wall-clock and engineering timestamps for **`Time to First Running Scenario`** (`RDY-EC-17`).
- Verify `git diff <FREEZE_CERTIFICATION_SHA> -- intergrax/` is strictly empty (`RDY-EC-01`).

### Phase P2 — Adapter Swap, Alternate Infrastructure Proof & Reproducible Local Deployment

- **Boundary Adapter Swap Proof (`RDY-EC-06`):** Swap at least one external boundary adapter (e.g., file input adapter <-> stub HTTP/JSON adapter) with **zero edits** to core scenario logic; measure and record adapter-swap time.
- **Representative Alternate Infrastructure / Provider Configuration Proof (`RDY-EC-23`, `RDY-EC-24`, `RDY-EC-25`):**
  - Run the exact same scenario code under a second sanctioned platform configuration/composition for the `OD-RDY-02`-selected replaceable concern (no vendor/backend branching in scenario code).
  - Verify governance, execution, tenant isolation, traceability, and recovery guarantees hold identically under both configurations.
  - Verify any SQLite or fixed substrate used is configuration-ratified or consumed exclusively through platform contracts (never scenario-local persistence authority).
- **Clean Reproducible Local Deployment & Full Automated E2E (`RDY-EC-02`, `RDY-EC-15`, `RDY-EC-16`, `RDY-EC-18`):**
  - Execute a clean-state local deployment from scratch, record manual steps count, measure **`Time to Reproducible Local Deployment`**, and run full automated E2E suite serially (`uv run pytest`) capturing proof artifacts under `.tmp/proof/scenario_readiness_x/<run_id>/`.

### Phase P3 — Delivery Velocity Metrics, Friction Inventory & Scenario Delivery Kit Analysis

- **Measure & Publish All 11 Mandatory Delivery Velocity Metrics** (`docs/project/maintainers/qualification/scenario_readiness_x/READINESS_METRICS_REPORT.md`):
  1. `Time to First Running Scenario`
  2. `Time to Reproducible Local Deployment`
  3. Core Intergrax changes count (`target: 0`)
  4. New platform mechanisms count (`target: 0`)
  5. Amount of scenario-specific code (LOC / files)
  6. Amount of adapter-specific code (LOC / files)
  7. Share of configuration / declarative specification
  8. Reuse ratio of existing platform capabilities
  9. Time to swap one external boundary adapter
  10. Manual deployment steps count
  11. Count of friction points impeding fast deployment
- **Inventory & Classify Every Friction Point (`RDY-EC-19`, `RDY-EC-20`, `RDY-EC-21`):**
  - Record every friction finding in `READINESS_FRICTION_REGISTER.md` with **exactly one** canonical roadmap class:
    1. `PLATFORM DEFICIENCY`
    2. `TOOLING / DX DEFICIENCY`
    3. `ADAPTER / INTEGRATION WORK`
    4. `BUSINESS-SCENARIO WORK`
    5. `DEPLOYMENT / OPERABILITY DEFICIENCY`
    6. `STOP — FROZEN ARCHITECTURE VIOLATION`
  - Platform deficiencies MUST NOT be masked as scenario-local workarounds. Confirm unresolved platform-level architecture blockers = `0` prior to `SCENARIO-GATE`.
- **Author `Intergrax Scenario Delivery Kit` Standardization Analysis (`READINESS_DELIVERY_KIT_ANALYSIS.md`):**
  - Analysis **only** (implementation explicitly deferred per roadmap §3.0.7). Evaluate standardization needs for: scenario template, typed input/output contract templates, boundary adapter template, deterministic rules module convention, fixture convention, qualification template, local/Docker deployment template, environment/configuration template, and one-command validation/deployment workflow.
- Run `EVOL-X` / `ENT-AUDIT-X` drift & S8 conformance verification plus `uv run python -m scripts.governance.igx_gov ticket`.

### Phase P4 — Independent Exact-SHA Closure Audit & `SCENARIO-GATE` Handoff

- Hand off pinned candidate SHA, metrics report, friction register, alternate-infra proof, and Delivery Kit analysis to a **separate new conversation** (`independent-quality-auditor`) for exact-SHA independent audit.
- Implementer, tester, and reviewer sessions MUST NOT declare `SCENARIO-READINESS-X = CLOSED`, `G6 PASS`, `IIAC PASS`, or `SCENARIO-GATE PASS`.
- Upon independent exact-SHA PASS and Operator closure signature, `SCENARIO-READINESS-X = CLOSED` unlocks formal entry to **`SCENARIO-GATE`**.

---

## 4. Complete 25-Criterion Exit Criteria Traceability Matrix (Roadmap §3.0.7)

`SCENARIO-READINESS-X` reaches `CLOSED` **only** when all 25 criteria hold at an independently audited exact SHA:

| ID | Roadmap §3.0.7 Exit Criterion | Phase | Verification Evidence |
|---|---|---|---|
| `RDY-EC-01` | Frozen Intergrax core delta = 0 | P1–P4 | `git diff <FREEZE_CERTIFICATION_SHA> -- intergrax/` empty + `ENT-AUDIT-X` drift check |
| `RDY-EC-02` | Reference scenario runs as a separately deployable component | P1–P2 | Standalone package build & clean local launch proof |
| `RDY-EC-03` | Input has typed contract | P1 | Typed input schema/model + contract validation unit tests |
| `RDY-EC-04` | Output has typed contract | P1 | Typed output schema/model + contract validation unit tests |
| `RDY-EC-05` | External systems are behind adapters/contracts | P1 | Boundary adapter isolation review + tier/import scan |
| `RDY-EC-06` | An adapter can be swapped without changing core scenario logic | P2 | Adapter swap test + `0` diff in core scenario rules/flow |
| `RDY-EC-07` | Deterministic rules use deterministic mechanisms | P1 | Repeatability / determinism test across all 7 classifications; zero LLM calls on deterministic paths |
| `RDY-EC-08` | HITL uses existing platform mechanisms | P1 | Integration test exercising existing platform human-decision contract |
| `RDY-EC-09` | Trace/evidence uses existing platform mechanisms | P1 | Evidence persistence test via existing `EvidencePersistencePort` / observability spine |
| `RDY-EC-10` | Governance/Execution boundaries remain intact | P1–P2 | Tier boundary & governance conformance tests |
| `RDY-EC-11` | Tenant boundary remains intact | P1–P2 | Cross-tenant negative tests + fail-closed tenant verification |
| `RDY-EC-12` | No scenario-local execution/governance/provider/configuration authority | P1–P3 | Static scan + independent reviewer audit |
| `RDY-EC-13` | No duplicated mechanisms | P1–P3 | DUP-X / no-duplicate audit against frozen platform owners |
| `RDY-EC-14` | No bypasses | P1–P3 | Policy/auth/impact negative tests + `EVOL-X` S8 check |
| `RDY-EC-15` | Clean local deployment is reproducible | P2 | Clean-environment local deploy transcript + exit 0 proof |
| `RDY-EC-16` | Full E2E can run automatically | P2 | Serial automated E2E pytest run transcript |
| `RDY-EC-17` | `Time to First Running Scenario` measured | P1, P3 | Timestamped entry in `READINESS_METRICS_REPORT.md` |
| `RDY-EC-18` | `Time to Reproducible Deployment` measured | P2, P3 | Timestamped entry in `READINESS_METRICS_REPORT.md` |
| `RDY-EC-19` | Friction points inventoried | P3 | `READINESS_FRICTION_REGISTER.md` |
| `RDY-EC-20` | Every friction point classified (6-class taxonomy) | P3 | 1:1 canonical class assignment verified in friction register |
| `RDY-EC-21` | Unresolved platform-level architecture blockers = 0 before `SCENARIO-GATE` | P3–P4 | Zero open `PLATFORM DEFICIENCY` / `STOP` blockers verified |
| `RDY-EC-22` | Scenario code contains no vendor/backend selection logic (sanctioned platform config/composition only) | P1–P2 | Code audit + import/branching scan across scenario package |
| `RDY-EC-23` | Representative alternate infrastructure/provider configuration demonstrated for >=1 platform-claimed replaceable concern | P2 | Dual-configuration E2E proof under `OD-RDY-02` |
| `RDY-EC-24` | Governance, execution, tenant, traceability, and recovery guarantees hold under alternate configuration | P2 | Parity assertion matrix across primary & alternate configs |
| `RDY-EC-25` | SQLite or other fixed substrates are configuration-ratified or consumed only through platform contracts — not scenario-local persistence authority | P2–P3 | Composition & persistence contract audit |

---

## 5. Out of Scope & Change Control

- **Out of Scope in P0 (pre-freeze):** Any code under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`; any customer production rollout; any implementation of `Intergrax Scenario Delivery Kit` templates; any modification of `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`.
- **Execution Discipline (P1–P3 post-freeze):** Isolated worktree branch, serial `uv run pytest` only (never parallel workers on Windows), English-only commits with `Ref: governance/tickets/T-IGX-RDY-A.md` and zero AI attribution, human-only `git push` / PR / merge.
