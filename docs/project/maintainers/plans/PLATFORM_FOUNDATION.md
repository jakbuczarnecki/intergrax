# Platform Foundation — Implementation Plan

**Architecture (1:1):** [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Architecture governance:** [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (PLATFORM_FOUNDATION plan).

- **Implement / audit default:** §6.1 gate maintenance (default) · §4.0a scope split. **On demand:** [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) · [`plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md) (re-validate closed only) · [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) (Band 2ae HEP only)
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## LCI-7A — LangChain optional extras packaging

**Status:** `APPROVED`
**Owner:** PLATFORM_FOUNDATION

The package's direct core dependencies contain no normalized `langchain*` or
`langgraph` names. Compatibility/provider ownership remains explicit in
`rag-langchain-loaders`, `rag-langchain-embeddings`,
`rag-langchain-splitters`, `llm-langchain-ollama`, and
`langgraph-legacy`. The clean-core and compatibility installation gates are
closed under LCI-7B and LCI-7C; documentation closeout is LCI-7D.

## LCI-7D — LangChain independence documentation closeout

**Status:** `READY_FOR_REVIEW`

The feature architecture and dependency inventory now distinguish native/core
defaults, optional compatibility providers, and legacy optional LangGraph
paths. LCI-8A remains the next task after acceptance; no production runtime,
RAG implementation, or packaging declaration changes belong to this closeout.

---

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](.#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**. **Open qualification:** **[§6.1ax PF-TIER-ENFORCEMENT](.#61ax-pf-tier-enforcement--production-tier-boundary-qualification)** — tier-boundary enforcement remediation (audit `4c92e0a`, verdict `CONDITIONALLY SOUND — ENFORCEMENT REMEDIATION REQUIRED`).

**Maintenance-only mode (qualified):** Feature/platform implementation backlogs remain **closed**; ongoing work = §6.1 gate maintenance. Tier-boundary enforcement remediation is **not** closed — see §6.1ax. If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance (enforcement qualification still open).

**Out of scope:** product/application implementation work and business backlogs — **[§6.3](.#63-end-of-platform-plan)** · **[§4.0a](.#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](.#61z-harness-implementation-queue-consolidated) / [§6.1aa](.#61aa-harness-implementation-queue-memory-platform).

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed** for feature/platform implementation. Ongoing work = keep the harness green. **Exception (open qualification):** tier-boundary enforcement proof is incomplete — **[§6.1ax](.#61ax-pf-tier-enforcement--production-tier-boundary-qualification)** must complete before Foundation may return to unqualified maintenance-only/closed status for this area. **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](.#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Band 2ay M.12** = **Done** — see **[§6.1an](.#61an-harness-implementation-queue--llm-guardrail-integrations-closed)**. **Optional harness extension (after gate green):** **[Band 2ae Phase HEP](.#61aw-phase-hep--harness-evidence-pack-band-2ae)** — runtime evidence packaging.

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/maintenance/check_harness_no_getattr.py
  python scripts/maintenance/check_legacy_modules_removed.py
  python scripts/maintenance/check_agent_skill_resolution.py
  python scripts/maintenance/check_harness_registry_resolution.py
  python scripts/maintenance/check_harness_capability_graph_wiring.py
  python scripts/maintenance/check_legacy_tool_plan_booleans.py
  python scripts/maintenance/check_trace_bridge_event_catalog.py
  python scripts/maintenance/check_plugin_catalog.py
  python scripts/maintenance/check_llm_adapter_typed_returns.py
  python scripts/maintenance/check_agents_llm_adapter_response.py
  uv run python scripts/release/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/release/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/release/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/maintenance/check_scaffold_harness_alignment.py
  python scripts/maintenance/check_agents_no_tier3_imports.py
  python scripts/maintenance/check_intergrax_no_applications_imports.py
  python scripts/check_no_upward_application_imports.py
  uv run python scripts/maintenance/check_harness_prompt_golden_catalog.py
  uv run python scripts/maintenance/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/release/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/release/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/release/phase_v_capability_graph_guard.py --enforce
  python scripts/maintenance/check_agents_no_inline_prompts.py
  python scripts/maintenance/check_agents_no_vendor_sdk_imports.py
  uv run python scripts/gates/check_ideal_harness_l3_gates.py
  uv run python scripts/gates/harness_maturity_report.py --enforce-l3-critical
```

**Out of scope for §6.1:** product/application implementation work and business backlogs. This document does not track product application roadmaps.

### 6.1av Harness implementation queue — Platform Foundation audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `PLATFORM_FOUNDATION` layers 1, 2, 32 · [`../audit_results/2026-06-19/PLATFORM_FOUNDATION.md`](../../../audit_results/2026-06-19/PLATFORM_FOUNDATION.md) · prior: [`../audit_results/2026-06-18/PLATFORM_FOUNDATION.md`](../../../audit_results/2026-06-18/PLATFORM_FOUNDATION.md)
**Priority ladder:** **Band 1** (§6.1) — doc hygiene + optional legacy cleanup; runs **in parallel** with gate maintenance

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **PF-MAINT-DOC-01** | Docs | P2 | **Done** | Remove stale M.6 P6 from audit prompt known-gaps; sync audit result file | Audit prompt + result match plan §6.1y (**Done** 32/32) |
| 2 | **PF-MAINT-DOC-02** | Docs | P2 | **Done** | Sync §6.1au + §4.0 Band 2az counter with `AUDIT_IDEAL_2026.md` | Plan shows **90/90 Done** · **0 Planned** |
| 3 | **PF-MAINT-DX-01** | Docs | P3 | **Done** | Implementer quick-start in `intergrax_runtime_architecture.md` hub | §4.0 ladder + scaffold flow linked |
| 4 | **PF-MAINT-LEG-01** | Code | P3 | **Done** | Remove `use_rag`/`use_websearch` from LLM planner schema (`EnginePlan`) | `check_legacy_tool_plan_booleans.py` green; `tool_ids` only |
| 5 | **PF-MAINT-DOC-03** | Docs | P3 | **Done** | Sync §0.5 regression gate counter with live `pytest -m gate` snapshot | Plan §0.5 shows **1498 passed** (2026-06-19) |
| 6 | **PF-MAINT-LEG-02** | Code | P3 | **Done** | Remove legacy `use_rag`/`use_websearch` shims from `ToolInvocationPlan` (`tool_runtime.py`) | Zero DeprecationWarning in gate; `tool_ids` only at runtime bridge |
| 7 | **PF-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/legacy/2026-06-19` | `PLATFORM_FOUNDATION.md` + `legacy campaign README` present |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** §50 marketplace, new Tier-0 mechanisms beyond §6.1 — see [§6.3](.#63-end-of-platform-plan).

### 6.1ax PF-TIER-ENFORCEMENT — Production tier-boundary qualification

**Status:** `PLANNED`
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** PLATFORM_FOUNDATION enforcement audit — snapshot `4c92e0a08f92341f559408c234d213a8ac482d76`
**Verdict:** `CONDITIONALLY SOUND — ENFORCEMENT REMEDIATION REQUIRED` — conceptual Tier-0..3 architecture is sound; no confirmed current upward Tier-3 import violation was found in the audited scope; enforcement/proof is incomplete and must be strengthened before PLATFORM_FOUNDATION can truthfully be treated as fully closed for tier boundaries.

**Audit findings (persisted):**

| ID | Severity | Area | Summary |
|----|----------|------|---------|
| FND-01 | HIGH | ARCH / PROOF | Documented “full lower-layer scan” is not full — `check_no_upward_application_imports.py` uses manually enumerated `SCAN_ROOTS` |
| FND-02 | HIGH | ARCH / CONTRACT / PROOF | No single authoritative package→tier classification model; `DeploymentTier` is label/metadata; knowledge duplicated across docs and scanner lists |
| FND-03 | HIGH | WIRE / DOC / PROOF | Plan requires three tier guards per harness PR; audited CI runs only `check_agents_no_tier3_imports.py` in relevant tier-boundary gate paths |
| FND-04 | HIGH | ARCH / PROOF | Guards focus on application/Tier-3 imports; no one complete mechanism for full forbidden upward dependency matrix |
| FND-05 | MEDIUM | IMPL / PROOF | Inspected guards use regex/text matching, not semantic import/dependency analysis |
| FND-06 | HIGH | REL / WIRE | Main regression workflow is push-triggered for `main`, not shared `development` — integration-branch protection gap |
| FND-07 | MEDIUM | TEST / PROOF | No dedicated contract tests found in audited scope proving guard fail/pass/completeness behavior |
| FND-08 | LOW | CONTRACT / LEGACY | `DeploymentTier.PRODUCT` deprecated alias appears unused — optional cleanup subordinate to enforcement |
| FND-09 | HIGH | DOC / PROOF | “Closed / maintenance-only” language too strong while enforcement gaps remain |

**Scope (one implementation unit):**

| | Deliverable |
|---|-------------|
| **A** | **Authoritative package→tier classification** — one canonical source of truth for Tier-0..3 production packages; eliminate duplicated manual `SCAN_ROOTS` knowledge |
| **B** | **Complete dependency matrix enforcement** — validate every forbidden upward tier dependency, not only Tier-3 application imports |
| **C** | **Fail-closed package discovery** — newly introduced production packages must not silently escape classification/enforcement |
| **D** | **Semantic import/dependency analysis** — replace or subsume regex-only proof with structurally reliable analysis (implementation choice left to implementation task; production-grade, dependency-conscious) |
| **E** | **Tests of the architectural guard** — deterministic tests proving forbidden upward dependency → FAIL; allowed dependency → PASS; unclassified package → FAIL; representative relative/import syntax cannot silently bypass enforcement |
| **F** | **CI wiring** — canonical tier-boundary enforcement runs in relevant PR/full gate; active shared `development` integration path receives appropriate automated protection |
| **G** | **Guard consolidation** — after canonical mechanism owns enforcement, remove duplicate/obsolete tier guard scripts or reduce to thin wrappers only with justified compatibility need |
| **H** | **Legacy cleanup** — assess/remove `DeploymentTier.PRODUCT` if confirmed unused (subordinate to A–G) |
| **I** | **Documentation/status closeout** — architecture and plan wording match actual enforcement; only then may Foundation return to unqualified maintenance-only/closed status for tier boundaries |

**Acceptance criteria:**

1. One authoritative production package→tier classification exists.
2. Every production package in relevant repository roots is classified or causes fail-closed failure.
3. Full Tier-0..3 forbidden upward dependency matrix is enforced.
4. Guard is not solely regex/text based.
5. Contract/unit tests prove allowed and forbidden dependency cases plus unclassified-package behavior.
6. Canonical CI invokes the new enforcement.
7. Active integration workflow protects `development` appropriately.
8. Old duplicate tier guards are removed or have an explicitly justified remaining role.
9. Architecture and plan wording match actual enforcement.
10. Independent audit is required before marking this remediation **Done**.

**Explicitly excluded from this block:** product/application feature work; Tier-0..3 conceptual redesign.

<a id="61ax-tl-fix-a--executable-tier-ownership-protocol-v221-2026-08-18"></a>

### TL-FIX-A — Executable tier ownership (Protocol v2.1 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-TIER_LAYER_BOUNDARIES-01`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md), [`AUDIT-20260818-TIER_LAYER_BOUNDARIES-05`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- one authoritative production package→tier classifier
- complete semantic forbidden-edge enforcement
- unclassified production package fails closed
- deterministic tests for allowed/forbidden/unclassified/relative-import cases
- canonical CI wiring on relevant development/PR/integration path
- Tier-3 `applications/` consumer/static-contract coverage (`check_harness_no_getattr` scope or successor)
- old duplicated guards removed or reduced only after canonical mechanism owns proof

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- May reference/absorb §6.1ax PF-TIER-ENFORCEMENT conceptually; do **not** mark PF-TIER-ENFORCEMENT Done merely because this block exists.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-TIER-LAYER-PERSIST.
- **Cross-ref:** [`AUDIT-20260818-PLATFORM_FOUNDATION-01`](../../audit_results/2026-08-18/PLATFORM_FOUNDATION.md) revalidates TL-FIX-A / §6.1ax scope; [`AUDIT-20260818-PLATFORM_FOUNDATION-04`](../../audit_results/2026-08-18/PLATFORM_FOUNDATION.md) maps to deliverable **F** (integration-path protection).

<a id="61ax-pf-proof-integrity--foundation-proof-gate-contract-protocol-v222-2026-08-18"></a>

### PF-PROOF-INTEGRITY — Foundation proof and gate-contract parity (Protocol v2.2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P1
**Type:** Wire / Proof / Doc
**Source:** [`AUDIT-20260818-PLATFORM_FOUNDATION-02`](../../audit_results/2026-08-18/PLATFORM_FOUNDATION.md), [`AUDIT-20260818-PLATFORM_FOUNDATION-03`](../../audit_results/2026-08-18/PLATFORM_FOUNDATION.md), [`AUDIT-20260818-PLATFORM_FOUNDATION-05`](../../audit_results/2026-08-18/PLATFORM_FOUNDATION.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- foundation proof runners (`intergrax doctor --ci`, umbrella gates such as `check_audit_ideal_gates.py`) resolve scripts through one canonical path registry or equivalent strongly owned mechanism (`scripts/ci/script_paths.py` or successor)
- required checks fail closed when a declared script cannot be resolved or executed — no PASS-like `skip missing` for required guards
- umbrella gates execute the complete intended check set and collect failure state without short-circuiting after the first non-zero result
- documented §6.1 harness PR gate contract and actual CI smoke/full wiring describe the same required enforcement — do not weaken the documented target to match current CI subset

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- PF-02/PF-03 are proof-runner defects — do **not** treat them as tier-model redesign.
- PF-04 integration-path protection remains owned by **TL-FIX-A** / §6.1ax deliverable **F** — this block does not duplicate that work.
- PF-06 (`DeploymentTier.PRODUCT`) remains subordinate cleanup under §6.1ax deliverable **H**.
- **Not implemented** by audit persistence task AUDIT-20260818-PLATFORM_FOUNDATION-PERSIST.

<a id="protocol-v2-pcm-persistence-topology-integrity-2026-08-18"></a>

### PCM-PERSISTENCE-TOPOLOGY-INTEGRITY — Cross-layer persistence topology qualification (Protocol v2 · 2026-08-18)

**Status:** `ACCEPTED / PLANNED`
**Priority:** P0
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-01`](../../audit_results/2026-08-18/PERSISTENCE_CONCURRENCY_MULTIHOST.md), [`AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-06`](../../audit_results/2026-08-18/PERSISTENCE_CONCURRENCY_MULTIHOST.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- persistence capability classes: `PROCESS_LOCAL`, `DURABLE_SINGLE_HOST`, `SHARED_MULTI_HOST`
- each stateful runtime mechanism declares required persistence capability for its deployment topology
- STRICT/multi-host composition mechanically rejects process-local or otherwise insufficient stores
- domain persistence ports own concurrency semantics (CAS, lease/claim, transactional commit, required isolation)
- provider catalog supplies implementations that satisfy domain port guarantees — not merely minimal `RelationalStore` facades
- cross-link [`PROVIDER_BACKEND_ABSTRACTION`](../../audit_results/2026-08-18/PROVIDER_BACKEND_ABSTRACTION.md), [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) recovery-side blocks, Agent Distribution CAS target pattern ([`AGENT_DISTRIBUTION.md`](../../architecture/AGENT_DISTRIBUTION.md) §§23–25, §34)

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Platform Foundation coordinates topology qualification — does **not** become a persistence implementation domain.
- Redis distributed idempotency and other provider capabilities remain valid when they prove required semantics — stay provider-neutral.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-PERSISTENCE-CONCURRENCY-MULTIHOST-PERSIST.

### 6.1aw Phase HEP — Harness Evidence Pack (Band 2ae)

**Status:** HEP-1 **Done**; HEP-2 Trace Evidence Path **Done**; HEP-3 Evidence Posture / Scoreboard **Done**; EVID-CORE-FU-01 Selected Live Tier-0 Probes **Done** — `certify core` → `trace export` → `evidence live-core` → `evidence posture` / `evidence posture export`. EVID-CORE-FU-01 adds selected local no-network live Tier-0 probes with mock LLM/tools. It does not replace deterministic CORE certification and is not full runtime certification.
EVID-EVAL Eval Regression Evidence **Done** — `evidence eval` writes deterministic eval evidence artifacts and optionally enriches `evidence posture` via `EVAL_REGRESSION` when the report exists. It is not a new eval framework and does not run real LLM/provider evaluation.
EVID-COST Cost Evidence **Done** — `evidence cost` writes deterministic local cost evidence artifacts and optionally enriches `evidence posture` via `COST_EVIDENCE` when the report exists. It is not a billing engine, provider pricing system, cloud cost estimator, or real LLM usage meter.
Evidence platform proof path **Done** — the canonical local proof path is now documented in architecture and README: `certify core` → `trace export` → `evidence live-core` → `evidence eval` → `evidence cost` → `evidence posture` / `posture export`.
Evidence smoke audit: **Done** — canonical local proof path verified (see `HARNESS_EVIDENCE_PACK.md` § A2 closeout). README / onboarding update after smoke audit: **Done** — operator-facing proof path in README (see `HARNESS_EVIDENCE_PACK.md` § A3 closeout). Evidence artifact sanity checker / docs checker: **Done** — `scripts/maintenance/check_evidence_artifacts.py` validates expected artifacts and README proof-path references (see `HARNESS_EVIDENCE_PACK.md` § A4 closeout). External one-page harness narrative: **Done** — `docs/project/technical/guides/INTERGRAX_HARNESS_NARRATIVE.md` (see `HARNESS_EVIDENCE_PACK.md` § A5 closeout). **Strong ROI and polished/adopter-ready ROI are closed.** No immediate HEP evidence ROI task remains; deferred evidence waves remain deferred until explicitly prioritized. The detailed task count and roadmap live in `HARNESS_EVIDENCE_PACK.md` § Evidence ROI roadmap. **Boundary:** HEP remains §6.1 harness/platform evidence extension — not product/application work.
**Priority ladder:** **Band 2ae** — §6.1 extension (harness evidence / runtime proof / onboarding); runs **after** gate green; **not** product/application work
**Source:** External infrastructure audit (2026-06) + operator decision B → A → C

| Wave | Scope | IDs | Status |
|------|-------|-----|--------|
| HEP-0 | Mapping & contracts | EVID-CORE-01 … EVID-CORE-03 | **Done** (2026-06-21 C1) |
| HEP-1 | Core evidence path (`intergrax certify core`) | EVID-CORE-04 … EVID-CORE-06 | **Done** (2026-06-21 C2–C3) |
| HEP-2 | Trace evidence path (`intergrax trace show` · `intergrax trace export`) | EVID-TRACE-01 … EVID-TRACE-04 | **Done** (2026-06-21 C4–C6) |
| HEP-3 | Evidence posture / scoreboard (`intergrax evidence posture`) | EVID-POSTURE-01 … EVID-POSTURE-04 | **Done** (2026-06-21 C10) |

**Explicitly excluded:** product/application implementation work · W-ADAPT L4 replacement · duplicate doctor CI gates · Band 2ad (M.7 P7 — **Done**).

**Suggested PR order:** see [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) § Implementation waves.

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages`) | P0 |
| 2 | P-Ext.0.6 | EP discovery tests (all three groups) | P0 |
| 3 | P-Ext.1.6 | Integration EP test via fixture | P0 |
| 4 | P-Ext.1.10 | Tier-3 `integration_wiring` → `bootstrap_catalogs()` | P0 |
| 5 | P-Ext.2.9–2.11 | External tool example + unit + EP tests | P0 |
| 6 | P-Ext.3.6–3.8 | External skill example + unit + EP tests | P0 |
| 7 | P-Ext.0.7 | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | P1 |
| 8 | P-Ext.4.3, 4.5, 1.8 | Conflict policy + CI smoke (incl. integration counts) | P1 |
| 9 | P-Ext.1.5, 1.7, 5.5–5.6 | Slug/docs cleanup + author guide matrix | P2 |
| 10 | P-Ext.2.12, 3.9–3.11 | Tool/skill lazy bootstrap, scaffold plugin template, importer docs | P2 |
| 11 | P-Ext.1.3a, 1.4, 1.9, 1.11–1.12 | Typed resolve expansion, health API, integration wiring helper | P3 |
| 12 | P-Ext.5.1, 3.10, 3.12 | Scaffold CLI (all three catalogs) + harness `requires_skills` demo | P3 |

Full task register: [Appendix I](plan/satellites/PLATFORM_FOUNDATION_appendices.md).

**Out of scope for §6.1:** product/application implementation work and business backlogs. This document does not track product application roadmaps. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](.#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](.#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.3 End of platform plan

This plan is platform-only. Product applications, product agents, and business backlogs are intentionally out of scope and are not tracked here.

Platform Foundation tracks only harness/platform maintenance, platform gates, platform evidence, and reusable platform closeouts.
