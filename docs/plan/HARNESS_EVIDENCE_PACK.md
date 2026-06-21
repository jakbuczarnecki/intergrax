# Harness Evidence Pack — Implementation Plan

**Phase:** HEP (Harness Evidence Pack)  
**Band:** 2ae  
**Hub register:** [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1aw  
**Architecture (DX owns smoke/e2e evidence):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **Placement:** §6.1 harness infrastructure extension — **not** §6.3 product work.  
> **Naming:** Do **not** use `IDEAL-L4-EVIDENCE` — L4 in repo is W-ADAPT closed-loop semantics (`l4_runtime_evidence.py`). Do **not** reuse Band 2ad (M.7 P7 integrations — **Done**).

**Last updated:** 2026-06-21 — HEP-1 **Done** (EVID-CORE-01…06); HEP-2 Trace Evidence Path **Done** (EVID-TRACE-01…04; C4–C6); HEP-3 Evidence Posture / Scoreboard **Done** (EVID-POSTURE-01…04; C8–C10); EVID-CORE-FU-01 Selected Live Tier-0 Probes **Done** (EVID-CORE-FU-01A…E; C12–C16); **EVID-EVAL** Eval Regression Evidence **Done** (EVID-EVAL-01…05; N1–N5); **EVID-COST** Cost Evidence **Done** (C1–C5 Done; EVID-COST-01…05 Done); **A2** End-to-end evidence smoke audit **Done**.

---

## Cursor read scope (token budget)

- **Implement HEP-1 (closed):** § Mode I summary · § Certification semantics · § CORE levels · **EVID-CORE-*** rows.
- **Implement HEP-2 Trace:** § Mode I — HEP-2 · § Trace semantics · open **EVID-TRACE-*** rows only.
- **HEP-3 Posture (closed):** § HEP-3 closeout · § HEP-3 operator path · **EVID-POSTURE-*** rows.
- **EVID-CORE-FU-01 (closed):** § EVID-CORE-FU-01 closeout · **EVID-CORE-FU-01A…E** rows.
- **EVID-EVAL (closed):** § Mode I — EVID-EVAL · § EVID-EVAL closeout · **EVID-EVAL-01…05** rows · § Future waves.
- **EVID-COST (Done):** § Mode I — EVID-COST · **EVID-COST-01…05 Done** · § EVID-COST closeout · § Evidence ROI roadmap · § Future waves.
- **Skip** HEP-4+ unless implementing those waves.
- **Architecture:** DX read-scope block only — smoke/e2e evidence owns list.
- **Audit slice:** [`guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).

---

## Mode I summary (approved 2026-06-21)

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-0-1` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `improvement` |
| **Tier** | Tier-0 (`intergrax/cli/`, `intergrax/runtime/evidence/`) |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` (primary) · `PLATFORM_FOUNDATION` (hub) |

**Reason:** Mechanisms exist across IDEAL-* gates, doctor, trace, policy, eval, cost, W-OPS, and W-ADAPT — but no single **productized runtime certification path** (`intergrax certify core`). Value = packaging + executable proof + onboarding clarity.

**ADR:** No ADR needed for HEP-0 + HEP-1 (packaging existing contracts; no Nexus/UAEP/HarnessKernel semantic change).

---

## Problem statement

Intergrax is **architecture-heavy** (32/32 L3 scorecard, extensive CI gates) but **evidence-heavy adoption** is fragmented:

- Repo health (`pytest -m gate`, `intergrax doctor --ci`) ≠ live harness runtime proof.
- W-ADAPT L4 evidence measures adaptive utility improvement — not core E2E certification.
- Agent certification (`agent_certification.py`) is per-agent — not harness core.

**Goal:** One operator path — `intergrax certify core` — with JSON/Markdown report under `build/evidence/core_certification/`.

---

## Certification semantics (EVID-CORE-02)

| Mechanism | Question | Scope | Output |
|-----------|----------|-------|--------|
| `pytest -m gate` | Do tests pass? | Unit/integration matrix | pytest report |
| `intergrax doctor` / `--ci` | Is repo wiring healthy? | ~15 check scripts | PASS/FAIL per script |
| `phase_v_closeout_gate --enforce-l4` | Governance artifacts OK? | Capability graph + maturity evidence | closeout stdout |
| `check_l4_runtime_evidence.py` | W-ADAPT closed-loop OK? | 30-day utility / rollback on golden scenarios | `l4_runtime_evidence.json` |
| `check_mvp_promotion_gates.py` G0–G2 | MVP runnable baseline exists? | File existence checks | G0–G2 OK |
| **`intergrax certify core`** | Does the CORE evidence contract path execute end-to-end? | Deterministic mock evidence, no network/real LLM; not live HarnessKernel/Nexus/provider E2E | CORE-L* + JSON/Markdown certification report |

**CI placement (operator decision):** `certify core` is **operator-local** at HEP-1 launch. **Not** a default PR gate. Optional later: CORE-L1 as nightly or release preflight — not in scope for HEP-1.

---

## CORE certification levels

| Level | Scenarios | Use case |
|-------|-----------|----------|
| **CORE-L1** | `basic_run_completed`, `trace_persisted`, `tool_denied_by_policy`, `certification_report_emitted` | Quick smoke (~2 min) |
| **CORE-L2** | CORE-L1 + `high_risk_tool_hitl`, `budget_exceeded_handled`, `retry_executed`, `domain_signal_emitted` | Default `certify core` target |
| **CORE-L3** | CORE-L2 + `llm_error_classified`, `memory_read_write_recorded`, `rag_context_event_recorded`, `cost_report_generated` | Full evidence matrix |

CLI: `intergrax certify core --level L1|L2|L3` (default **L2**).

---

## Output artifacts (EVID-CORE-05)

```text
build/evidence/core_certification/
  report.json      # structured certification report
  report.md        # human-readable summary
```

Future waves may add sibling dirs (not HEP-1):

```text
build/evidence/trace/
build/evidence/eval/
build/evidence/cost/
build/evidence/posture/
```

---

## Non-goals

- **Not** §6.3 product work (K.1, K.2, LKW, Legal, new Tier-3 apps).
- **Not** W-ADAPT L4 replacement — keep `check_l4_runtime_evidence.py` separate semantics.
- **Not** duplicate doctor — do not add certify scenarios as new `check_*.py` entries in `doctor.py`.
- **Not** per-agent or business agent certification — see `agent_certification.py`, `business_agent_certification.py`.
- **Not** full Trace Explorer, Policy DSL, Extension SDK — deferred HEP-2+ (see § Future waves).
- **Not** new Tier-0 parallel subsystems — reuse event spine, reference harness, golden scenario IDs.

---

## External audit mapping (EVID-CORE-01)

Condensed matrix from external infrastructure audit → existing evidence → gap → proposed item.

| Audit # | Finding | Existing (plan / code) | Gap type | HEP item |
|---------|---------|------------------------|----------|----------|
| 1 | Core Certification Pack | IDEAL-27.3, §6.1 gates, `doctor.py`, `l4_runtime_evidence.py` | runtime proof + packaging | EVID-CORE-01…06 |
| 2 | Trace Explorer product | IDEAL-27.1 **Done**, `trace_explorer_routes.py` | UX depth | EVID-TRACE-* (HEP-2) |
| 3 | Policy-as-Product | IDEAL-5.x, `RuntimePolicyBundle` | mechanism + UX | EVID-POL-* (HEP-4) |
| 4 | Eval regression path | IDEAL-25.x, `check_eval_scenario_library.py` | packaging | EVID-EVAL-* (HEP-2) |
| 5 | Capability compatibility | capability graph gates, plugin catalog | UX | EVID-CAP-* (HEP-5) |
| 6 | Runtime replay | MVP-EVOL.3, `intergrax mvp replay` | product depth | EVID-REPLAY-* (HEP-5) |
| 7 | Cost intelligence | IDEAL-24.x, `assembly_cost.py` | UX | EVID-COST-* (HEP-2) |
| 8 | Context compiler | ContextProfile, CE-MAINT-02 | mechanism | EVID-CTX-* (HEP-5) |
| 9 | Extension SDK | P-Ext **Done**, scaffold | formal contract | EVID-EXT-* (HEP-5) |
| 10 | Security hardening pack | IDEAL-23.x, injection gates | packaging | EVID-SEC-* (HEP-5) |
| 11 | Attestation standard | ExecutionBoundaryExport (architecture) | mechanism | EVID-ATT-* (HEP-5) |
| 12 | Harness scoreboard | `harness_maturity_report.py` (static) | runtime proof | EVID-POSTURE-* (HEP-3) |

---

## Implementation register — Wave HEP-0 + HEP-1

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-CORE-01** | HEP-0 | P1 | **Done** | External audit → evidence matrix in plan | § External audit mapping (2026-06-21) |
| **EVID-CORE-02** | HEP-0 | P1 | **Done** (doc) | Certification spec: CI gate vs certify-core vs W-ADAPT L4 | § Certification semantics + § CORE levels delivered in docs; **code enums** (`CoreCertificationLevel`, etc.) ship with EVID-CORE-03 in C1 — not a separate reopen of this row |
| **EVID-CORE-03** | HEP-0 | P1 | **Done** | Runtime scenario contracts (12 scenarios) | `intergrax/runtime/evidence/` — Pydantic contracts; deterministic; no network; `validate_core_scenario_catalog()` |
| **EVID-CORE-04** | HEP-1 | P0 | **Done** | `intergrax certify core` CLI | `intergrax/cli/certify.py`; `--level L1\|L2\|L3`; exit non-zero on failure |
| **EVID-CORE-05** | HEP-1 | P1 | **Done** | JSON + Markdown certification report | `build/evidence/core_certification/report.json` + `report.md` |
| **EVID-CORE-06** | HEP-1 | P2 | **Done** | README / HARNESS_ENVIRONMENT evidence path | Quick start + HARNESS_ENVIRONMENT § Core certification; mock vs live narrative |

---

## Scenario contracts (EVID-CORE-03)

| Scenario ID | CORE min level | Reuse path |
|-------------|----------------|------------|
| `basic_run_completed` | L1 | `EchoAgent` + `AgentEngine` — `tests/integration/agents/test_agent_engine_uaep_echo.py` |
| `trace_persisted` | L1 | `SQLiteRunTraceStore` readback — IDEAL-27.1 trace store |
| `tool_denied_by_policy` | L1 | Policy deny — IDEAL-5.5 adversarial fixtures; `test_kernel_policy_pre_deny` |
| `certification_report_emitted` | L1 | EVID-CORE-05 report models (new) |
| `high_risk_tool_hitl` | L2 | IDEAL-11.3 HITL policy pattern |
| `budget_exceeded_handled` | L2 | IDEAL-24.5 quota hard-stop |
| `retry_executed` | L2 | REL retry path tests |
| `domain_signal_emitted` | L2 | `RuntimeEventBus` + event catalog |
| `llm_error_classified` | L3 | Error handling / REL classification |
| `memory_read_write_recorded` | L3 | MEM wiring events |
| `rag_context_event_recorded` | L3 | RAG/CTX `CONTEXT_ASSEMBLED` events |
| `cost_report_generated` | L3 | `assembly_cost.py` / budget policy |

Golden scenario IDs for echo baseline: `golden-echo`, `golden-policy`, `golden-routing` (from `l4_runtime_evidence.py` — reuse IDs only, not W-ADAPT utility semantics).

---

## Delivered modules (HEP-1 complete)

```text
intergrax/cli/certify.py
intergrax/runtime/evidence/
  __init__.py
  core_certification_spec.py    # EVID-CORE-02 code enums + CORE_LEVEL_SCENARIOS
  scenario_contracts.py         # EVID-CORE-03 contracts + validate_core_scenario_catalog()
  scenario_runner.py            # deterministic mock evidence runner
  certification_report.py       # JSON + Markdown report
tests/unit/runtime/evidence/
  test_core_certification_spec.py
  test_scenario_contracts.py
  test_scenario_runner.py
  test_certification_report.py
  test_certify_cli.py
```

**Docs (EVID-CORE-06):** README Quick start + `guides/HARNESS_ENVIRONMENT.md` § Core certification evidence path (HEP).

**Follow-up (EVID-CORE-FU-01):** **Done** — selected live Tier-0 probes alongside deterministic CORE certification; see § EVID-CORE-FU-01 closeout. Does not replace CORE-L* certification.

**Tier boundaries:** Tier-0 only — no `applications/` imports in evidence runner; use `reference_harness.py` / echo agent patterns.

**Optional preflight:** `certify core --with-doctor` may run doctor checks before scenarios — does not add scenarios to doctor.

---

## Implementation notes (C1 + C2)

| Artifact | Path |
|----------|------|
| CORE levels + surfaces | `intergrax/runtime/evidence/core_certification_spec.py` |
| Scenario contracts (12) | `intergrax/runtime/evidence/scenario_contracts.py` |
| Scenario runner (mock proof) | `intergrax/runtime/evidence/scenario_runner.py` |
| Report JSON/Markdown | `intergrax/runtime/evidence/certification_report.py` |
| CLI | `intergrax/cli/certify.py` · `intergrax certify core` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |
| Unit tests | `tests/unit/runtime/evidence/` (incl. `test_certify_cli.py`) |

**Verify:** `uv run pytest tests/unit/runtime/evidence -q` · `uv run intergrax certify core --level L2`

---

## Implementation waves (suggested PR order)

| PR | IDs | Scope | Status |
|----|-----|-------|--------|
| **PR1** | EVID-CORE-02 (code), EVID-CORE-03 | Spec enums + scenario contracts + unit tests | **Done** (C1) |
| **PR2** | EVID-CORE-04, EVID-CORE-05 | CLI + report generation | **Done** (C2) |
| **PR3** | EVID-CORE-06 | Docs: README, HARNESS_ENVIRONMENT evidence path | **Done** (C3) |

**Verify after PR2:** `uv run intergrax certify core --level L2` + `uv run pytest -m gate -q`

---

## Mode I — HEP-2 Trace Evidence Path

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-2-trace` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `developer_experience` · `observability` |
| **Tier** | Tier-0 / Tier-1 boundary, depending on existing trace package placement |
| **Domains** | `OBSERVABILITY` · `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` |

**Verdict:** `partial_overlap`

**Reason:** Trace routes, event spine, trace stores, and debug endpoints already exist; HEP-1 report path is delivered. The gap is a lightweight, canonical evidence timeline contract and operator path (`intergrax trace show`) — not a full Trace Explorer UI in HEP-2.

**ADR:** No ADR needed for HEP-2 Trace (packaging timeline over existing evidence artifacts; no Nexus/UAEP/HarnessKernel semantic change).

### Problem statement (HEP-2)

HEP-1 answers whether the core certification contract passed. HEP-2 must show **how** it passed.

The gap is not raw observability infrastructure. The gap is a small, canonical evidence timeline that can be rendered in CLI, Markdown, JSON, and later Trace Explorer.

**Strategic goal:** After `certify core`, the operator sees pass/fail and a report. After the trace evidence path, the operator sees a step-by-step timeline: certification started → scenario started → evidence emitted → policy/budget/HITL marker → scenario passed → report written → certification completed.

### Trace semantics

| Surface | Existing mechanism | Question | HEP-2 role |
|---------|-------------------|----------|------------|
| Runtime events | `RuntimeEvent` / `RuntimeEventBus` | What happened inside runtime? | Source input, not operator format |
| Trace store | `PersistedRunTraceEventStore` / persisted trace records | What was recorded? | Source input / future adapter |
| Trace Explorer routes | `trace_explorer_routes.py` · `/ops/trace/*` or debug trace routes | Can operators inspect traces in host? | Existing infra, not HEP-2 CLI contract |
| Debug formatters | `build_trace_payload` (`intergrax/debug/formatters.py`) | How are persisted traces shaped for UI? | Existing infra; HEP-2 does not replace |
| Replay bridge | `trace_replay_bridge.py` | How are trace events replayed? | Existing infra; not HEP-2 timeline contract |
| HEP-1 report | `build/evidence/core_certification/report.json` / `report.md` | Did certification pass? | Input artifact for timeline |
| Trace Evidence Path | planned `intergrax trace show` | What happened step-by-step? | Canonical operator timeline |

### Planned trace evidence artifacts

```text
build/evidence/trace/
  timeline.json
  timeline.md
```

HEP-2 may read `build/evidence/core_certification/report.json` as an input artifact.

### HEP-2 non-goals

- Not full Trace Explorer UI.
- Not replacement for existing trace routes (`trace_explorer_routes.py`, debug trace endpoints).
- Not new event bus or `RuntimeEventBus` wiring.
- Not new trace store (`PersistedRunTraceEventStore` remains separate).
- Not live provider tracing.
- Not W-ADAPT L4 evidence (`check_l4_runtime_evidence.py` semantics).
- Not `doctor` changes.
- Not a new default CI gate.
- Not changing `intergrax certify core` semantics.
- Not implementing EVID-CORE-FU-01 live runtime probes.

---

## Implementation register — Wave HEP-2 Trace Evidence Path

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-TRACE-01** | HEP-2 | P1 | **Done** | Trace timeline contract | Canonical timeline models for evidence runs; stable event kinds; artifact refs; no CLI yet |
| **EVID-TRACE-02** | HEP-2 | P1 | **Done** | Policy/budget/HITL facets | Timeline can represent policy decisions, budget markers, HITL markers, scenario lifecycle |
| **EVID-TRACE-03** | HEP-2 | P1 | **Done** | `intergrax trace show` CLI | Renders timeline from HEP-1 report/evidence artifacts; no UI |
| **EVID-TRACE-04** | HEP-2 | P2 | **Done** | Trace export JSON/Markdown | Writes timeline JSON/Markdown under `build/evidence/trace/` |

---

## Suggested implementation order — HEP-2

| Step | IDs | Scope |
|------|-----|-------|
| C4 | EVID-TRACE-01 | Timeline contracts only | **Done** |
| C5 | EVID-TRACE-02 | Facets for policy/budget/HITL/evidence | **Done** |
| C6 | EVID-TRACE-03/04 | CLI + export | **Done** |

---

## Implementation notes (C4 · EVID-TRACE-01)

| Artifact | Path |
|----------|------|
| Trace timeline contracts | `intergrax/runtime/evidence/trace_timeline_contracts.py` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_contracts.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_contracts.py -q`

---

## Implementation notes (C5 · EVID-TRACE-02)

| Artifact | Path |
|----------|------|
| Trace timeline facets | `intergrax/runtime/evidence/trace_timeline_facets.py` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_facets.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_facets.py -q`

---

## Implementation notes (C6 · EVID-TRACE-03/04)

| Artifact | Path |
|----------|------|
| Certification report adapter | `intergrax/runtime/evidence/trace_timeline_adapter.py` |
| Timeline CLI/export | `intergrax/runtime/evidence/trace_timeline_export.py` |
| CLI | `intergrax/cli/trace.py` · `intergrax trace show` · `intergrax trace export` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_adapter.py` · `test_trace_timeline_export.py` · `test_trace_cli.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Operator path:**

```bash
uv run intergrax certify core --level L2
uv run intergrax trace show
uv run intergrax trace export
```

**Operator semantics (C6 hotfix):**

- Timeline is **report-derived only** from `build/evidence/core_certification/report.json`.
- Evidence basis matches HEP-1: `deterministic_mock` — not live runtime trace, event bus, or trace store.
- `trace show` renders to stdout; `trace export` writes `build/evidence/trace/timeline.{json,md}`.
- Policy/budget/HITL facets are descriptive markers mapped from scenario evidence refs, not runtime probes.

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_adapter.py tests/unit/runtime/evidence/test_trace_timeline_export.py tests/unit/runtime/evidence/test_trace_cli.py -q`

---

## HEP-2 closeout (2026-06-21)

HEP-2 Trace Evidence Path: **Done**  
EVID-TRACE-01…04: **Done**

HEP-2 delivered the report-derived operator timeline path:

```bash
uv run intergrax certify core --level L2
uv run intergrax trace show
uv run intergrax trace export
```

`trace show` renders the timeline to stdout.
`trace export` writes `build/evidence/trace/timeline.json` and `timeline.md`.

The timeline is derived from `build/evidence/core_certification/report.json` and uses deterministic mock evidence. It is not live RuntimeEventBus, persisted trace store, or provider tracing.

The `certification_report_emitted` scenario is represented in the timeline by a `REPORT_WRITTEN` event kind — not a separate scenario lifecycle triplet.

---

## Mode I — HEP-3 Evidence Posture / Scoreboard

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-3-posture` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `developer_experience` · `evidence_packaging` |
| **Tier** | Tier-0 CLI / evidence package, with read-only references to existing gates |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` · `OBSERVABILITY` |

HEP-1 answers: did the core certification path pass?

HEP-2 answers: what happened step-by-step in the report-derived timeline?

HEP-3 should answer: what is the current evidence posture of the harness?

The goal is not to create new runtime proof. The goal is to aggregate existing evidence surfaces into a small operator-facing posture summary that clearly distinguishes:

- repo health,
- test gates,
- core certification,
- trace artifacts,
- deterministic mock evidence,
- deferred live runtime probes,
- W-ADAPT L4 as separate semantics.

The posture surface should make Intergrax easier to evaluate without reading the entire architecture. It should show what evidence exists, what it proves, what it does not prove, and which follow-ups are deferred.

### Evidence posture semantics

| Surface | Source | Question | HEP-3 role |
|---------|--------|----------|------------|
| Repo health | `intergrax doctor` | Is repository wiring healthy? | Input signal / optional status |
| Pytest gate | `pytest -m gate` | Does the unit/integration gate pass? | Input signal, not runtime proof |
| Core certification | `build/evidence/core_certification/report.json` | Did CORE-L* deterministic contract evidence pass? | Primary evidence source |
| Trace timeline | `build/evidence/trace/timeline.json` | Is report-derived timeline available? | Secondary evidence source |
| W-ADAPT L4 | `check_l4_runtime_evidence.py` | Is adaptive utility/rollback evidence healthy? | Separate semantics, not CORE posture |
| Live Tier-0 probes | EVID-CORE-FU-01 | Are selected runtime probes live? | Deferred / future input |

### HEP-3 non-goals

- Not a new CI gate.
- Not a replacement for `doctor`.
- Not a replacement for `pytest -m gate`.
- Not live runtime certification.
- Not W-ADAPT L4 evidence.
- Not Trace Explorer UI.
- Not policy DSL.
- Not cost engine.
- Not eval regression runner.
- Not EVID-CORE-FU-01 live Tier-0 probes.
- Not a production readiness certification for business applications.

---

## Implementation register — Wave HEP-3 Evidence Posture / Scoreboard

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-POSTURE-01** | HEP-3 | P1 | **Done** | Evidence posture contract | Pydantic/read-model contract for posture summary; no CLI yet |
| **EVID-POSTURE-02** | HEP-3 | P1 | **Done** | Evidence posture collector | Reads existing artifacts and optional command outputs; no new runtime proof |
| **EVID-POSTURE-03** | HEP-3 | P1 | **Done** | `intergrax evidence posture` CLI | Renders current evidence posture to stdout |
| **EVID-POSTURE-04** | HEP-3 | P2 | **Done** | Posture export JSON/Markdown | Writes posture artifacts under `build/evidence/posture/` |

---

## Suggested implementation order — HEP-3

| Step | IDs | Scope |
|------|-----|-------|
| C8 | EVID-POSTURE-01 | Posture contracts only | **Done** |
| C9 | EVID-POSTURE-02 | Artifact collector / read-only aggregation | **Done** |
| C10 | EVID-POSTURE-03/04 | CLI + export | **Done** |

---

## Implementation notes (C8 · EVID-POSTURE-01)

| Artifact | Path |
|----------|------|
| Evidence posture contracts | `intergrax/runtime/evidence/evidence_posture_contracts.py` |
| Unit tests | `tests/unit/runtime/evidence/test_evidence_posture_contracts.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_evidence_posture_contracts.py -q`

---

## Implementation notes (C9 · EVID-POSTURE-02)

| Artifact | Path |
|----------|------|
| Evidence posture collector | `intergrax/runtime/evidence/evidence_posture_collector.py` |
| Unit tests | `tests/unit/runtime/evidence/test_evidence_posture_collector.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_evidence_posture_collector.py -q`

---

## Implementation notes (C10 · EVID-POSTURE-03/04)

| Artifact | Path |
|----------|------|
| Posture CLI/export | `intergrax/runtime/evidence/evidence_posture_export.py` |
| CLI | `intergrax/cli/evidence.py` · `intergrax evidence posture` · `intergrax evidence posture export` |
| Unit tests | `tests/unit/runtime/evidence/test_evidence_posture_export.py` · `test_evidence_cli.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_evidence_posture_export.py tests/unit/runtime/evidence/test_evidence_cli.py -q`

---

## HEP-3 closeout (2026-06-21)

HEP-3 Evidence Posture / Scoreboard: **Done**  
EVID-POSTURE-01…04: **Done**

**Operator path:**

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

**Operator semantics:**

- Posture reads existing artifacts (`report.json`, `timeline.json`); it does not run doctor, pytest, certify core, or trace export.
- Posture is not live runtime proof — `LIVE_TIER0_PROBES` remains **DEFERRED**; `W-ADAPT L4` remains **SEPARATE**.
- `evidence posture` renders to stdout; `evidence posture export` writes `build/evidence/posture/posture.{json,md}`.

---

## Posture artifacts

```text
build/evidence/posture/
  posture.json
  posture.md
```

HEP-3 may read:

```text
build/evidence/core_certification/report.json
build/evidence/trace/timeline.json
```

It should not require live providers, network access, runtime event bus, or trace store.

---

## HEP-3 operator path

Full operator path (preflight → evidence artifacts → read-only posture):

```bash
uv run intergrax doctor
uv run pytest -m gate -q
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

`intergrax doctor` and `pytest -m gate` are **optional preflight** surfaces — repo health and test-gate signals, not required to run posture.

`certify core` and `trace export` produce the artifacts posture reads (`report.json`, `timeline.json`).

`intergrax evidence posture` and `intergrax evidence posture export` are read-only over existing evidence artifacts. They do not execute `doctor`, `pytest`, `certify core`, `trace export`, live runtime probes, RuntimeEventBus, trace store, or provider calls.

Expected posture summary should clearly state:

- repo health: available / unknown / failed,
- gate status: available / unknown / failed,
- core certification: available / missing / failed,
- core level: CORE-L1 / CORE-L2 / CORE-L3 / unknown,
- trace timeline: available / missing,
- evidence basis: deterministic_mock,
- live Tier-0 probes: deferred,
- W-ADAPT L4: separate,
- overall posture: onboarding-ready / partial / missing-evidence.

---

## Follow-up Mode I — EVID-CORE-FU-01 Selected Live Tier-0 Probes

| Field | Value |
|-------|-------|
| **Idea label** | `selected-live-tier0-probes` |
| **Verdict** | `approved_for_small_follow_up` (Mode I approved — small follow-up only) |
| **Type** | `harness_capability` · `improvement` · `evidence_packaging` |
| **Tier** | Tier-0 (`intergrax/runtime/evidence/`, future live probe runner) |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` |

**Context (do not conflate):**

| Path | What it proves |
|------|----------------|
| HEP-1 `certify core` | Deterministic mock contract evidence (CORE-L*) |
| HEP-2 `trace show` / `trace export` | Report-derived deterministic mock timeline |
| HEP-3 `evidence posture` | Read-only aggregation over existing evidence artifacts |
| **EVID-CORE-FU-01** | Future **selected live Tier-0 probes** — small controlled follow-up only |

### Problem statement

HEP-1/2/3 provide deterministic mock evidence, report-derived timeline, and read-only posture. The remaining explicit gap is `LIVE_TIER0_PROBES: DEFERRED`. EVID-CORE-FU-01 defines a small, controlled live-probe follow-up that checks whether selected Tier-0 runtime paths can execute without network, providers, or real LLM calls.

### Target outcome

After the full EVID-CORE-FU-01 implementation, `LIVE_TIER0_PROBES` is no longer only **DEFERRED** in posture when a live probe report exists. Posture can show **PASSED** / **FAILED** / **UNKNOWN** for selected probes, while still clearly stating that this is selected live Tier-0 evidence, not full runtime certification.

### First selected probes (C13–C15 scope)

Only these three probes are approved for the first wave:

| Probe ID | Rationale |
|----------|-----------|
| `basic_run_completed_live` | Proves a minimal runtime path can complete. |
| `trace_persisted_live` | Proves execution leaves an inspectable evidence/trace artifact. |
| `tool_denied_by_policy_live` | Proves policy denial works through the live probe path. |

**Future candidates only** (not C13–C15 scope): `retry_executed_live`, `budget_exceeded_handled_live`, `hitl_required_live`, `memory_read_write_live`, `rag_context_recorded_live`.

### Non-goals

- Not full live runtime certification.
- Not production readiness certification.
- Not business application certification.
- Not provider proof.
- Not real LLM proof.
- Not network execution.
- Not external API execution.
- Not replacement for CORE-L1/L2/L3 deterministic certification.
- Not replacement for `pytest -m gate`.
- Not replacement for `intergrax doctor`.
- Not W-ADAPT L4.
- Not EVID-EVAL.
- Not EVID-COST.
- Not EVID-POL.
- Not replay system.
- Not Trace Explorer UI.
- Not policy DSL.

### Runtime constraints

- No network.
- No provider calls.
- No real LLM calls.
- Mock LLM only.
- Mock tools only.
- Local or in-memory stores only.
- Deterministic execution where possible.
- No dependency on external credentials.
- No dependency on user environment outside the repo.

### Evidence semantics

| Field | Value |
|-------|-------|
| Evidence basis | `LIVE_RUNTIME` for the selected probe path only |
| LLM basis | `MOCK_LLM` |
| Execution boundary | `LOCAL_NO_NETWORK` |
| Scope | `SELECTED_TIER0_PROBES` |

`LIVE_RUNTIME` here means the selected probe path exercises real Tier-0 runtime mechanisms locally — **not** full production runtime, **not** a real provider, **not** real LLM proof.

### Planned artifacts

```text
build/evidence/live_core_probes/
  live_core_report.json
  live_core_report.md
```

### Planned command (not implemented in C12)

```bash
uv run intergrax evidence live-core
```

This command was implemented in C15.

### Posture integration semantics

After EVID-CORE-FU-01 implementation, `intergrax evidence posture` should read the live core probe report when present and map `LIVE_TIER0_PROBES` from **DEFERRED** to **PASSED** / **FAILED** / **UNKNOWN**. If the live probe report is missing, posture should continue to show `LIVE_TIER0_PROBES` as **DEFERRED**.

Posture must still label this surface as selected live Tier-0 evidence — not full runtime certification, not production certification, not provider-validated proof.

### Suggested implementation order

| Step | Scope | Status |
|------|-------|--------|
| C12 | Mode I / planning docs | **Done** |
| C13 | Live probe contracts | **Done** |
| C14 | Live probe runner | **Done** |
| C15 | CLI + report + posture integration | **Done** |
| C16 | Docs closeout / cleanup | **Done** |

---

## Implementation register — EVID-CORE-FU-01 Selected Live Tier-0 Probes

| ID | Priority | Status | Deliverable | Acceptance criteria |
|----|----------|--------|-------------|---------------------|
| **EVID-CORE-FU-01A** | P1 | **Done** | Live probe contracts | Contracts for selected live Tier-0 probe result/report; no runner yet |
| **EVID-CORE-FU-01B** | P1 | **Done** | Live probe runner | Executes selected probes locally with mock LLM/tools; no network/provider calls |
| **EVID-CORE-FU-01C** | P1 | **Done** | `intergrax evidence live-core` CLI | Writes live core probe report JSON/Markdown |
| **EVID-CORE-FU-01D** | P1 | **Done** | Posture integration | Posture maps live probe report to LIVE_TIER0_PROBES PASSED/FAILED/UNKNOWN |
| **EVID-CORE-FU-01E** | P2 | **Done** | Closeout docs | Documents final operator path and semantics |

**Implementation note (C13):** C13 added live core probe contracts only; no runner, CLI, export, runtime execution, provider calls, or posture integration.

**Implementation note (C14):** C14 added a controlled local live-core probe runner. It produces an in-memory LiveCoreProbeReport only. No CLI, no export, no posture integration, no provider calls, no network, no real LLM.

**Implementation note (C15):** C15 added `intergrax evidence live-core`, live core probe report JSON/Markdown export, and posture integration. When `live_core_report.json` exists, posture maps `LIVE_TIER0_PROBES` from DEFERRED to PASSED/FAILED/UNKNOWN based on selected live probe results. This remains selected local no-network evidence with mock LLM/tools, not full runtime certification.

**Implementation note (C16):** C16 closeout docs — final operator path and semantics documented; no code changes.

---

## EVID-CORE-FU-01 closeout

EVID-CORE-FU-01 Selected Live Tier-0 Probes: **Done**

Delivered:

- live probe contracts,
- controlled local live-core probe runner,
- `intergrax evidence live-core`,
- `live_core_report.json` / `live_core_report.md`,
- posture integration for `LIVE_TIER0_PROBES`.

### Final operator path

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence live-core
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

### Semantics

- `certify core` remains deterministic mock contract evidence.
- `trace export` remains report-derived deterministic mock timeline.
- `evidence live-core` runs selected local Tier-0 probes with mock LLM/tools.
- `evidence live-core` does not use network, providers, real LLM calls, event bus, trace store, or business applications.
- `evidence posture` maps `LIVE_TIER0_PROBES` from `DEFERRED` to `PASSED` / `FAILED` / `UNKNOWN` when `live_core_report.json` exists.
- This is selected live Tier-0 evidence only; it is not full runtime certification and does not replace CORE-L1/L2/L3 deterministic certification.

---

## Mode I — EVID-EVAL Eval Regression Evidence

| Field | Value |
|-------|-------|
| **Idea label** | `eval-regression-evidence` |
| **Verdict** | `approved_for_small_hep_wave` |
| **Type** | `harness_capability` · `evidence_packaging` · `developer_experience` |
| **Tier** | Tier-0 evidence / eval packaging |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` |

**Context (do not conflate):**

| Path | What it proves |
|------|----------------|
| HEP-1 `certify core` | Deterministic mock contract evidence (CORE-L*) |
| HEP-2 `trace show` / `trace export` | Report-derived deterministic mock timeline |
| HEP-3 `evidence posture` | Read-only aggregation over existing evidence artifacts |
| EVID-CORE-FU-01 `evidence live-core` | Selected local no-network live Tier-0 probes |
| **EVID-EVAL** | **Done** — eval regression evidence packaging; read-only wrapper over existing eval checks (`check_eval_scenario_library.py`, IDEAL-25.x); not a new eval framework |

### Problem statement

HEP-1/2/3 and EVID-CORE-FU-01 prove core certification, trace timeline, posture aggregation, and selected live Tier-0 probes. EVID-EVAL closes the eval regression evidence gap: operators get a stable artifact showing whether the eval scenario library / regression surface is healthy without turning this into a new eval framework.

### Target outcome

After EVID-EVAL, Intergrax should provide a small operator-facing eval evidence path that packages existing eval checks into deterministic JSON/Markdown artifacts under `build/evidence/eval/`.

### Non-goals

- Not a new eval framework.
- Not a benchmark suite.
- Not model quality certification.
- Not provider comparison.
- Not real LLM evaluation.
- Not network execution.
- Not CI policy replacement.
- Not full product analytics.
- Not replacement for `pytest -m gate`.
- Not replacement for `intergrax certify core`.

### Artifacts

```text
build/evidence/eval/
  report.json
  report.md
```

### Command

```bash
uv run intergrax evidence eval
```

### Suggested implementation order

| Step | Scope | Status |
|------|-------|--------|
| N1 | Mode I / planning docs | **Done** |
| N2 | Eval evidence contracts | **Done** |
| N3 | Eval evidence runner | **Done** |
| N4 | CLI + report export | **Done** |
| N5 | Optional posture integration + closeout docs | **Done** |

---

## Implementation register — EVID-EVAL Eval Regression Evidence

| ID | Priority | Status | Deliverable | Acceptance criteria |
|----|----------|--------|-------------|---------------------|
| **EVID-EVAL-01** | P1 | Done | Eval evidence contracts | Report/result contracts for eval scenario library evidence; no runner yet |
| **EVID-EVAL-02** | P1 | Done | Eval evidence runner | Read-only wrapper over existing eval check mechanism; no new eval framework |
| **EVID-EVAL-03** | P1 | Done | `intergrax evidence eval` CLI + export | Writes `build/evidence/eval/report.json` and `report.md` |
| **EVID-EVAL-04** | P2 | Done | Posture integration | Optional read-only signal in evidence posture, if report exists |
| **EVID-EVAL-05** | P2 | Done | Closeout docs | Final operator path and semantics |

**Implementation note (N2):** N2 added eval evidence contracts only; no runner, CLI, export, posture integration, real LLM evaluation, provider comparison, or new eval framework.

**Implementation note (N3):** N3 added a read-only eval evidence runner that produces an in-memory EvalEvidenceReport from existing local eval/check availability. It does not execute real LLM evaluation, compare providers, use network, write artifacts, add CLI, add posture integration, or create a new eval framework.

**Implementation note (N4):** N4 added `intergrax evidence eval`, eval evidence JSON/Markdown export, and operator CLI rendering. It does not add posture integration, execute real LLM evaluation, compare providers, use network, execute the eval source script, or create a new eval framework.

**Implementation note (N5):** N5 added optional read-only posture integration for eval evidence and closed EVID-EVAL docs. Missing eval report does not fail posture; existing eval report maps to EVAL_REGRESSION PASSED/FAILED/UNKNOWN.

---

## EVID-EVAL closeout

EVID-EVAL Eval Regression Evidence: **Done**

Delivered:

- eval evidence contracts,
- read-only eval evidence runner,
- `intergrax evidence eval`,
- `build/evidence/eval/report.json`,
- `build/evidence/eval/report.md`,
- optional posture integration via `EVAL_REGRESSION`.

### Final operator path

```bash
uv run intergrax evidence eval
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

### Semantics

* `evidence eval` packages existing local eval/check availability into deterministic evidence artifacts.
* `evidence eval` does not execute real LLM evaluation.
* `evidence eval` does not compare providers.
* `evidence eval` does not use network.
* `evidence eval` does not create a new eval framework.
* `evidence posture` includes `EVAL_REGRESSION` only when `build/evidence/eval/report.json` exists.
* Missing eval evidence report does not fail posture and does not make posture missing.

---

## Mode I — EVID-COST Cost Evidence

| Field | Value |
|-------|-------|
| **Idea label** | `cost-evidence` |
| **Verdict** | `approved_for_small_hep_wave` |
| **Type** | `harness_capability` · `evidence_packaging` · `developer_experience` |
| **Tier** | Tier-0 evidence / cost packaging |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` · `OBSERVABILITY` |

**Context (do not conflate):**

| Path | What it proves |
|------|----------------|
| HEP-1 `certify core` | Deterministic mock contract evidence |
| HEP-2 `trace show` / `trace export` | Report-derived deterministic trace evidence |
| HEP-3 `evidence posture` | Read-only aggregation over existing evidence artifacts |
| EVID-CORE-FU-01 `evidence live-core` | Selected local no-network live Tier-0 probes |
| EVID-EVAL `evidence eval` | Eval regression evidence packaging |
| **EVID-COST** | Future cost evidence packaging over existing local budget/cost/trace information; not a billing engine |

### Problem statement

The current evidence path proves core certification, trace evidence, posture aggregation, selected live Tier-0 probes, and eval regression evidence. The remaining HEP-2 ROI gap is cost evidence: operators need a deterministic artifact that summarizes cost/budget posture from existing local evidence surfaces without introducing a billing engine, provider pricing model, or new runtime accounting framework.

### Target outcome

After EVID-COST, Intergrax should provide a small operator-facing cost evidence path that packages existing local budget/cost/trace information into deterministic JSON/Markdown artifacts under `build/evidence/cost/`.

### Non-goals

- Not a billing engine.
- Not provider pricing.
- Not cloud cost estimation.
- Not token accounting for real providers.
- Not real LLM usage metering.
- Not financial reporting.
- Not a cost dashboard.
- Not a new budget policy framework.
- Not replacement for trace budget facets.
- Not replacement for `intergrax evidence posture`.
- Not network execution.
- Not provider calls.

### Planned artifacts

```text
build/evidence/cost/
  report.json
  report.md
```

### Planned command

```bash
uv run intergrax evidence cost
```

This command is planned for implementation after C1.

### Suggested implementation order

| Step | Scope | Status |
|------|-------|--------|
| C1 | Mode I / planning docs | **Done** |
| C2 | Cost evidence contracts | **Done** |
| C3 | Cost evidence runner / collector | **Done** |
| C4 | CLI + report export | **Done** |
| C5 | Optional posture integration + closeout docs | **Done** |

**Implementation note (C1):** C1 approved EVID-COST as the next small HEP evidence packaging wave. Scope is read-only packaging over existing local budget/cost/trace information (`TraceBudgetFacet`, trace timeline budget facets, certification cost/budget signals). No contracts, runner, CLI, export, posture integration, billing engine, provider pricing, or real LLM usage metering. EVID-COST-01…05 remain Planned.

**Implementation note (C2):** C2 added cost evidence contracts only; no runner, CLI, export, posture integration, provider pricing, billing engine, real LLM usage metering, network execution, or new budget policy framework.

**Implementation note (C3):** C3 added a read-only cost evidence runner / collector that produces an in-memory CostEvidenceReport from existing local trace budget facets. It does not write artifacts, add CLI, add posture integration, compute provider pricing, implement billing, meter real LLM usage, use network, or create a new budget policy framework.

**Implementation note (C4):** C4 added `intergrax evidence cost`, cost evidence JSON/Markdown export, and operator CLI rendering. It does not add posture integration, compute provider pricing, implement billing, meter real LLM usage, use network, execute trace export, or create a new budget policy framework.

**Implementation note (C5):** C5 added optional read-only posture integration for cost evidence and closed EVID-COST docs. Missing cost report does not fail posture; existing cost report maps to COST_EVIDENCE PASSED/FAILED/UNKNOWN. It does not compute provider pricing, implement billing, meter real LLM usage, use network, execute trace export, or create a new budget policy framework.

---

## Implementation register — EVID-COST Cost Evidence

| ID | Priority | Status | Deliverable | Acceptance criteria |
|----|----------|--------|-------------|---------------------|
| **EVID-COST-01** | P1 | Done | Cost evidence contracts | Report/result contracts for local cost evidence; no runner yet |
| **EVID-COST-02** | P1 | Done | Cost evidence runner / collector | Read-only packaging over existing local budget/cost/trace information; no billing engine |
| **EVID-COST-03** | P1 | Done | `intergrax evidence cost` CLI + export | Writes `build/evidence/cost/report.json` and `report.md` |
| **EVID-COST-04** | P2 | Done | Posture integration | Optional read-only `COST_EVIDENCE` signal in evidence posture when report exists |
| **EVID-COST-05** | P2 | Done | Closeout docs | Final operator path and semantics |

---

## EVID-COST closeout

EVID-COST Cost Evidence: **Done**

Delivered:

- cost evidence contracts,
- read-only cost evidence runner / collector,
- `intergrax evidence cost`,
- `build/evidence/cost/report.json`,
- `build/evidence/cost/report.md`,
- optional posture integration via `COST_EVIDENCE`.

### Final operator path

```bash
uv run intergrax evidence cost
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

### Semantics

* `evidence cost` packages existing local budget/cost/trace information into deterministic evidence artifacts.
* `evidence cost` does not compute provider pricing.
* `evidence cost` does not implement billing.
* `evidence cost` does not estimate cloud costs.
* `evidence cost` does not meter real LLM usage.
* `evidence cost` does not use network.
* `evidence cost` does not create a new budget policy framework.
* `evidence posture` includes `COST_EVIDENCE` only when `build/evidence/cost/report.json` exists.
* Missing cost evidence report does not fail posture and does not make posture missing.

---

## Completed waves

| Wave | IDs | Status |
|------|-----|--------|
| HEP-1 Core Certification | EVID-CORE-01 … EVID-CORE-06 | **Done** — `certify core` report path delivered |
| HEP-2 Trace Evidence Path | EVID-TRACE-01 … EVID-TRACE-04 | **Done** — `trace show` / `trace export` report-derived timeline delivered |
| HEP-3 Evidence Posture / Scoreboard | EVID-POSTURE-01 … EVID-POSTURE-04 | **Done** — `evidence posture` / `evidence posture export` delivered |
| EVID-CORE-FU-01 Selected Live Tier-0 Probes | EVID-CORE-FU-01A … EVID-CORE-FU-01E | **Done** — `evidence live-core` + posture integration delivered |
| EVID-EVAL Eval Regression Evidence | EVID-EVAL-01 … EVID-EVAL-05 | **Done** — `evidence eval` + optional posture `EVAL_REGRESSION` delivered |
| EVID-COST Cost Evidence | EVID-COST-01 … EVID-COST-05 | **Done** — `evidence cost` + optional posture `COST_EVIDENCE` delivered |

---

## Evidence ROI roadmap

This roadmap tracks the remaining highest-ROI HEP / evidence work after the completed evidence waves. It exists to keep implementation planning in repo documentation rather than in external chat context.

### Current completed evidence surface

| Area | Status | Operator value |
|------|--------|----------------|
| HEP-1 Core Certification | **Done** | `intergrax certify core` produces deterministic CORE-L* evidence |
| HEP-2 Trace Evidence Path | **Done** | `intergrax trace show` / `trace export` produce report-derived trace evidence |
| HEP-3 Evidence Posture / Scoreboard | **Done** | `intergrax evidence posture` aggregates evidence artifacts |
| EVID-CORE-FU-01 Selected Live Tier-0 Probes | **Done** | `intergrax evidence live-core` adds selected local no-network live Tier-0 probes |
| EVID-EVAL Eval Regression Evidence | **Done** | `intergrax evidence eval` packages eval regression evidence and optionally enriches posture |
| EVID-COST Cost Evidence | **Done** | `intergrax evidence cost` packages cost evidence and optionally enriches posture via `COST_EVIDENCE` |

### Minimal remaining ROI

Minimal ROI is **closed**. Cost evidence and the operator-facing evidence path closeout are complete.

| Order | Work item | Expected task count | Status | Purpose |
|-------|-----------|---------------------|--------|---------|
| 1 | EVID-COST Mode I / spec | 1 | **Done** | Approve the cost evidence wave and define scope/non-goals |
| 2 | EVID-COST contracts | 1 | **Done** | Define report/result contracts for cost evidence |
| 3 | EVID-COST runner / collector | 1 | **Done** | Package existing cost/budget information into evidence results |
| 4 | EVID-COST CLI + JSON/Markdown export | 1 | **Done** | Add `intergrax evidence cost` and write artifacts under `build/evidence/cost/` |
| 5 | EVID-COST posture integration + closeout | 1 | **Done** | Add optional posture signal and close EVID-COST docs |
| 6 | Final evidence operator path closeout | 1 | **Done** | Document one canonical evidence onboarding flow |

Estimated remaining tasks for minimal ROI: **0**.

**Progress note:** The final evidence operator path closeout is Done. Minimal ROI is now closed. A2 end-to-end evidence smoke audit is Done. A3 README / onboarding update after smoke audit is Done. Strong ROI is closed. Remaining estimated tasks: polished/adopter-ready ROI **2**.

### Strong ROI / onboarding-ready evidence path

After minimal ROI, two additional tasks make the evidence path stronger for external developers and early adopters.

| Order | Work item | Expected task count | Status | Purpose |
|-------|-----------|---------------------|--------|---------|
| 7 | End-to-end evidence smoke audit (A2) | 1 | **Done** | Verify the full local evidence command sequence and artifact consistency |
| 8 | README / onboarding update after smoke audit (A3) | 1 | **Done** | Full onboarding documentation after end-to-end evidence smoke audit |

Estimated remaining tasks for strong ROI: **0**.

### Optional presentation/adopter polish

These are useful but not required for the core ROI path.

| Order | Work item | Expected task count | Status | Purpose |
|-------|-----------|---------------------|--------|---------|
| 9 | Evidence artifact sanity checker / docs checker | 1 | Optional | Validate expected evidence artifacts and docs consistency |
| 10 | External one-page harness narrative | 1 | Optional | Explain why Intergrax is a harness, not just an agent framework |

Estimated remaining tasks for polished adopter-ready ROI: **2 total**.

### Deferred from highest-ROI path

The following waves remain valuable, but are not part of the immediate highest-ROI evidence onboarding path:

| Candidate | Reason deferred |
|-----------|-----------------|
| EVID-POL | High value, but heavier mechanism/UX boundary than cost evidence |
| EVID-CAP | Useful for capability graph UX, but not required for the first evidence onboarding path |
| EVID-REPLAY | Higher risk of product-depth overbuild |
| EVID-CTX | Depends on broader context engineering architecture |
| EVID-EXT | Useful for extension SDK maturity, but lower immediate evidence ROI |
| EVID-SEC | Important enterprise value, but not next unless security proof becomes the priority |
| EVID-ATT | Too early; depends on deeper attestation architecture |

### Recommended next wave

Recommended next wave: **A4 — Evidence artifact sanity checker / docs checker** — A3 README / onboarding update after smoke audit **Done**.

Reason:

Minimal ROI and strong ROI are closed. A2 verified the canonical local evidence command sequence and artifact consistency. A3 updated README and plan onboarding so early adopters can run and interpret the proof path. The highest-ROI remaining evidence work is optional artifact/docs consistency validation. See § Evidence ROI roadmap, § A2 closeout, and § A3 closeout.

---

## A3 — README / onboarding update after smoke audit closeout

A3 README / onboarding update after smoke audit: **Done**

### Documentation updated

| File | Change |
|------|--------|
| `README.md` | Strengthened § Proof of platform — framing, canonical commands, artifact table, prove/does-not-prove boundaries, next steps |
| `HARNESS_EVIDENCE_PACK.md` | A3 closeout; ROI counters; recommended next wave → A4 |
| `PLATFORM_FOUNDATION.md` | §6.1aw — strong ROI closed; polished/adopter-ready path |

### Operator onboarding outcome

Early developers/adopters can answer from README alone: what the proof path is, why to run it, exact commands, expected artifacts, what it proves, what it explicitly does not prove, and where to go next (`posture.md` first, then individual reports).

**Implementation note (A3):** Docs-only. No code, tests, smoke audit rerun, or provider/network execution. Canonical command sequence unchanged from A2.

---

## A2 — End-to-end evidence smoke audit closeout

A2 End-to-end evidence smoke audit: **Done**

### Command sequence executed

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence live-core
uv run intergrax evidence eval
uv run intergrax evidence cost
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

### Artifacts verified

| Surface | Artifacts |
|---------|-----------|
| Core certification | `build/evidence/core_certification/report.json`, `report.md` |
| Trace evidence | `build/evidence/trace/timeline.json`, `timeline.md` |
| Live Tier-0 probes | `build/evidence/live_core_probes/live_core_report.json`, `live_core_report.md` |
| Eval evidence | `build/evidence/eval/report.json`, `report.md` |
| Cost evidence | `build/evidence/cost/report.json`, `report.md` |
| Evidence posture | `build/evidence/posture/posture.json`, `posture.md` |

### Posture surfaces verified

Posture `posture.json` includes: `CORE_CERTIFICATION`, `TRACE_TIMELINE` (trace surface), `LIVE_TIER0_PROBES`, `EVAL_REGRESSION`, `COST_EVIDENCE`.

**Implementation note (A2):** Local smoke audit executed the canonical proof path; all expected artifacts exist and posture reflects available evidence surfaces. No code changes. No network, provider calls, real LLM evaluation, billing, provider pricing, or cloud cost estimation.

---

## Future waves

| Wave | IDs | Audit priority |
|------|-----|----------------|
| HEP-2 / EVID-COST | **EVID-COST** | **Done** — cost evidence packaging + optional posture `COST_EVIDENCE` |
| HEP-4 | EVID-POL | External audit #3 — not Mode I approved yet |
| HEP-5 | EVID-CAP, EVID-REPLAY, EVID-CTX, EVID-EXT, EVID-SEC, EVID-ATT | External audit #5–11 — not Mode I approved yet |

Register future rows in this file when Mode I approves each wave.

---

## Definition of Done (HEP-1)

1. **Contract** — Pydantic report + scenario contracts in `intergrax/runtime/evidence/`
2. **Trace** — scenarios assert `RuntimeEvent` / trace persistence where applicable
3. **Test** — integration tests deterministic, no network
4. **Documentation** — this plan + EVID-CORE-06 onboarding path
5. **No regression** — `pytest -m gate` green; certify-core does not break doctor
6. **Reuse Tier-0** — extend existing harness paths; no parallel certification stack
