# Harness Evidence Pack — Implementation Plan

**Phase:** HEP (Harness Evidence Pack)  
**Band:** 2ae  
**Hub register:** [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1aw  
**Architecture (DX owns smoke/e2e evidence):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **Placement:** §6.1 harness infrastructure extension — **not** §6.3 product work.  
> **Naming:** Do **not** use `IDEAL-L4-EVIDENCE` — L4 in repo is W-ADAPT closed-loop semantics (`l4_runtime_evidence.py`). Do **not** reuse Band 2ad (M.7 P7 integrations — **Done**).

**Last updated:** 2026-06-21 — Mode I approved (operator); doc apply only; implementation deferred.

---

## Cursor read scope (token budget)

- **Implement default:** This file § Mode I summary · § Certification semantics · § CORE levels · open **EVID-CORE-*** rows only.
- **Skip** future waves (HEP-2+) unless implementing that wave.
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
| **`intergrax certify core`** | Does **live harness** pass controlled E2E scenarios? | Mock providers, deterministic, no network | CORE-L* + certification report |

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
| **EVID-CORE-04** | HEP-1 | P0 | **Planned** | `intergrax certify core` CLI | `intergrax/cli/certify.py`; `--level L1\|L2\|L3`; exit non-zero on failure; registered in `cli/main.py` |
| **EVID-CORE-05** | HEP-1 | P1 | **Planned** | JSON + Markdown certification report | Writes `build/evidence/core_certification/report.json` + `report.md`; per-scenario PASS/FAIL + evidence refs |
| **EVID-CORE-06** | HEP-1 | P2 | **Planned** | README / HARNESS_ENVIRONMENT evidence path | 30-min onboarding: doctor → certify core → trace; linked from hub + `guides/HARNESS_ENVIRONMENT.md` |

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

## Proposed modules (implementation — C1 partial)

**Delivered (C1 · EVID-CORE-03):**

```text
intergrax/runtime/evidence/
  __init__.py
  core_certification_spec.py    # EVID-CORE-02 code enums + CORE_LEVEL_SCENARIOS
  scenario_contracts.py         # EVID-CORE-03 contracts + validate_core_scenario_catalog()
tests/unit/runtime/evidence/
  test_core_certification_spec.py
  test_scenario_contracts.py
```

**Deferred (C2+):**

```text
intergrax/cli/certify.py
intergrax/runtime/evidence/
  scenario_runner.py
  certification_report.py
  scenarios/
tests/integration/evidence/test_core_certification.py
```

**Tier boundaries:** Tier-0 only — no `applications/` imports in evidence runner; use `reference_harness.py` / echo agent patterns.

**Optional preflight:** `certify core --with-doctor` may run doctor checks before scenarios — does not add scenarios to doctor.

---

## Implementation notes (C1)

| Artifact | Path |
|----------|------|
| CORE levels + surfaces | `intergrax/runtime/evidence/core_certification_spec.py` |
| Scenario contracts (12) | `intergrax/runtime/evidence/scenario_contracts.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |
| Unit tests | `tests/unit/runtime/evidence/` |

**Verify:** `uv run pytest tests/unit/runtime/evidence -q`

---

## Implementation waves (suggested PR order)

| PR | IDs | Scope | Status |
|----|-----|-------|--------|
| **PR1** | EVID-CORE-02 (code), EVID-CORE-03 | Spec enums + scenario contracts + unit tests | **Done** (C1) |
| **PR2** | EVID-CORE-04, EVID-CORE-05 | CLI + report generation | **Planned** |
| **PR3** | EVID-CORE-06 | Docs: README, HARNESS_ENVIRONMENT evidence path | **Planned** |

**Verify after PR2:** `uv run intergrax certify core --level L2` + `uv run pytest -m gate -q`

---

## Future waves (not approved for implementation)

| Wave | IDs | Audit priority |
|------|-----|----------------|
| HEP-2 | EVID-TRACE, EVID-EVAL, EVID-COST | External audit #2, #4, #7 |
| HEP-3 | EVID-POSTURE | External audit #12 |
| HEP-4 | EVID-POL | External audit #3 |
| HEP-5 | EVID-CAP, EVID-REPLAY, EVID-CTX, EVID-EXT, EVID-SEC, EVID-ATT | External audit #5–11 |

Register future rows in this file when Mode I approves each wave.

---

## Definition of Done (HEP-1)

1. **Contract** — Pydantic report + scenario contracts in `intergrax/runtime/evidence/`
2. **Trace** — scenarios assert `RuntimeEvent` / trace persistence where applicable
3. **Test** — integration tests deterministic, no network
4. **Documentation** — this plan + EVID-CORE-06 onboarding path
5. **No regression** — `pytest -m gate` green; certify-core does not break doctor
6. **Reuse Tier-0** — extend existing harness paths; no parallel certification stack
