# Reasoning and Cognition — Implementation Plan

**Architecture (1:1):** [`architecture/REASONING_AND_COGNITION.md`](../../architecture/REASONING_AND_COGNITION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (REASONING_AND_COGNITION plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/REASONING_AND_COGNITION_appendices.md`](plan/satellites/REASONING_AND_COGNITION_appendices.md) · [`plan/satellites/REASONING_AND_COGNITION_audit_history.md`](plan/satellites/REASONING_AND_COGNITION_audit_history.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/REASONING_AND_COGNITION.md`](../../architecture/REASONING_AND_COGNITION.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/REASONING_AND_COGNITION.md`](../../technical/guides/audit_slices/REASONING_AND_COGNITION.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/REASONING_AND_COGNITION_appendices.md`](plan/satellites/REASONING_AND_COGNITION_appendices.md) | appendices |
| [`plan/satellites/REASONING_AND_COGNITION_audit_history.md`](plan/satellites/REASONING_AND_COGNITION_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5, §16 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — incremental after IDEAL-L3 W2 closeout; closed COG-LC (2026-06-17)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-7.1 | §7 Cognition | Ship `ReasoningProfile` contract + environment wire | P1 | **Done** (COG-5.1 · COG-PROD.1) |
| AUDIT-IDEAL-7.2 | §7 Cognition | Complete `allow_dynamic_replan` runtime path | P1 | **Done** |
| AUDIT-IDEAL-7.3 | §7 Cognition | Reasoning failure taxonomy on all planner kinds | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

(Global)

1. **Contract** — Pydantic / Protocol public API
2. **Trace** — cognition transitions emit `TraceEvent` / `RuntimeEvent` (`ops:planning`, `DECISION_EMITTED`)
3. **Test** — unit + integration, deterministic, no network
4. **Documentation** — update this plan + architecture pair when contracts change
5. **No regression** — `pytest -m gate` green; Echo through NexusLoop
6. **Reuse Tier-0** — extend existing planner modules; no parallel LLM/log/trace stacks
7. **Separation** — reasoning/planning docs stay in this pair; orchestration scheduling stays in `ORCHESTRATION` / `NEXUS_EXECUTION_FLOW`
8. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts

---

## Phase COG-PROD — Production reasoning plane hardening (Band 2au)

**Status:** **Done** (2026-06-12) — doc↔code drift closed; production wiring complete  
**Prerequisites:** Phase COG-DEPTH **Done**  
**Goal:** L3+ reasoning plane with honest production wiring — planner LLM separation, parse retries, full prompt template binding, planning `DecisionRecord` enrichment, missing wiring helpers  
**Priority ladder:** Band 2au (maintenance queue after COG-DEPTH)

**Delivery rule:** One **COG-PROD.\*** ID per PR → update architecture §21 + this register → gate green.

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| COG-PROD.1 | **`resolve_planner_llm_adapter()`** — producer/planner separation (mirror critic) | **Done** | **Critical** | `reasoning_wiring.py`, `orchestration_wiring.py`, `nexus_factory.py` | Separate adapter when `planner_llm_profile` set |
| COG-PROD.2 | **`planner_parse_retries` wire** + `nexus_task_planner` `user_template` | **Done** | High | `nexus_plan_bridge.py`, `nexus_planner_prompts.py`, `prompts/nexus_task_planner/` | Retries exercised in unit test |
| COG-PROD.3 | **Planning `DecisionRecord` enrichment** + `resolve_engine_planner_prompt_config()` | **Done** | High | `planning_runner.py`, `reasoning_wiring.py` | Gate test asserts classification + policy fields |
| COG-PROD.4 | **Doc reconciliation** — architecture stale gaps removed | **Done** | High | `architecture/REASONING_AND_COGNITION.md` | No §2/§14 contradictions vs §21 |
| COG-PROD.5 | **`check_reasoning_gates.py` uplift** — registry-only planner prompts | **Done** | Medium | `scripts/maintenance/check_reasoning_gates.py` | CI script green |

### COG-PROD — Sprint execution order

| Sprint | IDs | Files (primary) |
|--------|-----|-----------------|
| S1 Doc | COG-PROD.4 | `docs/project/architecture/REASONING_AND_COGNITION.md`, `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` |
| S2 Planner LLM | COG-PROD.1 | `reasoning_wiring.py`, `orchestration_wiring.py`, `nexus_factory.py`, `contracts/reasoning_profile.py` |
| S3 Parse + prompt | COG-PROD.2 | `nexus_plan_bridge.py`, `nexus_planner_prompts.py`, `prompts/nexus_task_planner/1.yaml` |
| S4 Decision + engine prompt | COG-PROD.3, COG-PROD.5 | `planning_runner.py`, `reasoning_wiring.py`, `scripts/maintenance/check_reasoning_gates.py`, tests |

---
