# Orchestration — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) · [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md)  
**Audit map layers:** 3, 9 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with **full repository access**.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: ORCHESTRATION
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Orchestration (`ORCHESTRATION`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Orchestration** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **intake normalization**, **scheduling**, **ExecutionGraph** execution, parallelism, merge policies, coordination patterns, resilience layers, and CFG configuration completeness — as formal Tier-1 responsibilities, not agent-implemented orchestration.

## Key symbols and contracts

TaskEnvelope · OrchestrationProfile · ApplicationGraphSpec · ExecutionGraph · NexusPlan/PlanStep · CoordinationPattern · MergeStrategy (concat/last_wins/structured_json) · SubtaskContract · IntentRoute

## Active plan phases (verify status vs code reality)

ORCH Done · ORCH-STRAT · ORCH-CONFIG (11/11) · ORCH-5.1 swarm · ORCH-6 sync/async · H-APP-WIRING surface parity

## Known open gaps — re-validate every item (closed / still open / partial)

CFG-14 LKW hybrid E2E deferred · active-active node redundancy L0 · QueuedNexusExecutionAdapter not scaffold-default · semantic merge ORCH-5.4 future

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/ORCHESTRATION.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/ORCHESTRATION.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 3, 9
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix I (orchestration control plane)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/runtime/nexus/orchestration/ (intake, planning, graph_runner)
intergrax/runtime/nexus/execution/graph_executor.py
intergrax/runtime/architecture/multi_agent_coordination.py (CoordinationPattern)
intergrax/runtime/nexus/orchestration_capabilities.py
intergrax/queueing/ · intergrax/distributed/
applications/_shared/task_intake.py · orchestration_wiring.py
applications/contracts/graph_builder.py (AgentGraph)
scripts/check_orchestration_config_docs.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. All tasks enter via UnifiedTaskRunner / normalized TaskEnvelope — no API bypass.
2. ExecutionGraph has typed nodes/edges — graph not implicit in agent methods.
3. DELEGATES_TO expands to child node (ADR-FLOW-001) — not function-call subagents.
4. Parallel batches specify merge_strategy — deterministic merge verified.
5. max_delegation_depth enforced.
6. Scheduler: priority, concurrency caps, backpressure (GRAPH_BACKPRESSURE event).
7. Three retry layers A/B/C documented and not conflated in code.
8. CoordinationPattern explicit per graph/host (§50 catalog).
9. classifier_kind rules|llm for free-text intake when required.
10. graph_spec respects trigger_capabilities (ADR-FLOW-004).
11. CFG-01–CFG-20 cases documented with honest host matrix §59.2.
12. Tier-2 agents do not call other agents directly — Nexus delegates.
13. Fan-out/fan-in with concurrency limits — not unbounded asyncio.gather in agents.
14. Long-running recovery (CFG-19) and strict mode (CFG-20) paths inspected.
15. OrchestrationProfile fields wired — no orphan CFG knobs.
16. Sync/async/streaming postures share same Nexus core path.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- CFG simulation tests (orchestration config matrix).
- Deep graph + wide parallel fan-out + stuck-node recovery.
- Swarm CFG-17 budget envelope.
- GRAPH_BACKPRESSURE at max_inflight_nodes.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

OrchestrationProfile (planner_kind, classifier_kind, merge_strategy, caps) · ApplicationGraphSpec · trigger_capabilities · strict_multi_agent_defaults() · apply_long_running_from_profile

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production baseline comparison

Compare against: **LangGraph/CrewAI coordination · Viktor-style long-running workflows · enterprise multi-agent orchestration (IDEAL §6.4)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Subtasks as plain function calls · implicit graph in agent code · missing merge policy · scheduler logic in Tier-2

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/check_orchestration_config_docs.py
uv run pytest tests/unit/runtime/nexus/orchestration/ -q
uv run pytest tests/acceptance/agent_os/ -q -k orchestration
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/ORCHESTRATION.md` gap rows + `docs/architecture/ORCHESTRATION.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
