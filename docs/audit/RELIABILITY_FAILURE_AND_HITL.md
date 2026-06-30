# Reliability, Failure Model, and HITL — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](../architecture/RELIABILITY_FAILURE_AND_HITL.md) · [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md)  
**Audit map layers:** 22 · compact slice: [`audit_slices/RELIABILITY_FAILURE_AND_HITL.md`](../guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: RELIABILITY_FAILURE_AND_HITL
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Reliability, Failure Model, and HITL (`RELIABILITY_FAILURE_AND_HITL`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Reliability, Failure Model, and HITL** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **failure taxonomy**, three retry layers, circuit breakers, checkpoint recovery, HITL gates, autonomy levels, and ReliabilityProfile wiring — safe-failure across runtime.

## Key symbols and contracts

RetryPolicy · RetryRecord · RetryHint · ResiliencePolicy · AutonomyLevel (MANUAL|ASK|AUTONOMOUS) · PauseRecord · RuntimeCheckpoint · HumanRequest · failure taxonomy (UserError, PolicyError, DependencyError, RuntimeError, QualityError)

## Active plan phases (verify status vs code reality)

REL Done · REL-ADV Done · H-APP-WIRING.1 HTTP surfaces

## Known open gaps — re-validate every item (closed / still open / partial)

REL-LC Done · §6.1av REL-MAINT Done · durable async queue → ORCH-MAINT-04 · LLM failover → LLM-MAINT-03

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md`](../guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md) — compact slice (layers **22**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/RELIABILITY_FAILURE_AND_HITL.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/RELIABILITY_FAILURE_AND_HITL.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
5. `@docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H (risk/HITL)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md` — then inspect:

```text
intergrax/runtime/nexus/retry/retry_engine.py
intergrax/runtime/resilience/ · intergrax/runtime/human/
applications/_shared/reliability_wiring.py
intergrax/runtime/sandbox/ · intergrax/runtime/shadow/
autonomy_middleware · CancellationCoordinator · ActiveTaskRegistry
tests/acceptance/agent_os/ (04, 05, 05b HITL/checkpoint)
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Three retry layers A/B/C not conflated (ORCH §52.1 cross-check).
2. Agents emit RETRY hints — not internal adapter while-loops.
3. HITL via Nexus/policy — not Slack webhook in agent.
4. Checkpoint includes plan+graph+UAEP cursor — recoverable.
5. Cancel cooperative via CancellationCoordinator.
6. Guardrail denial composes with HITL escalation path.
7. idempotency_key on side-effect tool retries.
8. Circuit breaker from IntegrationProfile/resilience registry.
9. AutonomyLevel obeys policy ceiling (MANUAL|ASK|AUTONOMOUS).
10. PARTIALLY_COMPLETED only when policy allows.
11. Trace shows retry reason and attempt count.
12. ReliabilityProfile wired via reliability_wiring at Tier-3.
13. Incident-worthy failures emit ops:alert-class signals.
14. Recovery reboot strategy documented for long-running runs.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Flaky integration with circuit breaker open.
- HITL queue backlog scenario.
- Cascading failure across graph nodes.
- 30-day long-running monitor (ORCH §26 cross-ref).

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ReliabilityProfile · OrchestrationProfile.max_run_retries · apply_reliability_task_defaults · require_human_approval · mid-run autonomy API (lab hosts)

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

Compare against: **PagerDuty/Opsgenie escalation · enterprise approval queues · AWS well-architected retry/backoff**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Unbounded agent retry loops · HITL bypass for HIGH risk · checkpoint without UAEP cursor · conflated retry layers

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/acceptance/agent_os/ -q -k 'hitl or checkpoint'
uv run pytest tests/unit/runtime/nexus/retry/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- **O1 terse** checkpoint unless operator requests full report.
- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7–§8 for final write-up.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update plan/arch gap rows; **no code** unless operator requests separately.

Begin the audit now.

---END PROMPT---
