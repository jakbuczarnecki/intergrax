# Critic and Verification (CVL) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/CRITIC_VERIFICATION.md`](../architecture/CRITIC_VERIFICATION.md) · [`plan/CRITIC_VERIFICATION.md`](../plan/CRITIC_VERIFICATION.md)  
**Audit map layers:** 25 (depth) · compact slice: [`audit_slices/CRITIC_VERIFICATION.md`](../guides/audit_slices/CRITIC_VERIFICATION.md)  
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

domain: CRITIC_VERIFICATION
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Critic and Verification (CVL) (`CRITIC_VERIFICATION`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Critic and Verification (CVL)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **Critic Verification Layer**: L0/L1 gateways, evaluator loops, LLM-as-judge via ToolRuntime, trajectory eval, HITL on borderline — integrated in runtime, not bolt-on scripts.

## Key symbols and contracts

CriticProfile · CriticRequest · CriticVerdict · L0Gateway · L1Gateway · EvaluatorLoopSpec · RubricSpec · ValidationResult · eval.judge · eval.trajectory · eval.record_observation

## Active plan phases (verify status vs code reality)

CRIT-V 0–7 + FOLLOWUP Done · CVL-LC-1/2 layer completion (2026-06-13) · FAUDIT-EVAL.1 closed

## Known open gaps — re-validate every item (closed / still open / partial)

CVL-LC Done · §6.1av CVL-MAINT Done · L4 thresholds Frozen → AHI · FLOW-8 host → §6.3

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/CRITIC_VERIFICATION.md`](../guides/audit_slices/CRITIC_VERIFICATION.md) — compact slice (layers **25 (depth)**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/CRITIC_VERIFICATION.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/CRITIC_VERIFICATION.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/CRITIC_VERIFICATION.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/CRITIC_VERIFICATION.md` — then inspect:

```text
intergrax/runtime/critic/critic_orchestrator.py · contracts.py · policy_bridge.py
intergrax/runtime/critic/evaluator_loop_executor.py · CriticTraceEmitter
intergrax/runtime/nexus/validation_engine.py
intergrax/tools/providers/eval/judge.py
applications/_shared/critic_runtime_bridge.py · critic_assembly_resolver.py
eval/nexus_eval_runner.py
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. L0 static/rule gateway before L1 LLM judge always.
2. Judge LLM ≠ producer LLM (separate profile/ref).
3. eval.judge invoked via ToolRuntime — not direct adapter in agent.
4. ValidatorAgents as graph nodes allowed — not parallel eval stack.
5. No parallel SQLite eval store per agent.
6. require_critic_on_completion fail-closed when profile set.
7. Critic steps in trace (CriticTraceEmitter).
8. Registry observations via eval.record_observation.
9. Domain rubrics live in Tier-2 — not Nexus business rules.
10. guardrail_scan merges into L0 where configured.
11. node_partial vs graph_final verify scopes correct.
12. EvaluatorLoopExecutor wired for CoordinationPattern.EVALUATOR_LOOP.
13. Semantic NexusEvalRunner mode for harness eval.
14. False positive/negative handling and retry semantics documented.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Evaluator-loop until budget exhausted.
- CFG-16/CFG-20 strict multi-agent critic.
- High-volume eval latency impact on user path.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

CriticProfile + EvaluationProfile · require_critic_on_completion · separate critic LLMProfile · CoordinationPattern.EVALUATOR_LOOP

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

Compare against: **Guardrails AI Hub validators · Braintrust/Phoenix LLM-as-judge · legal/finance human sign-off workflows**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Judge same model as producer · critic as optional script · duplicate eval store · skipping L0 for speed

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/critic/ -q
uv run pytest tests/unit/tools/providers/eval/ -q
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
