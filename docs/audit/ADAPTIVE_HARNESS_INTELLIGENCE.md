# Adaptive Harness Intelligence (L4) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../plan/ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Audit map layers:** L4 AHI · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: ADAPTIVE_HARNESS_INTELLIGENCE
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Adaptive Harness Intelligence (L4) (`ADAPTIVE_HARNESS_INTELLIGENCE`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Adaptive Harness Intelligence (L4)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **L4 adaptive loops**: bounded self-tuning, utility function, shadow→canary→prod promotion, policy-bounded routing, signal emission — honest W-ADAPT Done vs product-gated L4 thresholds.

## Key symbols and contracts

HarnessOutcomeSignal · AdaptiveLoopEnvelope · AdaptiveLoopKind · ProfileVersion · Utility U · AdaptationEngine · AdaptationExecutor · AdaptiveProfile · LLMCallSummary on signals

## Active plan phases (verify status vs code reality)

W-ADAPT W0–W7 Done (70/70) · Phase V L4 evidence · L4 adaptive critic thresholds product-gated

## Known open gaps — re-validate every item (closed / still open / partial)

AHI-LC Done · §6.1av AHI-MAINT Done · L4 auto-apply requires explicit product gate · live routing owner LLM-MAINT-02

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers L4 AHI
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/runtime/adaptive/ (signal_emission.py, SignalCollector, adaptive_governance.py, VerificationLoop)
intergrax/runtime/adaptive/cost_optimization.py
intergrax/runtime/architecture/ (ProcessPatternMiner W-ADAPT-6, ExecutionStrategyEngine)
runtime_governance_bridge.py
scripts/phase_w_adapt_report.py · scripts/phase_v_closeout_gate.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Adaptations versioned with rollback pointer.
2. PolicyEngine never bypassed by adaptive executor.
3. Post-task outcome signals emitted (HarnessOutcomeSignal).
4. Utility U computed from documented function.
5. Proposals pass capability graph impact analysis.
6. Shadow mode before canary — evidence in registry.
7. Human gate on apply in production profiles.
8. Tier-1 remains domain-agnostic — no Problem Radar business logic in core.
9. Classical RL explicitly NOT the adaptation model.
10. Evaluation registry consumes adaptive outcomes.
11. Cost optimization under policy cap.
12. Process miner emits proposals — not auto Tier-2 code generation.
13. Kill switches and cooldowns on AdaptiveProfile.
14. Observability: why route/model/tool changed.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- ≥10% utility improvement target on golden scenarios.
- Rollback <5 min evidence.
- Feedback delay vs adaptation lag.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

AdaptiveProfile on Tier-3 · shadow_eval_enabled · observe/recommend/apply/verify lifecycle modes

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

Compare against: **Canary/feature-flag systems (LaunchDarkly/Unleash) · contextual bandits + regression gates — NOT OpenAI RLHF**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Unapproved policy mutation · unconstrained model switching · adaptive loop without shadow · RLHF-style training in Tier-1

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/phase_w_adapt_report.py
uv run pytest tests/unit/runtime/adaptive/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` gap rows + `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
