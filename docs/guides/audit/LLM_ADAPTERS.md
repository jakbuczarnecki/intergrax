# LLM Adapters — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) · [`plan/LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md)  
**Audit map layers:** 6 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with repository access.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus`).
4. Output must follow [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: LLM_ADAPTERS
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — LLM Adapters (`LLM_ADAPTERS`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **LLM Adapters** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Audit LLM abstraction: provider replaceability, response envelopes, routing, metering, retries, guardrails, and structured output validation.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/LLM_ADAPTERS.md` — current architecture canon
3. `docs/plan/LLM_ADAPTERS.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 6
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
intergrax/llm_adapters/, LLMAdapter, LLMProfile, streaming, structured output
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. All LLM calls through LLMAdapter — no direct OpenAI/Anthropic/Gemini SDK in agents/runtime business code.
2. LLMAdapterResponse / LLMStructuredResult typed envelopes on all completion paths.
3. LLMProfile: model selection by cost/latency/quality/risk/capability.
4. Streaming events (LLMStreamEvent) parity with non-streaming.
5. Token/cost usage metering per call, aggregated per run/tenant.
6. Retries, fallbacks, timeout, rate-limit handling.
7. Structured output schema validation — not manual JSON parse.
8. Guardrail middleware integration (AFTER_LLM_OUTPUT).

---

## 4. Workload and scale probes

Evaluate behaviour for:

High token volume, long contexts, tool-call heavy turns, provider failover.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

LLMProfile per host/agent step, provider plugins, model routing policy.

Confirm overrides are **wired**, not documentation-only.

---

## 6. Cross-cutting checklist (mandatory)

Apply every item in `docs/guides/audit/README.md` §Shared production Harness checklist:

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

## 7. Production comparison

Compare the implementation to **production-grade systems** in this domain (commercial and open-source). State clearly:

- What Intergrax already matches at L3 production Harness OS level
- What is L2 or below with specific gaps
- What is intentionally deferred (design boundary) vs **niedoróbka** / missing wiring

---

## 8. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5:

```text
L0 — Fragmented
L1 — Operational MVP
L2 — Scalable Harness
L3 — Production Harness OS
L4 — Adaptive Agent OS
```

Report **score before**, **target for current milestone**, evidence, and **remaining risks**.

---

## 9. Verification commands

Run applicable checks; cite results:

```bash
uv run pytest -m gate -q
uv run pytest tests/unit/<relevant>/ -q
python scripts/check_harness_no_getattr.py
# plus domain-specific scripts discovered during inspection
```

---

## 10. Output and mode rules

- Follow output format in `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 (Audit Result template).
- End with §8 Completion Summary.
- `audit-only`: **no file edits**
- `audit-and-fix`: update `docs/plan/LLM_ADAPTERS.md` gap rows and `docs/architecture/LLM_ADAPTERS.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
