# LLM Adapters — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) · [`plan/LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md)  
**Audit map layers:** 6 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: LLM_ADAPTERS
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — LLM Adapters (`LLM_ADAPTERS`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **LLM Adapters** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **LLMAdapter** abstraction: typed response envelopes (M-LLM-R), 19 provider slugs, streaming, structured output, metering, tenant scope, guardrail middleware, and planner≠producer discipline.

## Key symbols and contracts

LLMAdapter · LLMAdapterResponse · LLMFinishReason · LLMTokenUsage · LLMToolCall · LLMStructuredResult[T] · LLMProfile · LLMStreamEvent · LLMCallConfig

## Active plan phases (verify status vs code reality)

M-LLM-R envelope Done · W-ML.1 capability flags · Phase V FAUDIT-LLM.1 residual · COG cross-ref planner≠producer

## Known open gaps — re-validate every item (closed / still open / partial)

Planner LLM ≠ producer discipline incomplete at Nexus boundary · distributed rate limit needs Redis wiring · usage tracking layers not auto-merged

---

## 0. Context budget (mandatory — quality without bulk loading)

Deep audit = **targeted reads + code/gate evidence**, not loading entire plan files.

### Session rules
- **One domain per chat** unless the operator explicitly batches.
- **Never** read a file >500 lines in full — grep section headers, then `Read` with offset/limit.
- **Never** re-read the same file in one session unless it changed.
- Prefer **grep with path filters** over repo-wide semantic search for known symbols.
- Run **only** scripts in section 10 — no full-suite pytest unless this prompt lists a domain slice.
- Do **not** load `docs/audit_results/` unless RESUME/bootstrap says so.
- Respect **`.cursorignore`** — excluded paths are out of scope unless the operator points to them.

### Scoped plan read (`docs/plan/{DOMAIN}.md`)
Read **only**: `## 6.` open queue rows only · gap/remediation registers tied to **Known open gaps** and **Active plan phases** · skip `(closed)`, `(complete)`, `Archived` unless re-validating a listed gap

### Scoped architecture read (`docs/architecture/{DOMAIN}.md`)
Table of contents + sections for audit-map layers **6** + registers tied to **Known open gaps**. Skip historical paydown logs unless a gap ID points there.

### Scoped guide reads
- **Prefer** [`docs/guides/audit_slices/{DOMAIN}.md`](../guides/audit_slices/{DOMAIN}.md) — compact slice for this domain (replaces bulk IDEAL + AUDIT_MAP load)
- Otherwise: `IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for layers **6** only
- `INTEGRAX_HARNESS_AUDIT_MAP.md` — layers **6** + maturity §5 only
- `SYSTEM_INVARIANTS.md` — skim invariant IDs referenced in section 3 dimensions only

---


## 1. Canonical reads (scoped — in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — **layers 6 only** (see §0)
2. `docs/architecture/LLM_ADAPTERS.md` — **scoped sections** (see §0)
3. `docs/plan/LLM_ADAPTERS.md` — **scoped sections only** (see §0) — do **not** load the full file
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — **layers 6** + §5 maturity
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/llm_adapters/ (registry/, providers/*, call_lifecycle.py, tracking/)
intergrax/llm/messages.py (AttachmentRef)
intergrax/runtime/replay/trace_replay_bridge.py
intergrax/runtime/adaptive/llm_call_summary.py
scripts/check_llm_adapter_typed_returns.py · scripts/check_agents_llm_adapter_response.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. All completions return LLMAdapterResponse / LLMStructuredResult — not bare str.
2. Agents do not annotate LLM returns as str — CI check_agents_llm_adapter_response.
3. Vendor SDK only inside provider modules — check_agents_vendor_imports.
4. refusal/content_filter surfaced on envelope.
5. Streaming LLMStreamEvent parity with non-streaming paths.
6. LLMProfile drives model selection — not hardcoded model per agent.
7. Token/cost usage on LLMTokenUsage; aggregated per run/tenant.
8. Retries, timeout, circuit breaker via LLMCallConfig.
9. Structured output schema validation — Pydantic/generic T.
10. Guardrail middleware AFTER_LLM_OUTPUT when profile configured.
11. llm_tenant_scope and INTERGRAX_LLM_TENANT_MAX_TOKENS quota.
12. Metrics plugin on TASK_COMPLETED; register_llm_metrics_routes.
13. Attachments respect ModalityProfile.max_media_bytes.
14. Capability flags default false until provider tested (W-ML.1).
15. Secrets via SecretsStore llm/<provider>/api_key paths.
16. Replay bridge maps historical trace to adapter calls.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- High token volume run with cost aggregation.
- Tool-call-heavy turns with streaming.
- Provider failover / rate-limit storm.
- 19 provider slug registry bootstrap time.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

LLMProfile per host/step · SecretsStore paths · options.use_distributed_rate_limit · guardrail middleware stack

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

Compare against: **OpenAI/Anthropic/Azure/Bedrock enterprise adapters · Helicone/LangSmith proxies · SaaS token metering gateways**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- str returns from adapters · model hardcoded in agent · direct SDK in Tier-2 · manual JSON parse for structured output

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
python scripts/check_llm_adapter_typed_returns.py
python scripts/check_agents_llm_adapter_response.py
python scripts/check_agents_vendor_imports.py
uv run pytest tests/unit/llm_adapters/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/LLM_ADAPTERS.md` gap rows + `docs/architecture/LLM_ADAPTERS.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
