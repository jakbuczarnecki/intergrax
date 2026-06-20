# LLM_ADAPTERS — audit register

**Parent hub:** [`LLM_ADAPTERS.md`](../LLM_ADAPTERS.md)

## Audit register (2026-06-14)

### AUDIT-IDEAL §6 (LLM layer — master register cross-ref)

| ID | Gap | Priority | Status | Phase / owner |
|----|-----|----------|--------|---------------|
| AUDIT-IDEAL-6.1 | Structured output validation on reference + certified paths | P1 | **Done** | M-LLM-R — §Structured output |
| AUDIT-IDEAL-6.2 | Live cost/latency/quality routing (AHI prod path) | P2 | **Done** | M-LLM-X.5 — LC-3 hot-path wiring |
| AUDIT-IDEAL-6.3 | Central `ModelCatalog` + context window resolution | P0 | **Done** | M-LLM-X.1 — LC-1 |
| AUDIT-IDEAL-6.4 | Tokenizer-consistent context preflight | P0 | **Done** | M-LLM-X.3 — LC-2/LC-2b |
| AUDIT-IDEAL-6.5 | Profile failover chain | P1 | **Done** | M-LLM-X.4 — LC-3 |
| AUDIT-IDEAL-6.6 | ACP `StepLLMRouter` backed by `LLMAdapter` | P1 | **Done** | M-LLM-X.5 — LC-3 |
| AUDIT-IDEAL-6.7 | Developer `USAGE.md` + startup validation | P2 | **Done** | M-LLM-X.7 + LLM-MAINT-01 — `validate_runtime()` + `intergrax doctor` LLM checks |

### Production audit gaps (LLM-AUDIT-*)

| ID | Gap | Severity | Phase | Status |
|----|-----|----------|-------|--------|
| LLM-AUDIT-1 | No central `ModelCatalog`; per-adapter context dicts stale | **P0** | M-LLM-X.1 | **Done** |
| LLM-AUDIT-2 | `context_window_tokens` override only on Ollama | **P0** | M-LLM-X.1 | **Done** |
| LLM-AUDIT-3 | Preflight token estimate uses chars/4 not adapter tokenizer | **P0** | M-LLM-X.3.1–3.4 | **Done** |
| LLM-AUDIT-4 | `ModelRouter` not on Nexus hot path | **P1** | M-LLM-X.4–5 | **Done** |
| LLM-AUDIT-5 | No provider failover chain | **P1** | M-LLM-X.4 | **Done** |
| LLM-AUDIT-6 | ACP `StepLLMRouter` disconnected from `LLMAdapter` | **P1** | M-LLM-X.5 | **Done** |
| LLM-AUDIT-7 | OpenRouter / gateway models default 32k context | **P1** | M-LLM-X.2 | **Done** — catalog + gateway merge + miss diag trace |
| LLM-AUDIT-8 | No `intergrax/llm_adapters/USAGE.md` | **P2** | M-LLM-X.7 | **Done** — USAGE + doctor hook (LLM-MAINT-01) |
| LLM-AUDIT-9 | AUDIT-IDEAL-6.2 wiring ceremonial — no runtime swap | **P1** | M-LLM-X.5 | **Done** |
| LLM-AUDIT-10 | Plugin provider story undocumented | **P2** | M-LLM-X.6 | **Done** — USAGE §Extension · enum-free profile **M-LLM-X.14.3** |
| LLM-AUDIT-11 | `ContextBudgetPolicy` default 4k decoupled from adapter window | **P0** | M-LLM-X.3.3 | **Done** |
| LLM-AUDIT-12 | Prefix context heuristics only on Bedrock (not Claude/OpenAI/Gemini) | **P0** | M-LLM-X.1.2–1.3 | **Done** |
| LLM-AUDIT-13 | Cohere dual slug (`cohere` vs `cohere_native`) confuses developers | **P2** | M-LLM-X.7.5 | **Done** |
| LLM-AUDIT-14 | Capability flags not catalog-driven (`supports_vision`, tools, structured) | **P2** | M-LLM-X.14.1 | **Done** |
| LLM-AUDIT-15 | `engine_history_layer` token count inconsistent with preflight (chars/4) | **P0** | M-LLM-X.3.5 | **Done** — history already used adapter; preflight aligned in LC-2 |
| LLM-AUDIT-16 | No unified LLM routing rule contract — static hints only; no custom author logic | **P1** | M-LLM-X.9 | **Done** — ADR-LLM-003 |
| LLM-AUDIT-17 | Routing enterprise E2E — start-of-run + ACP (auto context, trace, reference host) | **P1** | M-LLM-X.10 | **Done** |
| LLM-AUDIT-18 | Routing mid-run Nexus — live re-eval, context refresh, full trace loop, true E2E run | **P1** | M-LLM-X.11 | **Done** (X-11 scope) |
| LLM-AUDIT-19 | Routing strict L5 — budget meter accuracy, all Nexus paths, tier boundary, production E2E, ACP trace parity | **P1** | M-LLM-X.12 | **Done** |
| LLM-AUDIT-20 | Post-L5 polish — ACP Plane A trace, tier bridge, concurrent test, secondary LLM + auxiliary Nexus paths | **P2** | M-LLM-X.13 | **Done** |
| LLM-AUDIT-21 | Domain closeout — audit register, AUDIT_IDEAL sync, implementation journal | **P1** | M-LLM-X.8 | **Done** |
| LLM-AUDIT-22 | Capability flags not catalog-driven | **P2** | M-LLM-X.14.1 | **Done** |
| LLM-AUDIT-23 | Dynamic gateway metadata (OpenRouter `/models`) not on catalog hot path | **P1** | M-LLM-X.14.2 | **Done** |
| LLM-AUDIT-24 | ACP mid-run budget routing — `AcpInvocationUsageView` not mapped to `RoutingContext` | **P2** | M-LLM-X.14.4 | **Done** |
| LLM-AUDIT-25 | Secondary LLM surfaces lack opt-in evaluating wrap (planner / websearch / critic) | **P2** | M-LLM-X.14.5 | **Done** |
| LLM-AUDIT-26 | Plugin provider story — `LLMProfile.provider` enum coupling | **P2** | M-LLM-X.14.3 | **Done** |

**Deferred (documented, no blocking X-phase task):** tiktoken OpenAI-centric token estimate for non-OpenAI models — **M-LLM-X.14.7** documents limitation and optional vendor tokenizer plugins; not blocking L5 routing.

**By design:** two-layer usage model (`LLMAdapterUsageLog` + `LLMUsageTracker`) — do not merge without explicit bridge (ADR-LLM-001).

**Ops (host responsibility):** distributed Redis rate limit requires `set_llm_distributed_rate_limiter` at Tier-3 bootstrap — not a Tier-0 code gap.

**Single adapter per Nexus run:** `RuntimeConfig.llm_adapter` holds one primary instance today; multi-model via profile chain + routing is M-LLM-X.4–5 (not a separate LLM-AUDIT ID).

**Closed baselines:** M-LLM (13/13), M-LLM-R (39/39), M-LLM-X LC-1…LC-3 **Done**; routing X-9…X-13 **Done**.

**Audit revalidation (2026-06-19, post X-16):** routing **L5** · catalog miss **L5 ops** · whole domain **L4 enterprise** · LLM-AUDIT-1…26 **Done**.

---

## Anti-patterns

| Anti-pattern | Why forbidden | Correct approach |
|--------------|---------------|------------------|
| Adapter returns bare `str` | Breaks metering, guardrails, replay | `LLMAdapterResponse` / `LLMStructuredResult[T]` |
| Hardcoded model in Tier-2 agent | Bypasses profile, catalog, routing | `LLMProfile` + host resolver + `LLMRoutingProfile` |
| Custom routing logic without Protocol | Untraceable, bypasses allowlist | `LLMRoutingRule` subclass in Tier-3 |
| String eval / dynamic rule paths | Security + audit failure | Importable typed rule classes only |
| Direct OpenAI/Anthropic SDK in agents | Tier violation | Injected `LLMAdapter` via Nexus |
| Manual JSON parse for structured output | No schema validation | `generate_structured(output_model=...)` |
| Per-adapter context dict without catalog entry | Stale windows for new models | `ModelCatalog` + ADR-LLM-002 resolution |
| Silent 32k fallback on OpenRouter ids | Wrong history trim | Profile override or M-LLM-X.2 metadata |
| Merging adapter + run usage counters | Double-count or lost attribution | Two-layer model per ADR-LLM-001 |

---

## CI

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
python scripts/check_llm_adapter_typed_returns.py
python scripts/check_agents_llm_adapter_response.py
python scripts/check_agents_vendor_imports.py
```

**Target gates (M-LLM-X):** `check_model_catalog_coverage.py`, `check_context_preflight_uses_adapter_tokens.py`, `check_llm_routing_rules.py`, `check_llm_routing_tier_boundary.py`, `check_llm_routing_context_wiring.py`.

Workflows: `unit-tests.yml`, `llm-adapters-guard.yml`, optional `llm-network-smoke.yml`.

---

## Related work (other domains)

| Item | Owner domain |
|------|----------------|
| Central LLM gateway service (single egress) | Platform ADR — not M-LLM-X |
| Multimodal attachment mapping (W-ML.1) | LLM_ADAPTERS + MODALITY |
| Cost envelopes (V-COST.*) | UNIFIED_EXECUTION_RUNTIME |
| AHI routing tuning production loop | ADAPTIVE_HARNESS_INTELLIGENCE + M-LLM-X.5 · **M-LLM-X.9** |
| LLM routing rules (author + custom classes) | LLM_ADAPTERS M-LLM-X.9 · ADR-LLM-003 |
| LLM routing enterprise closeout (start-of-run + ACP) | M-LLM-X.10 · LLM-AUDIT-17 |
| LLM routing enterprise hardening (mid-run Nexus) | M-LLM-X.11 · LLM-AUDIT-18 |
| LLM routing strict enterprise closeout (L5 honest) | M-LLM-X.12 · LLM-AUDIT-19 **Done** |
| LLM routing post-L5 polish (secondary surfaces, ACP Plane A) | M-LLM-X.13 · LLM-AUDIT-20 **Done** |
| LLM enterprise domain maturity (catalog caps, gateway meta, ACP budget, plugin DX) | M-LLM-X.14 · LLM-AUDIT-22…26 **Done** |
| LLM catalog miss observability spine (L4) | M-LLM-X.15 **Done** |
| LLM catalog miss L5 ops (alerts, runbook, umbrella CI) | M-LLM-X.16 **Done** |
| LLM domain closeout (register + journal) | M-LLM-X.8 · LLM-AUDIT-21 **Done** |
| `BudgetReactionProfile.degrade_model` unification | AGENT_CONTRACTS + M-LLM-X.9.6 |
| AHI `ProfileVersion` llm_routing persistence | AHI-MAINT-06 |
| Product HTTP API DTOs | Tier-3 applications |

**Out of scope:** per-business-agent adapter code in `llm_adapters/`, YOLO/ONNX engines, Phase K business agents.
