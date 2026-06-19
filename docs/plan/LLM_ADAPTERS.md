# Llm Adapters — Implementation Plan

**Architecture (1:1):** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-18) — all AUDIT-IDEAL §6 rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.1 | §6 LLM | Structured output validation on 100% reference + certified agent paths | P1 | **Done** |
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality model routing (AHI prod path) | P2 | **Done** — `check_live_model_routing_wiring.py` |
| AUDIT-IDEAL-6.3 | §6 LLM | Central `ModelCatalog` + unified context window resolution | P0 | **Done** — `CatalogCapabilityAdapter` |
| AUDIT-IDEAL-6.4 | §6 LLM | Tokenizer-consistent context preflight (adapter path) | P0 | **Done** — `count_message_tokens(adapter=)` |
| AUDIT-IDEAL-6.5 | §6 LLM | Profile failover chain on retriable provider errors | P1 | **Done** — LC-3 |
| AUDIT-IDEAL-6.6 | §6 LLM | ACP `StepLLMRouter` backed by `LLMAdapter` (single DX) | P1 | **Done** — M-LLM-X.5.4 |
| AUDIT-IDEAL-6.7 | §6 LLM | Developer `USAGE.md` + startup validation | P2 | **Done** — `check_llm_profile_runtime.py` + doctor |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

## Phase M-LLM-X — LLM Developer Excellence (post-audit 2026-06-14)

**Source:** Deep production audit 2026-06-14 — contract L3, model metadata L1–L2, routing L1–L2, DX L2.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) §Model catalog · §Routing · §Audit register  
**Goal:** Elevate Tier-0 LLM layer from **production L3 foundation** to **best-in-class developer engine** — correct context for any model string, unified token accounting, runtime routing/failover, single agent API.  
**Status:** **LC baseline Done** (2026-06-14) — P0/P1 closed · **M-LLM-X partial waves** = P2+ backlog (not blocking layer maturity)  
**Priority ladder:** Band **2ba** (after M-LLM-R closeout) · queue [§6.1ax](#61ax-harness-implementation-queue--llm-developer-excellence-m-llm-x)  
**Execution order:** [§6.2af](#62af-phase-m-llm-x-execution-order)  
**Target maturity:** Model metadata **L3**, routing **L3**, DX **L3+** (see architecture maturity table).

**Hard rules (non-negotiable):**

- **No** new per-adapter hardcoded context dicts without catalog entry or prefix rule — migrate to `ModelCatalog`.
- **`LLMProfile.options["context_window_tokens"]`** MUST override catalog for **all** providers (not Ollama-only).
- **Preflight / history budget** MUST use `adapter.count_messages_tokens` when adapter is in scope.
- **No** vendor SDK imports in Tier-2 — unchanged tier boundary.
- One **M-LLM-X.\*** task group per PR → update master table + architecture audit register → gate green.
- **ADR:** [ADR-LLM-002](../adr/entries/2026-06-14/ADR-LLM-002.md) **Done** — prerequisite for M-LLM-X.1 code merge.  
**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) **Accepted** — prerequisite for M-LLM-X.9 code merge.

**Explicitly excluded:** Central LLM gateway microservice (needs separate platform ADR), rewriting all 19 SDK clients, product HTTP DTOs, Phase K agents.

---

## Layer Completion Mode — Sprint register (2026-06-14)

**Mode:** Intergrax Layer Completion · **Domain:** LLM_ADAPTERS only  
**Baseline:** M-LLM-R **Done** (contract L3) · M-LLM-X code in progress  
**Status:** **LC-3 closeout** — P0/P1 audit gaps closed · backlog P2+ only  
**Target state:** **State B — Architecturally Mature** (no open P0/P1)

### Step 1 — Audit summary (2026-06-14, re-validated)

| ID | Problem | Class | Sprint | Status |
|----|---------|-------|--------|--------|
| LLM-AUDIT-1 … 2, 11, 3, 12, 15 | ModelCatalog + token accounting | **P0** | LC-1/LC-2/LC-2b | **Done** |
| LLM-AUDIT-4 … 6, 9, 7 | Routing, failover, ACP, OpenRouter default | **P1** | LC-3/LC-4 | **Done** |

**Closed (no sprint):** M-LLM-R envelope · AUDIT-IDEAL-6.1 structured output · LLM-AUDIT-13 Cohere DX · tier boundaries · observability L3.

### Sprint LC-1 — ModelCatalog + resolver (P0) — **Done**

### Sprint LC-2 — Token accounting (P0) — **Done**

### Sprint LC-2b — Nexus budget adoption + CI guard (P0) — **Done**

| Field | Value |
|-------|-------|
| **Scope** | M-LLM-X.3.3 rollout, M-LLM-X.3.4 |
| **Goal** | `resolve_context_budget_policy(from adapter)`, RuntimeConfig bridge, preflight CI guard |
| **DoD** | `test_context_window_wiring.py`; `check_context_preflight_uses_adapter_tokens.py` green |

### Sprint LC-3 — Failover + runtime routing (P1) — **Done**

| Field | Value |
|-------|-------|
| **Scope** | M-LLM-X.4.1–4.3, 5.1–5.5 |
| **Goal** | Profile chain, Nexus hot path, ACP bridge |
| **DoD** | `test_failover_adapter.py`, `test_model_router.py`, ACP adapter port test green |

### Sprint LC-4 — Gateway metadata + DX closeout — **Partial** (static OpenRouter default Done; dynamic fetch → backlog)

### Backlog (P2–P4 — does not block layer completion)

| ID | Item | Class |
|----|------|-------|
| LLM-AUDIT-14 | Capability flags z catalog | **P2** |
| LLM-AUDIT-8 | `validate_runtime()` + doctor | **P2** |
| LLM-AUDIT-10 | Plugin provider enum-free profile | **P2** |
| M-LLM-X.6.1–6.2 | Custom gateway example | **P2** |
| Streaming tool-call parity | Provider-specific gaps | **P2** |
| Vendor-native tokenizer plugins | Post-X deferred | **P4** |
| Central LLM gateway microservice | Separate ADR | **P4** |

---

### 6.1ax Harness implementation queue — LLM Developer Excellence (M-LLM-X)

**Purpose:** Ordered backlog for **Phase M-LLM-X**. Pull **P0 waves first** — context window accuracy blocks context engine quality.

| Order | Wave | IDs | Priority | Status |
|-------|------|-----|----------|--------|
| 0 | X-0 | M-LLM-X.0.1–0.3 | **P0** | **Done** |
| 1 | X-1 | M-LLM-X.1.1–1.6 | **P0** | **Done** |
| 2 | X-2 | M-LLM-X.2.1–2.4 | P1 | **Backlog** (static catalog covers OpenRouter default) |
| 3 | X-3 | M-LLM-X.3.1–3.5 | **P0** | **Done** |
| 4 | X-4 | M-LLM-X.4.1–4.5 | P1 | **Partial** (4.1–4.4 Done; 4.5 Tier-3 wiring → backlog) |
| 5 | X-5 | M-LLM-X.5.1–5.5 | P1 | **Done** |
| 6 | X-6 | M-LLM-X.6.1–6.3 | P2 | **Partial** (6.3 Done) |
| 7 | X-7 | M-LLM-X.7.1–7.5 | P2 | **Partial** (7.1, 7.5 Done; 7.2–7.4 Planned) |
| 8 | X-8 | M-LLM-X.8.1–8.3 | Medium | **Planned** |
| 9 | X-9 | M-LLM-X.9.1–9.9 | **P1** | **Done** (2026-06-19) — ADR-LLM-003 · routing rule Protocol |
| 10 | X-10 | M-LLM-X.10.1–10.8 | **P1** | **Done** — routing enterprise closeout (start-of-run + ACP) |
| 11 | X-11 | M-LLM-X.11.1–11.8 | **P1** | **Done** — routing enterprise hardening (mid-run Nexus) |
| 12 | X-12 | M-LLM-X.12.1–12.12 | **P1** | **Planned** — routing strict enterprise closeout (honest L5) |

**Closeout gate:** All M-LLM-X.* Done + architecture audit register all **Done** + `tests/unit/llm_adapters/` green + new CI scripts green. **X-10** closed LLM-AUDIT-17 (start-of-run scope). **X-11** closed LLM-AUDIT-18 (X-11 scope). **X-12** required for **LLM-AUDIT-19** (strict L5). **X-8** domain closeout follows **X-12**.

---

### 6.2af Phase M-LLM-X execution order

**Status:** **Planned**

```text
Wave M-LLM-X-0 (planning):     M-LLM-X.0.1 → 0.2 → 0.3
Wave M-LLM-X-1 (catalog):      M-LLM-X.1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6
Wave M-LLM-X-2 (dynamic meta): M-LLM-X.2.1 → 2.2 → 2.3 → 2.4
Wave M-LLM-X-3 (tokens):       M-LLM-X.3.1 → 3.2 → 3.3 → 3.4 → 3.5
Wave M-LLM-X-4 (failover):     M-LLM-X.4.1 → 4.2 → 4.3 → 4.4 → 4.5
Wave M-LLM-X-5 (routing):      M-LLM-X.5.1 → 5.2 → 5.3 → 5.4 → 5.5
Wave M-LLM-X-6 (plugins):      M-LLM-X.6.1 → 6.2 → 6.3
Wave M-LLM-X-7 (DX):           M-LLM-X.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave M-LLM-X-8 (closeout):     M-LLM-X.8.1 → 8.2 → 8.3
Wave M-LLM-X-9 (routing rules): M-LLM-X.9.1 → 9.2 → 9.2b → 9.3 → 9.4 → 9.5 → 9.6 → 9.7 → 9.8 → 9.9
Wave M-LLM-X-10 (routing enterprise): M-LLM-X.10.1 → 10.2 → 10.3 → 10.4 → 10.5 → 10.6 → 10.7 → 10.8
Wave M-LLM-X-11 (routing hardening): M-LLM-X.11.1 → 11.2 → 11.3 → 11.4 → 11.5 → 11.6 → 11.7 → 11.8
Wave M-LLM-X-12 (strict L5): M-LLM-X.12.1 → 12.2 → 12.3 → 12.4 → 12.5 → 12.6 → 12.7 → 12.8 → 12.9 → 12.10 → 12.11 → 12.12
```

**Prerequisites:** M-LLM + M-LLM-R **Done**; CONTEXT_ENGINE preflight paths stable. **X-9** depends on X-4.2 (failover) and X-5.1 (ModelRouter hints) — both **Done**. **X-10** depends on **X-9 Done**. **X-11** depends on **X-10 Done**. **X-12** depends on **X-11 Done**.

**Parallelism:** X-2 (OpenRouter fetch) may run after X-1.3; X-5 depends on X-4.2; X-7 may start after X-1.1 (docs partial).

---

#### Wave M-LLM-X-0 — Planning and ADR

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.0.1 | **Plan register** — §6.1ax, §6.2af, AUDIT-IDEAL-6.3–6.7 rows | **Done** | Critical | This file + architecture audit register | Cross-linked |
| M-LLM-X.0.2 | **`docs/adr/entries/2026-06-14/ADR-LLM-002.md`** — ModelCatalog, resolution order, override semantics | **Done** | Critical | `docs/adr/` | Linked from architecture + hub |
| M-LLM-X.0.3 | **Master register sync** — `AUDIT_IDEAL_2026.md` rows 6.2–6.7 | **Done** | High | `plan/AUDIT_IDEAL_2026.md` | 6.2 + 6.7 Partial; 6.3–6.6 Planned |

---

#### Wave M-LLM-X-1 — ModelCatalog and context window (P0)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.1.1 | **`ModelRecord` + `ModelCatalog`** frozen types | **Done** | Critical | `registry/model_catalog.py` | Unit: load, immutability |
| M-LLM-X.1.2 | **Bundled `model_catalog.yaml`** — OpenAI, Claude, Gemini, Mistral, Bedrock, Groq, common OpenRouter ids; family prefix rules | **Done** | Critical | `registry/model_catalog.yaml` | ≥50 entries; prefix rules for `claude-*`, `gpt-*`, `gemini-*` |
| M-LLM-X.1.3 | **`resolve_context_window_tokens(provider, model, options)`** — single resolver per ADR-LLM-002 order | **Done** | Critical | `registry/context_window.py` | Unit: override > exact > prefix > default |
| M-LLM-X.1.4 | **Wire all 19 adapters** — remove inline dicts; call resolver at `__init__` | **Done** | Critical | `providers/*` | Resolver at ctor; legacy dicts as fallback only |
| M-LLM-X.1.5 | **`LLMProfile` propagates `context_window_tokens`** to every adapter ctor via `create_adapter()` | **Done** | Critical | `registry/profile.py` | Test: Claude override |
| M-LLM-X.1.6 | **Env `INTERGRAX_LLM_MODEL_CATALOG_PATH`** optional YAML overlay | **Done** | Medium | `model_catalog.py` | Operator merge without code deploy |
| M-LLM-X.1.7 | **Capability flags from catalog** — populate `supports_vision` / tools / structured when `ModelRecord` known | **Planned** | Medium | `providers/*`, `modality_capabilities.py` | Conformance subset; W-ML.1 alignment |

---

#### Wave M-LLM-X-2 — Dynamic gateway metadata (P1)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.2.1 | **`OpenRouterModelMetadataClient`** — optional `/models` fetch, TTL cache | **Planned** | High | `registry/gateway_metadata/` | Mocked unit tests |
| M-LLM-X.2.2 | **Merge gateway metadata into catalog session cache** | **Planned** | High | `model_catalog.py` | context from API when present |
| M-LLM-X.2.3 | **`LLMProfile.options["fetch_gateway_metadata"]=True`** opt-in | **Planned** | Medium | `profile.py` | Default false (no network in unit gate) |
| M-LLM-X.2.4 | **Diagnostic `ModelCatalogMissDiagV1`** when fallback default used | **Planned** | Medium | tracing | Trace on first miss per model/run |

---

#### Wave M-LLM-X-3 — Token accounting consistency (P0)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.3.1 | **`verify_context_preflight`** uses `adapter.count_messages_tokens(messages)` | **Done** | Critical | `context_preflight.py` | Unit: adapter path exercised |
| M-LLM-X.3.2 | **`count_message_tokens` helper** — delegate to adapter when provided | **Partial** | High | `context_preflight.py` | Adapter default when counter None |
| M-LLM-X.3.3 | **`ContextBudgetPolicy.from_adapter(adapter)`** factory — derive `max_tokens_estimate` from `resolve_input_budget_tokens` | **Done** | High | `context_budget.py` | Unit test |
| M-LLM-X.3.4 | **`scripts/check_context_preflight_uses_adapter_tokens.py`** CI guard | **Done** | High | `scripts/` | Added to §6.1 maintenance |
| M-LLM-X.3.5 | **`engine_history_layer`** history token count via `adapter.count_messages_tokens` | **Done** | High | `engine_history_layer.py` | Already delegated; preflight aligned LC-2 |

---

#### Wave M-LLM-X-4 — Profile failover chain (P1)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.4.1 | **`LLMProfile.fallback_profiles: tuple[LLMProfile, ...]`** | **Done** | High | `registry/profile.py` | Pydantic validation; extra=forbid |
| M-LLM-X.4.2 | **`FailoverLLMAdapter`** — wraps primary; tries chain on retriable errors | **Done** | Critical | `registry/failover_adapter.py` | Unit: 429 → fallback success |
| M-LLM-X.4.3 | **`LLMProfile.create_adapter_with_failover()`** | **Done** | High | `profile.py` | Integration with `LLMCallConfig.retry_on_status` |
| M-LLM-X.4.4 | **Trace `LLMRoutingAttemptDiagV1`** per failover attempt | **Done** | Medium | observability | Fields: profile_id, provider, model, error |
| M-LLM-X.4.5 | **Tier-3 wiring** — `ApplicationEnvironmentProfile` optional fallback list | **Planned** | Medium | `environment_profile.py`, `nexus_factory.py` | Host smoke test |

---

#### Wave M-LLM-X-5 — Runtime routing and AHI integration (P1)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.5.1 | **Expand `ModelRouter`** — hints: `balanced`, `cheapest`, `fastest`, `quality`; map to profile index | **Done** | High | `registry/model_router.py` | Unit: each hint |
| M-LLM-X.5.2 | **`resolve_runtime_llm_profile(env, policy_hint)`** — single entry for Nexus factory | **Done** | Critical | `applications/_shared/llm_resolver.py` | Replaces ceremonial AHI-only wiring |
| M-LLM-X.5.3 | **Wire `resolve_live_model_routing_wiring` → actual adapter selection** | **Done** | Critical | `llm_routing_wiring.py`, `llm_resolver.py` | AUDIT-IDEAL-6.2 → **Done** |
| M-LLM-X.5.4 | **`StepLLMRouter` → `LLMAdapter` bridge** — async wrapper over `generate_messages` | **Done** | High | `agents/authoring/llm_router.py` | ACP tests use real adapter port |
| M-LLM-X.5.5 | **Remove stub echo path** when `llm_port` unset in production hosts | **Done** | Medium | `acp_run.py` | Fail-fast if adapter missing in STRICT profile |

---

#### Wave M-LLM-X-6 — Plugin provider story (P2)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.6.1 | **`LLMProfile` accepts `provider: str`** without enum (validated against registry) | **Planned** | Medium | `profile.py` | `register()` slug works without enum PR |
| M-LLM-X.6.2 | **Example custom gateway** in `tests/unit/llm_adapters/test_custom_provider_register.py` | **Planned** | Medium | tests | Conformance subset |
| M-LLM-X.6.3 | **Architecture §Extension** + AGENT_CREATION_GUIDE cross-link | **Done** | Low | docs | USAGE + canon §Extension |

---

#### Wave M-LLM-X-7 — Developer experience (P2)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.7.1 | **`intergrax/llm_adapters/USAGE.md`** — quickstart, env matrix, overrides, failover, catalog | **Done** | High | Tier-0 module root | Linked from architecture |
| M-LLM-X.7.2 | **`LLMProfile.validate_runtime()`** — catalog hit, key, context > 0 | **Done** | Medium | `profile.py` | `check_llm_profile_runtime.py` + `intergrax doctor` |
| M-LLM-X.7.3 | **`scripts/check_model_catalog_coverage.py`** — gate warns on adapter default models missing from YAML | **Planned** | Medium | CI | §6.1 maintenance |
| M-LLM-X.7.4 | **Scaffold `new-agent` template** — comment block pointing to USAGE + catalog override | **Planned** | Low | `scaffold/new_agent.py` | Scaffold test |
| M-LLM-X.7.5 | **Cohere slug guidance** — document `cohere` (compat) vs `cohere_native` selection in USAGE | **Done** | Low | `USAGE.md` §Providers | Reduces dual-slug confusion |

---

#### Wave M-LLM-X-8 — Closeout

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.8.1 | **Architecture audit register** — all LLM-AUDIT-* → Done | **Planned** | Critical | `architecture/LLM_ADAPTERS.md` | Matches code |
| M-LLM-X.8.2 | **AUDIT-IDEAL-6.2–6.7** → Done in master register | **Planned** | High | `AUDIT_IDEAL_2026.md` | Gate scripts |
| M-LLM-X.8.3 | **Implementation journal** + maturity re-score L3+ | **Planned** | Medium | `implementation-journal/` | `check_implementation_journal.py` |

---

#### Wave M-LLM-X-9 — LLM routing rules (Protocol + custom classes)

**Source:** Mode I idea audit (2026-06-19) — dynamic model selection by task state, budget, and author logic.  
**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) **Accepted**  
**Goal:** Single developer-facing routing surface — built-in parametric rules and Tier-3 custom classes on one `LLMRoutingRule` Protocol; AHI L4 remains optional overlay.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.9.1 | **`LLMRoutingRule` Protocol + `LLMRoutingRuleBase` ABC** — `matches()`, `resolve()`, `rule_id`, `priority` | **Done** | **Critical** | `intergrax/llm_adapters/routing/contracts.py` | Unit: Protocol structural subtyping |
| M-LLM-X.9.2 | **`RoutingContext`, `RoutingTarget`, `LLMRoutingProfile`** Pydantic models | **Done** | **Critical** | `routing/contracts.py` | Immutable context snapshot |
| M-LLM-X.9.2b | **`LLMRoutingEvaluator`** — priority sort, first-match, allowlist guard | **Done** | **Critical** | `routing/evaluator.py` | Unit: priority + allowlist rejection |
| M-LLM-X.9.3 | **Built-in rules package** — `BudgetBelowRule`, `TaskClassRule`, `TokenThresholdRule`, `BudgetExceededDegradeRule` | **Done** | High | `routing/builtin_rules.py` | Each implements same Protocol |
| M-LLM-X.9.4 | **Tier-3 `LLMRoutingProfile` on `ApplicationEnvironmentProfile`** + `CapabilityBundle` wire | **Done** | **Critical** | `environment_profile/`, `llm_resolver.py` | `test_llm_routing_resolver.py` |
| M-LLM-X.9.5 | **Hot path wire** — `resolve_llm_adapter()` evaluates rules using budget meter + `task_class` | **Done** | **Critical** | `llm_resolver.py`, `llm_routing_wiring.py` | Integration test: rule triggers profile swap |
| M-LLM-X.9.6 | **Unify `BudgetReactionProfile.degrade_model`** with `BudgetExceededDegradeRule` | **Done** | High | `budget_enforcing_llm_router.py`, `builtin_rules.py` | `cheapest_allowed_model_hint`; ACP-TOK-3 tests green |
| M-LLM-X.9.7 | **`DynamicLLMRouter` wrapper** — per-step model swap within run (extends budget-enforcing pattern) | **Done** | Medium | `agents/authoring/dynamic_llm_router.py` | `test_dynamic_llm_router.py` |
| M-LLM-X.9.8 | **USAGE.md cookbook** — built-in vs custom class, testing, allowlist, HF via vLLM | **Done** | High | `intergrax/llm_adapters/USAGE.md` | Linked from architecture §Routing rules |
| M-LLM-X.9.9 | **`scripts/check_llm_routing_rules.py`** — reference hosts validate allowlist conformance | **Done** | Medium | `scripts/` | `check_audit_ideal_gates.py` umbrella |

**Suggested PR order (X-9):** 9.1 → 9.2 → 9.2b → 9.3 → 9.4 → 9.5 → 9.6 → 9.7 → 9.8 → 9.9.

**Cross-domain:** AHI-MAINT-05 (`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`) — bandit arms → `ProfileVersion` `llm_routing`. **X-10** adds **AHI-MAINT-06** (persistent profile versions).

---

#### Wave M-LLM-X-10 — LLM routing enterprise closeout + predefined rule catalog

**Source:** Post X-9 enterprise readiness review (2026-06-19) — foundation L3+ Done; start-of-run enterprise closeout **Done** (2026-06-19).  
**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) — no new ADR unless `ProfileVersion` contract changes (**AHI-MAINT-06**).  
**Goal:** Parametric **platform catalog** (Tier-0 predefined classes) + **custom rules** (Tier-3 Protocol) + automatic runtime context on materialize + observability + reference host + acceptance proof.

Ship **12+** ready-made `LLMRoutingRule` implementations in `intergrax/llm_adapters/routing/builtin_rules.py` — each parametric via `__init__`, documented in USAGE. Authors **prefer** builtins for common cases; **custom `LLMRoutingRule` subclasses remain fully supported** (ADR-LLM-003).

| Class | Constructor params (examples) | `matches` semantics |
|-------|------------------------------|---------------------|
| `BudgetBelowRule` | `threshold: float`, `profile` or `hint` | `budget_remaining_ratio < threshold` — **Done** (X-9) |
| `BudgetAboveRule` | `threshold: float`, `profile` or `hint` | `budget_remaining_ratio > threshold` |
| `BudgetExceededDegradeRule` | — | `budget_degrade_active` → `CHEAPEST` — **Done** (X-9) |
| `TaskClassInRule` | `classes: tuple[str,…]`, `profile` or `hint` | `task_class in classes` — alias / extend `TaskClassRule` |
| `TaskClassNotInRule` | `classes`, `profile` or `hint` | `task_class not in classes` |
| `TokenUsedAboveRule` | `threshold: int`, `hint` | `tokens_used > threshold` — alias / extend `TokenThresholdRule` |
| `TokenUsedBelowRule` | `threshold: int`, `profile` or `hint` | `tokens_used < threshold` |
| `StepIndexAtLeastRule` | `min_step: int`, `profile` or `hint` | `step_index >= min_step` |
| `StepIndexBelowRule` | `max_step: int`, `profile` or `hint` | `step_index < max_step` |
| `AgentIdInRule` | `agent_ids: tuple[str,…]`, `profile` or `hint` | `agent_id in agent_ids` |
| `TenantIdInRule` | `tenant_ids: tuple[str,…]`, `profile` or `hint` | `tenant_id in tenant_ids` |
| `ModelHintPresentRule` | `profile` or `hint` | `model_hint` is non-empty |
| `PolicyHintRule` | `hint: RoutingHint` | always `matches` → resolve hint (use with low priority) |
| `CompositeAllRule` | `rules: tuple[LLMRoutingRule,…]`, `profile` or `hint` | all nested `matches()` true |
| `CompositeAnyRule` | `rules: tuple[LLMRoutingRule,…]`, `profile` or `hint` | any nested `matches()` true |
| `AlwaysRule` | `profile` or `hint`, `priority=-100` | unconditional fallback |

Export `BUILTIN_ROUTING_RULES: tuple[type[LLMRoutingRuleBase], ...]` + `builtin_rule_catalog.md` table in USAGE.

##### Enterprise closeout tasks

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.10.1 | **Predefined rule catalog** — 12+ parametric classes (table above); rename/alias X-9 rules where needed | **Done** | **Critical** | `routing/builtin_rules.py`, `routing/__init__.py` | Parametrized unit test per class |
| M-LLM-X.10.2 | **`build_routing_context_from_runtime()`** — auto-fill `task_class`, `budget_remaining_ratio`, `tokens_used`, `step_index`, `budget_degrade_active`, `tenant_id`, `agent_id` from Nexus / kernel / budget meter | **Done** | **Critical** | `llm_resolver.py`, `runtime_config_bridge.py`, `nexus_factory.py` | No manual `routing_context=` required on default host path |
| M-LLM-X.10.3 | **Routing observability** — emit `matched_rule_id`, `routing_reason`, selected `profile_id` on trace (`LLMRoutingAttemptDiagV1` extend or `LLMRoutingRuleDiagV1`) | **Done** | High | `llm_resolver.py`, `observability/` | Gate test maps schema |
| M-LLM-X.10.4 | **Reference host** — `lab_application` (or product reference) manifest with `LLMRoutingProfile` using **predefined classes only** | **Done** | High | `applications/lab_application/` | `check_llm_routing_rules.py` scans host |
| M-LLM-X.10.5 | **E2E acceptance** — Nexus run: `BudgetBelowRule` switches profile when budget crosses threshold | **Done** | High | `tests/acceptance/llm_routing/` | `-m gate` green |
| M-LLM-X.10.6 | **Global `DynamicLLMRouter` wire** — ACP / `harness_host_runtime` auto-wrap when `llm_routing_profile` set | **Done** | Medium | `acp_run.py`, `harness_host_runtime.py` | Per-step swap in agent run test |
| M-LLM-X.10.7 | **USAGE + architecture** — predefined rules matrix, enterprise readiness checklist, composition examples (`CompositeAllRule`) | **Done** | Medium | `USAGE.md`, architecture §Routing rules | Linked from AGENT_CREATION_GUIDE |
| M-LLM-X.10.8 | **CI gate extend** — `check_llm_routing_rules.py` validates catalog exports + reference host allowlist | **Done** | Medium | `scripts/` | Registered in `check_audit_ideal_gates.py` |

**Suggested PR order (X-10):** 10.1 → 10.2 → 10.3 → 10.4 → 10.5 → 10.6 → 10.7 → 10.8.

**Cross-domain:** **AHI-MAINT-06** — `ProfileVersionStore` `artifact_type=llm_routing` + persistent bandit (not `InMemoryBanditStateStore` only).

**Note on X-9.5:** X-9 delivered evaluator + optional `routing_context` parameter; **X-10.2** auto context on **materialize** path (not all `resolve_llm_adapter()` call sites — see **X-11.3**).

**X-10 scope delivered:** start-of-run routing + ACP per-step. **Mid-run Nexus `llm_adapter` re-eval** → **M-LLM-X.11** (LLM-AUDIT-18).

---

#### Wave M-LLM-X-11 — Routing enterprise hardening (mid-run Nexus)

**Source:** Post X-10 enterprise review (2026-06-19) — X-10 closed LLM-AUDIT-17; gaps remain for **live mid-run** routing on Nexus core adapter and full observability loop.  
**ADR:** ADR-LLM-003 (unchanged); **no new ADR** unless `RoutingEvaluatingLLMAdapter` introduces new cross-tier contract.  
**Goal:** Strict **enterprise grade** routing — budget/step/degrade changes during a Nexus run re-select profile; trace on every evaluation; true E2E proof.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.11.1 | **`RoutingEvaluatingLLMAdapter`** — wrap `LLMAdapter`; before each `generate_messages` / `generate_with_tools` / `generate_structured` re-run `LLMRoutingEvaluator` with live context; rebuild inner adapter when profile/hint changes | **Done** | **Critical** | `intergrax/llm_adapters/routing/evaluating_adapter.py`, `llm_resolver.py` | Unit: budget threshold crossing swaps profile identity |
| M-LLM-X.11.2 | **`refresh_llm_routing_context()`** — update `RuntimeConfig.llm_routing_context` from budget meter (`ResolvedBudgetLimits` / usage), `step_index`, `budget_degrade_active`, `task_class` during Nexus step loop | **Done** | **Critical** | `context_bridge.py`, `llm_routing_runtime_bridge.py`, `uaep.py` | Context fields change between synthetic steps in test |
| M-LLM-X.11.3 | **Unify `resolve_llm_adapter()` call sites** — `nexus_factory`, `environment_wiring`, `critic_tool_wiring`, `harness_host_runtime` pass `build_routing_context_from_runtime()` (tenant, metadata, budget when available) | **Done** | **Critical** | `applications/_shared/*.py` | No bare `resolve_llm_adapter(env)` when `llm_routing_profile` set |
| M-LLM-X.11.4 | **Observability loop** — emit `LLMRoutingRuleDiagV1` on **every** evaluation (evaluating adapter + `DynamicLLMRouter`); `trace_bridge` gate for `routing_rule` schema; `LLMRoutingAllowlistViolationDiagV1` on `AllowlistViolationError` | **Done** | High | `llm_routing_attempt.py`, `trace_bridge.py`, `runtime_state.py` | `test_trace_bridge_maps_llm_routing_rule_schema` green |
| M-LLM-X.11.5 | **True E2E acceptance** — full ACP or Nexus run: seed budget → execute LLM step(s) → budget crosses `BudgetBelowRule` threshold → assert model/profile change in trace or adapter meter | **Done** | High | `tests/acceptance/llm_routing/` | `-m gate` green; evaluating adapter mid-run swap |
| M-LLM-X.11.6 | **Harness host parity** — when `llm_routing_profile` set, wire evaluating adapter or document ACP-only; minimum: `harness_host_runtime` uses evaluating wrapper on `llm_adapter` passed to `build_nexus_loop_from_environment` | **Done** | Medium | `harness_host_runtime.py`, `runtime_config_bridge.py` | Host integration test |
| M-LLM-X.11.7 | **Docs maturity re-score** — architecture L4→L5 criteria; USAGE §mid-run routing; clarify X-10 vs X-11 scope; custom + builtin authoring paths | **Done** | Medium | `architecture/LLM_ADAPTERS.md`, `USAGE.md` | Linked from hub |
| M-LLM-X.11.8 | **CI gate** — `scripts/check_llm_routing_context_wiring.py` (or extend `check_llm_routing_rules.py`) — static scan: no `resolve_llm_adapter(env)` without context bridge on Tier-3 wiring modules | **Done** | Medium | `scripts/` | Registered in `check_audit_ideal_gates.py` |

**Suggested PR order (X-11):** 11.1 → 11.2 → 11.3 → 11.4 → 11.5 → 11.6 → 11.7 → 11.8.

**Cross-domain:** `NEXUS_EXECUTION_FLOW` (step loop hook) · `OBSERVABILITY` (trace schema) · `AGENT_CONTRACTS_AND_ASSEMBLY` (ACP budget meter).

**Closes:** **LLM-AUDIT-18**. **Prerequisite for:** **M-LLM-X.12** (strict L5 gaps).

---

#### Wave M-LLM-X-12 — Routing strict enterprise closeout

**Source:** Post X-11 architecture audit (2026-06-19) — evaluating wrapper delivered; production review found budget-meter drift, narrow path coverage, tier import violation.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) § Routing strict enterprise closeout  
**ADR:** ADR-LLM-003 amendment or **ADR-LLM-004** if evaluating factory moves to Tier-3-only surface (decide in 12.2).  
**Goal:** Honest **L5** — budget rules fire on real usage; sync on all Nexus hot paths; tier-clean; ACP trace parity; production E2E.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-X.12.1 | **Budget meter ↔ routing context** — `RoutingEvaluatingLLMAdapter` aggregates inner `usage` for tracker **or** re-registers inner on swap; `sync_llm_routing_snapshot` uses accurate `tokens_used` | **Planned** | **P0 Critical** | `evaluating_adapter.py`, `llm_routing_runtime_bridge.py`, `runtime_state.py` | Unit: after N inner calls, `budget_remaining_ratio` drops; `BudgetBelowRule` fires without mock |
| M-LLM-X.12.2 | **Tier boundary refactor** — move evaluating wrapper + `create_adapter_for_routing_evaluation` to `applications/_shared/`; Tier-0 keeps Protocol + evaluator only; inject `AdapterFactory` callback | **Planned** | **P1 Critical** | `evaluating_adapter.py` → `applications/_shared/`, `llm_resolver.py` | `intergrax/llm_adapters/` has zero `applications/` imports |
| M-LLM-X.12.3 | **Nexus-wide context sync** — `sync_llm_routing_snapshot_for_state` (or equivalent) on graph step kernel / CE pre-LLM hooks, not only UAEP | **Planned** | **P1 Critical** | `NEXUS_EXECUTION_FLOW` hook, `context_bridge` consumers | Integration: graph node run updates `step_index` in snapshot |
| M-LLM-X.12.4 | **Per-call context refresh** — context provider or `_refresh_inner_adapter` triggers sync when `RuntimeState` available (multi-LLM-call steps) | **Planned** | **P1 Critical** | `llm_routing_runtime_bridge.py`, `evaluating_adapter.py` | Test: two LLM calls in one step see updated budget ratio |
| M-LLM-X.12.5 | **AHI live context on swap** — pass current `RoutingContext` to `resolve_live_model_routing_wiring` in `create_adapter_for_routing_evaluation` | **Planned** | **P1** | `llm_resolver.py` | AHI hint respects budget snapshot in unit test |
| M-LLM-X.12.6 | **`budget_degrade_active` in Nexus sync** — map from runtime policy / cost envelope to `LLMRoutingRuntimeSnapshot` | **Planned** | **P1** | `llm_routing_runtime_bridge.py` | `BudgetExceededDegradeRule` test on Nexus path |
| M-LLM-X.12.7 | **Per-run observability** — replace process-global `set_routing_evaluation_observer` with instance-bound callbacks on `RuntimeState` / evaluating adapter | **Planned** | **P2** | `llm_resolver.py`, `runtime_state.py` | Concurrent run test: traces do not cross-contaminate |
| M-LLM-X.12.8 | **ACP trace parity** — wire `on_evaluated` → `emit_llm_routing_rule_diag` in `acp_run` for `DynamicLLMRouter` | **Planned** | **P2** | `acp_run.py`, `dynamic_llm_router.py` | ACP integration test emits `llm_routing_rule` step |
| M-LLM-X.12.9 | **First-eval profile correction** — evaluating adapter swaps on first call when rule profile ≠ resolver materialized profile | **Planned** | **P2** | `evaluating_adapter.py` | Unit: mismatched start profile corrected on first `generate_messages` |
| M-LLM-X.12.10 | **Production E2E acceptance** — UAEP or Nexus run with real usage meter: budget crosses threshold → model change + trace events (no factory mock) | **Planned** | **P2** | `tests/acceptance/llm_routing/` | `-m gate` green |
| M-LLM-X.12.11 | **Docs + audit honesty** — architecture L4+ label, strict L5 checklist; USAGE known limitations; close **LLM-AUDIT-19** | **Planned** | **P2** | `architecture/`, `USAGE.md` | Hub + audit register synced |
| M-LLM-X.12.12 | **Secondary LLM surfaces policy** — document or extend evaluating wrap to planner / critic / websearch LLM (explicit product decision) | **Planned** | **P3** | `reasoning_wiring.py`, architecture § | Policy row in architecture; no silent bypass |

**Suggested PR order (X-12):** 12.1 → 12.2 → 12.3 → 12.4 → 12.5 → 12.6 → 12.7 → 12.8 → 12.9 → 12.10 → 12.11 → 12.12.

**Cross-domain:** `NEXUS_EXECUTION_FLOW` (graph sync) · `OBSERVABILITY` (trace payload enrichment) · `AGENT_CONTRACTS_AND_ASSEMBLY` (ACP trace) · `UNIFIED_EXECUTION_RUNTIME` (budget degrade).

**Closes:** **LLM-AUDIT-19**. **Blocks:** **M-LLM-X.8** honest domain closeout.

---

### M-LLM-X — Suggested PR order

```text
PR-1:  M-LLM-X.0.2 ADR-LLM-002 — Done
PR-2:  M-LLM-X.1.1 → 1.3 (catalog + resolver — no adapter wire yet)
PR-3:  M-LLM-X.1.4 → 1.5 (adapter migration + profile override)
PR-4:  M-LLM-X.3.1 → 3.4 (preflight — highest user-visible win after catalog)
PR-5:  M-LLM-X.1.2 + 1.6 (YAML seed + operator overlay)
PR-6:  M-LLM-X.4.1 → 4.3 (failover adapter)
PR-7:  M-LLM-X.5.1 → 5.3 (runtime routing + AHI)
PR-8:  M-LLM-X.5.4 → 5.5 (ACP bridge)
PR-9:  M-LLM-X.2.* (OpenRouter metadata — optional network)
PR-10: M-LLM-X.6.* + 7.* (plugins + DX)
PR-11: M-LLM-X.9.* (routing rule Protocol — ADR-LLM-003)
PR-12: M-LLM-X.10.* (routing enterprise closeout — start-of-run + ACP) — Done
PR-13: M-LLM-X.11.* (routing hardening — mid-run Nexus) — Done
PR-14: M-LLM-X.12.* (routing strict L5 closeout)
PR-15: M-LLM-X.8.* closeout (after X-12)
```

**Estimated effort:** X-12 · ~2–3 PRs harness cadence · 2–3 weeks.

---

### M-LLM-X — Traceability (audit gap → task ID)

| Audit gap (2026-06-14) | Task IDs |
|------------------------|----------|
| LLM-AUDIT-1 — No central ModelCatalog | M-LLM-X.1.1–1.6 |
| LLM-AUDIT-2 — context override Ollama-only | M-LLM-X.1.5 |
| LLM-AUDIT-3 — Preflight chars/4 | M-LLM-X.3.1–3.4 |
| LLM-AUDIT-4 — ModelRouter not on hot path | M-LLM-X.5.1–5.3 |
| LLM-AUDIT-5 — No failover chain | M-LLM-X.4.1–4.5 |
| LLM-AUDIT-6 — StepLLMRouter stub | M-LLM-X.5.4–5.5 |
| LLM-AUDIT-7 — OpenRouter 32k fallback | M-LLM-X.2.*, X-1.2 |
| LLM-AUDIT-8 — No USAGE.md | M-LLM-X.7.1, 7.5 |
| LLM-AUDIT-9 — AUDIT-IDEAL-6.2 partial | M-LLM-X.5.3, X-8.2 |
| LLM-AUDIT-10 — Plugin provider undocumented | M-LLM-X.6.* |
| LLM-AUDIT-11 — ContextBudgetPolicy fixed 4k decoupled from adapter | M-LLM-X.3.3 |
| LLM-AUDIT-12 — Prefix heuristics only on Bedrock today | M-LLM-X.1.2, 1.3 |
| LLM-AUDIT-13 — Cohere dual slug (`cohere` / `cohere_native`) DX | M-LLM-X.7.5 |
| LLM-AUDIT-14 — Capability flags not catalog-driven | M-LLM-X.1.7 |
| LLM-AUDIT-15 — History layer token count inconsistent with preflight | M-LLM-X.3.5 |
| LLM-AUDIT-16 — No unified routing rule contract (idea audit 2026-06-19) | M-LLM-X.9.* |
| LLM-AUDIT-17 — Routing enterprise E2E start-of-run + ACP (context, trace, reference host) | M-LLM-X.10.* **Done** |
| LLM-AUDIT-18 — Routing mid-run Nexus live re-eval, context refresh, full trace, true E2E | M-LLM-X.11.* **Done** (X-11 scope) |
| LLM-AUDIT-19 — Routing strict L5: budget meter, all Nexus paths, tier boundary, production E2E | M-LLM-X.12.* **Planned** |
| tiktoken OpenAI-centric estimate (all providers) | **Deferred** — document limitation in USAGE; vendor tokenizer plugins post-X |
| Single `RuntimeConfig.llm_adapter` per run (multi-model) | M-LLM-X.4–5 (profile chain + routing); no multi-adapter pool in X |
| Distributed Redis rate limit host wiring | **Ops** — document in USAGE X.7.1; not LLM-AUDIT tier-0 code |

---

### 6.1v Harness implementation queue — LLM completion response envelope (closed)

**Purpose:** Single ordered list for **Phase M-LLM-R** (Band 2z). **Closed 2026-06-06** — **39/39 Done**. Runs **in parallel** with W-ADAPT waves 5–7 (Tier-0 LLM contract; independent of L4 runtime loop).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **M-LLM-R.0.2–0.3** | Docs | **Done** | ADR-LLM-001 + canon §5.2.2 addendum | Linked from plan |
| 2 | **M-LLM-R.1.1–1.8** | Code | **Done** | Contract types + builders + public exports | Import smoke; no dict returns |
| 3 | **M-LLM-R.2.1–2.6** | Code | **Done** | `LLMAdapter` ABC typed signatures | ABC compiles; stubs updated |
| 4 | **M-LLM-R.3.1–3.7** | Code | **Done** | All provider adapters return envelope | Conformance per provider family |
| 5 | **M-LLM-R.4.1–4.6** | Code | **Done** | Nexus runtime consumers | `test_context_preflight + ACP agent tests` + tool planner |
| 6 | **M-LLM-R.5.1–5.3** | Code | **Done** | RAG + websearch + legacy | RAG unit tests green |
| 7 | **M-LLM-R.6.1–6.4** | Code | **Done** | Agents + scaffold + CI lint | `check_llm_adapter_typed_returns.py` + `check_agents_llm_adapter_response.py` |
| 8 | **M-LLM-R.7.1–7.5** | Code | **Done** | Usage alignment + replay/trace bridge | `test_replay_engine` + diagnostics |
| 9 | **M-LLM-R.8.1–8.4** | Docs/CI | **Done** | Docs + conformance + closeout | M-LLM.14 Done; Appendix L complete |

**Suggested PR order:** See [Phase M-LLM-R — Suggested PR order](plan/LLM_ADAPTERS.md).

**Explicitly excluded:** K.1, K.2, product HTTP API DTOs, provider SDK rewrites — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1w Harness implementation queue — Integration expansion (M.6 P4 closed)

**Purpose:** Ordered backlog for **Phase M.6 P4** (Band 2aa). **Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127**.  
**Register:** [M.6 P4 — Master register](#m6-p4--master-register-28-slugs) · **Execution order:** [§6.2ae](#62ae-phase-m6-p4-execution-order--done)  
**Policy:** One slug per PR; runs **in parallel** with §6.1 maintenance — pull only when harness ops/adaptive/INT health needs the slug.

| Order | Wave | IDs | Slugs | Priority | Status |
|-------|------|-----|-------|----------|--------|
| 0 | CAT | M-P4-CAT.1, M-P4-CAT.2 | *(categories)* | **P0** | **Done** (beta) |
| 1 | H-INT-1 | M-P4.1–M-P4.4 | `pgvector`, `duckdb`, `influxdb`, `timescaledb` | P0/P1 | **Done** |
| 2 | H-INT-2 | M-P4.5–M-P4.7 | `grafana`, `loki`, `tempo` | **P0** | **Done** |
| 3 | H-INT-3 | M-P4.8–M-P4.11 | `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`, `doppler` | P0/P1 | **Done** |
| 4 | H-INT-4 | M-P4.12–M-P4.16 | `unleash`, `launchdarkly`, `github_actions`, `redpanda`, `cloudflare_r2` | P0/P1 | **Done** |
| 5 | H-INT-5 | M-P4.17–M-P4.28 | `memgraph`, `falkordb`, `incident_io`, `kubernetes`, `servicenow`, `bitbucket`, `asana`, `sendgrid`, `mailgun`, `mlflow`, `huggingface_hub`, `ollama` | P1/P2 | **Done** |

**Per-slug checklist (M.4):** contract → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → `architecture/INTEGRATIONS.md` → canon §7.1.3 row → gate green → paydown log row.

**Explicitly excluded:** CRM, payments, blockchain, duplicate vector SaaS, LLM vendor APIs — see [M.6 P4 register](#m6-p4--harness-platform-expansion-planned).

---

### 6.2ad Phase M-LLM-R execution order (Band 2z — closed 2026-06-06)

**Status:** **Done** · register: [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

```text
Wave M-LLM-R-0 (planning):     M-LLM-R.0.2 → 0.3  (0.1 **Done**)
Wave M-LLM-R-1 (contracts):    M-LLM-R.1.1 → 1.8
Wave M-LLM-R-2 (ABC):          M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave M-LLM-R-3 (providers):    M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7
Wave M-LLM-R-4 (Nexus):        M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave M-LLM-R-5 (RAG/web):      M-LLM-R.5.1 → 5.2 → 5.3
Wave M-LLM-R-6 (agents):       M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave M-LLM-R-7 (obs/replay):   M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave M-LLM-R-8 (closeout):     M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Prerequisites:** Phase M-LLM **Done** (M-LLM.1–13); no dependency on W-ADAPT runtime L4 gate.

**Parallelism:** May run alongside W-ADAPT-5+; coordinate M-LLM-R.7.5 with W-ADAPT signal work if both touch `signal_collector.py`.

**Closeout gate:** `scripts/check_llm_adapter_typed_returns.py` + `scripts/check_agents_llm_adapter_response.py` + full `tests/unit/llm_adapters/` gate green (M-LLM-R.8.3, M-LLM-R.6.4).### 6.2ac Phase W-ADAPT execution order (Band 2y — closed)

**Status:** **Done** (2026-06-02) · register: [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · queue: [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)

```text
Wave W-ADAPT-0 (planning):        W-ADAPT-0.2 → 0.3 → 0.4 → 0.5  (**Done**)
Wave W-ADAPT-1 (observe L4-O):    W-ADAPT-1.1 → 1.12  (**Done**)
Wave W-ADAPT-2 (recommend L4-R):  W-ADAPT-2.1 → 2.12  (**Done**)
Wave W-ADAPT-3 (shadow L4-S):      W-ADAPT-3.1 → 3.2 → 3.3 → 3.4 → 3.6 → 3.7 → 3.5  (**Done**)
Wave W-ADAPT-4 (apply L4-A):       W-ADAPT-4.1 → 4.10  (**Done**)
Wave W-ADAPT-5 (verify L4-V):      W-ADAPT-5.1 → 5.3 → 5.4 → 5.5 → 5.2 → 5.11 → 5.6 → 5.7 → 5.8 → 5.9 → 5.10 → 5.12  (**Done**)
Wave W-ADAPT-6 (patterns):         W-ADAPT-6.2 → 6.1 → 6.3 → 6.5 → 6.4  (**Done**)
Wave W-ADAPT-7 (Tier-3 + docs):    W-ADAPT-7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7  (**Done**)
```

**Prerequisites:** Phase V + V-REM + W-OPS + EVAL + COST + CG closeouts **Done**.

**Runtime L4 gate:** `uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime` (added in W-ADAPT-5.6).

---

### Phase M-LLM — LLM Adapter Layer (Tier-0)

**Canon:** §5.2.2 · **Doc:** [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md)  
**Goal:** One `LLMAdapter` contract, lazy registry, streaming + native tools + structured output across commercial and self-hosted providers.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-LLM.1 | Shared `_shared/` (messages, tools, retry, conformance) | **Done** | 2026-05-30 |
| M-LLM.2 | Seven core providers hardened | **Done** | OpenAI, Claude, Azure, Gemini, Mistral, Bedrock, Ollama |
| M-LLM.3 | Groq + vLLM (OpenAI-compatible) | **Done** | `openai_compat_providers.py` |
| M-LLM.4 | Bedrock Converse + tools + stream | **Done** | `INTERGRAX_BEDROCK_USE_CONVERSE`, `converse_stream` |
| M-LLM.5 | Conformance tests in CI gate | **Done** | `tests/unit/llm_adapters/` |
| M-LLM.6 | `architecture/LLM_ADAPTERS.md` + README section | **Done** | 19 providers |
| M-LLM.7 | OpenAI-compat expansion + Vertex + `LLMProfile` | **Done** | Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere, Vertex |
| M-LLM.8 | Optional network smoke workflow | **Done** | Weekly schedule + `workflow_dispatch` |
| M-LLM.9 | Azure refactor (Chat Completions base) | **Done** | Thin `AzureOpenAIChatAdapter` |
| M-LLM.10 | Production hardening | **Done** | Metrics, builtin conformance, `LLMProfile`, Bedrock tools stream, `cohere_native`, `azure_ai_inference` |
| M-LLM.11 | Production ops layer | **Done** | OTLP/Prometheus routes, tenant metrics, rate limit + circuit breaker, secrets map, PR guard, extended network smoke |
| M-LLM.12 | Nexus + governance wiring | **Done** | `llm_tenant_scope`, runtime metrics plugin, `INTERGRAX_LLM_TENANT_MAX_TOKENS` quota |
| M-LLM.13 | Observability + secrets + distributed limits | **Done** | Pushgateway, `architecture/LLM_ADAPTERS.md` § Observability, Vault loader, Redis rate limit, governance warn |
| M-LLM.14 | Typed completion envelope (`LLMAdapterResponse`) | **Done** | Phase M-LLM-R — [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed) · gate **776** |

---

### Phase M-LLM-R — LLM Completion Response Envelope (audit 2026-06-06)

**Source:** Tier-0 LLM adapter audit (2026-06-06) — `generate_messages` returns `str`; `generate_with_tools` returns untyped dict via `make_tool_result`; SDK metadata (`finish_reason`, `response_id`, cached/reasoning tokens, refusal) discarded; usage only via side-channel `LLMAdapterUsageLog`; replay `LLMCallInfo` not fed from adapter returns.  
**Canon:** §5.2.2 · **Doc:** [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) · **Traceability:** [Appendix L](#appendix-l--llm-completion-response-envelope-traceability-phase-m-llm-r)  
**Status:** **Done** (2026-06-06) — **39/39 Done**  
**Priority ladder:** **Band 2z** (§4.0) — **parallel with W-ADAPT waves 5–7** (Tier-0; no Nexus primitive changes beyond consumer wiring)  
**Execution order:** [§6.2ad](#62ad-phase-m-llm-r-execution-order-band-2z--closed-2026-06-06) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)  
**Goal:** Replace plain `str` and `Dict[str, Any]` LLM adapter returns with a **single strongly typed envelope** — `LLMAdapterResponse` — carrying `content: str` plus production-standard metadata, extensible without dict soup.

**Hard rules (non-negotiable):**

- **No** public adapter method returns bare `str` or `Dict[str, Any]` for completions.
- **No** `make_tool_result` dict factory — delete after migration; use typed builders only.
- **No** untyped `tool_calls: list[dict]` — use frozen `LLMToolCall` (+ `LLMToolCallArgument` where needed).
- Per-call `usage` **must** be present on every `LLMAdapterResponse` (sync with `LLMAdapterUsageLog.end_call`; prefer SDK counts over estimates when available).
- `LLMAdapterUsageLog` remains for run-level aggregation; response envelope is the **per-call source of truth** for callers.
- One **M-LLM-R.\*** ID per PR → update master table + Appendix L + paydown log → `pytest -m gate` + `tests/unit/llm_adapters/` green.

**Canonical type (target contract):**

| Type | Role |
|------|------|
| `LLMAdapterResponse` | Primary return for `generate_messages`, `generate_with_tools`, final stream event |
| `LLMTokenUsage` | `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, `reasoning_tokens` |
| `LLMFinishReason` | Enum: `completed`, `length`, `tool_calls`, `content_filter`, `refusal`, `error`, … |
| `LLMToolCall` | Typed native tool call (`id`, `name`, `arguments_json` or validated args model) |
| `LLMStreamEvent` | Streaming partial/final chunks (`event_kind`, `delta_content`, optional `completion` on final) |
| `LLMStructuredResult[T]` | `generate_structured` → `(parsed: T, response: LLMAdapterResponse)` |
| `LLMProviderExtensions` | Tagged optional extensions (OpenAI / Anthropic / Gemini slices) — **no** open `dict` bag |

**Naming note:** `LLMAdapterResponse` (not bare `LLMResponse`) — Tier-0 adapter return type; avoids collision with HTTP transport and product API DTOs.

#### M-LLM-R — Traceability (audit gap → task ID)

| Audit gap | Task IDs |
|-----------|----------|
| `generate_messages` → `str` | M-LLM-R.2.1, M-LLM-R.3.*, M-LLM-R.4.*, M-LLM-R.5.*, M-LLM-R.6.* |
| `generate_with_tools` → `Dict[str, Any]` | M-LLM-R.1.7, M-LLM-R.2.2, M-LLM-R.3.*, M-LLM-R.4.2 |
| `stream_messages` → `Iterable[str]` | M-LLM-R.1.5, M-LLM-R.2.3, M-LLM-R.3.* |
| `stream_with_tools` → `Iterable[Dict]` | M-LLM-R.1.5, M-LLM-R.2.4, M-LLM-R.3.* |
| `generate_structured` untyped | M-LLM-R.1.6, M-LLM-R.2.5 |
| SDK metadata discarded (`finish_reason`, `response_id`, refusal) | M-LLM-R.1.1, M-LLM-R.3.1–3.6 |
| Usage only side-channel | M-LLM-R.1.2, M-LLM-R.2.6, M-LLM-R.7.1 |
| Inconsistent token counting (estimate vs SDK) | M-LLM-R.3.5, M-LLM-R.3.6 |
| Replay `LLMCallInfo` not fed from adapter | M-LLM-R.7.2, M-LLM-R.7.3 |
| `CoreLLMAdapterReturnedDiagV1.adapter_return_type="str"` | M-LLM-R.7.4 |
| Conformance asserts `isinstance(text, str)` | M-LLM-R.8.2 |
| Public API missing response types | M-LLM-R.1.8, M-LLM-R.8.1 |

#### Wave M-LLM-R-0 — Planning and canon sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.0.1 | **Plan register** — Phase M-LLM-R, §4.0 Band 2z, §6.1v, §6.2ad, Appendix L; M-LLM follow-up pointer | **Done** | **Critical** | This section | Cross-links from `architecture/LLM_ADAPTERS.md` |
| M-LLM-R.0.2 | **`docs/adr/entries/2026-06-06/ADR-LLM-001.md`** — typed completion envelope vs plain string; two-layer usage model preserved | **Done** | High | `docs/adr/` | ADR linked from plan + `architecture/LLM_ADAPTERS.md` |
| M-LLM-R.0.3 | **Canon §5.2.2 addendum** — `LLMAdapterResponse` contract paragraph in `intergrax_runtime_architecture.md` | **Done** | Medium | Architecture canon | No duplicate full spec in README |

#### Wave M-LLM-R-1 — Contract types (Tier-0)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.1.1 | **`LLMAdapterResponse`** — frozen dataclass: `content`, `finish_reason`, `usage`, `model`, `provider`, `response_id`, `refusal`, `tool_calls`, `provider_extensions` | **Done** | **Critical** | `llm_adapters/contracts/adapter_response.py` | Unit: construction + immutability |
| M-LLM-R.1.2 | **`LLMTokenUsage`** — frozen dataclass with cached/reasoning token fields | **Done** | **Critical** | same module | `total_tokens` derived or validated |
| M-LLM-R.1.3 | **`LLMFinishReason`** enum + **`LLMToolCall`** (+ argument typing) | **Done** | **Critical** | `llm_adapters/contracts/tool_call.py` or same package | No raw tool dicts in public API |
| M-LLM-R.1.4 | **`LLMProviderExtensions`** — tagged union slices (OpenAI / Anthropic / Gemini / Bedrock) | **Done** | High | `llm_adapters/contracts/provider_extensions.py` | Extensibility without `Dict[str, Any]` |
| M-LLM-R.1.5 | **`LLMStreamEvent`** — partial/final streaming envelope | **Done** | High | `llm_adapters/contracts/stream_event.py` | Final event carries full `LLMAdapterResponse` |
| M-LLM-R.1.6 | **`LLMStructuredResult[T]`** generic wrapper for structured output | **Done** | High | `llm_adapters/contracts/structured_result.py` | Typed generic; mypy/pyright clean |
| M-LLM-R.1.7 | **Typed builders** — replace `make_tool_result` with `build_adapter_response(...)` / `merge_stream_events(...)` | **Done** | **Critical** | `llm_adapters/_shared/adapter_response_builders.py` | Delete `tool_results.py` dict factory |
| M-LLM-R.1.8 | **Public re-exports** — response types from `llm_adapters/__init__.py` | **Done** | Medium | `llm_adapters/__init__.py` | Import smoke test in gate |

#### Wave M-LLM-R-2 — `LLMAdapter` ABC refactor

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.2.1 | **`generate_messages` → `LLMAdapterResponse`** | **Done** | **Critical** | `contracts/llm_adapter.py` | ABC + all stubs updated |
| M-LLM-R.2.2 | **`generate_with_tools` → `LLMAdapterResponse`** | **Done** | **Critical** | same | `tool_calls` on response, not dict key |
| M-LLM-R.2.3 | **`stream_messages` → `Iterable[LLMStreamEvent]`** | **Done** | High | same | Final event mandatory |
| M-LLM-R.2.4 | **`stream_with_tools` → `Iterable[LLMStreamEvent]`** | **Done** | High | same | Tool deltas typed |
| M-LLM-R.2.5 | **`generate_structured` → `LLMStructuredResult[T]`** | **Done** | High | same | Return type annotated |
| M-LLM-R.2.6 | **`_finalize_call` helper** — unify `begin_call`/`end_call` + populate `LLMTokenUsage` on response from same counters | **Done** | **Critical** | `llm_adapter.py` or `_shared/call_lifecycle.py` | Single path; no duplicate counting |

#### Wave M-LLM-R-3 — Provider adapters (all 19 slugs)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.3.1 | **OpenAI Responses + Chat Completions** — map SDK usage, `finish_reason`, `response.id` / choice metadata | **Done** | **Critical** | `openai_responses_adapter.py`, `openai_chat_completions_adapter.py` | Mocked unit tests per method |
| M-LLM-R.3.2 | **Claude + Mistral + Cohere native** — SDK usage where available; map stop_reason / refusal | **Done** | **Critical** | `claude_adapter.py`, `mistral_adapter.py`, `cohere_native_adapter.py` | Stop using estimate-only when SDK exposes usage |
| M-LLM-R.3.3 | **Gemini + Vertex** — candidate finish reason, usage metadata, typed tool calls | **Done** | High | `gemini_adapter.py`, `vertex_gemini_adapter.py` | Conformance green |
| M-LLM-R.3.4 | **AWS Bedrock** — Converse + legacy paths; map stopReason, usage, toolUse blocks | **Done** | High | `aws_bedrock_adapter.py` | Existing bedrock tool tests updated |
| M-LLM-R.3.5 | **Ollama + OpenAI-compat family** — best-effort usage; document estimate fallback in `provider_extensions` | **Done** | Medium | `ollama_adapter.py`, `openai_compat_*` | Explicit `usage.source` flag on extensions |
| M-LLM-R.3.6 | **Streaming parity** — all `supports_streaming()` adapters emit typed `LLMStreamEvent` | **Done** | High | all streaming providers | No `yield str` remaining |
| M-LLM-R.3.7 | **Structured output parity** — return `LLMStructuredResult[T]` with raw completion preserved | **Done** | Medium | adapters with `supports_structured_output()` | JSON parse failures attach to response metadata |

#### Wave M-LLM-R-4 — Nexus runtime consumers (Tier-1)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.4.1 | **agent LLM step (`on_next_step`)** — adapter response in pattern/act; trace finish_reason + token snapshot | **Done** | **Critical** | `intergrax/agents/authoring/patterns/` | ACP agent tests |
| M-LLM-R.4.2 | **`ToolPlanningService`** — native tools path uses `completion.tool_calls`; planner text path uses `completion.content` | **Done** | **Critical** | `tools/tool_planning_service.py` | Tool plan tests green |
| M-LLM-R.4.3 | **`plan_sources` + `engine_history_layer`** — consume `.content` | **Done** | High | `planning/plan_sources.py`, `context/engine_history_layer.py` | Unit tests updated |
| M-LLM-R.4.4 | **User/org profile services + session consolidation** — all `generate_messages` call sites | **Done** | High | `runtime/user_profile/*`, `runtime/organization/*` | Grep: zero `.generate_messages` → str assignment |
| M-LLM-R.4.5 | **`supervisor.py`** — all LLM call sites | **Done** | Medium | `intergrax/supervisor/supervisor.py` | Supervisor unit tests |
| M-LLM-R.4.6 | **Optional: store last adapter response on `RuntimeState`** — `last_llm_adapter_response: LLMAdapterResponse \| None` for trace/replay | **Done** | Medium | `engine/runtime_state.py` | Enables per-step cost attribution |

#### Wave M-LLM-R-5 — RAG, websearch, legacy (Tier-0 consumers)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.5.1 | **RAG LLM paths** — `query_refiner`, `query_expander`, `chunk_enricher`, `llm_graph_indexer` | **Done** | **Critical** | `intergrax/rag/` | RAG unit tests use typed mocks |
| M-LLM-R.5.2 | **Websearch** — `websearch_context_generator`, `websearch_answerer` | **Done** | High | `intergrax/websearch/` | Tests updated |
| M-LLM-R.5.3 | **Legacy `rag_answers`** — migrate or mark deprecated path to `.content` | **Done** | Low | `legacy/rag_answers/` | No str assumption in active Nexus paths |

#### Wave M-LLM-R-6 — Agents, scaffold, test support (Tier-2)

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.6.1 | **Agent pipeline mocks** — echo, legal, research, problem_radar, signoff_probe, organization_worker, lab mocks | **Done** | High | `agent cognitive patterns (`on_next_step`)`, `agents/lab/mock_agents.py` | Agent unit tests green |
| M-LLM-R.6.2 | **`scaffold/new_agent.py` template** — generated stub returns `LLMAdapterResponse` | **Done** | High | `intergrax/scaffold/new_agent.py` | New-agent scaffold test |
| M-LLM-R.6.3 | **`testing_support/builder.py` fake adapter** | **Done** | Medium | `testing_support/builder.py` | Shared test helper |
| M-LLM-R.6.4 | **Tier-2 rule check** — agents must not assume `str` from adapter | **Done** | Low | `scripts/check_agents_llm_adapter_response.py` | CI script in §6.1 maintenance list |

#### Wave M-LLM-R-7 — Observability, replay, trace bridge

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.7.1 | **Align `LLMAdapterUsageLog.end_call` with response `usage`** — same integers; optional validation assert in debug | **Done** | High | `llm_adapter.py` | Metrics unchanged; no double-count |
| M-LLM-R.7.2 | **Emit `LLM_CALL` trace events from runtime** — populate `LLMCallInfo` fields from `LLMAdapterResponse` | **Done** | **Critical** | `core_llm_call_recorded.py`, `trace_replay_bridge.py`, `persisted_trace_event_store.py` | Gate: `test_trace_replay_bridge.py` |
| M-LLM-R.7.3 | **`LLMCallInfo` typed bridge** — map `LLMAdapterResponse` → replay model (no loose dict payloads) | **Done** | High | `runtime/replay/models.py` + mapper | Frozen mapper function |
| M-LLM-R.7.4 | **Update diagnostics** — `CoreLLMAdapterReturnedDiagV1`: `finish_reason`, token fields, drop `adapter_return_type="str"` | **Done** | Medium | `tracing/adapters/core_llm_adapter_returned.py` | PII-safe payload |
| M-LLM-R.7.5 | **Adaptive harness signal hook (optional)** — expose per-call tokens/refusal for W-ADAPT cost/quality signals | **Done** | Low | `llm_call_summary.py`, `signal_collector.py`, `HarnessOutcomeSignal.last_llm_*` | Optional `SignalAssemblyInput.last_llm_call` |

#### Wave M-LLM-R-8 — Docs, conformance, CI closeout

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| M-LLM-R.8.1 | **`architecture/LLM_ADAPTERS.md` rewrite** — response envelope section; migration guide; two-layer usage clarified | **Done** | **Critical** | `docs/architecture/LLM_ADAPTERS.md` | Examples use `.content` |
| M-LLM-R.8.2 | **Conformance suite** — `assert_generate_messages_returns_completion`; tools/stream/structured typed asserts | **Done** | **Critical** | `_shared/conformance.py`, `tests/unit/llm_adapters/` | Gate + `llm-adapters-guard.yml` |
| M-LLM-R.8.3 | **`check_llm_adapter_typed_returns.py`** — CI guard: no `-> str` / `-> Dict[str, Any]` on adapter public methods | **Done** | High | `scripts/` | Added to §6.1 maintenance |
| M-LLM-R.8.4 | **Phase closeout** — Appendix L paydown complete; M-LLM table row M-LLM.14 **Done**; remove audit follow-up pointer | **Done** | Medium | This plan | All M-LLM-R.* Done |

**Suggested PR order:**

```text
Wave 0:  M-LLM-R.0.2 → 0.3
Wave 1:  M-LLM-R.1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8
Wave 2:  M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave 3:  M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7  (may split 1 PR per provider family)
Wave 4:  M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave 5:  M-LLM-R.5.1 → 5.2 → 5.3
Wave 6:  M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave 7:  M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave 8:  M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Explicitly out of scope:** K.1/K.2, new product Tier-3 apps, rewriting provider SDK clients, HTTP API response DTOs for product routes (Tier-3 owns those separately).

---

## Phase LLM-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates 2026-06-14 Layer Completion; no open P0/P1  
**Prerequisites:** LC-1–LC-3 **Done** · M-LLM-R **Done**  
**Goal:** Formal Full Harness LC closeout — audit prompt sync, gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| LLM-LC-S1 | **Audit prompt sync** — planner≠producer Done; P0/P1 gaps closed | **Done** | High | `docs/audit/LLM_ADAPTERS.md` |
| LLM-LC-S2 | **Plan/architecture sync** — AUDIT-IDEAL header + M-LLM-X backlog clarity | **Done** | High | Domain pair consistent |
| LLM-LC-S3 | **Gate verification** — typed returns, preflight, agents LLM response | **Done** | High | 3 scripts green · 110 unit tests |
| LLM-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** M-LLM-X.2 dynamic OpenRouter fetch · X.4.4/X.4.5 trace DTO + Tier-3 failover list · Redis distributed rate limit · doctor hook (AUDIT-IDEAL-6.7)

### 6.1av Harness implementation queue — LLM adapters audit maintenance (planned)

**Source:** Layer 7 audit (2026-06-18) — `LLM_ADAPTERS` layer 6 · [`../audit_results/2026-06-18/LLM_ADAPTERS.md`](../audit_results/2026-06-18/LLM_ADAPTERS.md)  
**Priority ladder:** **Band 1** (§6.1) — DX + Tier-3 wiring hygiene; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **LLM-MAINT-01** | DX | P2 | **Done** | Close AUDIT-IDEAL-6.7 — add LLM subset (`check_llm_adapter_typed_returns` + optional catalog smoke) to `intergrax doctor check` | `intergrax doctor --ci` runs LLM checks; 6.7 **Done** |
| 2 | **LLM-MAINT-02** | CI | P2 | **Done** | M-LLM-X.7.3 — `scripts/check_model_catalog_coverage.py` warns on adapter default models missing from YAML | Gate registered in CI umbrella |
| 3 | **LLM-MAINT-03** | Code | P2 | **Done** | M-LLM-X.4.5 — Tier-3 `ApplicationEnvironmentProfile` optional LLM failover list wiring | `LLMProfile.fallback_profiles` + `resolve_llm_adapter` |
| 4 | **LLM-MAINT-04** | Docs | P3 | **Done** | Redis distributed rate limit bootstrap pattern — reference host wiring doc + cross-ref ECP/TIER3 | `intergrax/llm_adapters/USAGE.md` §Distributed rate limiting |

**Suggested PR order:** LLM-MAINT-01 → LLM-MAINT-02 → LLM-MAINT-03 → LLM-MAINT-04.

**Cross-domain (not LLM-owned):** M-LLM-X.2 dynamic OpenRouter fetch · AUDIT-IDEAL-6.2 live routing (AHI) — remain in M-LLM-X register.

---

### Phase LLM-MAINT-vllm — vLLM production ops (idea audit 2026-06-19)

**Source:** Mode I idea audit — vLLM as production self-hosted LLM; provider slug **already Done** (M-LLM.3).  
**Goal:** Ops parity with Ollama (Docker + env docs) without duplicating the OpenAI-compat adapter.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **LLM-MAINT-vllm-1** | Docs + env | P2 | **Done** | Fix canon env vars; `.env.example` + architecture self-hosted table; cross-link `infra/PORTS.md` | Operator can wire `INTERGRAX_DEFAULT_VLLM_BASE_URL` without doc/code mismatch |
| 2 | **LLM-MAINT-vllm-2** | Infra | P2 | **Done** | `infra/docker/vllm/docker-compose.yml`; integration profile **`vllm`**; host port **8100** | `./manage.sh start vllm` starts OpenAI API; no Chroma 8000 conflict |
| 3 | **LLM-MAINT-vllm-3** | CI | P3 | **Done** | `test_vllm_live_one_shot` + `require_vllm_reachable`; workflow env vars | Skip when vLLM unreachable; live smoke via `-m network` |
| 4 | **LLM-MAINT-vllm-4** | Catalog | P3 | **Done** | Expand `model_catalog.yaml` + adapter legacy windows for common vLLM ids | `check_model_catalog_coverage.py` green; parametrized catalog tests |

**Explicitly deferred:** P5 `interaction_surface/vllm` slug (M-P4.29) — adapter + Docker health sufficient until harness lab needs catalog probe.

**Phase status:** **Done** (2026-06-19) — 4/4 Done.

**ADR:** no ADR needed — reuses M-LLM.3 OpenAI-compat adapter; infra-only addition.

---

### Phase LLM-MAINT-llama-cpp — llama.cpp production ops (idea audit 2026-06-19)

**Source:** Mode I idea audit — llama.cpp self-hosted LLM parity with `LLM-MAINT-vllm`; provider slug **already Done** (M-LLM.7).  
**Goal:** Ops parity (Docker + env docs + smoke tests) without duplicating the OpenAI-compat adapter.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **LLM-MAINT-llama-cpp-1** | Docs + env | P2 | **Done** | Fix canon env vars; `.env.example` + architecture self-hosted table; cross-link `infra/PORTS.md` | Operator can wire `INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL` without doc/code mismatch |
| 2 | **LLM-MAINT-llama-cpp-2** | Infra | P2 | **Done** | `infra/docker/llama-cpp/docker-compose.yml`; integration profile **`llama-cpp`**; host port **8102** | `./manage.sh start llama-cpp` starts OpenAI API; no Weaviate 8080 conflict |
| 3 | **LLM-MAINT-llama-cpp-3** | E2E verify | P3 | **Done** | `tests/e2e/llama_cpp/` + `infra/docker/llama-cpp/verify.{sh,ps1}`; **excluded from GitHub CI** | `INTERGRAX_LLAMA_CPP_VERIFY=1` hard-fails when stack down; not in `llm-network-smoke.yml` |
| 4 | **LLM-MAINT-llama-cpp-4** | Catalog | P3 | **Done** | Expand `model_catalog.yaml` + adapter legacy windows for common llama.cpp ids | `check_model_catalog_coverage.py` green; parametrized catalog tests |

**Explicitly deferred:** P5 `interaction_surface/llama_cpp` slug — adapter + Docker health sufficient until harness lab needs catalog probe.

**Phase status:** **Done** (2026-06-19) — 4/4 Done.

**ADR:** no ADR needed — reuses M-LLM.7 OpenAI-compat adapter; infra-only addition.

---

### Phase M-LLM-X-9 — LLM routing rules (idea audit 2026-06-19)

**Source:** Mode I idea audit — adaptive LLM model switching with developer-defined custom rule classes.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) § LLM routing rules  
**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) **Accepted**  
**Goal:** `LLMRoutingRule` Protocol on Tier-0; built-in + custom Tier-3 classes; AHI L4 overlay unchanged.  
**Phase status:** **Done** (2026-06-19) — 10/10 Done · see [Wave M-LLM-X-9](#wave-m-llm-x-9--llm-routing-rules-protocol--custom-classes)

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **M-LLM-X.9.1** | Contract | P1 | **Done** | `LLMRoutingRule` Protocol + `LLMRoutingRuleBase` | `routing/contracts.py` importable |
| 2 | **M-LLM-X.9.2** | Contract | P1 | **Done** | `RoutingContext`, `RoutingTarget`, `LLMRoutingProfile` | Pydantic validation |
| 3 | **M-LLM-X.9.2b** | Code | P1 | **Done** | `LLMRoutingEvaluator` first-match + allowlist | Unit tests green |
| 4 | **M-LLM-X.9.3** | Code | P1 | **Done** | Built-in rule classes | Same Protocol as custom |
| 5 | **M-LLM-X.9.4** | Tier-3 | P1 | **Done** | `ApplicationEnvironmentProfile` field + wiring | `test_llm_routing_resolver.py` |
| 6 | **M-LLM-X.9.5** | Wire | P1 | **Done** | `resolve_llm_adapter()` hot path | Integration test |
| 7 | **M-LLM-X.9.6** | Cross-ref | P2 | **Done** | Unify `degrade_model` with routing rule | ACP-TOK-3 paths |
| 8 | **M-LLM-X.9.7** | Code | P2 | **Done** | `DynamicLLMRouter` per-step swap | Unit test |
| 9 | **M-LLM-X.9.8** | Docs | P2 | **Done** | USAGE.md cookbook | Architecture cross-link |
| 10 | **M-LLM-X.9.9** | CI | P2 | **Done** | `check_llm_routing_rules.py` | CI umbrella |

**Suggested PR order:** 9.1 → 9.2 → 9.2b → 9.3 → 9.4 → 9.5 → 9.6 → 9.7 → 9.8 → 9.9.

**Cross-domain:** AHI-MAINT-05 · `TIER3_APPLICATION_ENVIRONMENT` · `AGENT_CONTRACTS_AND_ASSEMBLY` (degrade_model).

---

### Phase M-LLM-X-10 — LLM routing enterprise closeout (2026-06-19)

**Source:** Post X-9 enterprise readiness review.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) § LLM routing rules · § Enterprise routing  
**ADR:** ADR-LLM-003 (unchanged); **AHI-MAINT-06** Done.  
**Goal:** Platform catalog + custom Protocol + auto context on materialize + trace + lab reference + acceptance (resolver-level).  
**Phase status:** **Done** — 8/8 · closes **LLM-AUDIT-17** (start-of-run scope). Mid-run gaps → **X-11**.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **M-LLM-X.10.1** | Catalog | P1 | **Done** | 12+ predefined parametric rule classes + custom Protocol | Unit test per class |
| 2 | **M-LLM-X.10.2** | Wire | P1 | **Done** | `build_routing_context_from_runtime()` on materialize path | `runtime_config_bridge` |
| 3 | **M-LLM-X.10.3** | Obs | P1 | **Done** | Trace `rule_id` + `routing_reason` at tracker init | Schema unit test |
| 4 | **M-LLM-X.10.4** | Tier-3 | P1 | **Done** | Lab reference host (predefined demo) | CI gate scan |
| 5 | **M-LLM-X.10.5** | E2E | P1 | **Done** | Acceptance: budget rule switches model (resolver) | `tests/acceptance/llm_routing/` |
| 6 | **M-LLM-X.10.6** | Wire | P2 | **Done** | `DynamicLLMRouter` on ACP hosts | Agent router test |
| 7 | **M-LLM-X.10.7** | Docs | P2 | **Done** | USAGE matrix + checklist | Architecture sync |
| 8 | **M-LLM-X.10.8** | CI | P2 | **Done** | `check_llm_routing_rules.py` | Umbrella gate |

**Suggested PR order:** 10.1 → 10.2 → … → 10.8 — **Done**.

**Cross-domain:** **AHI-MAINT-06** (`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`).

---

### Phase M-LLM-X-11 — Routing enterprise hardening (2026-06-19)

**Source:** Post X-10 enterprise review — honest maturity L4; strict enterprise requires mid-run Nexus.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) § Enterprise routing hardening  
**ADR:** ADR-LLM-003 (unchanged unless evaluating adapter changes tier contract).  
**Goal:** Live mid-run profile swap on Nexus `llm_adapter`; full observability loop; true E2E; unified call sites.  
**Phase status:** **Done** — 8/8 · closes **LLM-AUDIT-18** (X-11 declared scope). Strict L5 gaps → **X-12** · LLM-AUDIT-19.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **M-LLM-X.11.1** | Code | P1 | **Done** | `RoutingEvaluatingLLMAdapter` | Unit: profile swap on context change |
| 2 | **M-LLM-X.11.2** | Wire | P1 | **Done** | `refresh_llm_routing_context()` in runtime loop | Step loop test |
| 3 | **M-LLM-X.11.3** | Wire | P1 | **Done** | All `resolve_llm_adapter` call sites use context bridge | CI static gate |
| 4 | **M-LLM-X.11.4** | Obs | P1 | **Done** | Per-eval trace + allowlist violation diag + trace_bridge gate | Schema tests |
| 5 | **M-LLM-X.11.5** | E2E | P1 | **Done** | Full run budget threshold → model change | `tests/acceptance/llm_routing/` |
| 6 | **M-LLM-X.11.6** | Wire | P2 | **Done** | Harness host evaluating adapter parity | Host test |
| 7 | **M-LLM-X.11.7** | Docs | P2 | **Done** | USAGE mid-run section + architecture sync | Linked from hub |
| 8 | **M-LLM-X.11.8** | CI | P2 | **Done** | `check_llm_routing_context_wiring.py` | Umbrella gate |

**Suggested PR order:** 11.1 → 11.2 → 11.3 → 11.4 → 11.5 → 11.6 → 11.7 → 11.8.

**Closes:** **LLM-AUDIT-18** (X-11 scope). **Blocks:** **M-LLM-X.12** (strict L5).

---

### Phase M-LLM-X-12 — Routing strict enterprise closeout (2026-06-19)

**Source:** Post X-11 architecture audit — honest maturity **L4+**; strict **L5** requires budget-accurate mid-run on all Nexus paths.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) § Routing strict enterprise closeout  
**ADR:** ADR-LLM-003 or new ADR-LLM-004 if tier split changes public contract (decide in 12.2).  
**Goal:** Close LLM-AUDIT-19 — production-trustworthy budget routing, tier-clean evaluating path, full observability loop.  
**Phase status:** **Planned** — 0/12 Done · see [Wave M-LLM-X-12](#wave-m-llm-x-12--routing-strict-enterprise-closeout)

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **M-LLM-X.12.1** | Wire | P0 | **Planned** | Budget meter ↔ routing context | Budget rule fires on real token usage |
| 2 | **M-LLM-X.12.2** | Arch | P1 | **Planned** | Tier boundary refactor (evaluating → Tier-3) | No Tier-0 → `applications/` import |
| 3 | **M-LLM-X.12.3** | Wire | P1 | **Planned** | Nexus graph / CE context sync | Graph step updates snapshot |
| 4 | **M-LLM-X.12.4** | Wire | P1 | **Planned** | Per-call context refresh | Multi-LLM step budget accuracy |
| 5 | **M-LLM-X.12.5** | Wire | P1 | **Planned** | AHI live context on adapter swap | AHI hint uses live snapshot |
| 6 | **M-LLM-X.12.6** | Wire | P1 | **Planned** | `budget_degrade_active` Nexus mapping | Degrade rule test green |
| 7 | **M-LLM-X.12.7** | Obs | P2 | **Planned** | Per-run observers (no globals) | Concurrent run isolation |
| 8 | **M-LLM-X.12.8** | Obs | P2 | **Planned** | ACP `DynamicLLMRouter` trace | `llm_routing_rule` on ACP path |
| 9 | **M-LLM-X.12.9** | Code | P2 | **Planned** | First-eval profile correction | Mismatch fixed on first call |
| 10 | **M-LLM-X.12.10** | E2E | P2 | **Planned** | Production acceptance (no mocks) | `tests/acceptance/llm_routing/` |
| 11 | **M-LLM-X.12.11** | Docs | P2 | **Planned** | L4+ label + LLM-AUDIT-19 closeout | Architecture + plan synced |
| 12 | **M-LLM-X.12.12** | Policy | P3 | **Planned** | Secondary LLM surfaces policy | Architecture policy row |

**Suggested PR order:** 12.1 → 12.2 → 12.3 → 12.4 → 12.5 → 12.6 → 12.7 → 12.8 → 12.9 → 12.10 → 12.11 → 12.12.

**Closes:** **LLM-AUDIT-19**. **Blocks:** **M-LLM-X.8** honest closeout.

---

*End of LLM Adapters Implementation Plan.*
