# LLM_ADAPTERS — routing and failover

**Parent hub:** [`LLM_ADAPTERS.md`](../LLM_ADAPTERS.md)

Deep technical canon for **profile routing** and **profile-chain failover** only. Hub public front, adapter envelope, `ModelCatalog`, provider catalog, and routing wave maturity register live elsewhere — do not duplicate them here.

| Depth | Route |
| ----- | ----- |
| Hub summary | [`LLM_ADAPTERS.md`](../LLM_ADAPTERS.md) §Routing and failover |
| Wave / audit register | [`LLM_ADAPTERS_audit_register.md`](LLM_ADAPTERS_audit_register.md) |
| Plan open rows | [`maintainers/plans/LLM_ADAPTERS.md`](../../maintainers/plans/LLM_ADAPTERS.md) |
| ADR | [ADR-LLM-003](../../technical/adr/entries/2026-06-19/ADR-LLM-003.md) |
| Developer wiring | [`intergrax/llm_adapters/USAGE.md`](../../../../intergrax/llm_adapters/USAGE.md) |

---

## Ownership boundaries

| Layer | Owns | Does not own |
| ----- | ---- | ------------ |
| **Reasoning** | Role / reasoning intent; `ReasoningProfile.planner_llm_profile` → separate planner adapter | Provider protocol; profile failover execution |
| **LLM Adapters (Tier-0)** | `LLMRoutingEvaluator`, `ModelRouter`, `FailoverLLMAdapter`, rule Protocol, built-in rule catalog | Autonomous "pick best model" without declarative policy |
| **ACP `StepLLMRouter`** | Per-step `model_hint` resolution within allowlists/contracts (`intergrax/agents/authoring/llm_router.py`) | Tier-3 routing rule authoring |
| **Tier-3** | `LLMProfile`, `LLMRoutingProfile`, `fallback_profiles`, env/host resolver wiring | Vendor SDK imports in agents |
| **Token Optimization** | Consumes cost/token/cache signals from adapter envelope and catalog | Adapter routing core or parallel router |

**Anti-patterns:** string `eval` for rules; provider selection inside Tier-2 agents; routing target outside `allowed_profiles`; parallel router subsystem.

---

## Deterministic mental model

Routing is **policy-driven and evidence-backed** — declarative rules over `RoutingContext` signals (budget, task class, step index, model hint, tenant/agent identity). It is **not** "AI automatically chooses the best model."

```text
RoutingContext snapshot
      ↓
LLMRoutingEvaluator (rules sorted by priority, descending → first match)
      ↓
RoutingEvaluation (selected_profile, policy_route_hint, routing_reason)
      ↓
ModelRouter (order primary + fallback_profiles per policy hint)
      ↓
LLMProfile.create_adapter_with_failover()  — or single adapter when no chain
      ↓
FailoverLLMAdapter (when chain length > 1)
      ↓
adapter call(s)
```

Live cost/latency/quality routing on AHI product paths is **wired** (AUDIT-IDEAL-6.2 **Done**) — hints and approved profiles feed `ModelRouter`; AHI `RoutingTuningEngine` may **recommend** profile order; policy engine **approves**; runtime executes the chain in-process (M-LLM-X.4 **Done**). Central LLM gateway microservice is an explicit non-goal.

---

## Core contracts

**Location:** `intergrax/llm_adapters/routing/` (contracts, built-in rules, evaluator).

| Type | Role |
| ---- | ---- |
| `RoutingContext` | Immutable snapshot: `task_class`, `budget_remaining_ratio`, `tokens_used`, `step_index`, `model_hint`, `tenant_id`, `agent_id`, `budget_degrade_active` |
| `RoutingTarget` | Rule output: `LLMProfile \| None`, `RoutingHint \| None`, `reason: str` |
| `LLMRoutingRule` | **Protocol** — `rule_id`, `priority`, `matches(ctx)`, `resolve(ctx)` |
| `LLMRoutingRuleBase` | Optional ABC with helpers (`budget_below`, `task_is`, `tokens_above`, …) |
| `LLMRoutingProfile` | Tier-3: `default_profile`, `allowed_profiles`, `rules: tuple[LLMRoutingRule, ...]` |
| `RoutingEvaluation` | Evaluator result: `matched_rule_id`, `selected_profile`, `policy_route_hint`, `routing_reason` |
| `LLMRoutingEvaluator` | Sort rules by priority → first `matches()` → allowlist guard |
| `ModelRouter` | Order `primary` + `fallback_profiles` using `routing_policy_hint` / rule hint |
| `ModelRoutingDecision` | Resolved route metadata (`profile_id`, `routing_reason`, `fallback_profile_id`) |
| `FailoverLLMAdapter` | Execute ordered adapter chain on retriable provider errors |

**ADR:** [ADR-LLM-003](../../technical/adr/entries/2026-06-19/ADR-LLM-003.md).

---

## Rule evaluation semantics

```text
1. allowlist = allowed_profiles or (default_profile,)
2. rules sorted by priority (descending)
3. first rule where matches(context) is True
4. target = rule.resolve(context)
5. guard: target.profile ∈ allowlist (AllowlistViolationError + trace on violation)
6. if no rule matches → default_profile
```

Built-in parametric rules (Tier-0 catalog) and custom Tier-3 `LLMRoutingRule` subclasses share the same Protocol. Representative built-ins (`intergrax/llm_adapters/routing/builtin_rules.py`):

| Class | Signal |
| ----- | ------ |
| `BudgetBelowRule` / `BudgetAboveRule` | `budget_remaining_ratio` |
| `BudgetExceededDegradeRule` | `budget_degrade_active` → cheapest hint |
| `TaskClassInRule` / `TaskClassNotInRule` | `task_class` |
| `TokenUsedAboveRule` / `TokenUsedBelowRule` | `tokens_used` |
| `StepIndexAtLeastRule` / `StepIndexBelowRule` | `step_index` |
| `AgentIdInRule` / `TenantIdInRule` | identity |
| `ModelHintPresentRule` | honour `model_hint` |
| `PolicyHintRule` | force `RoutingHint` |
| `CompositeAllRule` / `CompositeAnyRule` | AND / OR |
| `AlwaysRule` | explicit catch-all |

Full wave delivery checklist (M-LLM-X.9–16): audit register satellite — not repeated here.

### Custom rule example (Tier-3)

```python
class LowBudgetForceLocalRule(LLMRoutingRuleBase):
    rule_id = "my_app.low_budget_vllm"
    priority = 10

    def matches(self, context: RoutingContext) -> bool:
        return (
            context.budget_remaining_ratio is not None
            and context.budget_remaining_ratio < 0.2
            and context.task_class in {"contract_review", "due_diligence"}
        )

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            profile=LLMProfile(provider="vllm", model="meta-llama/Llama-3.1-8B"),
            reason=f"budget_low for {context.task_class}",
        )
```

Wired in manifest: `LLMRoutingProfile(rules=(LowBudgetForceLocalRule(), BudgetBelowRule(...), ...))`.

### Three-layer model

```text
Layer 1 — Author rules (LLMRoutingProfile on Tier-3)     → explicit logic; wins over tuning hints
Layer 2 — LLMRoutingEvaluator + ModelRouter (Tier-0)    → hot path (single router — SYS-INV-10)
Layer 3 — AHI ROUTING_TUNING (optional)                  → bandit proposes ProfileVersion;
                                                           does not execute arbitrary author code
```

---

## `ModelRouter` and profile selection

`ModelRouter` (`intergrax/llm_adapters/registry/model_router.py`) orders profiles **before** adapter construction:

| `routing_policy_hint` | Order when fallbacks exist |
| --------------------- | -------------------------- |
| `balanced` | first fallback, primary, rest |
| `cheapest` | all fallbacks, then primary |
| `fastest` | primary, then fallbacks |
| `quality` | primary, then fallbacks reversed |
| (none) | primary, then fallbacks in declaration order |

`LLMProfile.create_adapter_with_failover()` builds `ModelRouter.from_profiles(primary, fallbacks=fallback_profiles, policy_route_hint=…)`, materializes one adapter per ordered profile, and wraps with `FailoverLLMAdapter` when `len(adapters) > 1`.

`routing_policy_hint` may come from `LLMProfile.routing_policy_hint` or from rule `RoutingTarget.hint`.

---

## Profile fields (routing-relevant)

| Field | Purpose |
| ----- | ------- |
| `provider` / `model` | Vendor identity |
| `options` | Passed to adapter ctor; includes `LLMCallConfig` fields |
| `fallback_profiles` | Ordered tuple for failover chain |
| `routing_policy_hint` | `balanced` \| `cheapest` \| `fastest` \| `quality` |

Tier-3 wiring: `ApplicationEnvironmentProfile.llm_profile`, `resolve_llm_adapter()` precedence (agent > env > `INTERGRAX_LLM_*`), optional `LLMRoutingProfile` on environment capabilities.

---

## Nexus run precedence

Within a single Nexus run (hub canon — not reordered here):

```text
1. RuntimeConfig.llm_adapter           — primary producer
2. resolve_planner_llm_adapter()       — optional separate planner adapter (Reasoning)
3. CriticProfile / EvaluationProfile   — separate LLMProfile for judge paths
4. ModelRouter + fallback_profiles     — runtime selection before adapter create (Done — M-LLM-X.4)
```

`RoutingEvaluatingLLMAdapter` (Tier-3 wrapper, M-LLM-X.11–12 **Done**) re-evaluates routing before LLM calls on core Nexus/UAEP paths; context refresh via `refresh_llm_routing_context()` at step boundaries. Secondary surfaces (tool planner, websearch, critic) receive routing snapshot sync or explicit routing metadata per M-LLM-X.12.12 / X-13 policy — see audit register.

---

## ACP `StepLLMRouter`

**Done** — AUDIT-IDEAL-6.6 / M-LLM-X.5.4.

`StepLLMRouter` (`intergrax/agents/authoring/llm_router.py`) resolves per-step `model_hint` against catalog + `LLMAdapter` allowlists. It is a **step-scoped hint port** within ACP contracts — not a replacement for `LLMRoutingProfile` rule authoring. When `llm_port` is wired, completion delegates to `LLMAdapter.generate_messages` (single DX).

---

## Profile failover semantics

Failover switches **profile / provider / model** on the ordered chain — not the same as per-call retry on one adapter.

```text
primary profile
      ↓
adapter call (with per-call resilience on that adapter)
      ↓
retriable provider failure?
   yes → next fallback profile (deterministic order)
   no  → raise — failure remains visible; do not mask
```

**Implementation:** `FailoverLLMAdapter` (`intergrax/llm_adapters/registry/failover_adapter.py`) + `LLMProfile.create_adapter_with_failover()` (**Done** — AUDIT-IDEAL-6.5 / LC-3).

Retriability uses `is_retriable_provider_error()` (`intergrax/llm_adapters/_shared/retry.py`):

- HTTP status in `LLMCallConfig.retry_on_status` (default `429, 500, 502, 503, 504`)
- Exception type name contains `timeout`, `connection`, `rate`, or `overloaded`

**Non-retriable failures** (validation errors, auth failures, content policy, malformed requests, etc.) **do not** advance the chain — they propagate to the caller per current contracts.

Primary adapter owns `context_window_tokens` and token estimation for the wrapper. Streaming selects the first chain member that `supports_streaming()`.

---

## Per-call resilience ≠ profile failover

Two separate layers — do not merge mentally or in wiring.

| Layer | Mechanism | Scope |
| ----- | --------- | ----- |
| **Per-call resilience** | `LLMCallConfig`: `max_retries`, `retry_backoff_sec`, `retry_on_status`, `timeout_sec`, circuit breaker, rate limit (`intergrax/llm_adapters/_shared/resilience.py`) | Same adapter / same profile |
| **Profile failover** | `fallback_profiles` + `FailoverLLMAdapter` | Switch to next `LLMProfile` after retriable failure exhausts or bypasses single-adapter retry |

`call_with_retry()` may retry transient errors **on the current adapter** before failover sees the exception (adapter implementations apply call policy). Quota / tenant hard stops (`check_llm_tenant_quota`) are a third layer — not adapter retry or profile failover.

---

## Observability

| Diagnostic | When | Location |
| ---------- | ---- | -------- |
| `LLMRoutingRuleDiagV1` | Rule evaluation / profile selection | `intergrax/runtime/nexus/tracing/adapters/llm_routing_attempt.py` |
| `LLMRoutingAttemptDiagV1` | Failover attempt (provider, model, error) | same module; step `llm_routing_attempt` |
| `LLMRoutingAttemptRecord` | In-process failover audit on `FailoverLLMAdapter.routing_attempts` | `failover_adapter.py` |

Trace bridges wire rule id, `routing_reason`, and attempt records on Nexus/UAEP/ACP paths (M-LLM-X.10.3–X-13.2 **Done**). Catalog-miss spine (`ModelCatalogMissDiagV1`) is adjacent — owned by hub §Model catalog; see audit register M-LLM-X.15–16.

CI gates: `scripts/maintenance/check_llm_routing_rules.py`, `check_llm_routing_context_wiring.py`, `check_llm_routing_tier_boundary.py`, `check_live_model_routing_wiring.py`.

---

## Implementation evidence (current)

| Capability | Status | Primary modules |
| ---------- | ------ | --------------- |
| `ModelCatalog` | **Done** | `registry/model_catalog.py` — routing consumes metadata; catalog miss diagnostics separate |
| `LLMRoutingEvaluator` + built-in rules | **Done** | `routing/evaluator.py`, `routing/builtin_rules.py` |
| `ModelRouter` | **Done** | `registry/model_router.py` |
| Profile failover chain | **Done** | `registry/failover_adapter.py`, `profile.create_adapter_with_failover()` |
| Live cost/latency/quality routing (AHI path) | **Done** | AUDIT-IDEAL-6.2; `check_live_model_routing_wiring.py` |
| Mid-run re-eval (`RoutingEvaluatingLLMAdapter`) | **Done** | Tier-3 `applications/_shared/` evaluating wrapper + runtime bridges |
| `StepLLMRouter` + `LLMAdapter` DX | **Done** | `agents/authoring/llm_router.py` |
| Planner adapter separation | **Done** | `resolve_planner_llm_adapter()` in Nexus factory |
| Full provider plugin ecosystem | **Planned** | `LLM-PROVIDER-PLUGIN-1` — registry `register()` exists; thin plugin layer backlog |

Resolver entry points: `resolve_llm_adapter()`, `resolve_environment_llm_adapter()`, `resolve_live_model_routing_wiring()` (product hosts), runtime bridges under `intergrax/runtime/wiring/llm_routing_*` and `intergrax/applications/_shared/llm_routing_*`.

---

## Provider coverage (routing targets)

| Model source | Routing path |
| ------------ | ------------ |
| Cloud API (OpenAI, Claude, Groq, …) | `LLMProfile(provider=…)` |
| Local (Ollama, vLLM, llama.cpp) | Existing `LLMProvider` slugs |
| HF self-hosted weights | Model id on `vllm` / `llama_cpp` profile |
| HF Inference API (chat) | Optional provider plugin backlog or gateway via `openrouter` |
