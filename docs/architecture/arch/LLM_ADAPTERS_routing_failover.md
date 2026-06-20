# LLM_ADAPTERS — routing failover

**Parent hub:** [`LLM_ADAPTERS.md`](../LLM_ADAPTERS.md)

## Model routing and failover

### Current state

- `ModelRouter` + `FailoverLLMAdapter` wired via `resolve_llm_adapter()` and `LLMProfile.create_adapter_with_failover()` (**Done** LC-3).
- `resolve_live_model_routing_wiring()` applies routing hints on product hosts; decision drives adapter creation (**Done** LC-3).
- `LLMCallConfig` — retry, circuit breaker, rate limit; cross-provider failover via profile chain (**Done**).

### Target state (M-LLM-X.4 / X.5)

```text
LLMProfileChain
  primary: LLMProfile
  fallbacks: tuple[LLMProfile, ...]
  policy: RoutingPolicyHint

LLMAdapterFactory.create_with_failover(chain) → LLMAdapter
  on retriable error (429, 502, 503, timeout): next profile in chain
  emit LLMRoutingDiagV1 on trace (provider, model, reason, attempt)
```

AHI `RoutingTuningEngine` **recommends** profile order; policy engine **approves**; runtime executes chain — aligns IDEAL §3.5 model selection by cost/latency/quality.

**Explicit non-goal:** Central LLM gateway microservice (§5.2.4) — separate ADR if pursued; M-LLM-X stays in-process Tier-0.

### LLM routing rules (M-LLM-X.9 — Done)

**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md)

Authors configure dynamic model selection through a **Protocol contract** — not a rigid enum DSL. Built-in parametric rules and **custom classes** from Tier-3 share the same interface.

#### Contracts (target location: `intergrax/llm_adapters/routing/`)

| Type | Role |
|------|------|
| `RoutingContext` | Immutable snapshot: `task_class`, `budget_remaining_ratio`, `tokens_used`, `step_index`, `model_hint`, `tenant_id`, `agent_id`, … |
| `RoutingTarget` | Rule output: `LLMProfile \| None`, `RoutingHint \| None`, `reason: str` (for trace) |
| `LLMRoutingRule` | **Protocol** — `rule_id`, `priority`, `matches(ctx)`, `resolve(ctx)` |
| `LLMRoutingRuleBase` | Optional ABC with helper methods (`budget_below`, `task_is`, …) |
| `LLMRoutingProfile` | Tier-3: `default_profile`, `allowed_profiles`, `rules: tuple[LLMRoutingRule, ...]` |
| `LLMRoutingEvaluator` | Sort by priority → first `matches()` → validate allowlist → `ModelRouter` |

#### Evaluation flow

```text
RoutingContext (snapshot from Nexus / budget meter / classifier)
    → sort rules by priority (descending)
    → first rule where matches(ctx) is True
    → target = rule.resolve(ctx)
    → guard: target.profile ∈ allowed_profiles (reject + trace on violation)
    → ModelRouter + FailoverLLMAdapter
    → trace: rule_id, reason, profile_id (`LLMRoutingRuleDiagV1` + failover `LLMRoutingAttemptDiagV1`)
```

#### Built-in rules (same Protocol)

**Done (X-9 + X-10):** full parametric **platform catalog** (Tier-0) — constructor params replace former enum DSL. Tier-3 authors may always add **custom** `LLMRoutingRule` subclasses; see custom example below and [plan M-LLM-X.10](../plan/LLM_ADAPTERS.md#wave-m-llm-x-10--llm-routing-enterprise-closeout--predefined-rule-catalog).

| Class | Status | Covers |
|-------|--------|--------|
| `BudgetBelowRule(threshold, profile\|hint)` | **Done** | `budget_remaining_ratio < threshold` |
| `BudgetAboveRule(threshold, profile\|hint)` | **Done** | `budget_remaining_ratio > threshold` |
| `BudgetExceededDegradeRule()` | **Done** | `budget_degrade_active` → `CHEAPEST` |
| `TaskClassInRule(classes, profile\|hint)` | **Done** (alias `TaskClassRule`) | `task_class in classes` |
| `TaskClassNotInRule(classes, …)` | **Done** | negated task class |
| `TokenUsedAboveRule(threshold, hint)` | **Done** (alias `TokenThresholdRule`) | `tokens_used > threshold` |
| `TokenUsedBelowRule(threshold, …)` | **Done** | low token usage |
| `StepIndexAtLeastRule` / `StepIndexBelowRule` | **Done** | per-step routing |
| `AgentIdInRule` / `TenantIdInRule` | **Done** | identity-based routing |
| `ModelHintPresentRule` | **Done** | honour agent `model_hint` |
| `PolicyHintRule(hint)` | **Done** | force `RoutingHint` |
| `CompositeAllRule` / `CompositeAnyRule` | **Done** | AND / OR composition |
| `AlwaysRule(profile\|hint)` | **Done** | explicit catch-all fallback |

#### Enterprise routing (M-LLM-X.10 — Done · scope)

**Delivered (2026-06-19):** predefined catalog, Protocol + custom rules, auto context on **materialize** path, reference lab host, CI gate, ACP `DynamicLLMRouter`, initial `LLMRoutingRuleDiagV1`.

| Capability | Task ID | Status | Scope note |
|------------|---------|--------|------------|
| Predefined catalog (Tier-0) + custom rules (Tier-3) | M-LLM-X.10.1 | **Done** | Builtin + `LLMRoutingRule` Protocol |
| Auto `RoutingContext` on materialize / ACP start | M-LLM-X.10.2 | **Done** | `runtime_config_bridge`, `acp_run` — not all call sites |
| Trace `rule_id` + `routing_reason` | M-LLM-X.10.3 | **Done** | Start-of-run; per-eval on UAEP/Nexus via X-11.4/X-12.7; ACP Plane A parity via **X-13.2** |
| Reference lab host (predefined demo; products may use custom) | M-LLM-X.10.4 | **Done** | CI gate lab-only |
| Acceptance: budget rule switches profile | M-LLM-X.10.5 | **Done** | Resolver + materialize — not full Nexus run |
| `DynamicLLMRouter` on ACP hosts | M-LLM-X.10.6 | **Done** | ACP only |
| USAGE + architecture checklist | M-LLM-X.10.7 | **Done** | |
| CI `check_llm_routing_rules.py` | M-LLM-X.10.8 | **Done** | |
| AHI persistent bandit + ProfileVersion read | AHI-MAINT-06 | **Done** | Hint path; no full canary apply |

**Maturity label:** **L4+ enterprise-ready** — start-of-run + ACP + UAEP mid-run evaluating adapter (M-LLM-X.11). **Strict L5** delivered in **M-LLM-X.12** · LLM-AUDIT-19 **Done**.

#### Enterprise routing hardening (M-LLM-X.11 — Done)

**Delivered (2026-06-19):** live re-eval on Nexus `llm_adapter`, `refresh_llm_routing_context()` in UAEP step loop, unified Tier-3 resolver call sites, per-evaluation trace + allowlist violation diag, mid-run acceptance (mocked adapter factory), harness host parity, CI `check_llm_routing_context_wiring.py`.

| Capability | Task ID | Status |
|------------|---------|--------|
| `RoutingEvaluatingLLMAdapter` — re-eval before each LLM call | M-LLM-X.11.1 | **Done** |
| `refresh_llm_routing_context()` + `llm_routing_snapshot` on `RuntimeConfig` | M-LLM-X.11.2 | **Done** (UAEP step boundary) |
| `resolve_environment_llm_adapter()` on all Tier-3 wiring modules | M-LLM-X.11.3 | **Done** |
| Per-eval `LLMRoutingRuleDiagV1` + allowlist violation diag | M-LLM-X.11.4 | **Done** (UAEP/Nexus state; ACP partial) |
| Mid-run budget threshold → profile swap (acceptance) | M-LLM-X.11.5 | **Done** (evaluating adapter; mocked factory) |
| Harness host + materialize evaluating adapter parity | M-LLM-X.11.6 | **Done** |
| USAGE mid-run section | M-LLM-X.11.7 | **Done** |
| CI `check_llm_routing_context_wiring.py` | M-LLM-X.11.8 | **Done** |

**Closes:** LLM-AUDIT-18 (declared X-11 scope). **Does not claim strict L5** — see post-audit register **LLM-AUDIT-19**.

#### Routing strict enterprise closeout (M-LLM-X.12 — Done)

**Delivered (2026-06-19):** Tier-3 `RoutingEvaluatingLLMAdapter` with injected factory; Tier-0 `metering.py` + `runtime_sync.py`; Nexus graph/CE sync hooks; per-run observability; ACP routing diagnostics; production metering E2E; CI `check_llm_routing_tier_boundary.py`.

| Deliverable | Task | Status |
|-------------|------|--------|
| Inner usage metering + tracker re-register on swap | M-LLM-X.12.1 | **Done** |
| Tier-clean evaluating wrapper (Tier-3) | M-LLM-X.12.2 | **Done** |
| Graph + CE routing snapshot sync | M-LLM-X.12.3 | **Done** |
| Per-call context refresh via provider | M-LLM-X.12.4 | **Done** |
| Live `RoutingContext` on adapter swap / AHI | M-LLM-X.12.5 | **Done** |
| `budget_degrade_active` in Nexus sync | M-LLM-X.12.6 | **Done** |
| Instance-bound observers (no globals) | M-LLM-X.12.7 | **Done** |
| ACP `DynamicLLMRouter` routing diagnostics | M-LLM-X.12.8 | **Done** |
| First-eval profile correction | M-LLM-X.12.9 | **Done** |
| Production metering E2E | M-LLM-X.12.10 | **Done** |
| Docs + audit register | M-LLM-X.12.11 | **Done** |
| Secondary LLM surfaces policy (documented) | M-LLM-X.12.12 | **Done** |

**Maturity label:** **L5 strict enterprise-ready** for declarative routing on core + UAEP + ACP paths.

**Secondary LLM policy (M-LLM-X.12.12 · X-13.4–13.6):** Tool planner, websearch map/reduce/rerank, and critic paths receive **routing snapshot sync** or explicit **critic routing metadata** when `llm_routing_profile` is enabled. They are not auto-wrapped in `RoutingEvaluatingLLMAdapter`; hosts may still opt into dedicated evaluating wraps per surface.

**Closes:** **LLM-AUDIT-19** (X-12), **LLM-AUDIT-20** (X-13).

#### Post-L5 follow-up register (M-LLM-X.13 — Done)

**Source:** Post X-12 enterprise audit (2026-06-19). All gaps below closed in **M-LLM-X.13** (2026-06-19).

| Gap (post-audit) | Severity | Task ID | Status |
|------------------|----------|---------|--------|
| `runtime_state.py` imports Tier-3 `RoutingEvaluatingLLMAdapter` for `isinstance` wiring | **P2** | M-LLM-X.13.1 | **Done** — `evaluating_hooks.py` Protocol |
| ACP records routing in `step.diagnostics` only — no Plane A `llm_routing_rule` trace step | **P2** | M-LLM-X.13.2 | **Done** — `acp_routing_trace_bridge.py` |
| No dedicated concurrent-run isolation test for per-run observers | **P2** | M-LLM-X.13.3 | **Done** |
| `tool_planning_service` LLM bypasses evaluating wrap | **P3** | M-LLM-X.13.4 | **Done** — snapshot sync |
| Websearch map/reduce/rerank LLM bypass evaluating wrap | **P3** | M-LLM-X.13.5 | **Done** — snapshot sync |
| Critic evaluator LLM bypass evaluating wrap | **P3** | M-LLM-X.13.6 | **Done** — routing policy metadata |
| `nexus_plan_bridge` / `llm_task_classifier` skip routing snapshot sync | **P2** | M-LLM-X.13.7 | **Done** |

**Plan:** [Wave M-LLM-X-13](../plan/LLM_ADAPTERS.md#phase-m-llm-x-13--post-l5-routing-polish-2026-06-19)

#### Enterprise domain maturity register (M-LLM-X-14 — Done)

**Delivered (2026-06-19):** gateway metadata client + session merge; catalog capability wire verified; ACP usage token bridge; enum-free `LLMProfile.provider`; opt-in secondary evaluating wrap (`llm_routing_evaluating_secondary`); multi-step routing soak; tokenizer plugin stub + USAGE; scaffold comment; **`ModelCatalogMissDiagV1` Plane A trace** (`llm_catalog_miss`).

| Gap | Severity | Task ID | Audit ID | Status |
|-----|----------|---------|----------|--------|
| Domain audit register + journal not formally closed | **P1** | M-LLM-X.8.1–8.3 | **LLM-AUDIT-21** | **Done** |
| Capability flags not catalog-driven | **P2** | M-LLM-X.14.1 | **LLM-AUDIT-22** | **Done** |
| OpenRouter / gateway live metadata not merged | **P1** | M-LLM-X.14.2 | **LLM-AUDIT-23** | **Done** |
| ACP budget token bridge incomplete | **P2** | M-LLM-X.14.4 | **LLM-AUDIT-24** | **Done** |
| Secondary LLM: sync only | **P2** | M-LLM-X.14.5 | **LLM-AUDIT-25** | **Done** |
| Plugin provider enum coupling | **P2** | M-LLM-X.14.3 | **LLM-AUDIT-26** | **Done** |
| Multi-step routing soak | **P2** | M-LLM-X.14.6 | — | **Done** |
| Tokenizer accuracy doc + plugin stub | **P3** | M-LLM-X.14.7 | — | **Done** |
| Scaffold DX | **P3** | M-LLM-X.14.8 | — | **Done** |

**Enterprise-grade domain DoD:** **Met** — X-8 + X-14 **Done** · LLM-AUDIT-21…26 **Done** · LLM CI gates green.

#### Catalog miss observability enterprise (M-LLM-X-15)

**Source:** Post X-14 maturity assessment (2026-06-19) — trace wiring **L4−**; OpenRouter unknown models masked by `provider_defaults`; no metrics/E2E/gate.

| Gap | Severity | Task ID | Status |
|-----|----------|---------|--------|
| Miss only on `fallback_default` — OpenRouter `provider_default` silent | **P1** | M-LLM-X.15.1 | **Done** |
| Trace sink wired only when `core_adapter is not None` | **P2** | M-LLM-X.15.2 | **Done** |
| No Prometheus counter for catalog misses | **P2** | M-LLM-X.15.3 | **Done** |
| No CI observability gate for `llm_catalog_miss` spine | **P2** | M-LLM-X.15.4 | **Done** — `check_llm_catalog_miss_observability.py` |
| No acceptance E2E trace → runtime bus | **P2** | M-LLM-X.15.5 | **Done** |
| ADR-LLM-002 resolution order drift vs code | **P3** | M-LLM-X.15.6 | **Done** |

**Target maturity:** catalog-miss spine **L4 enterprise** — **Met** (2026-06-19).

#### Catalog miss L5 enterprise ops (M-LLM-X-16 — Done)

**Source:** Post X-15 assessment — spine **L4+**; ops gaps (runbook, alerts, umbrella CI, run isolation, OBS-BUS, SLO canon).

| Gap | Severity | Task ID | Status |
|-----|----------|---------|--------|
| No operator runbook | **P2** | M-LLM-X.16.1 | **Done** — USAGE § Catalog miss operator runbook |
| No reference Prometheus alert rules | **P2** | M-LLM-X.16.2 | **Done** — USAGE § Alerting |
| LLM-MAINT-05 not in platform CI umbrella | **P2** | M-LLM-X.16.3 | **Done** — **LLM-MAINT-06** |
| Module-global dedupe / pending without run scope | **P2** | M-LLM-X.16.4 | **Done** — `begin_catalog_miss_run` + run observers |
| `llm_catalog_miss` absent from OBS-BUS emission coverage | **P3** | M-LLM-X.16.5 | **Done** |
| No SLO / severity guidance in observability canon | **P3** | M-LLM-X.16.6 | **Done** — OBSERVABILITY §7.1.1 |

**L5 ops criteria:** all six items **Done** (2026-06-19).  
**Target maturity:** catalog-miss **L5 enterprise ops** — **Met**.  
**Plan:** [Phase M-LLM-X-16](../plan/LLM_ADAPTERS.md#phase-m-llm-x-16--catalog-miss-l5-enterprise-ops-2026-06-19)

**Plan (X-15):** [Phase M-LLM-X-15](../plan/LLM_ADAPTERS.md#phase-m-llm-x-15--catalog-miss-enterprise-observability-2026-06-19)

#### Routing strict enterprise closeout — audit register (historical gaps, closed)

| Gap (audit) | Severity | Task ID |
|-------------|----------|---------|
| `LLMUsageTracker` reads wrapper; inner adapter accumulates tokens → `BudgetBelowRule` may not fire mid-run | **P0** | M-LLM-X.12.1 |
| `evaluating_adapter.py` imports `applications/_shared/llm_resolver` — Tier-0 → Tier-3 violation | **P1** | M-LLM-X.12.2 |
| `sync_llm_routing_snapshot` only in UAEP — Nexus graph / CE paths skip refresh | **P1** | M-LLM-X.12.3 |
| Context stale between multiple LLM calls within one step | **P1** | M-LLM-X.12.4 |
| `create_adapter_for_routing_evaluation` passes empty `RoutingContext()` to AHI wiring | **P1** | M-LLM-X.12.5 |
| `budget_degrade_active` not mapped in Nexus sync | **P1** | M-LLM-X.12.6 |
| Global `set_routing_evaluation_observer` — concurrent run risk | **P2** | M-LLM-X.12.7 |
| ACP `DynamicLLMRouter` without `on_evaluated` trace in `acp_run` | **P2** | M-LLM-X.12.8 |
| First eval trusts resolver profile even when rules disagree | **P2** | M-LLM-X.12.9 |
| E2E is mock-based — no production meter + trace proof | **P2** | M-LLM-X.12.10 |
| Docs claimed L5 prematurely | **P2** | M-LLM-X.12.11 |
| Planner / critic / websearch LLM bypass evaluating wrapper | **P3** | M-LLM-X.12.12 |

**Strict L5 criteria (checklist — all must pass before L5 label):**

1. `budget_remaining_ratio` in `RoutingContext` reflects **actual** run token usage on core adapter path.
2. Context sync runs on **UAEP + Nexus graph + context-engine** paths before routing eval.
3. No Tier-0 import from `applications/` for routing hot path.
4. Per-eval trace on **ACP and Nexus** with correlated `run_id` (no process-global observers).
5. Acceptance test: budget threshold → model swap **without** mocking `create_adapter_for_routing_evaluation`.
6. Audit register **LLM-AUDIT-19** → **Done**.

#### Custom rule example (Tier-3)

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

#### Three-layer model

```text
Layer 1 — Author rules (LLMRoutingProfile on Tier-3)     → explicit logic; always wins over L4
Layer 2 — LLMRoutingEvaluator + ModelRouter (Tier-0)    → hot path (SYS-INV-10 single router)
Layer 3 — AHI ROUTING_TUNING (AdaptiveProfile, optional) → bandit proposes ProfileVersion;
                                                           does not execute arbitrary author code
```

#### Provider coverage

| Model source | Routing path |
|--------------|--------------|
| Cloud API (OpenAI, Claude, Groq, …) | `LLMProfile(provider=…)` |
| Local (Ollama, vLLM, llama.cpp) | Existing `LLMProvider` slugs |
| HF self-hosted weights | Model id on `vllm` / `llama_cpp` profile |
| HF Inference API (chat) | Optional provider plugin (M-LLM-X.9.8) or gateway via `openrouter` |

**Anti-patterns:** string `eval` for rules; provider selection inside Tier-2 agents; routing target outside `allowed_profiles`; parallel router subsystem.

---

## Developer surfaces

### Nexus path (primary)

Inject `LLMAdapter` via `RuntimeConfig`; call `generate_messages` / tools / stream; read `.content` and `.usage`.

### ACP path (convergence target)

Today: `StepLLMRouter` in `agents/authoring/llm_router.py` — separate stub port when `llm_port` is None.

**Target (M-LLM-X.5):** `StepLLMRouter` delegates to `LLMAdapter.generate_messages` via `LLMAdapterCompletePort` — **Done** (LC-3).

### Documentation

| Artifact | Status |
|----------|--------|
| `docs/architecture/LLM_ADAPTERS.md` | This file |
| `docs/plan/LLM_ADAPTERS.md` | Phase M-LLM-X register |
| `intergrax/llm_adapters/USAGE.md` | **Done** — quickstart, env matrix, overrides, failover, extension |
| `docs/guides/AGENT_CREATION_GUIDE.md` § LLM | Cross-link only |

### Startup validation (target M-LLM-X.7)

`LLMProfile.validate_runtime()` — optional lightweight check: catalog hit, context window > 0, API key present, optional `adapter.validate()` ping.

---

## Modality plane A — generative multimodal (LLM)

LLM adapters own **Plane A** ([`MODALITY.md`](MODALITY.md) §7.1.9). Plane C (YOLO, ONNX, …) stays in `model_inference/`.

| Concern | Owner |
|---------|-------|
| Chat reasoning | `llm_adapters/` |
| Native vision/audio in dialog | `llm_adapters/` — capability flags (W-ML.1) |
| Deterministic CV / TTS tools | `model_inference/` + `speech_adapters/` |

### Capability flags

| Method | Meaning |
|--------|---------|
| `supports_vision()` | Image (optional video frame) input |
| `supports_audio_input()` | Audio in chat |
| `supports_audio_output()` | Spoken response |

Defaults **false** until mapping + conformance tests pass. **Target:** flags populated from `ModelCatalog` when known.

### Attachments

`intergrax/llm/messages.py` — `AttachmentRef`. Adapters map to vendor parts; `ModalityProfile.max_media_bytes` caps volume.

---
