# ADR-LLM-003: LLM routing rules — Protocol contract and custom rule classes

**Status:** Accepted (2026-06-19)  
**Phase:** M-LLM-X.9  
**Context:** Mode I idea audit (2026-06-19) — `ModelRouter` + `FailoverLLMAdapter` provide static hints and failover, but no unified developer-facing mechanism for dynamic model selection by task state, budget, or custom business logic. AHI `ROUTING_TUNING` operates at L4 (bandit proposals) and must not replace explicit author rules.

## Decision

Introduce a Tier-0 **`LLMRoutingRule` Protocol** (optional `LLMRoutingRuleBase` ABC) with:

- **`matches(context: RoutingContext) -> bool`** — author-defined `when` logic (no side effects).
- **`resolve(context: RoutingContext) -> RoutingTarget`** — author-defined `then` output (`LLMProfile` or `RoutingHint`).
- **`rule_id: str`** and **`priority: int`** — required for trace and first-match ordering.

Tier-3 hosts configure **`LLMRoutingProfile`** on `ApplicationEnvironmentProfile`:

- `default_profile: LLMProfile`
- `allowed_profiles: tuple[LLMProfile, ...]` — mandatory allowlist guardrail
- `rules: tuple[LLMRoutingRule, ...]` — built-in parametric rules **and** custom classes from `applications`

**`LLMRoutingEvaluator`** (Tier-0/1) sorts rules by descending priority, returns the first `matches()` hit, validates `resolve()` against `allowed_profiles`, then delegates to existing **`ModelRouter`** + **`FailoverLLMAdapter`**.

Built-in rules (`BudgetBelowRule`, `TaskClassRule`, `TokenThresholdRule`, `BudgetExceededDegradeRule`, …) are **implementations of the same Protocol** — not a parallel enum DSL.

## Three-layer model

```text
Layer 1 — Author rules (Tier-3 LLMRoutingProfile)     → always wins over L4 learning
Layer 2 — LLMRoutingEvaluator + ModelRouter (Tier-0)  → hot path; single canonical router
Layer 3 — AHI ROUTING_TUNING (optional)                 → proposes ProfileVersion changes; never bypasses allowlist
```

## Tier boundaries

| Tier | Responsibility |
|------|----------------|
| Tier-0 `intergrax/llm_adapters/routing` | Protocol, evaluator, built-in rules, `RoutingContext` |
| Tier-3 `applications/<app>` | Custom rule classes; wired in manifest / `LLMRoutingProfile` |
| Tier-2 `agents` | May supply `model_hint` via `StepLLMRouter`; MUST NOT import vendor SDKs or select providers directly |

**Forbidden:** string-based `eval` / dynamic import paths for rules — only importable typed classes registered in manifest.

## Integration with existing mechanisms

| Existing | Relationship |
|----------|--------------|
| `ModelRouter` | Orders profiles after rule resolves target |
| `FailoverLLMAdapter` | Retriable provider errors — unchanged |
| `BudgetReactionProfile.degrade_model` | Converges to `BudgetExceededDegradeRule` (M-LLM-X.9.5) — single code path |
| `resolve_live_model_routing_wiring` | Consumes active `ProfileVersion` from AHI when `live_model_routing_enabled` |
| `RoutingTuningEngine` | Proposes `artifact_type=llm_routing` profile versions — does not execute arbitrary author code |

## Non-goals

- New 23rd domain pair — extends `LLM_ADAPTERS` only.
- Parallel Tier-0 router subsystem (violates SYS-INV-10).
- HF Hub as LLM chat provider — self-hosted HF models via `vllm` / `llama_cpp` remain the supported path.
- AHI auto-mutating custom rule source code in production.

## Consequences

- Plan rows **M-LLM-X.9.1–9.9** registered in `plan/LLM_ADAPTERS.md`.
- Cross-ref in `ADAPTIVE_HARNESS_INTELLIGENCE` §11.1 and **AHI-MAINT-05**.
- `TIER3_APPLICATION_ENVIRONMENT` documents `LLMRoutingProfile` sub-profile.
- CI gate `check_llm_routing_rules.py` (M-LLM-X.9.9) validates allowlist conformance in reference hosts.

## References

- [architecture/LLM_ADAPTERS.md](../../../../architecture/LLM_ADAPTERS.md) § LLM routing rules
- [plan/LLM_ADAPTERS.md](../../../../maintainers/plans/LLM_ADAPTERS.md) Phase M-LLM-X-9
- [ADR-LLM-002](../2026-06-14/ADR-LLM-002.md) — ModelCatalog (unchanged)
- [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](../../../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) §11.1 ROUTING_TUNING
