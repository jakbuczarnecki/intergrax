# Enterprise Execution Scale & Resilience — W0 Guardrails

**Task:** Enterprise Scale & Resilience/W0 — Mandatory Host Capacity Caps & Deployment Guardrails  
**Status:** COMPLETE (guardrail wave; no distributed admission framework)

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `455d3b216f0ad56ea9cdf9db6e0f760b50063a81` |
| START_ORIGIN | `455d3b216f0ad56ea9cdf9db6e0f760b50063a81` |
| Branch | `development` |

Architecture companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).  
P0 inventory: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md).

## Canonical capacity owner

| Field | Owner |
|-------|--------|
| Schema / bounds (`ge=1`, `le=256`) | `OrchestrationProfile` in `intergrax/applications/contracts/environment_profile/sub_profiles.py` (mirror in `intergrax/contracts/host_profile_slices.py` for agent merge) |
| Resolution | `resolve_max_parallel_nodes`, `resolve_max_inflight_nodes`, `resolve_orchestration_runtime_settings` in `intergrax/applications/_shared/orchestration_wiring.py` |
| Strict guardrail | `validate_strict_host_execution_capacity` in `intergrax/applications/_shared/host_execution_capacity_policy.py` |
| Runtime consumer | `GraphExecutor` via `build_nexus_loop_from_environment` → `NexusLoop` |

Production mode in Tier-3 composition maps to `ExecutionMode.STRICT` (`production_mode=True` on Nexus when `execution_mode.value == "strict"`).

## Mandatory production caps

When `execution_mode=strict`:

- `orchestration_profile.max_parallel_nodes` **must** be set (not `None`).
- `orchestration_profile.max_inflight_nodes` **must** be set (not `None`).

Validation runs at canonical Nexus composition (`build_nexus_loop_from_environment`) and is surfaced by `ProfileInvariantValidator` for environment conformance messages.

W0 does **not** mandate numeric values for every deployment — only **explicit** deployment-selected limits. Platform maximum remains **256** per field (Pydantic contract). Reference product template `ApplicationEnvironmentProfile.product_defaults` sets **8 / 8** (same deployment class as existing `strict_multi_agent_defaults` / `async_batch_defaults` presets).

Unset caps in non-strict (lab/balanced) profiles remain allowed for unit tests.

## Local vs distributed semantics

| Mechanism | Semantics |
|-----------|-----------|
| `GraphExecutor` `max_parallel_nodes` / `max_inflight_nodes` | **Process-local** (`asyncio.Semaphore` per executor instance) |
| `asyncio.Lock` / registry locks | **Process-local** |
| Scheduler lease / checkpoint CAS | Cross-process durability — **not** execution capacity limits |

**Operator invariant:** `N workers × max_inflight_nodes=M` is **not** a global cap of `M`; it is up to **N×M process-local slots** unless a future wave adds distributed admission (W1+).

ECP `scaling_wiring` `resolve_max_inflight_nodes(env) or 8` applies only to orchestration **ceiling patcher** for scaling control plane — not a substitute for explicit strict-profile caps on Nexus.

## Deployment guardrails

1. Configure caps on `orchestration_profile` in the host/application environment profile (canonical path — no env-var side door to `GraphExecutor`).
2. Use `execution_mode=strict` only with both caps explicitly set.
3. Do not treat per-worker semaphores as cluster-wide throttles.

## Implemented changes

- `host_execution_capacity_policy.py` — strict-mode fail-closed validation.
- `nexus_factory.build_nexus_loop_from_environment` — invokes validation before `NexusLoop` construction.
- `ProfileInvariantValidator` — reuses same violation messages.
- `ApplicationEnvironmentProfile.product_defaults` / `strict_multi_agent_defaults` — explicit template caps (8/8).
- Qualification tests: `tests/unit/applications/test_host_execution_capacity_guardrails.py`.

## Deferred to W1

- ExecutionRuntime root admission; `ConcurrentExecutionWork` width cap; global deadline on all Nexus retry paths.

## Deferred to W2

- Provider/tool bulkheads; distributed rate limits; retry concurrency manager.

## Test evidence

```text
uv run pytest tests/unit/applications/test_host_execution_capacity_guardrails.py -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py -q
uv run pytest tests/unit/applications/test_orchestration_environment_presets.py -q
```

## Known limitations

- No distributed/global capacity sum or enforcement (by design for W0).
- `ProfileInvariantValidator` in `wire_application_environment` still uses `fail_on_violation=False` — strict rejection is authoritative at Nexus factory composition.
- Legacy duplicate `OrchestrationProfile` model in `host_profile_slices.py` remains for agent merge; application wiring owner is environment profile sub_profiles.
