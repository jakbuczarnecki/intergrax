# ADR-AGENT-005: Reference production process composition and AP store ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture only) |
| **Date** | 2026-08-17 |
| **Deciders** | Agent Platform / Harness architecture |
| **Related** | [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) section 34 · [`ADR-AGENT-004`](../2026-08-12/ADR-AGENT-004.md) · AGENT-CONSOLIDATION-3 |

## Context

AGENT-CONSOLIDATION-3 is blocked by ambiguous ownership of live AP-9/AP-10 store instances used across activation, registry projection, and production host startup.

**Broken pattern (pre-FIX-3):**

```text
application main.py
  -> build_production_agent_platform_runtime()   # new empty in-memory universe
  -> bootstrap_production_registry_projection()  # no active serving revision
```

Lifecycle preparation/activation and host serving must share the **same** `AgentPlatformRuntimeStores` bundle for the process lifetime. Application entrypoints are not suitable composition roots.

Alternatives considered:

| Option | Rejected because |
|--------|------------------|
| Application `main.py` owns runtime | Duplicates lifecycle universe per host module; breaks activation-to-serve continuity |
| Application factory owns runtime | Factories are per-product wiring; same duplication risk |
| Global singleton / service locator | Violates explicit typed composition; hidden coupling |
| Durable shared DB in this task | Out of scope; premature without topology freeze |

## Decision

Freeze **reference production V1** topology:

1. **Owner:** `ProductionProcessComposition` (`intergrax/applications/_shared/production_process_composition.py`) holds one `ProductionAgentPlatformRuntime` per process.
2. **Lifetime:** construct once at process composition start; pass the same instance through prepare, project, activate, and serve. Never reconstruct per request, factory call, or bootstrap.
3. **Store bundle:** `AgentPlatformRuntimeStores` with `ApplicationEnvironmentServingStore` (AP-9 surface) and `RuntimeRegistryProjectionStore` (AP-10 surface). Multi-application `(application_id, application_environment_id)` keys in one bundle (AP-11).
4. **Writers:**
   - Serving pointer: `ActivationService` (AP-9) via `ApplicationEnvironmentActivationStore.atomic_commit_activation()` on shared distribution state.
   - Projection: `ApplicationRegistryProjectionCoordinator` (AP-10) via `RuntimeRegistryProjectionStore.put()`.
5. **Consumer:** `resolve_active_registry_projection()` / `bootstrap_production_registry_projection()` — read-only; fail closed without active revision.
6. **Cold start:** fresh reference process without prior activate cannot start STRICT host. Deploy/activate is separate from serve.
7. **Adapter tier:** process-local in-memory adapters support **reference single-process production semantics** only. They do **not** provide durable multi-instance production readiness.
8. **Deferred:** durable / multi-instance production (Postgres, Redis, Cassandra, distributed activation, K8s controller topology) — migration when restart survival or horizontal scale is required.

`build_production_agent_platform_runtime()` (alias `create_process_local_agent_platform_runtime`) constructs a **new** process-local universe. Only the process composition root may call it once per process.

Application `main.py` and factories remain forbidden as canonical lifecycle owners (enforced in AC-3-FIX-3 wiring).

## Consequences

### Positive

- Unblocks AC-3-FIX-3 with explicit injection point for shared stores
- Clear fail-closed cold-start semantics
- Separates dev/lab manifest path from production lifecycle
- Documents process-local limitation without overstating production durability

### Negative

- Reference hosts still require FIX-3 to relocate construction out of `main.py`
- Restart drops activation state until durable adapters exist
- Single-process adapter tier cannot scale horizontally

## Compliance

- Tier boundaries preserved: Tier-0 stores, Tier-3 hosts consume projections
- No global service locator; typed `ProductionProcessComposition` only
- No new persistence implementation in this ADR
- Linked architecture section 34 and continuity tests

## Implementation notes

- Code: `production_process_composition.py`, `production_agent_platform_runtime.py`
- Tests: `tests/unit/applications/test_production_composition_store_continuity.py`
- Verification: `uv run pytest tests/unit/applications/test_production_composition_store_continuity.py -q`
- Next: AGENT-CONSOLIDATION-3-FIX-3 — wire reference hosts through `ProductionProcessComposition`
