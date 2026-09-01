# ADR-APP-003: Hierarchical profile bundles on ApplicationEnvironmentProfile

| Field | Value |
|-------|-------|
| **Status** | Accepted · **documentation** (`APP-EVOL-8-DOC`); implementation **planned** (M1–M3) |
| **Date** | 2026-06-17 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §22.6 · plan `APP-EVOL-8` · `P1-ARCH-01` |

## Context

`ApplicationEnvironmentProfile` is the Tier-3 composition root (`APP-INV-06`). It already composes **25+ typed sub-profiles**, but exposes them as **43+ flat top-level fields**. Each new harness domain (Code Craft, EBE, adaptive loops, governance slices) adds another top-level slot.

Symptoms:

- Author cognitive load - product teams must navigate a flat list to configure RAG, cost, and security.
- Preset duplication - `lab_defaults()` / `product_defaults()` copy large `model_copy` blocks.
- Growing merge surface - `runtime_config_bridge`, `merge_environment`, and snapshot digests scale with field count.

Alternatives considered:

1. **Keep flat forever** - simple wire path, unbounded namespace growth.
2. **Multiple peer roots** (`EnvironmentProfile` + `CapabilityProfile`) - breaks `APP-INV-06`, snapshots, and conformance gates.
3. **Nested bundles under single root** - groups fields; preserves one composition contract; backward compatible via shims.

## Decision

1. **Retain `ApplicationEnvironmentProfile` as the sole composition root** - name and wiring entrypoints unchanged.
2. Introduce **seven nested bundle containers** (plus `EnvironmentExtensions`):
   - `HostMeta`, `SecurityEnvelope`, `CapabilityBundle`, `CognitionBundle`, `GovernanceBundle`, `TopologyBundle`, `IsolationBundle`
3. **Sub-profile schemas are unchanged** - only nesting and authoring presets evolve.
4. **Migration in three phases:**
   - **M1:** nested models + flat `@property` shims + flat JSON deserializer (`spec_version` `1.x`, non-breaking).
   - **M2:** per-bundle presets and shared `CapabilityBundle` packs across hosts.
   - **M3:** `spec_version` `2.0.0` - nested JSON canonical; flat top-level deprecated.
5. **§41 primitives stay separate concepts** - `OrganizationalPolicyEnvelope` nests under `SecurityEnvelope` but remains its own type; `ApplicationGraphSpec` under `TopologyBundle`; `AgentBinding` merge unchanged.

Rejected: peer composition roots; bundle-local wiring logic; folding org envelope into `CapabilityBundle`.

## Consequences

### Positive

- Authoring groups align with harness mental model (security vs catalogs vs SRE vs topology).
- Reusable `CapabilityBundle` presets across product hosts.
- Controlled evolution path without Nexus fork or wiring rewrite in M1–M2.

### Negative

- Temporary dual access paths (flat shims + nested) until M3.
- Snapshot/diff must normalize bundle form for digest parity (`APP-EVOL-8.3`).
- Schema export grows nested depth - gate test required.

## Compliance

- Tier boundaries: contracts remain in `intergrax/applications/contracts`; wiring in `applications/_shared`.
- `APP-INV-06` preserved - bundles are grouping only.
- No Nexus runtime changes in M1–M2.
- Linked architecture §22.6 and plan `APP-EVOL-8` updated.

## Implementation notes

- Contract target: `intergrax/applications/contracts/environment_profile.py`
- Plan register: `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` - `APP-EVOL-8.1`–`APP-EVOL-8.7`
- Verification (when M1 lands):
  - `uv run pytest tests/unit/applications/test_environment_profile_bundles.py -q`
  - `uv run pytest tests/unit/applications/test_environment_profile.py -q`
  - `python scripts/maintenance/check_harness_adr.py`
