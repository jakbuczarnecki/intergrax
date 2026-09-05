# Stage 16 — Agent Platform Architecture Audit and Conformance Closure

**Audit ID:** STAGE-16-AGENT-ARCHITECTURE-AUDIT  
**Date:** 2026-09-05  
**Branch:** `development`  
**Start HEAD:** `ac83b20ffac689096b924300c3fce125451f93ea`  
**origin/development (start):** `ac83b20ffac689096b924300c3fce125451f93ea`

---

## Canonical architecture

```text
Agent package / definition
        ↓
Catalog / Agent Distribution
        ↓
install → bind → EffectiveRosterAuthority
        ↓
RuntimeRevision → materialization
        ↓
ReferenceProductionLifecycleLauncher / Activation
        ↓
traffic_serving_revision_id
        ↓
RegistryProjectionAuthority
        ↓
MaterializedRegistryProjection
        ↓
AgentRegistryRead
        ↓
canonical Execution (HostTaskExecutionPort)

Agent Manager: discovery/read + AgentPlatformAdminService (mutations)
Capability Map: derived discovery projection only
Nexus: internal orchestration — NOT Tier-3 public execution API
```

---

## Authority matrix

| Concern | Canonical authority | Duplicate found? |
| --- | --- | --- |
| install | `InstallationService` | No production duplicate |
| bind | `BindingService` / `EffectiveRosterAuthority` | No production duplicate |
| revision | `RuntimeRevisionService` | No production duplicate |
| materialization | `RuntimeMaterializationService` | No production duplicate |
| activation | `ActivationService` | No production duplicate |
| serving | `traffic_serving_revision_id` + serving store | No second serving authority |
| projection | `RegistryProjectionAuthority` | No duplicate; `AgentRegistry` is derived |
| execution | `HostTaskExecutionPort` / `execution.execute` | Nexus internal only at Tier-1 |
| discovery | Capability Map + federated catalog read models | No lifecycle authority leak |

---

## Findings

| ID | Severity | Location | Problem | Canonical invariant | Resolution | Status |
| --- | --- | --- | --- | --- | --- | --- |
| S16-001 | P1 | `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` §1, Step 4–5 | Guide taught `AgentRegistry.register()` + direct `NexusLoop` as integration model | Agent Distribution owns lifecycle; Execution is Tier-3 boundary | Bounded rewrite of mental model, Step 4–5, anti-pattern rows | **Fixed** |
| S16-002 | P1 | `intergrax/scaffold/new_agent.py` | Scaffold README/notebook emitted `AgentRegistry()` + `NexusLoop` quickstart | Scaffold must not teach lifecycle bypass | Updated `_readme`, `_notebook_stub`, integration wording | **Fixed** |
| S16-003 | P1 | `intergrax/scaffold/doc_templates.py` | Architecture doc template listed `AgentRegistry.register()` | Same as S16-001 | Registration → Integration section | **Fixed** |
| S16-004 | P1 | `agents/*/README.md` (9 active) | Public README quickstarts used mutable `AgentRegistry` | Public agent surfaces must not recommend local registration | Migrated to `agent.run()` + distribution lifecycle guidance | **Fixed** |
| S16-004b | P1 | `agents/*/README.md` (9 active) | Stage 16 migration dropped `## Capabilities`, corrupted `## Layout`, duplicated Step 4 | Active README must preserve capability ids and valid section headings | Restored Capabilities/Layout; extended conformance gate for structure | **Fixed** |
| S16-005 | P1 | Missing gate | No reusable guard on canonical authoring surfaces | Prevent regression on guide/scaffold/README | Added `check_canonical_authoring_surface_conformance.py` + unit gate | **Fixed** |
| S16-006 | P2 | `agents/*/notebooks/*.ipynb` | Historical notebooks still import `NexusLoop` | Active/canonical notebooks must be current | Left as historical; scaffold notebook now uses `agent.run()` | **Residual** |
| S16-007 | P2 | `agents/*/tests/*` (legacy packages) | Some agent tests still use `NexusLoop` smoke path | Unit tests should prefer `agent.run()` | Out of Stage 16 scope; existing packages not bulk-migrated | **Residual** |
| S16-008 | P2 | `AGENT_CREATION_GUIDE.md` appendices I–O | Deep appendix examples still reference `NexusLoop(...)` for orchestration internals | Appendices are control-plane reference, not Tier-3 quickstart | Bounded gate excludes appendices; Stage 17 doc cleanup | **Residual** |
| S16-009 | P2 | `platform_proofs/.../scenario.py` | Fallback `AgentRegistry()` when `not composition.is_platform_attached` | Scenarios must use platform-attached lifecycle | Covered by scenario architecture gate for new code; legacy fallback documented | **Residual** |
| S16-010 | P3 | `RuntimeContext` / `RuntimeRequest` in `intergrax.runtime.nexus.*` | Authoring API still on Nexus namespace | Neutral public contract pending | No broad move per Stage 16 stop rule | **Residual debt** |

**P0 findings:** none discovered in production serving paths during bounded scan.

---

## Residual debt (conscious)

1. **Legacy Nexus namespace contracts** (`RuntimeContext`, `RuntimeRequest`) remain in `intergrax.runtime.nexus.*` for agent authoring; neutral public re-export not attempted (requires Execution refactor).
2. **Historical agent package tests** (`signoff_probe`, `dispute_scenario`, etc.) still construct `NexusLoop` — migrate opportunistically when touching those agents.
3. **Historical experiment notebooks** under `agents/*/notebooks/` — not linked as canonical quickstarts; scaffold-generated notebooks now use `agent.run()`.
4. **AGENT_CREATION_GUIDE appendices** retain internal Nexus orchestration reference material (P2 cleanup in Stage 17).

---

## Inventory summary (bounded scan)

### AgentRegistry / register patterns

| Classification | Representative locations | Verdict |
| --- | --- | --- |
| INTERNAL_PROJECTION_BUILDER | `intergrax/applications/_shared/registry_projection.py`, `wiring.py` | Allowed — materialization |
| UNIT_TEST / TEST_SUPPORT | `testing_support/agent_registry_bootstrap.py`, runtime unit tests | Allowed — isolated bootstrap |
| SCENARIO (gated) | `platform_proofs/scenarios/...` | Legacy fallback when not platform-attached (P2) |
| DOCUMENTATION (fixed) | `AGENT_CREATION_GUIDE`, agent READMEs | Was P1 — corrected |
| SCAFFOLD (fixed) | `new_agent.py` | Was P1 — corrected |
| PRODUCTION | Tier-3 `applications/*/host`, `applications/*/serving` | No direct `AgentRegistry()` / `register()` in production host paths |

### Nexus public surface

| Category | Count (bounded) | Status |
| --- | --- | --- |
| A — Forbidden Tier-3 public entry | 0 in production host/serving after prior Stage 15 gates | Clear |
| B — Legacy namespace contract | `RuntimeContext`, `RuntimeRequest` in agent scaffold + authoring | Documented residual (S16-010) |

### Control plane

- `AgentManager` → read projection + `AgentPlatformAdminService` for mutations — **confirmed** (existing gate passes).
- No second `InstallationService` / `ActivationService` duplicate authority in application production composition paths.

### Identity chain / desired-vs-serving

- No new P0 violations found mixing `enabled == serving` in Agent Manager read models during bounded scan.
- Private vs public agent parity: no `private_agent_install()` alternate lifecycle discovered.

---

## Tests / gates (Stage 16)

| Suite | Result |
| --- | --- |
| `tests/unit/docs/test_canonical_authoring_surface_conformance.py` | 5 passed |
| `tests/unit/docs/test_agent_creation_guide_step_4e.py` | 1 passed |
| `tests/unit/scaffold/test_acp_pattern_scaffold.py` | 2 passed |
| `tests/unit/applications/architecture/test_application_lifecycle_conformance.py` | 13 passed |
| `tests/unit/agent_distribution/test_agent_manager_architecture_gate.py` | 2 passed |
| `tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py` | 2 passed |
| `tests/unit/scaffold/test_npsc2_canonical_scaffold_execution.py` | 4 passed |
| `tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py` | 3 passed (Stage 15 proof) |

---

## Explicit confirmations (post-Stage 16)

- **No production local `AgentRegistry` construction** in Tier-3 host/serving paths (bounded scan).
- **No public canonical `registry.register()` instruction** on guide/scaffold/active README surfaces (gate enforced).
- **No Tier-3 direct `NexusLoop` public quickstart** in canonical authoring surfaces.
- **No second lifecycle manager** introduced.
- **No Capability Map authority leak** found.
- **No Agent Manager lifecycle write leak** (existing gate).
- **Stage 15 proof gate** remains valid (no changes to proof targets).
- **active README structure integrity** = PASS (9 agents: Capabilities preserved, valid `## Layout`, no `## ##`, no duplicate Step 4).

---

## Changed artifacts (Stage 16 scope)

- `docs/project/technical/guides/AGENT_CREATION_GUIDE.md`
- `intergrax/scaffold/new_agent.py`
- `intergrax/scaffold/doc_templates.py`
- `agents/*/README.md` (9 agents)
- `scripts/maintenance/check_canonical_authoring_surface_conformance.py`
- `tests/unit/docs/test_canonical_authoring_surface_conformance.py`
- `docs/project/maintainers/audits/STAGE_16_AGENT_ARCHITECTURE_AUDIT.md`
