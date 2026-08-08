# Cross-Document Governance Consistency Audit

**Status:** Freeze audit (2026-06-11) — pre-architecture-freeze Tier-3 + ACP  
**Scope:** Semantic overlap and responsibility boundaries — **not** gap analysis  
**Canon pair:** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) · [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Ideal reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md) §19.4 · §18.4  

---

## 1. Purpose

After APP-CON / APP-EVOL / APP-OPS documentation tranches, the platform has many similarly named constructs (`*Registry`, `*Governance*`, `*Capability*`, `*Health*`). This audit answers:

1. Are there **two different definitions** of capability?
2. Are there **two registries** describing the same thing?
3. Does **ownership** duplicate lifecycle governance?
4. Does **health score** duplicate APP-PROD gates?
5. Does **TIER3 §50** conflict with **IDEAL** architecture?

**Verdict summary:** No structural conflicts. **Three naming overlap risks** require glossary discipline; **one table row** in TIER3 §22 was misleading (fixed). **CapabilityRegistry** must not be introduced as a new type.

### 1.1 Platform evolution governance checks

When reviewing new capabilities or adoption work, also verify ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)):

| Check | Principle |
|-------|-----------|
| One architectural owner per reusable capability | `PLATFORM-INV-003` |
| Generic capability not owned by an application | `PLATFORM-INV-001`, `PLATFORM-INV-011` |
| Architecture before adoption | `PLATFORM-INV-002` |
| Domain plan owns implementation tasks | §17 architecture/plan pairing |
| First adopter and proof do not redefine architecture | `PLATFORM-INV-005` |

---

## 2. Capability layer — ALIGNED (do not merge)

| Construct | Canonical home | Layer | Answers |
|-----------|----------------|-------|---------|
| **`AgentContract.capabilities[]`** | ACP §12 · §16 | Agent contract | What capability strings this agent implements |
| **`CapabilityDescriptor`** | UAEP §42.27 | Routing metadata | `(capability, semver, agent_id)` for versioned Nexus routing |
| **`CapabilityGraph`** | **ACP §19** (model) · `capability_graph.py` | Structural graph | Integration→Tool→Skill→Policy→Agent→Application dependency edges |
| **`EnvironmentCapabilityGraphView`** | TIER3 §50.1 · `capability_graph_wiring.py` | Tier-3 slice | Application-scoped subgraph for impact/lineage at host boundary |
| **`CapabilityAlias` / deprecation** | TIER3 §49.3 (**Done** APP-EVOL-3) | Sunset policy | Redirect deprecated capability strings during migration window |

### 2.1 Anti-pattern: `CapabilityRegistry`

**Does not exist** in codebase and **must not be added** as a parallel catalog.

| Need | Use instead |
|------|-------------|
| Route task to agent | `AgentRegistry` + `CapabilityMatchResult` (ACP §15) |
| Versioned capability metadata | `CapabilityDescriptor` on contract / UAEP routing table |
| Dependency / blast radius | `CapabilityGraph` + `build_capability_impact_report()` |
| Deprecation redirect | `CapabilityAlias` **Done** (`capability_alias_wiring.py` · intake middleware) |

**Risk:** Authors conflate “registry of capabilities” with `AgentRegistry` or invent `CapabilityRegistry`. Glossary term: **capability routing** = registry; **capability structure** = graph.

### 2.2 IDEAL §19.4 alignment

IDEAL requires typed nodes, lineage, blast radius, validation gates. Implemented as ACP §19 + `phase_v_capability_graph_guard.py`. TIER3 §50.1 adds **environment-scoped consumption** only — **no second graph model**.

---

## 3. Registry layer — LAYERED (different lifecycles)

| Name | Tier | Scope | Mutable? | Canonical doc |
|------|------|-------|----------|---------------|
| **`AgentRegistry`** | 1 runtime | In-process agent instances for one host boot | Per process | ACP §15 |
| **Tool / Skill / Integration / Prompt / Evaluation registry** | 0–1 | Catalog materialized from profile | Snapshot per deploy | ACP §18 |
| **`applications/README.md` index** | 3 docs | Human roster of monorepo apps | Manual | `applications/README.md` |
| **`ApplicationRegistry`** **Done** (APP-OPS-4) | Ops | Inventory of `ApplicationPackage` releases | Append-only versions | TIER3 §50.4 |
| **`EnvironmentRegistry`** **Done** (APP-OPS-4) | Ops | Deployed instances (region, image, endpoint) | Per deployment event | TIER3 §50.4 |

### 3.1 Rules

- **`AgentRegistry`** ≠ **`ApplicationRegistry`**. First is **runtime selection**; second is **platform inventory**.
- **`EnvironmentRegistry`** ≠ **`ApplicationEnvironmentProfile`**. Profile is config; registry entry is **where** that config is deployed. Nested profile bundles (§22.6 · ADR-APP-003) do **not** introduce a second config type — they group fields inside the same profile.
- README table is **documentation index**, not a registry contract — prefer `ApplicationRegistry` artifacts for ops automation (APP-OPS-4 **Done**).

**Overlap risk:** Low, if naming is preserved. **Conflict:** None.

---

## 4. Ownership & governance — LAYERED (watch naming)

| Construct | Home | Scope | Not the same as |
|-----------|------|-------|-----------------|
| **`AgentLifecycleState` + transition evaluator** | ACP §20 · `agent_lifecycle_governance.py` | experimental→retired **state machine** | Ownership |
| **`ProductionOwnerMetadata` / V-ALG.4** | ACP §20 · `production_ownership.py` | Per-**agent** on-call when `production_eligible` | Application owner |
| **`ApplicationOperationalOwnership`** **Done** (APP-OPS-2) | TIER3 §50.2 | Per-**application host** team / escalation | Agent lifecycle |
| **`GovernanceProfile`** | `environment_profile.py` | **Feature flags**: quarterly review, dashboard toggles | Ownership or lifecycle |
| **`IntegrationGovernanceProfile`** | `environment_profile.py` | **Feature flags**: marketplace catalog, hot-reload | Integration ownership |

### 4.1 Naming clarification (normative)

```text
GovernanceProfile           → platform cadence / dashboard FLAGS (not owner metadata)
IntegrationGovernanceProfile → integration marketplace FLAGS (not provider ownership)
AgentLifecycleGovernance    → agent state transitions + deprecation rules
ProductionOwnership         → agent on-call metadata gate (V-ALG.4)
ApplicationOperationalOwnership → application host ops contacts (APP-OPS-2)
```

**Overlap risk:** **Medium** on word “governance” — mitigated by glossary above. **Conflict:** None — different Pydantic models, different evaluators.

**Lifecycle vs ownership:** Lifecycle answers **may this agent run in prod?** Ownership answers **who is paged when it fails?** Application ownership answers **who owns the host deployment?** Orthogonal concerns.

---

## 5. Health & gates — LAYERED (complementary stack)

| Surface | Type | When | Home |
|---------|------|------|------|
| **APP-PROD-1..8** | Boolean CI gates | Deploy / merge | TIER3 §40 |
| **ACP-PROD-*** | Boolean agent gates | Agent certification | ACP §45 |
| **`EnvironmentHealthStatus`** | Runtime enum per Task | During execution | TIER3 §42 |
| **`ArchitectureMetricsReport`** (V-AM.1) | Graph structural metrics | Report-only CI | `architecture_metrics.py` · enabled via `GovernanceProfile.architecture_health_metrics_enabled` |
| **`EnvironmentHealthScore`** **Done** (APP-OPS-3) | Composite 0–1 ops score | Continuous / release | TIER3 §50.3 |

### 5.1 Stack (no duplication)

```text
APP-PROD gates     → pass/fail blockers (must be green to ship)
ArchitectureMetrics → structural debt signals from CapabilityGraph
EnvironmentHealthScore → rollup of gates + dimensions + staleness over time
EnvironmentHealthStatus → live task posture (budget, HITL, policy)
```

**Rule:** Health **score** may surface APP-PROD failures as dimensions — it does **not** replace gates. Gates remain authoritative for merge/deploy.

**Overlap risk:** Low with explicit stack. **Conflict:** None.

---

## 6. TIER3 §50 vs IDEAL — ALIGNED

| IDEAL topic | TIER3 §50 | Result |
|-------------|-----------|--------|
| §19.4 Capability graph | §50.1 references ACP §19 graph, adds env view | Aligned |
| §18.4 Evaluation registry | Not duplicated; eval stays ACP §18 | Aligned |
| Ownership in production L3 | §50.2 extends to application host | Aligned |
| Dependency blindness anti-pattern | §50.1 blast radius | Aligned |
| Marketplace / distribution | §49.7 + §50.4 deferred UI | Aligned (explicit deferral) |

**No contradictory definitions** found between §50 and IDEAL.

---

## 7. Canonical ownership matrix (freeze reference)

| Concept | Single source of truth |
|---------|------------------------|
| Agent cognition / `on_next_step` | ACP |
| Application composition / hooks / profile | TIER3 |
| Capability **graph** model | **ACP §19** |
| Capability **routing** | ACP §15 + UAEP §42.27 |
| Capability **graph at host** | TIER3 §50.1 (view only) |
| Runtime registries (agent/tool/skill/…) | ACP §18 |
| Ops registries (app/env inventory) | TIER3 §50.4 **Done** (APP-OPS-4) |
| Agent lifecycle + certification | ACP §20 |
| Application ops ownership | TIER3 §50.2 **Done** (APP-OPS-2) |
| Deploy boolean gates | TIER3 §40 APP-PROD |
| Continuous health scoring | TIER3 §50.3 **Done** (APP-OPS-3) |
| Graph structural metrics | V-AM.1 / `architecture_metrics.py` |

---

## 8. Freeze actions

| # | Action | Status |
|---|--------|--------|
| 1 | Publish this audit | **Done** |
| 2 | TIER3 §51 cross-ref matrix | **Done** |
| 3 | Fix §22 `GovernanceProfile` table description | **Done** |
| 4 | ACP §19 → TIER3 §50.1 cross-ref | **Done** |
| 5 | TIER3 §50.1 → ACP §19 canonical graph | **Done** |
| 6 | Ban `CapabilityRegistry` in glossary | **Done** (§2.1) |
| 7 | APP-OPS implementation | **Done** (APP-OPS-1…4) |

**Architecture freeze:** Tier-3 structural canon §24–§51 + this audit = **approved for freeze**. Further work is implementation (APP-*, ACP-TOK-*) and glossary discipline — not new composition primitives without ADR.

---

## 9. Re-run triggers

Re-run this audit when:

- A new `*Registry` or `*Governance*` type is proposed
- `CapabilityRegistry` or similar name appears in a PR
- APP-OPS-4 lands (validate ApplicationRegistry vs README index)
- ACP or TIER3 adds a second capability graph section without cross-ref

```bash
# Quick consistency smoke (manual)
rg "CapabilityRegistry" docs/ intergrax/
rg "GovernanceProfile" docs/project/architecture/TIER3*.md
python scripts/audit/check_docs_domain_pairs.py
```
