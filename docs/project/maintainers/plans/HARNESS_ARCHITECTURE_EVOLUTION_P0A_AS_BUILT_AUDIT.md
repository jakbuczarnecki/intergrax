# Harness Architecture Evolution — P0A As-Built Audit

## Status

**Audit type:** repository as-built classification only  
**Feature implementation:** NONE  
**Starting branch:** `development`  
**Starting SHA:** `61c5cd614d131340a3b83ca542b000d3ec054750`  
**Audit date:** 2026-09-02

This document is the evidence companion for P0A in [`HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md`](../../overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md).

Its purpose is to prevent already-shipped runtime foundations from being rebuilt and to separate:

- **DONE** — implemented and usable as a foundation,
- **PARTIAL** — implemented materially, but convergence/hardening/proof remains,
- **GAP** — materially missing from the canonical runtime path,
- **DOC-DRIFT** — code is ahead of canonical CURRENT documentation,
- **DO NOT DO** — explicitly avoid creating a competing subsystem.

Hard rule for implementation sessions:

> **ALREADY EXISTS => DO NOT REBUILD.**

A `DOC-DRIFT` classification is not an implementation gap. It is a documentation synchronization obligation.

---

# 1. Executive verdict

P0A confirms that Intergrax is materially further along in the Unified Execution spine than older CURRENT documentation states.

The repository already contains real implementations of:

- canonical `ExecutionId`,
- root `ExecutionRuntime`,
- `ExecutionBoundary`,
- `StrategyExecutionRouter`,
- `RuntimeEvent.execution_id`,
- child execution lineage through `parent_execution_id`,
- child authority narrowing,
- child budget allocation/reservation primitives,
- Execution-tree checkpoint snapshots,
- Execution-tree resume planning,
- ContextProvider contracts and providers,
- substantial sandbox runtime infrastructure,
- mature Skills catalog/resolver/plugin infrastructure,
- Platform Plugin coordination with a closed implementation roadmap.

Therefore the first implementation phase must **not** recreate these components.

The two most important real P0 safety gaps confirmed by the audit are:

1. **Tool Authority Intersection Integrity** — an explicit caller tool allow-list can currently replace, rather than intersect, a stricter `RuntimePolicyBundle.tool_access` allow-list.
2. **Skill Authority Integrity** — `extend_tool_profile_for_skills()` can currently expand `ToolProfile.enabled` from Skill requirements instead of validating that Skill requirements are already within host availability.

The most important documentation drift is the UER CURRENT description, which still describes `ExecutionId`, neutral `ExecutionBoundary`, and `RuntimeEvent.execution_id` as future target elements although they exist in current Python runtime code.

---

# 2. Frozen as-built evidence paths

The following implementation paths are primary P0A evidence:

## Unified Execution

- `intergrax/contracts/execution_identity.py`
- `intergrax/runtime/execution/runtime.py`
- `intergrax/runtime/execution/boundary.py`
- `intergrax/runtime/execution/child.py`
- `intergrax/runtime/execution/strategy_router.py`
- `intergrax/runtime/execution/execution_work_port.py`
- `intergrax/runtime/execution/host_task.py`
- `intergrax/runtime/execution/orchestration.py`
- `intergrax/runtime/events/runtime_event.py`
- `intergrax/runtime/events/emit_context.py`

## Execution budget / authority

- `intergrax/runtime/execution/budget/models.py`
- `intergrax/runtime/execution/budget/ledger.py`
- `intergrax/runtime/execution/budget/policy.py`
- `intergrax/runtime/execution/authority/policy.py`
- `intergrax/runtime/governance/active_execution_authority.py`

## Checkpoint / recovery

- `intergrax/runtime/long_running/runtime_checkpoint.py`
- `intergrax/runtime/long_running/execution_tree_checkpoint.py`
- `intergrax/runtime/execution/decision_checkpoint_persistence.py`
- `intergrax/runtime/execution/active_decision_checkpoint_persistence.py`

## Context

- `intergrax/context/protocols.py`
- `intergrax/context/contracts.py`
- `intergrax/context/orchestrator.py`
- `intergrax/context/providers/`
- `intergrax/context/session_history.py`
- `intergrax/context/providers/session_semantic_recall.py`

## Tool authority

- `intergrax/runtime/policy/tool_policy_resolution.py`
- `intergrax/runtime/nexus/tools/tool_runtime.py`
- `intergrax/runtime/tools/scope_policy.py`
- `docs/project/architecture/TOOLS.md`
- `docs/audit_results/2026-08-18/TOOLS.md`

## Skills authority

- `intergrax/applications/_shared/skill_tool_profile.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/skills/`
- `docs/project/architecture/SKILLS.md`
- `docs/audit_results/2026-08-18/SKILLS.md`

## Sandbox

- `intergrax/runtime/sandbox/`
- `intergrax/applications/_shared/sandbox_wiring.py`
- `intergrax/applications/_shared/sandbox_host_wiring.py`
- `intergrax/integrations/contracts/sandbox_host.py`
- `intergrax/runtime/codecraft/sandbox_resolver.py`

## Credentials / secrets

- `intergrax/integrations/contracts/secrets_store.py`
- `intergrax/integrations/providers/secrets_store/`
- `intergrax/llm_adapters/registry/secrets.py`
- `intergrax/applications/_shared/security_runtime_bridge.py`

## Meaningful side effects

- `intergrax/runtime/policy/meaningful_side_effect.py`
- `intergrax/runtime/policy/runtime_policy_engine.py`
- `intergrax/contracts/meaningful_side_effect_policy.py`
- `intergrax/collaborative_work/enforcement_gate.py`
- `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`

## Platform Plugins

- `docs/project/architecture/PLATFORM_PLUGINS.md`
- `intergrax/core/plugins/`
- domain-specific discovery/admission/materialization paths

---

# 3. P0A documentation drift classification

| Area | Status | As-built finding | Required P0A action |
|---|---|---|---|
| `ExecutionId` | **DONE + DOC-DRIFT** | canonical Python identity exists and is bound in root/child execution | update stale CURRENT docs; do not reimplement |
| root `ExecutionRuntime` | **DONE** | canonical root lifecycle owner exists | preserve; audit adoption coverage |
| `ExecutionBoundary` | **DONE + DOC-DRIFT** | strategy-neutral coordination boundary exists | update stale CURRENT docs |
| `StrategyExecutionRouter` | **DONE/PARTIAL** | typed inference/agentic/orchestration routing exists | audit remaining bypasses, not recreate router |
| `RuntimeEvent.execution_id` | **DONE + DOC-DRIFT** | field is canonical and required | update stale CURRENT docs |
| child execution lineage | **DONE/PARTIAL** | `ChildExecutionRunner` mints child IDs under active parent | audit all delegated/background/external paths |
| child authority | **DONE/PARTIAL** | child policy resolution and active authority binding exist | prove monotonicity on all paths |
| child budget | **DONE/PARTIAL** | grant/reservation/release primitives exist | audit remaining dimensions and recovery |
| Execution-tree checkpoint | **DONE/PARTIAL + DOC-DRIFT** | checkpoint v2 contains `ExecutionTreeSnapshot` | update stale checkpoint/UER CURRENT text |
| Execution-tree resume | **DONE/PARTIAL** | historical/active resume plan exists | audit provider/background/distributed recovery |
| ContextProvider seam | **DONE/PARTIAL** | protocol/context/providers exist | harden lifecycle/provenance/lazy semantics |
| Platform Plugins | **DONE for closed program / PARTIAL for new scoped runtime lifecycle** | package coordination program is closed | do not reopen closed Plugin program |
| Skills core | **DONE/PARTIAL** | mature resolver/catalog/plugin stack | close version/provenance/authority gaps |
| ToolRuntime core | **DONE/PARTIAL** | canonical ToolRuntime and policy layers exist | close monotonic authority bug and timeout/retry gaps |
| sandbox core | **DONE/PARTIAL** | runtime/session/manager/host wiring exist | converge execution environment and fail-closed semantics |
| credentials/secrets | **DONE/PARTIAL** | `SecretsStore` provider seam exists | add CredentialRef/late-resolution execution semantics |
| canonical runtime inspection | **GAP/PARTIAL** | diagnostics/read models exist in domains, no single canonical inspector | build projection only; no new truth authority |
| runtime invariant runner | **GAP** | invariants exist in docs/tests/domain validation but no shared runtime runner found | add domain-owned checks + shared runner |
| canonical ModelRequest evidence | **GAP** | LLM events exist; no canonical `ModelRequestId` evidence object found | create evidence identity under existing owners |
| outbound Human Continuation seam | **GAP/PARTIAL** | HITL and inbound InteractionAdapter exist; generic outbound provider-neutral continuation seam is not canonical | converge without overloading inbound adapter |

---

# 4. P0B — Safety and authority closure classification

## P0-SAFETY-1 — Tool Authority Intersection Integrity

**Status: GAP / BLOCKER**

### Current behavior

`intergrax/runtime/policy/tool_policy_resolution.py` currently returns `explicit` immediately when an explicit caller allow-list is provided.

Conceptually current behavior can become:

```text
RuntimePolicyBundle.tool_access = {A, B}
caller explicit                 = {A, B, C}

current effective = {A, B, C}
```

This violates the canonical monotonic authority invariant.

### Required target

```text
effective =
    host availability
    ∩ agent/skill requirement
    ∩ RuntimePolicyBundle
    ∩ modality/plan narrowing
    ∩ invoker/per-call narrowing
```

No downstream caller list may widen a stricter upstream list.

### Classification

- contract/tool architecture: **DONE**
- canonical invariant: **DONE in architecture**
- implementation enforcement: **GAP**
- regression/conformance proof: **GAP**

### P0B acceptance

A regression test must prove that an explicit caller allow-list can only narrow, never widen, `RuntimePolicyBundle.tool_access` or stricter upstream authority.

---

## P0-SAFETY-2 — Skill Authority Integrity

**Status: GAP / BLOCKER**

### Current behavior

`extend_tool_profile_for_skills()` expands host `ToolProfile.enabled` using Skill-declared tool requirements.

This conflates:

```text
Skill requires capability
```

with:

```text
host authorizes capability
```

### Required target

```text
Skill requires Tool X
        ↓
Is X already host-available?
  yes → resolution may continue
  no  → fail/degrade according to explicit environment rules
```

A Skill must never silently grant host availability.

### Classification

- Skills catalog/resolver: **DONE**
- Skill → Tool requirement modeling: **DONE**
- host availability authority separation: **GAP**
- version identity/provenance retention: **PARTIAL/GAP**
- prompt/policy bridge universal adoption: **PARTIAL**

### P0B acceptance

Skill resolution must not mutate or widen host ToolProfile authority. Missing required host capabilities fail with actionable diagnostics or enter an explicitly declared degraded path.

---

## Child authority monotonicity

**Status: PARTIAL**

`ChildExecutionRunner` resolves child authority from active parent authority through an `ExecutionAuthorityPolicy` and binds the result for child execution.

Foundation is real and must be preserved.

Remaining work:

- prove all child creation paths use the same authority path,
- prove external/delegated/background paths cannot bypass it,
- test requested child scopes larger than parent,
- test nested descendants,
- inspect effective delegation decisions.

---

## Meaningful-side-effect fresh authorization

**Status: DONE/PARTIAL**

Intergrax has:

- `MeaningfulSideEffectRequest`,
- `RuntimePolicyEngine.evaluate_meaningful_side_effect`,
- policy rules/actions,
- enforcement reuse in collaborative/reliability paths,
- HITL continuation concepts.

Do not invent a new policy engine.

Remaining audit/closure:

- enumerate all meaningful-effect production paths,
- prove fresh evaluation immediately before effect,
- prove approval grant matching/consumption,
- prove no host/tool shortcut bypass,
- connect recovery/redelivery semantics to fences/idempotency.

---

## Required causal evidence before meaningful work

**Status: PARTIAL**

The principle exists in background/transport and other durability paths, but it is not yet proven as a single cross-domain conformance invariant.

Required closure:

- identify boundaries where durable evidence is mandatory before work,
- fail closed on persistence failure at those boundaries,
- provide shared conformance tests without moving persistence ownership into UER.

---

## Credential resolution/exposure boundary

**Status: PARTIAL**

Existing foundation:

- tenant-scoped `SecretsStore`,
- provider implementations,
- LLM/tool integration consumption,
- security runtime bridges.

Missing canonical runtime semantics:

- stable `CredentialRef`,
- late/per-operation resolution,
- execution-scoped exposure,
- credential-use evidence without secret leakage,
- human credential authorization/acquisition flow,
- rotation semantics across in-flight vs future work.

Do not create a second secret store.

---

## Sandbox/isolation fail-closed boundary

**Status: PARTIAL**

Existing foundation:

- runtime sandbox models/session/manager,
- host sandbox integration,
- application sandbox wiring,
- CodeCraft isolation resolution,
- sandbox tools/skills.

Remaining closure:

- one effective execution-environment projection,
- explicit filesystem/network/process/credential restrictions,
- fail-closed behavior for required isolation,
- escalation semantics through Governance/Human Interaction,
- provider qualification and inspection,
- adversarial conformance by declared threat model.

Do not rebuild existing sandbox runtime.

---

## Retry/redelivery authorization semantics

**Status: PARTIAL**

Intergrax has multiple retry layers and background redelivery semantics.

Remaining closure:

- provider/tool internal retry vs execution retry vs transport redelivery vs whole-Run retry,
- same authorization/fence semantics across redelivery,
- no accidental reauthorization broadening,
- inspectable retry generation/relationship.

---

## Side-effect idempotency/fence integration

**Status: DONE/PARTIAL**

Existing idempotency/fencing primitives exist in persistence/concurrency and meaningful-effect domains.

Required work is **integration and ownership clarification**, not a new exactly-once runtime.

Hard distinction:

```text
at-most-once authorization
!= exactly-once external effect
```

---

# 5. P0C — Execution and durability convergence classification

| P0C item | Status | As-built result | Remaining work |
|---|---|---|---|
| remaining UER entry-path adoption | **PARTIAL** | root runtime/boundary/router and several host/system paths exist | find and remove/contain bypasses |
| Execution Tree queries/cancellation | **PARTIAL** | tree identity/checkpoint model exists | canonical query/read model + subtree cancellation across supported providers |
| pause/resume/cancel | **PARTIAL** | lifecycle/HITL/checkpoint primitives exist | provider-neutral cross-path convergence |
| budget dimensions/child accounting | **PARTIAL** | ledger + child grants/reservations exist | audit tokens/cost/tool-count/concurrency/wall-clock/step limits and restore semantics |
| retry ownership taxonomy | **PARTIAL** | multiple retry layers are known | make ownership explicit and converged |
| durability/recovery | **PARTIAL** | checkpoint tree/resume substantial | close transport/fence/budget/credential/delegated-child recovery |
| background/transport runtime identity | **PARTIAL** | background architecture and execution evidence exist | ensure all admitted work uses final execution identity semantics |
| delegated/external provider child semantics | **PARTIAL/GAP** | internal child runner exists | provider-neutral external child seam and recovery/interrupt semantics |
| effective profile revision bound to execution | **GAP/PARTIAL** | environment authority exists; some revision concepts exist in adjacent domains | canonical ProfileResolution snapshot/fingerprint per Execution |
| ModelRequest evidence foundation | **GAP** | `LLM_CALL` events are not equivalent to reconstructable request identity | introduce canonical request evidence identity/provenance |
| Human Continuation foundation | **GAP/PARTIAL** | HITL exists, inbound InteractionAdapter exists | provider-neutral outbound request/response seam |
| Runtime Inspection foundation | **GAP/PARTIAL** | domain diagnostics/projections exist | one shared read-model API over facts |
| Runtime Invariant runner | **GAP** | domain checks/tests exist | shared runner + domain-owned invariant definitions |

---

# 6. P1 — Composition, inspection, credentials, sandbox, health classification

## P1.1 Full ProfileResolution layering and provenance

**Status: CLOSED (P1.1)**

`ApplicationEnvironmentProfile` remains the sole Tier-3 composition authority.

Delivered in P1.1:

- typed `ProfileLayer` / `ProfileDelta` / `ProfileLayerInput` contracts,
- `resolve_profile(...)` → immutable `ProfileResolution` evidence,
- provenance decisions (`APPLIED` / `REJECTED` / `CLAMPED` / `UNCHANGED`),
- domain-owned field resolvers (`capabilities.tools`, `capabilities.llm`, `meta.execution_mode`, `governance.cost`),
- deterministic effective semantic fingerprint,
- canonical harness host adoption via `build_harness_host_runtime(profile_layers=...)`.

Deferred to P1.2+:

- full semantic diff UI,
- capability dependency graph population,
- broader field resolver coverage beyond initial domain seams.

Do not introduce `HarnessProfile`.

---

## P1.2 Effective profile diff/versioning

**Status: CLOSED (P1.2)**

Delivered in P1.2:

- typed `EffectiveProfileRevisionId` / `EffectiveProfileRevision` immutable snapshot contract,
- `materialize_effective_profile_revision(...)` from `ProfileResolution` evidence,
- domain-aware `EffectiveProfileDiff` (`meta.execution_mode`, `capabilities.llm`, `capabilities.tools`, `governance.cost`),
- append-only `EffectiveProfileRevisionStore` with in-memory adapter,
- execution pinning evidence (`EffectiveProfileExecutionBinding`) with checkpoint/resume/child inheritance,
- harness host adoption via `build_harness_host_runtime(revision_store=...)`.

Deferred to P1.3+:

- capability dependency graph population,
- broader field diff coverage beyond initial domain seams,
- Runtime Inspection API (P1.4).

---

## P1.3 Capability dependency validation

**Status: CLOSED (P1.3 + P1.3A)**

Delivered in P1.3:

- typed `CapabilityDependency` declarations with `REQUIRED` / `OPTIONAL` semantics,
- provider-neutral `CapabilityDependencyValidator` with domain-owned `CapabilityDependencyProvider` seam,
- Skill → Tool adoption via `SkillToolCapabilityDependencyProvider` (reuses ToolProfile effective availability),
- composition-time fail-closed gate in `environment_wiring`,
- `ProfileResolution.dependency_failures` / `degraded_capabilities` evidence population,
- deterministic dedup with `REQUIRED` dominance over `OPTIONAL`.

P1.3A correction:

- explicit `provider_id` routing identity separate from `source_domains` provenance,
- evaluate-before-merge with deterministic evaluation merge (`UNAVAILABLE` / `UNKNOWN` dominate `AVAILABLE`),
- duplicate `provider_id` fails closed via `CapabilityDependencyProviderConflictError`,
- provider registration order independence for merged semantic results.

Deferred to P1.4+:

- Runtime Inspection / explain API (P1.4 — CLOSED),
- cross-domain operational readiness projection (P1.5 — CLOSED),
- Integration → Provider / Credential chain adoption until domain contracts mature.

---

## P1.4 Runtime Inspection / explain

**Status: CLOSED (P1.4 + P1.4A)**

Delivered in P1.4:

- canonical read-only `RuntimeInspectionService` aggregator in `intergrax/applications/_shared/runtime_inspection/`,
- typed contracts in `intergrax/applications/contracts/runtime_inspection/`,
- explicit immutable `RuntimeInspectionProvider` extension seam (no global registry),
- profile inspection reuses existing `ProfileResolution.decisions` evidence (no precedence recompute),
- revision inspection/compare reuse `EffectiveProfileRevisionStore` and `diff_effective_profile_revisions`,
- execution inspection resolves exact pinned revision via `EffectiveProfileExecutionPinningStore` (no latest fallback),
- capability inspection reuses `CapabilityDependencyValidationResult` evidence and projects effective health via P1.5 (`CapabilityInspectionResult.health`),
- typed `InspectionCompleteness` / `InspectionInconsistency` / `InspectionExplanation` read models,
- deterministic ordering, partial provider-failure visibility, and profile redaction helpers.

Delivered in P1.4A (serialization safety correction):

- safe-by-construction inspection result serialization via `Field(exclude=True)` on canonical runtime objects plus typed safe projections (`SafeProfileResolutionView`, `SafeEffectiveProfileRevisionView`, `SafeEffectiveProfileDiffView`),
- reuse of `encode_provenance_value` / `redacted_profile_snapshot` as sole redaction authority (no second detector),
- sanitized provider failure reasons and defensive extension-evidence payload redaction,
- direct `model_dump` / `model_dump_json` proof tests for profile, revision, execution, compare, provider failure, and extension payload paths.

Deferred to post-P1.5:

- REST/CLI/dashboard operator surfaces,
- HOS execution-tree inspection adoption,
- live probes / background health monitors.

Add a read model only. It must not own runtime truth.

---

## P1.5 Effective capability health/readiness

**Status: CLOSED (P1.5 + P1.5A)**

Delivered in P1.5:

- canonical `CapabilityHealthStatus` (`READY` / `DEGRADED` / `UNAVAILABLE`) operational projection,
- typed `CapabilityHealthFact` / `CapabilityHealthReason` / `EffectiveCapabilityHealth` contracts in `intergrax/applications/contracts/capability_health/`,
- provider-neutral `CapabilityHealthProvider` seam and `EffectiveCapabilityHealthProjector` in `intergrax/applications/_shared/capability_health/`,
- real P1.3 adoption via `DependencyValidationHealthProvider` (reuses `CapabilityDependencyValidationResult` — no second validator),
- real Tool effective availability adoption via `ToolEffectiveAvailabilityHealthProvider` (reuses `available_tool_ids_for_profile`),
- Runtime Inspection integration on `inspect_capability(...)` (`health` + `safe_health` on `CapabilityInspectionResult`),
- deterministic fact merge/dominance, duplicate `provider_id` fail-fast (`CapabilityHealthProviderConflictError`),
- conservative provider-failure facts, tenant scope isolation, P1.4A-safe health serialization.

P1.5A correction (fail-closed missing evidence):

- no applicable canonical readiness evidence → `UNAVAILABLE` (never synthetic `READY`),
- typed missing-evidence reason `capability.health.evidence_missing` via projection-owned `READINESS_EVIDENCE` fact,
- decision taken after scope and capability filtering; `READY` requires at least one applicable positive canonical fact.

Explicitly not adopted (honest boundary):

- provider live operational health (`integrations.contracts.HealthStatus` exists; not wired into capability projection),
- credential operational health (deferred to P1.7),
- integration binding health facts,
- background monitors / polling / automatic fallback.

Operational read model only — does not grant capability, activate providers, or change governance.

---

## P1.6 Atomic activation lifecycle

**Status: CLOSED**

Canonical scoped active revision pointer with CAS semantics:

```text
EffectiveProfileRevisionStore (immutable authority)
        ↓
ActiveEffectiveProfileRevisionStore (atomic pointer)
        ↓
EffectiveProfileActivationService (validation/orchestration)
        ↓
EffectiveProfileExecutionPinningDependencies (future admission read seam)
```

Delivered:

- `ActiveEffectiveProfileRevisionBinding` — immutable scoped pointer/read contract
- `ActiveEffectiveProfileRevisionStore.compare_and_set_active` — expected-current CAS, no hidden retry
- `EffectiveProfileActivationService` — candidate existence/scope validation, P1.3 eligibility reuse hook
- `InMemoryActiveEffectiveProfileRevisionStore` — thread-safe reference adapter
- `KvActiveEffectiveProfileRevisionStore` — durable-capable KV adapter (production path when KV wired)
- Host admission resolves active revision atomically; execution pinning unchanged (P1.2)
- Runtime inspection `inspect_active_revision(scope)` — read-only active exposure

Durability statement:

```text
production durable activation store: PARTIAL — KvActiveEffectiveProfileRevisionStore when DistributedKVStore wired; in-memory default for lab/harness
```

Preserves INV-25 (atomic activation), INV-26 (in-flight pinning), INV-33 (revision immutability).

**Next: P1.7 — CredentialRef / late resolution**

---

## P1.7 CredentialRef / late resolution

**Status: GAP on canonical runtime contract, CURRENT/PARTIAL on secret storage**

`SecretsStore` already exists. `CredentialRef`-style per-operation late resolution/execution-scoped exposure was not found as one canonical runtime contract.

Add around existing secret/integration ownership.

---

## P1.8 Sandbox / ExecutionEnvironment convergence

**Status: PARTIAL**

Significant sandbox runtime exists. Remaining concern is convergence around execution-scoped effective environment, policy, provider inspection, credentials/network/process restrictions, and escalation.

Do not rebuild sandbox session/manager/provider infrastructure.

---

## P1.9 Context provider lifecycle/provenance hardening

**Status: CURRENT/PARTIAL**

ContextProvider contracts and providers already exist.

Remaining:

- provider registration/lifecycle convergence,
- fragment lifetime/replacement semantics,
- lazy activation,
- workspace instruction refresh semantics,
- exact provenance/version binding,
- cache/cost attribution,
- universal CE admission for all model-visible context.

---

## P1.10 Skill version/provenance bridge hardening

**Status: CURRENT/PARTIAL**

Skills are mature. Remaining gaps are specific:

- declared vs resolved version identity,
- retain canonical `ResolvedSkillPack` provenance,
- non-expanding host ToolProfile authority,
- prompt bridge production adoption,
- policy bridge production adoption.

Do not redesign the Skill system.

---

## P1.11 Governance permission presets

**Status: PARTIAL**

Governance, HITL, RuntimePolicyBundle, ToolRuntime and sandbox policy surfaces exist.

Need one ergonomic preset UX that expands only into canonical policies and can never grant authority beyond upstream configuration.

Presets are configuration shorthand, not a policy engine.

---

# 7. Explicit DO NOT DO list resulting from P0A

1. **Do not implement `ExecutionId` again.**
2. **Do not create a second `ExecutionBoundary`.**
3. **Do not create a second strategy router.**
4. **Do not add a parallel Execution Tree authority.**
5. **Do not build a second checkpoint tree model.**
6. **Do not create a new Job/background lifecycle beside UER + Background Tasks.**
7. **Do not create a second ContextProvider architecture.**
8. **Do not reopen Platform Plugins as a universal runtime semantics layer.**
9. **Do not redesign Skills; fix bounded gaps.**
10. **Do not replace `SecretsStore`; add runtime credential-reference semantics around it.**
11. **Do not replace sandbox runtime; converge effective ExecutionEnvironment semantics around it.**
12. **Do not create a new meaningful-side-effect policy engine.**
13. **Do not make runtime inspection or health/readiness a source of truth.**
14. **Do not conflate inbound `InteractionAdapter` with outbound human continuation interaction.**
15. **Do not claim exactly-once external effects merely because approval/idempotency/fencing exists.**

---

# 8. Canonical documentation drift to reconcile before code-changing sessions

## Highest priority

### `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`

Known stale CURRENT claims include descriptions that canonical `ExecutionId`, neutral Execution Boundary, and `RuntimeEvent.execution_id` are not yet implemented.

**Required correction:** mark shipped execution identity/boundary/event spine as CURRENT and narrow TARGET to remaining adoption/convergence work.

### UER satellites / maintainers plan

Review CURRENT/target language for:

- root execution,
- child execution,
- strategy routing,
- budget hierarchy,
- checkpoint tree,
- resume behavior.

### `docs/project/architecture/BACKGROUND_TASKS.md`

Reconcile older maturity statements with current TaskRegistry/WorkerRuntime/background proof state and final Execution identity requirements.

### `docs/project/architecture/OBSERVABILITY.md`

Ensure canonical execution-scoped event identity reflects required `execution_id` on current `RuntimeEvent`.

## Already substantially aligned

- `docs/project/architecture/SKILLS.md` already explicitly records ToolProfile expansion and version/provenance gaps.
- `docs/project/architecture/TOOLS.md` already states monotonic-intersection target invariant, although code still violates it.
- `docs/project/architecture/PLATFORM_PLUGINS.md` already records the closed coordination program and its boundaries.
- `docs/project/architecture/CONTEXT_ENGINEERING.md` already recognizes ContextProvider ownership and UCL/CE separation.

---

# 9. P0A outcome by roadmap phase

## P0B blockers entering implementation

### BLOCKER 1 — Tool authority

`resolve_allowed_tools_from_config()` must intersect caller narrowing with stricter upstream policy instead of allowing caller replacement.

### BLOCKER 2 — Skill authority

Skill requirements must not widen host ToolProfile availability.

These are the first code-changing tasks after P0A documentation synchronization.

## P0C major remaining work

P0C is a **convergence and proof program**, not a greenfield UER build.

Highest-value remaining areas:

1. remaining entry-path adoption/bypass removal,
2. canonical subtree cancellation/query projection,
3. provider-neutral pause/resume/cancel,
4. remaining budget dimensions and recovery,
5. retry taxonomy convergence,
6. durability across transport/background/external children,
7. effective profile revision binding,
8. ModelRequest evidence,
9. Human Continuation seam,
10. Runtime Inspection foundation,
11. Runtime Invariant runner.

---

# 10. P0A exit-gate assessment

| Exit condition | Result |
|---|---|
| starting SHA frozen | **PASS** |
| P0/P1 initiatives classified | **PASS** |
| concrete code/docs evidence paths recorded | **PASS** |
| already-shipped UER foundations separated from GAP | **PASS** |
| Tool authority gap identified precisely | **PASS** |
| Skill authority gap identified precisely | **PASS** |
| ContextProvider current state reconciled | **PASS** |
| Platform Plugins closed-program boundary reconciled | **PASS** |
| checkpoint current state reconciled | **PASS** |
| canonical stale CURRENT docs updated | **OPEN** |
| no implementation session can be misled by known stale docs | **OPEN until doc sync** |

## P0A verdict

**AUDIT COMPLETE — DOCUMENTATION SYNC REQUIRED BEFORE P0B CODE CHANGES.**

No feature code should be changed under this audit commit.

The next action is a bounded documentation synchronization of the canonical CURRENT sections listed in §8. After that sync, P0A can be marked CLOSED and P0B can start with the two authority blockers.
