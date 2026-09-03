# ADR-PLATFORM-SE-CONVERGENCE: Canonical Meaningful Side-Effect Authorization Model

| Field | Value |
|-------|-------|
| **Status** | **Accepted (architecture only)** — implementation pending |
| **Date** | 2026-09-03 |
| **Audit** | PLATFORM-ENTERPRISE-AUDIT-1 · FINDING-PLATFORM-SE-001 |
| **Baseline ancestor** | `b4b79779e1bd25125d93bb6462456337e8cad94c` |
| **Related ADRs** | ADR-POLICY-SIDE-EFFECT-001 · ADR-PLATFORM-PLUGIN-001 · ADR-GOVERNED-CONTINUATION-001 · ADR-MP-001 |
| **Next implementation** | PLATFORM-SE-FAIL-CLOSED-1 |

---

## 1. Status

**Accepted (architecture only).** This ADR defines the canonical platform rule for meaningful side-effect authorization convergence. **No production behavior changes** are included in this decision record. **P1 FINDING-PLATFORM-SE-001 remains OPEN** until PLATFORM-SE-FAIL-CLOSED-1 is implemented and verified.

---

## 2. Context

Intergrax hosts multiple legitimate execution surfaces that can mutate externally relevant or authoritative business/system state:

- **Generic tool execution** (`RuntimeToolInvoker`, currently implemented within Execution Engine/Nexus internals at `intergrax/runtime/nexus/tools/invoker.py`) — generic agent/tool operations governed by `ToolContract`, optional `ToolScopePolicy`, optional `DeclarativePolicyEnforcer`, optional declarative HITL, and optional idempotency coordination. **Nexus is an internal execution-engine mechanism; it is not the owner of platform governance, authorization strategy, or the public authorization domain.** Implementation location does not imply architectural ownership.
- **Collaborative Work / External Work** — workspace/tenant/principal/resource-scoped mutations governed by `MeaningfulSideEffectAuthorizationBoundary` → `CollaborativeWorkEnforcementGate`.
- **Control-plane mutations** — bundle-backed policy evaluation via `control_plane_mutation_authorization`.
- **Provider-backed writes** — mechanics owned by Integrations; authorization must occur above the provider boundary.
- **Autonomous Work** — lifecycle persistence emerging; real mutations must adopt a canonical strategy when introduced.

Both runtime declarative tool authorization (`DECLARATIVE_TOOL_AUTHORIZATION`, owned by platform/runtime authorization and governance) and Collaborative Work authorization (`COLLABORATIVE_WORK_AUTHORIZATION`) are mature, domain-appropriate mechanisms. The platform gap is not their coexistence but the absence of a **single invariant** guaranteeing that no meaningful side effect executes without an explicit, recognized authorization mechanism.

Prior art: ADR-POLICY-SIDE-EFFECT-001 established that meaningful external side effects require policy before execution for governed external work. This ADR generalizes and converges the **platform-level** rule across all mutation surfaces without merging domain policy semantics.

---

## 3. Problem

**FINDING-PLATFORM-SE-001:** `RuntimeToolInvoker._prepare_invocation` calls `resolve_declarative_policy_enforcer(state)`, which returns `None` when `state.context.config.policy_bundle` has no `declarative_policy_runtime`. `ToolScopePolicy` is also optional. When both are absent, tools with `ToolContract.side_effects=True` proceed to `ToolExecutor` without any recognized authorization mechanism.

Verified current path:

```text
ToolContract
  → side_effects flag
  → RuntimeToolInvoker._prepare_invocation
      → optional ToolScopePolicy (capability boundary, not authorization strategy)
      → optional DeclarativePolicyEnforcer (via resolve_declarative_policy_enforcer)
      → optional declarative HITL (REQUIRE_HITL when enforcer present and enforced)
  → optional IdempotencyPreEffectCoordinator (when side_effects=True AND idempotency_key present)
  → ToolExecutor
```

**Critical fact:** absence of `DeclarativePolicyEnforcer` does not block execution today. Tests confirm side-effecting tools execute without policy wiring (`test_side_effect_tool_without_retry_proof_executes_once` in `test_tools_side_effect_safety.py`).

Enterprise hosts require: **no authorization path ⇒ no meaningful side effect**.

---

## 4. Existing architectures

### 4.1 Declarative Tool Authorization (Strategy A)

**Canonical strategy name:** `DECLARATIVE_TOOL_AUTHORIZATION`

**Owned by:** platform/runtime authorization and governance (consumed by the generic execution/tool runtime).

**For:** generic tool-executing runtime contexts that do not require Collaborative Work workspace/principal/delegation semantics.

**Execution integration point:** `RuntimeToolInvoker` (currently under `intergrax/runtime/nexus/tools/invoker.py` within Execution Engine/Nexus internals).

| Component | Role |
|-----------|------|
| `ToolContract.side_effects` | Declares whether tool may perform meaningful external mutation |
| `ToolScopePolicy` | Capability allowlist — which tools an agent may invoke; **not** a substitute for side-effect authorization |
| `DeclarativePolicyEnforcer` | Evaluates declarative rules; returns ALLOW / DENY / REQUIRE_HITL |
| `PolicyEnforcementMode` | `ENFORCE` vs `AUDIT_ONLY` — see §20 |
| Declarative HITL bridge | `DeclarativeHitlApprovalGrant` scoped to invocation; canonical `runtime/human` owns pause/grant |
| `IdempotencyPreEffectCoordinator` | Claim/replay before external effect when idempotency key supplied |

Resolution: `resolve_declarative_policy_enforcer(state)` reads `policy_bundle.declarative_policy_runtime`; returns `None` if unwired.

### 4.2 Collaborative Work authorization (Strategy B)

| Component | Role |
|-----------|------|
| `MeaningfulSideEffectRequest` | Typed proposal with action, kinds, `side_effect_scope_id`, identity |
| `CollaborativeWorkEnforcementRequest` | Adds tenant, workspace, principal, membership, delegation, resource scope |
| `CollaborativeWorkEnforcementGate` | Authority resolver + collaborative policy + runtime policy composition |
| `MeaningfulSideEffectAuthorizationBoundary` | Fresh `evaluate()` before execute; HITL via governed continuation |
| `GovernedContinuationGrantCoordinator` | Grant match/consume; stale grant rejection |

Rich semantics: tenant, workspace, principal, membership, delegation, resource scope, authority profile. **These must not be imposed on every generic tool.**

### 4.3 Other mutation surfaces (inventory)

| Surface | Typical classification | Current authorization |
|---------|------------------------|----------------------|
| Generic tool (`side_effects=False`) | READ_ONLY or INTERNAL_STATE_MUTATION | Scope policy only; no side-effect authorization required |
| Generic tool (`side_effects=True`) | MEANINGFUL_SIDE_EFFECT | Optional declarative enforcer — **generic tool execution authorization gap when absent** |
| External Work adapter | MEANINGFUL_SIDE_EFFECT | `MeaningfulSideEffectAuthorizationBoundary` when wired |
| CW API / repository writes | MEANINGFUL_SIDE_EFFECT or INTERNAL_STATE_MUTATION | Caller authority + enforcement gate for governed operations |
| Control-plane mutations | MEANINGFUL_SIDE_EFFECT | `control_plane_mutation_authorization` (bundle-backed) |
| Task/checkpoint/HITL state | INTERNAL_STATE_MUTATION | Runtime lifecycle; not externally authoritative business state |
| Trace / diagnostics / evidence append | INTERNAL_STATE_MUTATION or observability | Not authorization mechanisms |
| Provider integration execute | MEANINGFUL_SIDE_EFFECT (when mutating) | Authorization must occur in host/runtime **before** provider call |
| Autonomous Work repos (current) | INTERNAL_STATE_MUTATION | In-memory lifecycle; no external mutation auth path yet |
| Qualification / proof execution | N/A (not authorization) | Qualification ≠ permission |

**Boundary rule:** not every database write is a meaningful side effect. Meaningful side effect means a mutation that changes **externally relevant or authoritative business/system state** and therefore requires explicit governance. Internal runtime bookkeeping, idempotency ledger updates, and observability emission are out of scope unless they themselves constitute the governed business effect.

---

## 5. Security finding

| ID | Summary |
|----|---------|
| **FINDING-PLATFORM-SE-001** | `side_effects=True` tools execute when declarative enforcer and CW boundary are both absent |
| **Impact** | Misconfigured host can permit unbounded external/DB mutation |
| **Root cause** | No platform invariant tying `ToolContract.side_effects` to mandatory authorization strategy presence |
| **ADR outcome** | Canonical invariant + staged fail-closed implementation (PLATFORM-SE-FAIL-CLOSED-1) |

---

## 6. Decision

The platform adopts a **multi-strategy, fail-closed** meaningful side-effect authorization model:

1. **One canonical invariant** (SE-INV-1…SE-INV-9) applies to all meaningful side effects.
2. **Two legitimate domain strategies** remain separate: Declarative Tool Authorization and Collaborative Work Authorization.
3. A **minimal shared coordination contract** (`MeaningfulSideEffectAuthorization` — concept only) will coordinate strategy presence and outcomes without becoming a third policy engine.
4. **Host/composition** selects strategy via typed DI; platform refuses meaningful effects when no strategy is active.
5. **Phase 1 implementation** closes the generic tool execution authorization gap (RuntimeToolInvoker fail-open path): `side_effects=True` without recognized strategy ⇒ DENY.

Collaborative Work is **not** mandatory for generic tools. Declarative Tool Authorization does **not** replace the CW boundary. Neither engine is merged.

### Architectural ownership

```text
Platform Governance / Runtime Authorization
        │
        ├── DECLARATIVE_TOOL_AUTHORIZATION
        │
        └── COLLABORATIVE_WORK_AUTHORIZATION
        │
        ▼
Authorization decision
        │
        ▼
Generic execution / tool runtime
        │
        ▼
Execution Engine
        │
        └── Nexus internal mechanics
        │
        ▼
Tool / domain / provider effect
```

Collaborative Work strategy may own richer domain authorization before execution; generic tool runtime uses declarative authorization. Collaborative Work is not forced through Nexus internals for authorization. Nexus remains below generic execution/runtime ownership as internal engine implementation.

---

## 7. Canonical invariants

| ID | Invariant |
|----|-----------|
| **SE-INV-1** | A meaningful side effect **MUST NOT** execute unless an explicit, recognized authorization mechanism is active for that execution. |
| **SE-INV-2** | Absence of authorization mechanism **MUST** fail closed. |
| **SE-INV-3** | Authorization mechanism is **domain-selectable**; bypassing all strategies is forbidden. |
| **SE-INV-4** | Authorization and execution are **separate**; authorization decision occurs before the effect boundary. |
| **SE-INV-5** | Observability/diagnostics failure **must not** grant authority. |
| **SE-INV-6** | Provider availability/fallback **must not** alter authorization semantics. |
| **SE-INV-7** | Authorization must be scoped sufficiently to the **actual effect**. |
| **SE-INV-8** | Approval/grant **must not** become transferable authority. |
| **SE-INV-9** | Meaningful side effects requiring retry/replay **must** preserve idempotency semantics. |

**Canonical invariant (exact wording):**

> **No authorization path ⇒ no meaningful side effect.**

---

## 8. Meaningful side-effect definition

A **meaningful side effect** is a proposed or performed mutation that:

- changes externally relevant or authoritative business or system state, **or**
- creates commitments, disclosures, or irreversible external consequences,

such that explicit governance is required before execution.

**Operational signals in current release:**

| Signal | Meaning |
|--------|---------|
| `ToolContract.side_effects == True` | Generic tool execution proposes meaningful side effect |
| `MeaningfulSideEffectRequest` | CW / runtime-policy evaluation input |
| `ControlPlaneMutationRequest` | Control-plane meaningful mutation |
| External Work domain actions | Mapped to `MeaningfulSideEffectRequest` at boundary |

**Not automatically meaningful:** runtime task state transitions, trace events, in-process caches, idempotency ledger bookkeeping, qualification record writes, diagnostic evidence capture.

**`ToolContract.side_effects` sufficiency (§12):** the boolean is **sufficient for current release** (Outcome A). Hosts and tool authors must set `side_effects=False` for genuinely read-only or purely internal tools. Future typed effect classification (e.g. commitment vs disclosure granularity) is **P2** and not required for Phase 1 fail-closed closure.

---

## 9. Domain authorization strategies

### Strategy A — `DECLARATIVE_TOOL_AUTHORIZATION`

**Owned by:** platform/runtime authorization and governance.

**For:** generic tool execution; agent/tool runtime contexts; non-workspace-scoped operations.

**Uses:** `DeclarativePolicyEnforcer`, `ToolScopePolicy` (capability, supplementary), canonical `runtime/human` HITL, idempotency coordinator.

**Execution integration:** `RuntimeToolInvoker` (currently implemented within Execution Engine/Nexus internals).

**Does not use:** workspace membership, delegation, CW resource profiles.

### Strategy B — `COLLABORATIVE_WORK_AUTHORIZATION`

**Owned by:** platform/runtime authorization and governance.

**For:** workspace/tenant/principal/resource-scoped mutations; External Work when collaboration semantics apply.

**Uses:** `CollaborativeWorkEnforcementGate`, `MeaningfulSideEffectAuthorizationBoundary`, `GovernedContinuationGrantCoordinator`.

**Does not use:** tool rule handlers as substitute for authority/delegation semantics.

Domain semantics **must not be merged**.

---

## 10. Shared platform contract (concept only — not implemented)

**Name:** `MeaningfulSideEffectAuthorization` (or `SideEffectAuthorizationStrategy`).

**Responsibilities:**

- declare whether an authorization mechanism is **configured** for the execution context;
- **authorize** a specific proposed effect;
- return typed outcome (`ALLOW` / `DENY` / `REQUIRE_HUMAN`);
- provide provenance/scope metadata for observability.

**Must NOT own:** provider execution, tool execution, workspace membership, policy storage, HITL storage, idempotency persistence.

This is an **authorization boundary contract**, not a new governance engine.

**Conceptual interface:**

```text
Protocol MeaningfulSideEffectAuthorizer:
    is_configured() -> bool
    authorize(request) -> SideEffectAuthorizationDecision
```

**Minimal request fields (shared denominator):** `tenant_id` (optional), principal/agent identity, `task_id`, `run_id`, execution/invocation id, effect/action identity, resource/effect scope, idempotency key, proposal digest — composed via domain adapters, not a universal context bag.

**Domain adapters (future):**

| Adapter | Maps | Delegates to |
|---------|------|--------------|
| `DeclarativeToolAuthorizationAdapter` | `ToolExecutionRequest` / `RuntimeState` | `DeclarativePolicyEnforcer` |
| `CollaborativeWorkAuthorizationAdapter` | `CollaborativeWorkEnforcementRequest` | `MeaningfulSideEffectAuthorizationBoundary` |

Adapters **wrap** existing mechanisms; they do not duplicate `RuntimePolicyEngine`, `CollaborativePolicyEvaluator`, `DeclarativePolicyEnforcer`, or `CollaborativeWorkAuthorityResolver`.

**Dependency direction:** shared contract lives in neutral `intergrax/contracts` or `intergrax/runtime` layer. Collaborative Work Authorization and Declarative Tool Authorization adapters depend on the neutral platform authorization contract; **not on each other**. Generic execution/tool runtime consumes the authorization decision. Nexus remains below generic execution/runtime ownership as internal engine implementation.

---

## 11. Strategy selection

**Rule:** host/composition **explicitly binds** the authorization strategy appropriate to the execution context via **typed DI/composition**.

| Execution context | Strategy |
|-------------------|----------|
| Generic tool-executing host/runtime | `DECLARATIVE_TOOL_AUTHORIZATION` |
| Workspace-aware External Work | `COLLABORATIVE_WORK_AUTHORIZATION` |
| Future Autonomous Work mutations | One of the canonical strategies — no bespoke engine |
| Control-plane API mutations | Bundle-backed runtime policy (existing path) |

**Forbidden selectors:** `if provider_id`, `if application_name`, tool name prefixes, filesystem location, reflection, service locator, global singleton.

**Example (conceptual):**

```text
ApplicationEnvironmentWiring
    meaningful_side_effect_authorization: SideEffectAuthorizationStrategy
```

Host chooses **which** strategy; platform invariant decides **that** a strategy is mandatory.

---

## 12. Fail-closed semantics

| Condition | Result |
|-----------|--------|
| Meaningful side effect + **no** recognized strategy | **DENY** (fail closed) |
| Meaningful side effect + strategy configured + DENY | **DENY** |
| Meaningful side effect + strategy configured + REQUIRE_HUMAN | **REQUIRE_HUMAN** → canonical HITL |
| Read-only tool (`side_effects=False`) | No side-effect authorization required; normal scope/policy rules apply |
| Default when uncertain | **DENY** — never implicit ALLOW |

**Forbidden production APIs:** `allow_side_effects_without_policy`, `unsafe_mode`, `skip_authorization`, `trusted=True` escape hatches.

Test-only bypass may exist in isolated test infrastructure; production runtime must not expose casual bypass.

---

## 13. HITL ownership

Canonical HITL remains **`runtime/human`** (`HumanPauseCoordinator`, governed continuation bridge).

Both strategies may **request** canonical HITL. Domain strategy determines **why** approval is required. `runtime/human` owns pause/response/grant mechanics.

**Do not create:** `ToolHumanApprovalManager`, `CollaborativeHumanApprovalManager`.

Outcome vocabulary: **`ALLOW`**, **`DENY`**, **`REQUIRE_HUMAN`** — map domain outcomes explicitly; no new synonyms unless required by existing contracts.

---

## 14. Fresh authorization

Authorization **must be evaluated as close as practical to the effect boundary**.

Approval obtained earlier **cannot** automatically skip fresh authorization when:

- policy changed
- authority changed
- resource changed
- operation changed
- run/task changed
- scope changed
- proposal changed

Reuse governed continuation principles (`matches_current_requirement`, grant consume). Target invariant applies platform-wide; identical implementation timing may vary per surface until converged.

**Verified today:** CW `authorize_and_execute` re-evaluates before execute; declarative HITL grant scoped to invocation; stale grants rejected (`test_g5c2b2b_governed_side_effect_reauthorization.py`).

---

## 15. Grant scope

Approval/grant binds relevant identity:

- task / run / execution / invocation
- tool or action identity
- resource or effect scope
- policy provenance / matched rules
- proposal digest where meaningful

**Do not require** workspace identity for generic tools. **Do not weaken** existing CW scope matching.

**`side_effect_scope_id` ≠ `resource_scope`:** independent dimensions where both exist. Generic tools may have invocation/effect scope without CW resource semantics. Governed-continuation matching on `side_effect_scope_id` remains valid. Do not define equality between `side_effect_scope_id` and `resource_scope` (P2-006).

---

## 16. Idempotency / retry ordering

Authorization and idempotency are **separate concerns:**

| Concern | Question |
|---------|----------|
| Authorization | May this effect occur? |
| Idempotency | Has this semantic effect already been requested/executed? |

**Canonical execution order (target):**

```text
authorize → acquire/validate idempotency claim → perform effect
```

**Current RuntimeToolInvoker behavior** (`RuntimeToolInvoker.invoke`, currently within Execution Engine/Nexus internals):

1. `_prepare_invocation` — scope + declarative policy (authorization layer)
2. `IdempotencyPreEffectCoordinator.before_external_effect` — when `side_effects=True` AND `idempotency_key` present
3. `_execute_external_effect` → `ToolExecutor`

Idempotency coordination is **not** required for all side-effecting tools — only when idempotency key is supplied. Authorization must gate **all** meaningful side effects regardless of idempotency key presence.

**Retry safety:** no retry may bypass reauthorization or idempotency constraints. Side-effect tools honor `side_effect_retry_safety`; `NOT_RETRY_SAFE` limits to one attempt.

---

## 17. Provider boundary

Providers execute **mechanics**. They **do not** decide business authorization.

```text
host/runtime → authorization → domain/tool execution → integration/provider
```

**Not:** `provider → authorization decision`.

No provider-specific side-effect policy. Qualification, functional proof, and provider availability do **not** imply effect permission.

---

## 18. Pluginability

External tool plugins declare side-effect nature via `ToolContract.side_effects` without core edits.

External authorization strategies are extensible only if a real future need emerges. **No generic strategy plugin registry** for symmetry. Current strategies are platform-owned.

---

## 19. Host responsibility

| Responsibility | Owner |
|----------------|-------|
| **Which** strategy to wire | Host / composition root |
| **That** a strategy is mandatory for meaningful effects | Platform invariant |

**Bad:** host may or may not wire safety.  
**Target:** host chooses strategy; platform refuses meaningful effect when none present.

### Application compatibility (audit — no host changes in this ADR)

| Host | Classification | Future implementation impact |
|------|----------------|------------------------------|
| **LKW** (`local_workspace_application`) | MIXED — declarative tool policy + deployment policy ports; CW durable adoption pending | Phase 1: ensure policy bundle wired for side-effecting tools; future CW durable path |
| **legal** | DECLARATIVE_TOOL_AUTH | Phase 1: verify `policy_rules` + ENFORCE mode for production profile |
| **governed_contractor** | CW_AUTH (External Work) + DECLARATIVE_TOOL_AUTH (tools) | External Work keeps CW; tools need declarative enforcer or fail-closed |
| **intergrax_assistant** | DECLARATIVE_TOOL_AUTH | Phase 1 fail-closed if policy unwired |
| **research** | DECLARATIVE_TOOL_AUTH | Phase 1 fail-closed if policy unwired |
| **attestation_demo** | DECLARATIVE_TOOL_AUTH + observability | Phase 1; PoC may use lab profile |
| **Autonomous Work** (library) | NO_SIDE_EFFECTS (current) | Phase 3: bind canonical strategy before real mutations |

### Backward compatibility / test drift

| Class | Description | Migration |
|-------|-------------|-----------|
| **TEST_FIXTURE_DRIFT** | Unit tests invoking `side_effects=True` without policy to assert retry/idempotency behavior | Update fixtures to wire declarative enforcer or use `side_effects=False` where semantically correct |
| **INTENTIONAL_TRUSTED_INTERNAL** | Rare internal mutations via tools that are not externally meaningful | Reclassify `side_effects=False` or define narrow explicit internal strategy — no `trusted=True` bypass |
| **REAL_HOST_GAP** | Hosts with tool bundles but `policy_rules=None` / `declarative_policy_runtime=None` | Wire `PolicyRulesProfile` with ENFORCE mode; Phase 1 will fail closed until fixed |

`build_declarative_invoker_from_tool_wiring` currently constructs `RuntimeToolInvoker` without scope policy; policy arrives via `state.context.config.policy_bundle` at invoke time. Phase 1 must fail closed when bundle lacks enforcer for side-effecting tools.

---

## 20. Production vs lab / audit-only

| Mode | Meaningful side-effect authorization |
|------|--------------------------------------|
| **Production profile** (`ExecutionMode.STRICT` / enterprise) | Mandatory **ENFORCE** — `AUDIT_ONLY` does **not** satisfy SE-INV-1 |
| **Lab / test profile** | `AUDIT_ONLY` permitted only via explicit environment/profile classification |

**Decision:** `AUDIT_ONLY` means authorization mechanism is **present** but **not enforcing**. For enterprise production execution of meaningful side effects, `AUDIT_ONLY` **does not** satisfy the mandatory authorization invariant. Policy may be evaluated and logged; effect must not proceed on would-deny unless host is explicitly non-production/test profile.

Distinguish:

- **Mechanism absent** → fail closed (SE-INV-2)
- **Mechanism present, AUDIT_ONLY** → does not satisfy production meaningful-effect authorization (SE-INV-1 in production)

Use existing typed `PolicyEnforcementMode` and `ExecutionMode`; do not key off arbitrary environment strings.

---

## 21. External Work

External Work **correctly** uses `MeaningfulSideEffectAuthorizationBoundary` today. **Do not replace** with declarative tool policy.

External Work retains CW authorization because business semantics include authority, resource scope, tenant/workspace context, and delegation. Generic tool execution does not inherit full CW semantics.

---

## 22. Autonomous Work

Autonomous Work **MUST NOT** invent its own generic side-effect authorization engine.

When AW begins executing real mutations, it must bind one of the canonical authorization strategies (likely declarative for agent-internal tools, CW when workspace-scoped).

Current AW library lifecycle (in-memory persistence) may remain unchanged. No AW-specific policy engine.

---

## 23. Observability / evidence

Every meaningful side-effect authorization should eventually emit common observability facts:

| Field | Purpose |
|-------|---------|
| authorization_strategy | `DECLARATIVE_TOOL` / `COLLABORATIVE_WORK` / … |
| decision | ALLOW / DENY / REQUIRE_HUMAN |
| task_id / run_id / execution_id | Correlation |
| effect / tool / action identity | What was proposed |
| matched policy / rule provenance | Why |
| HITL requirement | If paused |
| deny reason | If denied |

No secrets. Collaborative Work Authorization and Declarative Tool Authorization need not emit identical domain payload schemas immediately; define canonical projection target in platform observability (Phase 4 optional).

**Authorization evidence** is distinguishable from execution result, provider qualification, functional qualification, and public proof. No universal "evidence" blob.

Functional qualification and proof execution are **not** authorization mechanisms.

---

## 24. Migration plan

### Phase 1 — SECURITY CLOSURE (PLATFORM-SE-FAIL-CLOSED-1)

- `RuntimeToolInvoker`: when `contract.side_effects=True` and no recognized authorization strategy active ⇒ **fail closed**.
- Initial recognized Declarative Tool Authorization path: `DeclarativePolicyEnforcer` with `ENFORCE` mode at `RuntimeToolInvoker`.
- No CW import into invoker.
- No implicit allow.

### Phase 2 — EXPLICIT TYPED STRATEGY

- Introduce minimal `MeaningfulSideEffectAuthorization` coordination contract if needed for explicit strategy selection and reuse.

### Phase 3 — ADOPTION

- Autonomous Work real mutations
- New application hosts
- New external mutation domains

### Phase 4 — OPTIONAL CONVERGENCE

- Common observability/evidence projection
- Do **not** merge policy engines

---

## 25. Compatibility impact

- **Production code:** zero in this ADR task.
- **Tests:** some unit tests assume side-effecting tools run without policy — classify as TEST_FIXTURE_DRIFT; fix during Phase 1, not by preserving insecure behavior.
- **Hosts:** any host with `side_effects=True` tools and `declarative_policy_runtime=None` will fail closed after Phase 1 until policy wired.
- **P1 FINDING-PLATFORM-SE-001:** remains **OPEN** until Phase 1 ships.

---

## 26. Non-goals

- Merging all policy systems into one engine
- Making CW mandatory for all tools
- Replacing `DeclarativePolicyEnforcer`
- Replacing `CollaborativeWorkEnforcementGate`
- Redesigning HITL (`runtime/human`)
- Redesigning `ToolContract` (beyond boolean sufficiency note)
- Redesigning Integrations
- Creating generic ABAC/RBAC engine
- Creating another provider/plugin registry
- `UnifiedPolicyEngine` / `GlobalGovernanceEngine` / `EnterpriseAuthorizationFramework`

---

## 27. Rejected alternatives

### A — All meaningful side effects through Collaborative Work

**Rejected.** CW workspace/principal/membership/delegation semantics are not applicable to every generic tool. Would create inappropriate domain coupling.

### B — Declarative tool policy replaces CW boundary

**Rejected.** CW represents richer business authority/resource semantics. Tool rules cannot replace workspace/delegation authority.

### C — Keep current optional policy behavior

**Rejected.** Misconfigured host can execute meaningful side effects without authorization. Violates fail-closed enterprise invariant.

### D — One giant universal policy engine

**Rejected.** Duplicates mature domain semantics and creates god-object governance.

---

## 28. Implementation roadmap

| Task | Scope |
|------|-------|
| **PLATFORM-SE-FAIL-CLOSED-1** | Minimal fail-closed in `RuntimeToolInvoker` for `side_effects=True` without recognized strategy |
| PLATFORM-SE-TYPED-STRATEGY-1 | Optional Phase 2 shared contract + adapters |
| PLATFORM-SE-OBS-1 | Optional Phase 4 observability projection |

---

## 29. Acceptance criteria

| # | Criterion | ADR |
|---|-----------|-----|
| 1 | One canonical platform side-effect invariant | ✓ SE-INV-1…9 |
| 2 | No authorization path → no meaningful effect | ✓ |
| 3 | Multiple domain strategies allowed | ✓ A + B |
| 4 | CW not mandatory for generic tools | ✓ |
| 5 | Declarative policy does not replace CW | ✓ |
| 6 | No third policy engine | ✓ |
| 7 | HITL stays canonical runtime/human | ✓ |
| 8 | Idempotency remains separate | ✓ |
| 9 | Providers do not authorize business effects | ✓ |
| 10 | Host chooses strategy; cannot omit safety silently | ✓ |
| 11 | Production audit-only explicitly addressed | ✓ §20 |
| 12 | Strategy selection typed/DI-based | ✓ §11 |
| 13 | No app/provider string switches | ✓ |
| 14 | side_effect_scope_id ≠ resource_scope | ✓ §15 |
| 15 | Autonomous Work future adoption defined | ✓ §22 |
| 16 | Migration has minimal Phase 1 security closure | ✓ §24 |
| 17 | Next task clearly bounded | ✓ PLATFORM-SE-FAIL-CLOSED-1 |
| 18 | P1 remains open until implementation | ✓ |
| 19 | Nexus documented as internal Execution Engine implementation | ✓ R1 |
| 20 | Nexus is not authorization/governance owner | ✓ R1 |
| 21 | Strategy names are implementation-neutral | ✓ R1 |

---

## 30. Security consequences

**After Phase 1 implementation:**

- Misconfigured hosts cannot silently execute meaningful tool side effects.
- Attack surface for agent-driven external mutation reduces to explicitly wired, enforced policy paths.
- CW and declarative paths remain defense-in-depth for their respective domains.
- Production `AUDIT_ONLY` cannot be mistaken for authorized execution of meaningful effects.

**Residual risks until Phase 1:**

- FINDING-PLATFORM-SE-001 remains exploitable via host misconfiguration.
- Tool authors marking `side_effects=False` for externally mutating tools bypasses gate — contract discipline required.

**Failure types (conceptual, reuse existing where possible):**

| Category | When |
|----------|------|
| `AUTHORIZATION_NOT_CONFIGURED` | No recognized strategy for meaningful effect |
| `AUTHORIZATION_DENIED` | Strategy evaluated DENY |
| `HUMAN_APPROVAL_REQUIRED` | REQUIRE_HUMAN without valid grant |
| `AUTHORIZATION_CONTEXT_INVALID` | Missing required identity/scope |
| `STALE_APPROVAL` / `STALE_GRANT` | Grant does not match current proposal |

---

## Security boundary (canonical flow)

```text
UNTRUSTED / AGENT INTENT
  → validate tool/action
  → resolve execution identity
  → resolve authorization strategy
  → authorization decision
  → HITL if required
  → fresh reauthorization / grant validation
  → idempotency claim (when applicable)
  → effect execution
  → evidence / trace / diagnostics
```

No path may jump from intent → `ToolExecutor` / provider without authorization for meaningful effects.

---

*End of ADR-PLATFORM-SE-CONVERGENCE*
