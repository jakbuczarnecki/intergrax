# ADR-CTRL-X-001: Middleware Hook Contract Ownership & Typed Context

| Field | Value |
|-------|-------|
| **Status** | Proposed / Ready for Independent Architecture Audit |
| **Date** | 2026-10-03 |
| **Baseline HEAD** | `309baa99839960a9db2aae35b8a5deb0d0aabb4e` |
| **Parent** | CTRL-X — Enterprise Control-Plane Recertification |
| **Trigger** | CTRL-X-R2 independent audit — unauthorized ownership move, untyped hook context boundary, unowned security test debt, insufficient wide-pyright semantic classification |
| **Deciders** | Architecture / independent audit (not Cursor) |
| **Related** | `UNIFIED_EXECUTION_RUNTIME.md` · `TIER3_APPLICATION_ENVIRONMENT.md` · `RELIABILITY_FAILURE_AND_HITL.md` · `CTRL_X_ENTERPRISE_CONTROL_PLANE_RECERTIFICATION.md` · CX-01 semantic boundary modules |

---

## 1 Context

CTRL-X recertifies twelve enterprise control planes. R1 closed production security composition gaps (fail-closed defense, event port typing). R2 pursued contract-pure middleware composition: `HostOrchestrationMiddlewarePipelinePort`, single attach path, and relocation of `HookPoint` into Tier-0 `intergrax/contracts/`.

Independent audit **blocked** CTRL-X-R1, CTRL-X-R2, and the CTRL-X parent because:

1. **HookPoint ownership** moved from runtime to contracts without an ADR.
2. **`HostOrchestrationMiddlewareHookContext.runtime_state`** exposes `Mapping[str, object]` as a cross-layer semantic boundary — incompatible with freeze typing policy (FRZ-TYP-01/03).
3. **Fifteen security unit failures** are baselined as identical on A/B/C but lack canonical roadmap owners.
4. **Wide pyright (128 diagnostics)** are accounted mechanically; semantic boundary vs implementation-local classification relies too heavily on path/plane heuristics.

No further production implementation (CTRL-X-R3) may proceed until this ADR is **accepted** by independent architecture audit.

Steering: `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` (CTRL-X = CURRENT, STATE-X = NEXT, not entered), `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` (FRZ-CTL/TYP/TEN remain OPEN).

---

## 2 Problem

Middleware hook semantics sit on the boundary between Tier-0 host-orchestration contracts, Tier-1 runtime middleware pipeline, and Tier-3 application security wiring.

The platform requires **exactly one** semantic owner per hook concept, **exactly one** composition attach operation, **no** `contracts → runtime` import inversion, **no** duplicate hook enums/APIs, **strong typing** at boundaries, and **structural replaceability** of the pipeline port.

Current as-built code partially satisfies composition (port-based attach) but leaves ownership ambiguous and uses an unacceptable generic context bag at the contract.

---

## 3 Current as-built architecture

```text
intergrax/contracts/middleware_hook_point.py
    └── HookPoint (Enum)                    ← canonical definition (post-R2)

intergrax/runtime/hooks/hook_point.py
    └── re-export HookPoint only

intergrax/contracts/host_orchestration_wiring_capabilities.py
    ├── HostOrchestrationMiddlewareHookContext (Protocol)
    │       └── runtime_state: Mapping[str, object]   ← REJECTED final boundary
    ├── HostOrchestrationMiddlewareHookResult (Protocol)
    ├── HostOrchestrationRuntimeMiddlewareRegistration (Protocol)
    └── HostOrchestrationMiddlewarePipelinePort
            └── attach_runtime_middleware_if_absent(...)

intergrax/runtime/middleware/base.py
    └── RuntimeMiddleware(ABC)
            before/after(point: HookPoint, ctx: HostOrchestrationMiddlewareHookContext)

intergrax/runtime/hooks/hook_context.py
    └── HookContext(BaseModel) / HookResult(BaseModel)
            runtime_state: Dict[str, Any]                  ← runtime-internal today

intergrax/runtime/middleware/pipeline.py
    └── MiddlewarePipeline implements port + executes hooks

intergrax/applications/_shared/application_security_wiring.py
    └── register via HostOrchestrationMiddlewarePipelinePort (no isinstance pipeline)
```

Security invariants already accepted (must preserve):

- Production wiring rejects non-`FAIL_CLOSED` `SecurityDefensePlugin`.
- `security_events` targets `HostOrchestrationRuntimeEventPort | None` only.

---

## 4 Exact R1/R2 lineage

| Commit | Role |
|--------|------|
| `1178d6c7b` | R1: fail-closed production security, event port contract, wide pyright provenance committed |
| `d14078854` | R2: contract-pure middleware attach, `HookPoint` → contracts, alternate port proof |
| `309baa998` | R2 typing alignment (HookContext import cleanup) |

**Before R2:** `HookPoint` lived in `intergrax/runtime/hooks/hook_point.py`. Host registration protocols used a **parallel** representation; pyright mismatches arose because runtime middleware referenced runtime `HookPoint` while host contracts could not import runtime — motivating a **single shared enum** without an ADR.

**After R2:** Enum moved to `intergrax/contracts/middleware_hook_point.py`; runtime re-exports. Context/result remain split: Protocols in contracts, Pydantic models in runtime. `RuntimeMiddleware` already uses contract `HookPoint` + `HostOrchestrationMiddlewareHookContext`.

---

## 5 Hard constraints

- Tier-0 `intergrax/contracts` **must not** import `intergrax/runtime`, `agents/`, or `applications/`.
- Exactly-one composition owner; sanctioned attach: `attach_runtime_middleware_if_absent` only (`attach_tier1_middleware_if_absent` **not** canonical).
- Middleware **narrows/denies**; does not become Governance or Execution authority.
- Hook results do not grant execution permission by themselves.
- Structural replaceability: alternate `HostOrchestrationMiddlewarePipelinePort` without pretending to be `MiddlewarePipeline`.
- No `Any`, `object`, `dict[str, Any]`, or `Mapping[str, object]` as **final** cross-layer hook semantic boundaries.
- Security fail-closed and event-port directions frozen (§9–10).
- Do not solve global TENANT-X, STATE-X, TRACE-X, or full EBH-6 here.

---

## 6 Ownership conflict

**Fact:** R2 declared `HookPoint` “Tier-0 owner” in contracts. Historically runtime owned hook timeline semantics (UAEP / §42).

**Conflict:** Moving shared hook **enumeration** changes public ABI and documents an ownership shift that freeze policy requires to be ADR-governed.

**Resolution direction (see §13):** Cross-layer **hook vocabulary** (points, registration shape, context/result protocols) is owned by Tier-0 contracts; runtime owns **execution pipeline behavior** and **default concrete carriers** that structurally implement those protocols.

---

## 7 Generic context typing conflict

`HostOrchestrationMiddlewareHookContext.runtime_state: Mapping[str, object]` forces consumers to probe untyped keys (`prompt`, `tool_id`, `arguments`, `tenant_id`, `resource_tenant_id`, provider metadata).

This violates FRZ-TYP-01/03 and enables invalid substitution at the type level.

**Decision:** Reject `runtime_state` as the host-boundary contract. Replace with typed common invocation context + hook-scoped payload contracts (§16–17).

---

## 8 Middleware contract mismatch

| Surface | Location |
|---------|----------|
| `HostOrchestrationRuntimeMiddlewareRegistration` | Tier-0 Protocol |
| `RuntimeMiddleware` | Tier-1 ABC |

Both expose `before`/`after` with `HookPoint` and `HostOrchestrationMiddlewareHookContext`.

**Decision:** **Same semantic contract** (single contract, dual structural roles). `RuntimeMiddleware` is the **reference runtime implementation** of the registration protocol. `HookResult` structurally satisfies `HostOrchestrationMiddlewareHookResult`.

**Not required:** A second public adapter API or duplicate middleware base type.

**Required (internal):** Pipeline ensures `HookContext` instances satisfy the Tier-0 context protocol and typed payload accessors (R3).

---

## 9 Security fail-closed invariant

Frozen:

```text
Canonical production application security wiring accepts only FAIL_CLOSED SecurityDefensePlugin instances.
FAIL_OPEN plugins are rejected before middleware attach.
```

Lab/manual stacks may differ only where explicitly non-production.

---

## 10 Structural replaceability invariant

Frozen:

```text
application_security_wiring → HostOrchestrationMiddlewarePipelinePort
```

Forbidden: `isinstance(..., MiddlewarePipeline)` in production composition.

---

## 11 Options considered

### Option A — Hook semantics owned by Tier-0 contracts

Contracts own: `HookPoint`, hook context/result **protocols**, middleware registration protocol, pipeline **port**. Runtime implements pipeline, concrete context/result models, plugins.

- **Pros:** Correct dependency direction; stable Tier-0 surface; matches R2 attach path.
- **Cons:** Public ABI in contracts grows; requires COMPAT-X discipline.
- **Verdict:** **Selected** (§13).

### Option B — Hook semantics remain runtime-owned

- **Cons:** Requires `contracts → runtime` or duplicate enum.
- **Verdict:** **Rejected**.

### Option C — Neutral domain contract package

- **Verdict:** **Rejected** — unnecessary layer.

### Option D — Explicit public adapter

- **Verdict:** **Rejected** publicly; private pipeline normalization only.

### Typed context sub-options

| Sub-option | Verdict |
|------------|---------|
| A. Typed common context + typed payload contracts | **Selected** |
| B. Discriminated payloads only | Rejected |
| C. Generic parameterized context | Rejected |
| D. Untyped accessor / bag | Rejected |

---

## 12 Rejected options

See §11.

---

## 13 Proposed decision

**Select Option A** with **single semantic middleware registration contract** and **typed common context + hook payload contracts**.

Authorize R2 **HookPoint** location **retroactively**, conditional on R3 typed context replacement.

### Canonical ownership table

| Artifact | Owner |
|----------|-------|
| `HookPoint` | `intergrax/contracts/middleware_hook_point.py` (Tier-0) |
| Hook context **protocol** | Tier-0 (`host_orchestration_wiring_capabilities.py` or `middleware_hook_semantics.py`) |
| Hook result **protocol** | Tier-0 |
| Hook result **default concrete** | `intergrax/runtime/hooks/hook_context.py` |
| Hook context **default concrete** | `intergrax/runtime/hooks/hook_context.py` |
| Middleware registration contract | `HostOrchestrationRuntimeMiddlewareRegistration` |
| Reference middleware implementer | `RuntimeMiddleware` |
| Pipeline composition | `MiddlewarePipeline` |
| Host pipeline **port** | `HostOrchestrationMiddlewarePipelinePort` |
| Canonical attach operation | `attach_runtime_middleware_if_absent` |
| Application composition | `application_security_wiring` → port |
| Adapter required? | **Yes, internal only** |
| Public/stable ABI? | **Yes** for `HookPoint` + host middleware protocols + attach port |

`intergrax.runtime.hooks.hook_point.HookPoint` = **deprecated compatibility alias** (one definition). Removal: COMPAT-X.

---

## 14 Canonical ownership table

Same as §13 (audit duplicate).

---

## 15 Canonical dependency graph

### Before R2

```text
runtime.hooks.HookPoint (owner)
runtime.middleware.RuntimeMiddleware
host contracts (parallel hook types)
```

### Current (as-built)

```text
contracts.middleware_hook_point.HookPoint
        ↑ re-export
runtime.hooks.hook_point.HookPoint

HostOrchestrationMiddlewareHookContext → Mapping[str, object]

HostOrchestrationRuntimeMiddlewareRegistration
        ↓ attach_runtime_middleware_if_absent
HostOrchestrationMiddlewarePipelinePort → MiddlewarePipeline
```

### Proposed (post-R3)

```text
Tier-0: HookPoint, invocation context protocol, payload protocols, registration protocol, pipeline port
        ↓
Tier-1: MiddlewarePipeline, HookContext/HookResult, RuntimeMiddleware, plugins
        ↓
Tier-3: application_security_wiring → port.attach_runtime_middleware_if_absent
```

---

## 16 Typed context model

**Pattern:** Typed common invocation context + hook-scoped payload contracts.

### `MiddlewareHookInvocationContext` (protocol, Tier-0)

| Field | Semantics |
|-------|-----------|
| `task_id`, `run_id`, `node_id`, `agent_id`, `step_id` | Identities |
| `phase` | `ExecutionPhase` |
| `hook_point` | Active `HookPoint` |
| `payload` | `MiddlewareHookPayload` (family / discriminated union) |
| `subject` | `MiddlewareExecutionSubjectFacet` (tenant/resource scope when applicable) |

Rejected at boundary: `runtime_state` map, `Any`, `object`.

---

## 17 Hook-specific payload model

| Data | Classification |
|------|----------------|
| Identity spine fields | Common invocation context |
| `prompt` | `LlmInferenceHookPayload` |
| `tool_id`, `arguments` | `ToolCallHookPayload` |
| `tenant_id`, `resource_tenant_id` | Execution subject facet (not bag keys) |
| Provider/sandbox metadata | Hook-family payloads |

---

## 18 Authority model

Middleware does not become Governance or Execution authority. Hook results do not grant execution permission. Security remains deny/narrow only. Hooks cannot mint execution truth.

---

## 19 Tenant implications

Tenant on **execution subject facet** when applicable; explicit absence otherwise. `OnlineEvaluationObservation` lacks `tenant_id` — **TENANT-X OPEN**.

---

## 20 Compatibility implications

| Surface | Handling |
|---------|----------|
| `contracts.middleware_hook_point.HookPoint` | Primary stable import |
| `runtime.hooks.hook_point.HookPoint` | Deprecated alias |
| `runtime_state` | Remove from protocol in R3 |

COMPAT-X owns alias removal validation.

---

## 21 Pluginability / replaceability

Plugins use registration protocol + Tier-0 hook contracts; no `MiddlewarePipeline` / Nexus dependency for registration.

---

## 22 Security test-debt classification

15 failures at A/B/C (`test_ctrl_x_r2_security_baseline.py`). Provenance: `uv run pytest -p no:xdist tests/unit/runtime/security` at HEAD `309baa998`.

| Test | Concern | Root cause | Owner | Blocking CTRL-X? | Future stage |
|------|---------|------------|-------|------------------|--------------|
| `test_p0_safety_7_...::test_valid_sandbox_reaches_provider` | Sandbox/provider | `result.output.success` false in harness | **PROD-Q** | no | Live sandbox qualification |
| `test_p0_safety_8_...::test_default_side_effect_tool_executes_once_despite_retry_policy` | Retry safety | `active execution governance identity required` | **PROD-Q** | no | Reliability proofs |
| `test_p0_safety_8_...::test_explicit_retry_safe_side_effect_retries_when_authorized` | Retry auth | governance identity missing | **PROD-Q** | no | Reliability proofs |
| `test_p0_safety_8_...::test_fresh_sandbox_authorization_blocks_retry_when_unavailable` | Sandbox+retry | governance identity missing | **PROD-Q** | no | Reliability+sandbox |
| `test_p0_safety_8_...::test_side_effect_timeout_does_not_blind_retry` | Retry policy | governance identity missing | **PROD-Q** | no | Reliability |
| `test_p0_safety_8_...::test_plugin_default_side_effect_retry_safety_is_single_attempt` | Plugin retry | governance identity missing | **PROD-Q** | no | Reliability |
| `test_p0_safety_8_...::test_idempotency_key_preserved_across_retry_attempts` | Idempotency | governance identity missing | **PROD-Q** | no | Reliability |
| `test_sec_ent.py::test_resolve_encryptor_uses_valid_secrets_store` | Encryptor | `_FakeSecretsStore` fails `SecretsStoreIntegrationContract` guard | **PROD-Q** | no | secrets_store integration |
| `test_sec_ent.py::test_resolve_encryptor_fails_closed_on_resolution_error` | Fail-closed | exception type/path mismatch vs guard | **QUAL-X** | no | Test alignment |
| `test_sec_ent.py::test_resolve_encryptor_fails_closed_on_conformance_error` | Conformance | `TypeError` vs expected resolution error | **QUAL-X** | no | Test remediation |
| `test_sec_ent.py::test_harness_envelope_encryptor_not_selected_by_resolver` | Harness exclusion | expected exception not raised | **QUAL-X** | no | Security test refresh |
| `test_sec_ent.py::test_defense_middleware_blocks_cross_tenant_scope` | Cross-tenant | `active execution identity required` | **TENANT-X** | no | Tenant defense proofs |
| `test_sec_ent.py::test_security_spine_counters_increment` | Spine counters | execution identity required | **TRACE-X** | no | Security observability |
| `test_sec_planes_evol.py::test_defense_blocked_emits_platform_signal` | Defense signal | execution identity required | **TRACE-X** | no | Platform signals |
| `test_sec_planes_evol.py::test_encryption_denied_emits_platform_signal` | Encryption signal | execution identity required | **TRACE-X** | no | Platform signals |

---

## 23 Wide-pyright classification rule

### SEMANTIC BOUNDARY TYPE DEFECT

Any of: symbol in `CTRL_X_SEMANTIC_BOUNDARY_MODULES`; implements Tier-0 cross-layer contract; crosses tier; carries authority; carries tenant/execution identity; plugin substitution surface; invalid port substitution risk; composition uniqueness.

Path alone **insufficient**.

### IMPLEMENTATION-LOCAL TYPE DEBT

All of: internal implementation symbol; no cross-tier contract; no authority/tenant/identity/substitution; fix cannot change port replaceability; documented symbol role.

---

## 24 Representative wide-pyright audit

| Plane | Diagnostic | Role | Verdict |
|-------|------------|------|---------|
| CX-07 | `central_terminal_execution_diagnostic_port.py` `reportArgumentType` | Diagnostic adapter | **IMPLEMENTATION-LOCAL** |
| CX-08 | `catalog_dispatch.py` `reportArgumentType` | Tool dispatch helper | **IMPLEMENTATION-LOCAL** |
| CX-09 | `skills/core/contracts.py` `reportGeneralTypeIssues` | `SkillManifest` public contract | **SEMANTIC BOUNDARY** (if on exported manifest surface) |
| CX-10 | `agent_registry_read.py` `reportReturnType` | Registry read projection | **SEMANTIC BOUNDARY** |
| CX-12 | `context/bootstrap.py` `reportUndefinedVariable` | CE bootstrap helper | **IMPLEMENTATION-LOCAL** |

---

## 25 Evaluation / TENANT-X retained findings

Evaluation plane = `EvaluationProfile` + `OnlineEvaluationRegistry`. `evaluate_agent_promotion` separate. Decision Verification separate. Evaluation does not own Execution.

`OnlineEvaluationObservation` has no `tenant_id` — TENANT-X debt **OPEN**.

---

## 26 Migration shape for future CTRL-X-R3

**Precondition:** ADR **Accepted**.

1. Add Tier-0 invocation context + payload protocols; deprecate protocol `runtime_state`.
2. Align `HookContext` / middleware consumers.
3. Preserve invariants: single HookPoint, single attach, FAIL_CLOSED, event port, no pipeline isinstance.
4. Gates: `test_ctrl_x_typing_gates`, R2 middleware tests, independent GitHub audit.

**Likely files:** `host_orchestration_wiring_capabilities.py`, `hook_context.py`, `pipeline.py`, `defense_plugin.py`, `encryption_middleware.py`, `application_security_wiring.py`, CTRL-X qualification tests.

**Rollback risk:** Reverting typed context without ADR amend reopens tier mismatch.

---

## 27 Explicit non-goals

TENANT-X global redesign, STATE-X, TRACE-X global, CONFIG-X, PROD-Q deployment, EBH-6 full cleanup, middleware feature expansion.

---

## 28 Risks

Payload proliferation; R3 scope creep; premature alias removal.

---

## 29 Acceptance criteria (independent audit)

Ownership answered; one hook model; typed boundary; 15 tests owned; pyright rule + samples; Evaluation/TENANT-X retained; R3 bounded; fail-closed + event port preserved. Status **Accepted** only by independent audit.

---

## 30 Required independent audit

Self-audit: no second owner; no contracts→runtime; no public runtime leak; no boundary bag; replaceability intact; single HookPoint; no authority/tenant expansion; deprecated alias has COMPAT-X exit.

---

## Compliance

Tier boundaries preserved. FRZ-CTL-01, FRZ-CTL-04, FRZ-TYP-01/03, FRZ-TEN debt tracked — no PASS promotion from this ADR alone.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
