# Execution Engine — Security Boundary & Threat Model (EE-B3-A)

**Status:** Frozen certification baseline (audit-first).  
**Scope:** Execution identity, authority, tenant isolation, governance, admission, child execution, tools/side effects, retry/recovery, resource abuse boundaries.  
**Out of scope (parallel NPSC-5F session):** `causal_evidence*`, `export_boundary`, `background_execution/**`, NPSC-5F fingerprints/resignoff — findings only, no modification in EE-B3-A.

## 1. Protected assets

| Asset | Description |
| ----- | ----------- |
| Execution identity | `RunId`, `AttemptId`, `ExecutionId`, lineage tree |
| Tenant boundary | `tenant_id` on task, checkpoint, lifecycle, persistence keys |
| Authority context | `ParentExecutionAuthority`, effective delegation |
| Policy decisions | `PolicyDecision` / `EvaluatedPolicyDecision` from engine |
| Execution admission | Root policy + authority admission + boundary hooks |
| Tool credentials | Provider secrets; idempotency / external-operation ownership |
| External side effects | Email, DB, APIs, files, payments, publish |
| Checkpoints | Durable resume state; schema + identity binding |
| Retry authority | Same logical `run_id`; new `attempt_id` via lifecycle |
| Recovery authority | Resume re-validates tenant, authority, governance |
| Child execution lineage | Parent-bound `ExecutionId`; no anonymous children |
| Budget/capacity | `RunBudget`, EE-B1.2 admission permits |
| Audit/evidence integrity | Failure evidence, lineage seals (NPSC-5E plane) |

## 2. Trust boundaries

| Boundary | Input (untrusted/partial) | Trusted side | Validation owner |
| -------- | ------------------------- | ------------ | ---------------- |
| External request → application | HTTP/API payload, headers | Application service + principal resolution | Application + hosting |
| Application → Execution | Task options, metadata | `ExecutionRuntime` admission | Governance admission + `ExecutionBoundary` |
| Agent → Execution | Agent work requests | Nexus graph + runtime context | Agent governance port + policy |
| Workflow → Execution | Declarative graph nodes | `GraphExecutor` / coordinator | Nexus + execution boundary |
| Nexus → child execution | Child delegate request | `ChildExecutionRunner` | Authority policy + lineage hooks |
| Child → tool | Tool name + args | `RuntimeToolInvoker` | Policy enforcer + scope + sandbox gate |
| Execution → provider | Adapter calls | Provider SDK | Adapter; core does not trust provider for security |
| Recovery → resumed execution | Checkpoint blob | `checkpoint_resume_validation` | NPSC-5E validation pipeline |
| Retry → next attempt | Retry eligibility request | `ExecutionAttemptRetryService` | Retry policy + attempt lifecycle CAS |
| Plugin → platform | Plugin registration, capabilities | Tool registry, contracts | Plugin boundary; no arbitrary code authority |

## 3. Principal model

**Who is the principal?** The authenticated or runtime-trusted **subject** that initiated work, distinct from execution identifiers and from permission authority.

| Principal kind | Typical carrier | Notes |
| -------------- | --------------- | ----- |
| `user` | Request identity / HITL approver | Human tenant-bound |
| `service` | Service account principal | Hosting layer |
| `agent` | Agent runtime principal | Agent governance port |
| `child agent` | Delegated subtask | Authority narrowed via delegation |
| `workflow` | Task + graph root authority | Typed `ParentExecutionAuthority` on task |
| `system` | Internal maintenance | Explicit system scopes only |
| `recovery process` | Resume coordinator | Replays checkpoint under task tenant |
| `plugin` | Declared capabilities | No elevation without registry + policy |

**Frozen invariant:** Identity ≠ Authority ≠ Governance permission ≠ Execution admission.

## 4. Identity security

| Concern | Canonical owner | Enforcement |
| ------- | ----------------- | ----------- |
| Mint `RunId` / `AttemptId` / `ExecutionId` | `intergrax.runtime.execution.identity_authority` | `ExecutionRuntime`, `ChildExecutionRunner`, `AttemptLifecycleService` |
| Parse-only IDs | `intergrax.contracts.execution_identity` (`validate_*`) | Wire codecs, resume validation |
| Active identity | ContextVar `bind_active_execution_identity` | `ExecutionBoundary` |

**Rule:** Direct `mint_run_id()` / `mint_attempt_id()` / `mint_execution_id()` outside allowlist = **0** (EE-A2 gate).

Caller-forged IDs: rejected unless they pass `validate_*` canonical format; mint paths do not accept untrusted mint.

## 5. Authority security

| Concern | Canonical owner | Enforcement |
| ------- | ----------------- | ----------- |
| Root authority | Task / runtime typed `ParentExecutionAuthority` | `resolve_root_parent_execution_authority` |
| Child authority | `DefaultStrictAuthorityPolicy` + `mint_effective_delegation_authority` | `ChildExecutionRunner` |
| Active authority | `active_execution_authority` context | Required before child admission |

**Rule:** Child effective authority ≤ parent; overreach → `DelegationAuthorityError` (fail-closed).

NPSC semantic delegation does not grant execution permission scopes; Collaborative Work authority is a separate governance plane.

## 6. Tenant isolation

| Layer | Mechanism |
| ----- | --------- |
| Task / checkpoint | `tenant_id` on `Task`, `TaskCheckpoint` |
| Resume | `validate_checkpoint_identity_binding` → `REJECT_TENANT` on mismatch |
| Attempt lifecycle | Store key `(tenant_id, run_id)` |
| Retry orchestration | Caller must pass `tenant_id`; production coordinators bind `task.tenant_id` |
| Decision / correlation | `validate_correlation_tenant_scope` and related contracts |

**Rule:** No automatic tenant switch on child/tool paths; cross-tenant checkpoint resume is rejected.

## 7. Governance enforcement

| Concern | Canonical owner | Enforcement |
| ------- | ----------------- | ----------- |
| Root execution policy | `RuntimePolicyEngine` | `RuntimeExecutionPolicyAdmissionEvaluator` |
| Root authority admission | `root_execution_authority_admission` | Admission hooks |
| Physical / coordination delegation | `physical_delegation_governance`, `multi_agent_coordination_governance` | Typed requests |
| Side-effect policy | Declarative enforcer + `require_meaningful_side_effect_authorization` | `RuntimeToolInvoker` |

**Policy failure:** Engine and governance adapters default to **DENY / REQUIRE_HUMAN** — not silent ALLOW.

**Policy spoofing:** Untrusted callers cannot supply authoritative `EvaluatedPolicyDecision`; decisions are produced only inside `RuntimePolicyEngine` / governance evaluators with bundle binding.

## 8. Execution admission

Supported production entry: **`ExecutionRuntime.execute`** (and boundary-wrapped delegates) with admission hooks (policy, lineage, capacity per EE-B1.2).

**Bypass inventory:** Frozen P0 central inventory reports **supported production bypass = 0** (EE-A1 gate).

Test-only adapters (`AllowingRuntimeExecutionPolicyAdmission`) live under `runtime/governance/` and are not production wiring.

## 9. Child execution security

`ChildExecutionRunner`:

- Requires active parent identity + authority + lineage durability rules.
- Mints child `ExecutionId` via identity authority only.
- Applies authority policy and budget policy before `ExecutionBoundary`.

No anonymous child: lineage admission hook binds parent/child IDs.

## 10. Tool & side-effect authorization

`RuntimeToolInvoker` (Nexus-owned) enforces declarative policy, meaningful side-effect authorization, sandbox isolation gate, dependency concurrency admission, and registry-resolved tools — not arbitrary import paths.

Direct `ToolExecutor` in Nexus is constructed inside runtime context and invoked through the invoker wrapper on supported paths.

## 11. Retry & recovery security

| Path | Control |
| ---- | ------- |
| Retry | Preserves `run_id`; new `attempt_id` via lifecycle; optional active identity rebind |
| Recovery | Checkpoint schema, identity, tenant, authority, governance gates in `checkpoint_resume_validation` |
| Checkpoint tampering | Schema version, `validate_canonical()`, authority/governance rejection enums |

Retry does not expand authority or tenant scope by design; resume re-evaluates stored vs target tenant/authority.

## 12. Resource abuse (reference)

CPU/memory/task explosion, retry storms, fan-out: **EE-B1.2** (capacity admission), **EE-B1.3** (worker isolation), **NPSC-5B** (bounded fan-out). EE-B3-A documents threat model; does not add new limiters.

## 13. Dynamic code & shell (classified search)

| Pattern | Location | Classification |
| ------- | -------- | -------------- |
| `eval`/`exec` in user code guard | `codecraft/orchestrator` | Gated codecraft path; rejects eval in snippets |
| `subprocess` | `sandbox/session`, token proof scripts | Sandbox / proof — not canonical execution bypass |
| `importlib` / `__import__` on execution path | Not found under `runtime/execution` | — |
| `execute_hybrid_retrieval` | Named API (not Python `exec`) | INFO |

## 14. Security owner matrix

| Security concern | Canonical owner | Enforcement point | Bypass on supported prod? |
| ---------------- | --------------- | ----------------- | ------------------------- |
| Identity | `identity_authority` + EE-A2 | `ExecutionRuntime`, lifecycle | **No** |
| Tenant | Task/checkpoint/lifecycle keys | Resume validation, stores | **No** (resume); bind at coordinator |
| Authority | `delegation_authority` + policies | `ChildExecutionRunner` | **No** |
| Governance | `RuntimePolicyEngine` + governance modules | Admission evaluators | **No** |
| Execution admission | `ExecutionBoundary` + hooks | `ExecutionRuntime` | **No** |
| Child authority | `DefaultStrictAuthorityPolicy` | Child runner | **No** |
| Tool authorization | `RuntimeToolInvoker` | Nexus tool loop | **No** |
| Retry | `ExecutionAttemptRetryService` | Lifecycle CAS | **No** (wrong tenant = separate key space; prod binds task tenant) |
| Recovery | `checkpoint_resume_validation` | `LongRunningCoordinator` | **No** |
| Capacity | EE-B1.2 admission | Root execute | **No** |

## 15. Threat matrix (summary)

| Threat | Asset | Attack path | Control | Residual | Severity |
| ------ | ----- | ----------- | ------- | -------- | -------- |
| Identity spoofing | Run/attempt/exec IDs | Forged wire IDs | `validate_*` + mint allowlist | Low if codecs used | LOW |
| Cross-tenant resume | Checkpoint | Tenant B reads tenant A checkpoint | `REJECT_TENANT` | None on canonical resume | — |
| Authority escalation | Child execution | Request extra scopes | Parent ∩ requested; error on overreach | None | — |
| Governance bypass | Side effect | Direct tool without policy | Invoker + registry path | Test adapters only | LOW |
| Confused deputy | Provider creds | Low-priv caller triggers high-priv tool | Policy + execution context required | Adapter misconfig | MEDIUM (ops) |
| Retry tenant confusion | Lifecycle | Wrong `tenant_id` param | Key isolation; coordinator binds task | Mis-wired caller | LOW (integration) |
| Checkpoint tamper | Resume state | Mutate tenant/authority fields | Validation + schema | Store integrity out of band | LOW |
| Resource exhaustion | Workers/queue | Unbounded fan-out | NPSC-5B + B1.2/B1.3 | Authenticated abuse bounded | MEDIUM (accepted) |

No **CRITICAL** or **HIGH** open findings on supported production execution paths at certification time.

## 16. Bypass inventory (supported production)

| ID | Path | Supported production? | Impact | Decision |
| -- | ---- | --------------------: | ------ | -------- |
| B0 | P0 central inventory | Yes | Any bypass | **0 bypass** — frozen |
| B1 | `ExecutionRuntime.execute` without hooks | No | Full bypass | Not supported wiring |
| B2 | Raw `ToolExecutor` from application | No | Side effect | Nexus/context wiring required |
| B3 | `AllowingRuntimeExecutionPolicyAdmission` | Test only | Policy ALLOW | Not production default |

## 17. Accepted risks

- **Retry `tenant_id` parameter:** Port accepts explicit tenant; **must** be bound from trusted `Task`/principal in production coordinators (documented contract).
- **Confused deputy at provider adapters:** Mitigated by tool policy + execution context; adapter configuration remains operator responsibility.
- **DoS via authenticated load:** Bounded by EE-B1.2/B1.3 and NPSC-5B; not eliminated by EE-B3-A.

## 18. Open findings

None at **CRITICAL** / **HIGH** for supported paths. Follow-up hardening (if any) → **EE-B3-B**.

## 19. Cross-session NPSC-5F

Observability export, causal evidence v1/v2, background execution identity: **audit handoff only** in EE-B3-A qualification doc; no code changes in this task.
