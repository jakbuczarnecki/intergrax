# AW-7C-P1-ARCH — Scoped Adaptive Integration Execution Architecture Lock

## 1. Status and scope

| Field | Value |
| ----- | ----- |
| **Task** | `AW-7C-P1-ARCH` — **CLOSED / independently accepted through R1**; `AW-7C-P1-ARCH-R1` — **CLOSED / independently accepted** @ `fa5853ec009d015cb52c9de15dd7e283932a9b4d` |
| **Parent** | `AW-7C` — **CURRENT** — Scoped Adaptive Integration Execution |
| **Program baseline** | P1 blocked baseline `e055a5bdb3c120b90dad546dc33cc0219824ced7`; accepted R1 architecture anchor `fa5853ec009d015cb52c9de15dd7e283932a9b4d` (`development`) |
| **Source** | Scenario #24 GAP-03; roadmap §3.0.1 |
| **Status** | **CLOSED / independently accepted through R1 remediation** — design lock only; **no A2 production implementation** |
| **Production / tests / contracts in P1 / R1** | **0** |
| **Prerequisites** | AW-7C-P0-3B + AW-7C-P0-3B-PHYSQ **CLOSED / accepted** (egress substrate evidence) |
| **Next implementation** | `AW-7C-P2` + integrated P2 hardening — **READY FOR AUDIT**; `AW-7C-P3` — **READY FOR AUDIT** (reference strategy + replaceability; pending independent SHA audit) |
| **Program** | `AW-7C` — **CURRENT**; `AW-7C-P4` — **READY FOR AUDIT** |

**Scope:** lock reusable platform semantics for **A2 scoped adaptive integration**: from `SCOPED_ADAPTATION_CANDIDATE` through bounded adaptation, qualification, Governance/runtime admission, and **canonical Execution** — without a parallel runtime, AW-owned integration registry, or A1 bypass.

**Explicit non-closure:** this lock does **not** close global `FRZ-*` PASS, `FRZ-TEN-*` PASS, `TENANT-X`, `PROD-Q`, or parent `AW-7C` implementation/certification.

**UCA:** consumer only after public contract certification; UCA must **not** implement GAP-03.

---

## 2. Non-goals

- Production A2 service, executable adapter, provider rollout, or new runtime/registry
- CodeCraft A1 path reuse (`WorkerEphemeralCapabilityExecutionService` / `CodeCraftEphemeralCapabilityExecutionAdapter`) as A2 execution authority
- Second Execution Engine, direct ToolRuntime, direct Nexus import from AW production paths
- AW-owned Integration catalog/registry, provider switch, or global catalog mutation for A2
- AW-owned secret store (`WorkerSecretBroker`, `AWSecretStore`) or duplicate network policy (`A2HostAllowlist`, `WorkerNetworkPolicy`, `AdaptiveIntegrationFirewall`)
- Durable A3 promotion / `CONFIGURE_EXISTING` realization (separate INT-CONFIG-REAL-X path)
- Observability/Diagnostics minting permission or qualification truth
- Collapsing decision, artifact, qualification, Governance admission, and Execution into one boolean

---

## 3. Problem statement (closed-world gap)

**Verified on baseline `88e423109d7321e273a68fcc4be5e950dd29ffba`:**

| Capability | Present? | Owner / location |
| ---------- | -------- | ---------------- |
| A2 acquisition disposition | **YES** | `CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE`, `WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION`, `WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE` — `intergrax/contracts/autonomous_work/capability_acquisition.py`; AW-7A service emits decision only |
| A2 execution orchestration port/service | **NO** | No `ScopedAdaptive*` / `AdaptiveIntegration*Execution*` production surface |
| A1 ephemeral execution | **YES (separate)** | `WorkerEphemeralCapabilityExecutionService` — must not become A2 bypass |
| Canonical worker execution dispatch | **YES** | `WorkerExecutionDispatchService` → `RootExecutionLaunchPort` → `CanonicalExecutionIntakePort` |
| Scoped credentials | **YES** | `CredentialUseGrant`, `CredentialUseScope`, `ScopedCredentialBroker` — Integrations/credentials |
| Network egress + sandbox attestation | **YES** | `NetworkEgressHost` / `NetworkEgressAllowlist`; `SandboxSecurityCapable` / `SandboxSecurityCapabilities` |
| Integration adaptation strategy SPI | **NO** | Analog: `ExistingCapabilityConfigurationRealizationStrategy` (INT-CONFIG) — not adaptation |
| Capability qualification boundary | **YES** | `CapabilityQualificationProvider`, `CapabilityQualificationRequest`, `CapabilityQualificationResult` — UCA-4 contracts |

**Root architecture gap:** eligibility and prerequisites exist; **canonical typed A2 flow** (request → scope → strategy → artifact → qualification → admission → execution) is **designed here**, not implemented.

---

## 4. Canonical ownership matrix

| Concern | Semantic owner | Composition owner | Authority owner |
| ------- | -------------- | ----------------- | --------------- |
| `SCOPED_ADAPTATION_CANDIDATE` / AW-7A decision | Autonomous Work (acquisition) | AW-7A `WorkerCapabilityAcquisitionService` | Upstream policy + acquisition rules (no execution) |
| A2 orchestration (episode correlation, scope enforcement vs upstream decision) | **Autonomous Work** | Future `WorkerScopedAdaptiveIntegrationOrchestrationService` (P2) | None — orchestration only |
| Integration identity & domain adaptation semantics | **Integrations** | Integrations adaptation service (P2) | None |
| `ScopedIntegrationAdaptationScope` (adaptation envelope semantics) | **Integrations** (`intergrax/integrations/contracts/`) | Integrations validates at adaptation-port ingress; AW derives/supplies immutable instance | None |
| Adaptation strategy SPI (`ScopedIntegrationAdaptationStrategy`) | **Integrations contracts** | **Integrations** pure selection (`select_scoped_adaptation_strategy` pattern, mirror INT-CONFIG) | None |
| `CapabilityQualificationSubject` | **Capability Qualification** (`intergrax/contracts/capability_qualification/`) | Qualification coordinator — single provider-selection mechanism | None |
| Strategy/provider implementation | Integrations providers / extensions | Host wiring / plugin composition (Integrations) | None |
| Adaptation artifact immutability & fingerprint | Integrations contract + AW correlation refs | Strategy produces; AW stores correlation IDs only | None |
| Qualification truth | **Capability qualification** domain (`CapabilityQualificationProvider`) | Qualification coordinator (existing UCA-4 seam) | None — evidence only |
| Governance / runtime admission | **Runtime / Governance** | `RootExecutionAuthorityAdmissionPort`, `RuntimeExecutionPolicyAdmissionPort`, `WorkerExecutionAdmissionService` | Governance ALLOW only |
| Trusted execution lifecycle | **Execution** | `ExecutionRuntime` via `CanonicalExecutionIntakePort` | Runtime minted authority |
| Credential material | **Credential domain** | `ScopedCredentialBroker` under matching grant | Grant scope only |
| Network/isolation enforcement | **Sandbox / CodeCraft substrate** | Substrate + `SandboxSecurityCapable` attestation | Attested facts only |
| Integration Catalog | **Integrations** (existing catalog) | Catalog registry — **no AW registry** | Catalog read for identity; **no A2 global mutation** |
| Evidence / trace correlation | Observability consumers | AW links IDs; records do not authorize | **None** |

**Autonomous Work does NOT own:** integration implementation, provider selection logic inside strategies, secret storage, sandbox enforcement, execution lifecycle, authority minting, policy engine, qualification truth, observability truth.

**Integrations does NOT depend on AW concrete adaptation implementations** (no reverse dependency).

---

## 5. Architecture decision — adaptation strategy contract placement

**Question:** Is “how an existing integration is adapted” an Integrations semantic responsibility while AW only orchestrates when bounded adaptation is needed?

**Answer: YES** — consistent with INT-CONFIG-REAL-X (realization strategies owned by Integrations) and closed-world code (no AW adaptation SPI today).

**Preferred direction (locked):**

```text
AW orchestration (Autonomous Work)
  → consumes Integrations public adaptation port
  → Integrations-owned strategy SPI + composition
  → provider/domain strategy implementation
```

**Forbidden:**

- Integrations importing AW orchestration concrete types as authority
- AW registry of integration adapters or vendor branches in AW core
- AW selecting provider by string dispatch (`if provider == "x"`)

If implementation requires reverse layer dependency or AW-owned registry → **STOP — ARCHITECTURE DECISION REQUIRED**.

---

## 6. Before / after dependency graphs

### 6.1 Current path (baseline)

```text
WorkerCapabilityNeed
  → AW-7A discovery/acquisition
  → SCOPED_ADAPTATION_CANDIDATE (decision only)
  → [GAP: no canonical A2 execution orchestration]

Separate (must not merge):
  AW-7A EPHEMERAL_GENERATION_CANDIDATE
    → WorkerEphemeralCapabilityExecutionService (A1)
    → WorkerEphemeralCapabilityExecutionPort / CodeCraft adapter

Worker business execution (non-A2):
  → WorkerExecutionDispatchService
    → WorkerExecutionAdmissionService / budget
    → RootExecutionLaunchPort
    → CanonicalExecutionIntakePort
    → ExecutionRuntime (internal Nexus — not AW import)
```

### 6.2 Proposed A2 path (after P2–P4)

```text
SCOPED_ADAPTATION_CANDIDATE + upstream acquisition evidence
  → ScopedAdaptiveIntegrationExecutionRequest (AW contract)
  → ScopedIntegrationAdaptationScope (Integrations contract; immutable envelope ⊆ upstream authority)
  → ScopedIntegrationAdaptationPort.adapt (Integrations)
       → select exactly one ScopedIntegrationAdaptationStrategy
       → ScopedIntegrationAdaptationArtifact (not permission)
  → CapabilityQualificationSubject (projection from artifact)
  → CapabilityQualificationRequest/Result (Capability Qualification owner)
       → QUALIFIED subject-bound evidence only
  → Runtime/Governance admission (existing ports; scoped payload)
  → WorkerExecutionDispatchService or sanctioned RootExecutionLaunch
       → CanonicalExecutionIntakePort
       → ExecutionRuntime
  → CredentialUseEvidence + execution IDs + adaptation lineage (correlation)
```

**No edge:** artifact → ToolRuntime; AW → Nexus; A2 → A1 ephemeral service as execution authority.

---

## 7. Contract graph (names locked at semantic level; P2 implements)

### 7.1 AW orchestration input — `ScopedAdaptiveIntegrationExecutionRequest`

**Owner:** `intergrax/contracts/autonomous_work/` (P2).

**Carries (typed IDs / value objects only):**

- `WorkerInstanceId`, tenant id, `WorkerCapabilityNeed` correlation
- Recovery decision / episode id, selected `WorkerCapabilityCandidate` id + revision
- Integration identity ref (catalog-backed, not live client)
- Requested operations (typed enum/tuple — not free strings as authority)
- `adaptation_scope: ScopedIntegrationAdaptationScope` (Integrations-owned type; see §7.2)
- Upstream acquisition decision refs / `WorkerCapabilityAcquisitionDecision` snapshot
- Execution correlation (`RunId`, task refs as already used by dispatch)
- Authority **reference** compatible with existing execution admission (not raw ParentExecutionAuthority minting by AW)

**MUST NOT carry:** raw secrets, provider clients, sandbox handles, callables, unstructured bags used as contracts (forbidden patterns listed in §17).

### 7.2 Adaptation scope — `ScopedIntegrationAdaptationScope`

**Semantic owner:** **Integrations** only. **Future contract location:** `intergrax/integrations/contracts/`.

Autonomous Work does **not** own a second adaptation-scope contract. AW owns `ScopedAdaptiveIntegrationExecutionRequest` and constructs or supplies an immutable `ScopedIntegrationAdaptationScope` derived from upstream A2 decision/authority.

**Dependency direction (locked):**

```text
Autonomous Work → consumes Integrations public adaptation contracts
NOT: Integrations → imports AW concrete orchestration implementation
```

**Minimum fields (typed):**

- `tenant_id`
- Integration identity (`integration_category`, `provider_id`, resource/integration ref)
- Permitted operations (bounded set)
- `NetworkEgressAllowlist` (or explicit empty/deny) — **reuse canonical type**
- `credential_use_grant_ref` / grant id binding (reference only)
- Resource scope string (tenant-scoped)
- `expires_at` / lifetime bound
- Candidate id, configuration/adaptation revision constraints

**Responsibility split:**

| Domain | Responsibility |
| ------ | -------------- |
| **Autonomous Work** | Derive maximum permissible scope from upstream acquisition/recovery context; refuse request wider than upstream authority; pass immutable scope to Integrations; preserve tenant/candidate/recovery correlation |
| **Integrations** | Define `ScopedIntegrationAdaptationScope`; validate scope at adaptation-port ingress; ensure strategy output scope ≤ requested scope; reject strategy output widening; preserve integration identity and tenant continuity |
| **Execution** | `execution_scope ≤` qualified adaptation scope |
| **Credential domain** | `credential_use_scope ≤` adaptation scope |
| **Sandbox** | Effective egress ≤ adaptation scope |

No domain may silently broaden another domain's scope.

**Invariants:**

```text
adaptation_scope ⊆ upstream_capability_authority_scope
execution_scope ⊆ adaptation_scope
effective_network_scope = intersection(upstream_network, grant_target, artifact_declared_hosts)
effective_credential_scope = intersection(grant_scope, adaptation_scope) — never union widening
```

#### 7.2.1 Scope fingerprint (`scope_fingerprint`)

Deterministic semantic fingerprint of a `ScopedIntegrationAdaptationScope`. **Contributors (canonical ordered serialization; no raw secrets):** `tenant_id`; integration identity; permitted operations; resource scope; `NetworkEgressAllowlist` (canonical allowlist ordering); credential-grant binding/reference; `expires_at` / lifetime where semantically relevant; candidate/revision constraints.

Recorded on `CapabilityQualificationSubject` and used in qualification result/evidence binding.

#### 7.2.2 Artifact fingerprint (`artifact_fingerprint`)

**Distinct from** `scope_fingerprint`. `artifact_fingerprint` covers the immutable produced artifact/spec. `scope_fingerprint` covers the maximum approved adaptation envelope.

**Invariant:** artifact declares scope `S`; `artifact_fingerprint` binds the artifact to `scope_fingerprint(S)`. An artifact produced under `S1` cannot be re-qualified or reused as if produced under `S2`.

### 7.3 Integrations strategy SPI — `ScopedIntegrationAdaptationStrategy`

**Owner:** `intergrax/integrations/contracts/` (P2).

**Responsibilities:**

- `strategy_id: str` (unique in composition tuple)
- `supports(request, integration_target) -> bool`
- `adapt(...) -> ScopedIntegrationAdaptationArtifact`

**Forbidden responsibilities:** grant authority, execute, hidden ToolRuntime path, resolve secrets except via future explicit broker injection on Integrations side, mutate global catalog/provider state.

**Selection:** Integrations pure function/service — exactly one match; zero → `STRATEGY_UNAVAILABLE`; >1 → `STRATEGY_AMBIGUOUS`; duplicate `strategy_id` → fail closed at composition build.

### 7.4 Adaptation artifact — `ScopedIntegrationAdaptationArtifact`

**Owner:** Integrations contracts.

**Represents:** candidate realization spec **after** strategy success — **not** execution permission.

**Must include:** stable `artifact_fingerprint`, input candidate/integration correlation, typed operation/schema/protocol transformation description (typed structs / enums), required execution/sandbox profile **references**, evidence lineage refs, bounded scope copy ⊆ request scope.

**Must NOT grant:** network access, secret access, Execution authority.

### 7.5 Qualification — single mechanism, subject-oriented model

**Owner:** `intergrax/contracts/capability_qualification/` — `CapabilityQualificationProvider`, `CapabilityQualificationRequest`, `CapabilityQualificationResult`, `CapabilityQualificationEvidence`.

**Composition owner:** Capability Qualification domain only (existing UCA-4 coordinator). **Forbidden:** qualification provider registry in Integrations; `AWQualificationProviderRegistry`; separate A2 qualification engine.

#### 7.5.1 `CapabilityQualificationSubject` (canonical semantic subject)

Capability Qualification qualifies a canonical typed **subject**, not intrinsically a `CapabilityAcquisitionResult`.

| Concept | Owner |
| ------- | ----- |
| `CapabilityQualificationSubject` | **Capability Qualification** — identity/integrity envelope only; **no domain-specific adaptation semantics**; **no authority** |

**Required information (strongly typed fields / enums / value objects only):**

- `subject_kind` (enum)
- stable `subject_id`
- `subject_integrity_fingerprint`
- `tenant_id` when tenant-scoped (**mandatory** for A2 adaptation subjects)
- `scope_fingerprint` (see §7.2.1)
- typed `source_reference` / domain reference
- lineage/correlation references sufficient to prove the subject qualified is the subject later consumed (artifact id, integration identity ref, recovery/acquisition lineage ids as applicable)

**Forbidden at this boundary:** `dict[str, Any]`, `Mapping[str, Any]`, generic `object`, untyped metadata bags.

#### 7.5.2 Domain artifact vs qualification subject

| Type | Role |
| ---- | ---- |
| `ScopedIntegrationAdaptationArtifact` | Integrations-owned domain artifact |
| `CapabilityQualificationSubject` | CQ-owned immutable identity/integrity **projection** — does **not** replace the artifact |

**Deterministic projections (locked):**

```text
ScopedIntegrationAdaptationArtifact
  → project_adaptation_qualification_subject()
  → CapabilityQualificationSubject

CapabilityAcquisitionResult (outcome SUCCEEDED only — existing acquisition path)
  → project_acquisition_qualification_subject()
  → CapabilityQualificationSubject
```

Projection for A2 binds at minimum: artifact identity; `artifact_fingerprint`; tenant; integration identity/reference; `scope_fingerprint`; strategy identity; relevant lineage/correlation.

Qualification must be unable to qualify artifact A and later present evidence for artifact B.

**Explicitly forbidden:**

- Fabricating `CapabilityAcquisitionResult(outcome=SUCCEEDED)` for an A2 adaptation artifact
- `qualified=True` (or equivalent) on `ScopedIntegrationAdaptationArtifact`
- `A2QualificationEngine`, `ScopedAdaptationQualificationEngine`, or any second permanent qualification mechanism parallel to UCA-4

#### 7.5.3 `CapabilityQualificationRequest` — V1 legacy vs canonical target

**Baseline V1 (acquisition-specific legacy shape):** `CapabilityQualificationRequest` carries `acquisition_request_id`, `gap_id`, `strategy_id`, and `acquisition_result: CapabilityAcquisitionResult` with `requires acquisition_result.outcome == SUCCEEDED`. The acquisition result is the **legacy semantic subject** — adequate for acquisition-only handoff, **not** for A2 artifacts.

**Canonical target shape (locked for P2 contract evolution):** request qualifies `subject: CapabilityQualificationSubject`. Acquisition-specific ids may remain as **lineage** fields during controlled migration but must **not** remain the universal semantic authority defining what is qualified.

**Single-mechanism migration (locked):**

```text
V1 acquisition-specific request
  → deterministic subject projection (acquisition path)
  → canonical qualification path (same provider selection, same lifecycle policy)

A2 adaptation artifact
  → deterministic subject projection (adaptation path)
  → same canonical qualification path
```

**Forbidden:** permanent fork where providers select unrelated mechanisms; `if request.is_a2: new_engine else: old_engine`; two provider registries; two lifecycle policies; legacy fallback provider path when projection fails (F27).

P2 **implements** contracts and projection functions per this lock; P2 does **not** invent scope ownership, subject model, request semantic authority, result/evidence continuity, or migration direction.

#### 7.5.4 Result and evidence continuity (target semantics)

Target `CapabilityQualificationResult` and `CapabilityQualificationEvidence` bind:

- `qualification_request_id`
- subject identity + `subject_integrity_fingerprint` (and `subject_kind`)
- `tenant_id` where applicable
- `scope_fingerprint`
- `provider_id`
- qualification outcome
- evidence integrity chain

Acquisition identifiers may be retained as **lineage** for acquisition subjects; they are **not** mandatory universal identity for A2.

A2 evidence is **unusable** if any differ: tenant; artifact id; `artifact_fingerprint`; integration identity; `scope_fingerprint`; strategy/adaptation revision where applicable.

`CapabilityQualificationOutcome.QUALIFIED` ≠ Governance ALLOW ≠ Execution admission ≠ Execution success. Subject, provider, and evidence contain **no** authority and must not mint `ParentExecutionAuthority`, credential scope, network scope, or integration provider authority.

#### 7.5.5 Integrity chain (fail closed)

```text
request.subject
  == provider-evaluated subject
  == result subject identity/fingerprint
  == evidence subject identity/fingerprint
  == artifact identity bound for subsequent A2 orchestration
```

Stale subject fingerprint, missing subject, subject-kind mismatch, tenant mismatch, scope-fingerprint mismatch → fail closed.

#### 7.5.6 Semantic distinction (mandatory)

| State | Meaning |
| ----- | ------- |
| Artifact exists | Strategy returned `ScopedIntegrationAdaptationArtifact` |
| Artifact qualified | `CapabilityQualificationOutcome.QUALIFIED` with integrity-valid evidence bound to `CapabilityQualificationSubject` |
| Artifact authorized | Governance/runtime admission ALLOW for concrete execution |
| Executed | `CanonicalExecutionIntakePort` completed successfully |

### 7.6 Execution handoff — `ScopedAdaptiveIntegrationExecutionHandoff` (conceptual)

**Owner:** AW orchestration composes; Execution owns runtime.

**Path (locked):**

1. Qualified artifact + admission evidence + narrowed scope snapshot
2. `WorkerExecutionDispatchRequest` / `RootExecutionLaunchRequest` payload prepared by AW (typed execution intent — mirror existing dispatch patterns)
3. `RootExecutionLaunchPort.launch` → `CanonicalExecutionIntakePort.invoke`
4. `ExecutionRuntime` — sole side-effect engine

**Forbidden:** AW calling ToolRuntime, provider HTTP clients, or sandbox execute directly for business outcome.

### 7.7 Orchestration ports (P2 sketch)

| Contract | Owner | Consumer |
| -------- | ----- | -------- |
| `ScopedIntegrationAdaptationPort` | Integrations | AW orchestration |
| `WorkerScopedAdaptiveIntegrationOrchestrationPort` | AW | Recovery/worker runtime composition |
| `ScopedAdaptiveIntegrationOrchestrationOutcome` | AW | Callers (typed enum + reason codes) |

---

## 8. Semantic state distinctions (no collapse)

1. **SCOPED_ADAPTATION_CANDIDATE** — AW-7A acquisition/classification only  
2. **Adaptation request** — bounded proposal to produce artifact  
3. **Adaptation artifact** — strategy output spec  
4. **Qualification result** — evidence that rules satisfied  
5. **Governance/runtime admission** — permission for this execution attempt  
6. **Canonical Execution** — runtime side effects  

Forbidden: `candidate → execute`; `artifact → permission`; `qualification → permission`; Governance ALLOW executed outside Execution; absence of DENY treated as ALLOW.

---

## 9. Lifecycle model

Lifecycle values may be **contract outcomes / evidence statuses** without mandatory durable AW state machine (Execution/evidence may own durability).

**Happy path:**

```text
REQUESTED
  → STRATEGY_SELECTED
  → ARTIFACT_PRODUCED
  → QUALIFICATION_PENDING
  → QUALIFIED
  → EXECUTION_ADMITTED
  → EXECUTED
```

**Failure exits (typed reason codes — `ScopedAdaptiveIntegrationFailureReason` P2):**

`REJECTED`, `STRATEGY_UNAVAILABLE`, `STRATEGY_AMBIGUOUS`, `ARTIFACT_INVALID`, `QUALIFICATION_FAILED`, `AUTHORITY_DENIED`, `EGRESS_UNAVAILABLE`, `CREDENTIAL_SCOPE_DENIED`, `EXECUTION_DENIED`, `EXECUTION_FAILED`, plus matrix §10 codes.

---

## 10. Failure-mode matrix (F1–F30)

| ID | Condition | Resolution |
| -- | --------- | ---------- |
| F1 | No strategy supports request | Fail closed; `STRATEGY_UNAVAILABLE` |
| F2 | >1 equally valid strategy | Fail closed; `STRATEGY_AMBIGUOUS` — no first-wins |
| F3 | Artifact scope exceeds request | Reject; `ARTIFACT_INVALID` |
| F4 | Artifact host ∉ allowed allowlist / grant intersection | Reject before execution |
| F5 | Credential need outside `CredentialUseGrant` | Reject; `CREDENTIAL_SCOPE_DENIED` |
| F6 | Grant expired | Reject; broker raises / orchestration maps to denied |
| F7 | Tenant mismatch anywhere in chain | Reject; `TENANT_MISMATCH` |
| F8 | Integration identity mismatch | Reject; `IDENTITY_MISMATCH` |
| F9 | Candidate id/version mismatch | Reject |
| F10 | Qualification coordinator unavailable | No execution |
| F11 | Qualification negative | No execution |
| F12 | Governance admission unavailable/denied | No execution |
| F13 | Canonical Execution unavailable | Fail; **no** direct provider fallback |
| F14 | Sandbox attestation missing / allowlist not attested | Fail closed (`SandboxSecurityCapabilities` insufficient) |
| F15 | Provider/strategy unavailable | Fail closed |
| F16 | Strategy timeout/OSError/system failure | Typed failure; no alternate authority path |
| F17 | Partial artifact then failure | No reachable executable authority from partial output |
| F18 | Retry | Must not widen scope or silently pick broader strategy |
| F19 | Stale artifact/evidence vs request fingerprint | Reject |
| F20 | Duplicate invocation | Idempotency via deterministic request/artifact ids + execution correlation; duplicate qualified handoff must not double-mint authority (P4) |
| F21 | Cleanup failure | Substrate/provider that created resource owns cleanup; AW must not report success if physical cleanup failed |
| F22 | Qualification subject missing | Fail closed |
| F23 | Subject kind unsupported | Fail closed |
| F24 | Artifact fingerprint mismatch (subject vs artifact / evidence) | Fail closed |
| F25 | Scope fingerprint mismatch (subject vs scope / evidence / artifact binding) | Fail closed |
| F26 | Tenant mismatch across subject / artifact / qualification evidence | Fail closed |
| F27 | Legacy acquisition projection cannot produce valid canonical subject | Fail closed; no fallback legacy-only provider path |
| F28 | Qualification provider returns evidence for different subject than requested | Contract error; fail closed |
| F29 | Stale subject or artifact revision vs current fingerprint | Reject |
| F30 | Duplicate qualification request with conflicting subject identity/fingerprint | Reject deterministically |

---

## 11. Authority propagation

| Layer | What is carried | Widening allowed? |
| ----- | --------------- | ----------------- |
| Upstream acquisition / recovery authority | Max envelope for A2 candidacy | N/A |
| `ScopedIntegrationAdaptationScope` | Adaptation envelope | **No** — ⊆ upstream |
| `CredentialUseGrant` + scope validation | Secret use targets | **No** — intersection with adaptation/network |
| `NetworkEgressAllowlist` | Egress hosts | **No** — ⊆ upstream approved + attested |
| Execution dispatch payload | Runtime operation scope | **No** — ⊆ qualified artifact + admission |

AW must not mint `ParentExecutionAuthority` or trusted runtime roots.

---

## 12. Tenant propagation and isolation audit

**Verdict:** **PASS AT DESIGN INVARIANT LEVEL** — tenant is mandatory across A2; platform-wide **FRZ-TEN PASS is not claimed**.

**Invariant:**

```text
tenant_request == tenant_candidate == tenant_integration == tenant_credential_grant
  == tenant_qualification_evidence == tenant_execution
tenant_child_scope ⊆ tenant_parent_scope
```

Tenant A material must never authorize tenant B execution.

### Roadmap tenant questions (§1.2 items 1–16) — design answers

| # | Question | A2 design answer |
| - | -------- | ---------------- |
| 1 | Carry tenant identity? | **Yes** — required on request, scope, artifact, grant ref, qualification, dispatch |
| 2 | Where introduced? | Worker/recovery context at orchestration ingress |
| 3 | Who owns? | AW carries correlation; Integrations validates on adaptation port; Execution on dispatch |
| 4 | Propagated through boundaries? | **Yes** — explicit fields each hop; validated fail-closed |
| 5 | Child widen tenant? | **No** — strategy output continuity checks (mirror INT-CONFIG `_verify_strategy_output_continuity`) |
| 6 | Tenant disappear? | **No** — missing tenant fails closed |
| 7 | Missing → global/default? | **No** |
| 8 | State/cache scoped? | A2 default: no cross-tenant durable adaptation registry in AW |
| 9 | Provider/config scoped? | Integration target resolved under request tenant |
| 10 | Credentials scoped? | Grant must match tenant + integration + targets |
| 11 | Evidence preserves tenant? | Qualification evidence carries tenant; no cross-tenant truth |
| 12 | Retry/resume preserve tenant? | Episode correlation must re-validate same tenant |
| 13 | Recovery wrong tenant? | Reject |
| 14 | Plugins rewrite tenant? | Forbidden; mismatch → reject |
| 15 | Cross-tenant explicit? | Out of A2 scope unless future governed contract — not implicit |
| 16 | Adversarial A→B tested? | **Planned** in AW-7C-CERT (not executed in P1) |

**Adversarial test plan (CERT):** A candidate + B execution; A grant + B request; A qualification + B dispatch; missing tenant; strategy output tenant rewrite; host/credential widening.

---

## 13. Credential model

- Contracts carry **grant references / grant ids** only.
- Material resolution **only** through `ScopedCredentialBroker.resolve(..., grant=CredentialUseGrant)` with `assert_grant_matches_scope` and network intersection in `integrations/credentials/scope_validation.py`.
- Target host must satisfy **both** `NetworkEgressAllowlist` and grant target scope — effective scope = **intersection**, never union.
- Forbidden: raw secrets on AW public contracts; AW secret broker.

---

## 14. Egress model

Reuse **`NetworkEgressHost`**, **`NetworkEgressAllowlist`**, **`SandboxSecurityConfigurable`**, **`SandboxSecurityCapable`**, **`SandboxSecurityCapabilities`** — no duplicate A2 firewall types.

Execution requires substrate attestation matching declared allowlist (`network_egress_allowlist_enforced`, `enforced_network_hosts`). Missing attestation → fail closed (F14).

A2 network scope ⊆ upstream approved scope ⊆ attested enforcement.

---

## 15. Qualification boundary

**Owner:** Capability Qualification domain (`CapabilityQualificationProvider`); **composition owner** for provider selection — exactly one mechanism (see §7.5).

Integrations/AW **submit** qualification requests carrying `CapabilityQualificationSubject` (directly or via deterministic projection from artifact or successful acquisition result); provider **evaluates** subject facts; Governance **admits**; Execution **runs**.

Flow for A2:

```text
ScopedIntegrationAdaptationArtifact
  → project_adaptation_qualification_subject()
  → CapabilityQualificationRequest (canonical subject-oriented shape)
  → CapabilityQualificationProvider
  → CapabilityQualificationResult / Evidence
```

If qualification provider unavailable → orchestration stops at F10 (no execution).

---

## 16. CodeCraft / A1 boundary

CodeCraft may participate in **future** artifact synthesis as a **capability**, not as A2 execution authority.

Any synthesized output still flows: scope → qualification → Governance → `CanonicalExecutionIntakePort`.

**Explicit:** do not route A2 through `WorkerEphemeralCapabilityExecutionService`.

---

## 17. Integration Catalog boundary

A2 = **temporary scoped adapted use**. Default: **no** durable global integration registration.

Durable catalog/provider mutation → **A3 / INT-CONFIG / control-plane mutation** paths — not broadened A2.

---

## 18. Strong typing rules

Public semantic boundaries **must use** frozen dataclasses, typed Protocols, enums, existing typed IDs.

**Explicitly forbidden at boundaries (including P2 intent):**

- Untyped `object` as contract carrier
- `dict[str, Any]` / `Mapping[str, Any]` as pseudo-contract
- Reflection (`getattr`, `hasattr`) for semantic dispatch
- String-based strategy/provider selection in AW core
- Provider client objects or `Callable` as adapter contract
- `# type: ignore` / cast as design workaround

Document mentions of the above are **normative prohibitions**, not API shapes.

---

## 19. Pluginability / composition

| Element | Lock |
| ------- | ---- |
| Strategy SPI | `ScopedIntegrationAdaptationStrategy` in Integrations |
| Composition owner | Integrations adaptation service (strategy tuple injection at host) |
| AW core vendor branches | **Forbidden** |
| Duplicate strategy id | Fail closed at startup/composition |
| Ambiguous match | Fail closed |
| Global AW registry | **Forbidden** — reuse Integrations composition mechanisms; if insufficient, gap documented in P2 (do not invent in P1) |

Structural replaceability: alternate strategy implementation behind SPI without AW code changes.

---

## 20. Resource / cleanup ownership

| Phase | Owner |
| ----- | ----- |
| Request/candidate correlation | AW |
| Temporary strategy-internal resources | Creating strategy/provider/substrate |
| Immutable artifact | Integrations contract value; AW holds refs |
| Qualification evidence | Qualification domain |
| Execution lifecycle | Execution runtime |
| Sandbox/session | Substrate provider |

AW orchestration **does not** become physical resource owner because it initiated adaptation. Cleanup failure (F21) reported by owning domain.

---

## 21. Evidence / traceability lineage

Correlate across (typed ids):

`WorkerCapabilityNeed` → `WorkerCapabilityCandidate` → acquisition decision ref → `ScopedAdaptiveIntegrationExecutionRequest` → strategy/artifact ids → `CapabilityQualificationEvidence` → `CredentialUseEvidence` → `ExecutionId` / `RunId` / dispatch result.

Evidence **records truth only** — does not authorize, widen scope, or substitute Governance.

Post-effect evidence cannot retroactively justify pre-effect execution.

---

## 22. Forbidden bypasses (summary)

| Bypass | Status |
| ------ | ------ |
| Second Execution Engine | **Forbidden** |
| Direct ToolRuntime from AW | **Forbidden** |
| Direct Nexus import in AW production | **Forbidden** (existing gate) |
| A1 ephemeral service as A2 runtime | **Forbidden** |
| AW Integration registry | **Forbidden** |
| AW secret broker | **Forbidden** |
| Duplicate network policy model | **Forbidden** |
| Hidden qualification boolean on artifact | **Forbidden** |

---

## 23. Closed-world inventory (≤15 production files)

| # | File | Role |
| - | ---- | ---- |
| 1 | `intergrax/contracts/autonomous_work/capability_acquisition.py` | A2 disposition + candidate kinds |
| 2 | `intergrax/autonomous_work/capability_acquisition_service.py` | Emits `SCOPED_ADAPTATION_CANDIDATE` |
| 3 | `intergrax/autonomous_work/capability_acquisition_ports.py` | Discovery ports |
| 4 | `intergrax/autonomous_work/ephemeral_capability_execution.py` | A1 only — anti-pattern for A2 |
| 5 | `intergrax/autonomous_work/worker_execution_dispatch.py` | Canonical dispatch orchestration |
| 6 | `intergrax/contracts/autonomous_work/execution_dispatch.py` | Dispatch contracts |
| 7 | `intergrax/contracts/execution_intake.py` | `CanonicalExecutionIntakePort` |
| 8 | `intergrax/integrations/contracts/credential.py` | Grant/scope types |
| 9 | `intergrax/integrations/credentials/broker.py` | `ScopedCredentialBroker` |
| 10 | `intergrax/contracts/sandbox_network_egress.py` | Allowlist types |
| 11 | `intergrax/runtime/sandbox/contracts.py` | `SandboxSecurityCapabilities` |
| 12 | `intergrax/contracts/capability_qualification/provider.py` | Qualification SPI |
| 13 | `intergrax/contracts/capability_qualification/qualification_result.py` | Qualification outcomes |
| 14 | `intergrax/integrations/contracts/existing_capability_configuration.py` | Precedent for Integrations strategy SPI |
| 15 | `intergrax/integrations/existing_capability_configuration_service.py` | Precedent for strategy composition |

**Semantic role mapping (search terms):**

| Symbol | Role |
| ------ | ---- |
| `SCOPED_ADAPTATION_CANDIDATE` | AW-7A decision output |
| `ADAPTIVE_INTEGRATION` | Candidate kind |
| `A2_SCOPED_ADAPTIVE` | Autonomy level gate |
| `WorkerEphemeralCapabilityExecutionService` | A1 execution — not A2 |
| `CanonicalExecutionIntakePort` | Execution handoff sink |
| `ScopedCredentialBroker` | Secret material resolution |
| `NetworkEgressAllowlist` | Egress scope type |
| `SandboxSecurityCapabilities` | Attestation evidence |

---

## 24. Proposed implementation children

| ID | Scope |
| -- | ----- |
| **AW-7C-P2** | Typed AW orchestration request; Integrations `ScopedIntegrationAdaptationScope` + `ScopedIntegrationAdaptationStrategy` SPI + adaptation service; CQ `CapabilityQualificationSubject` + subject-oriented request/result/evidence evolution and V1→subject migration seam; deterministic acquisition/adaptation projections; scope/artifact fingerprints; no provider/sandbox/execution side effects |
| **AW-7C-P3** | One reference adaptation strategy; structural replaceability; tenant continuity; scope narrowing proofs |
| **AW-7C-P4** | Qualification coordinator wiring using canonical subject path; Governance/runtime admission; canonical Execution composition; broker + egress attestation; evidence correlation |
| **AW-7C-CERT** | Adversarial certification: authority non-amplification, tenant A→B, grant/expiry mismatch, host widening, ambiguity, stale qualification, bypass attempts, attestation gaps, provider failures, architecture regression gates |

---

## 25. FRZ evidence mapping (future — no PASS in P1)

| Family | Applicability |
| ------ | ------------- |
| FRZ-GOV-* | Governance admission on A2 handoff |
| FRZ-EXE-* | Canonical Execution path only |
| FRZ-CTR-* | Contract graph P2+ |
| FRZ-PLG-* / FRZ-RPL-* | Strategy SPI replaceability |
| FRZ-TEN-* | Tenant invariants + CERT adversarial proofs |
| FRZ-TYP-* / FRZ-TRC-* | Strong typing + lineage |
| FRZ-SEC-05/07 | Sandbox egress attestation consumption |
| FRZ-REG-02/03/08/09 | Architecture gates + qualification tests |

**new global FRZ PASS = 0; new FRZ-TEN PASS = 0**

---

## 26. STOP conditions

Stop and escalate **ARCHITECTURE DECISION REQUIRED** if implementation needs:

- Second Execution Engine or direct ToolRuntime from AW
- AW Integration registry or secret broker
- Duplicate network policy model
- Canonical Execution cannot express scoped work
- Widening `ParentExecutionAuthority`
- Tenant cannot traverse boundaries with strong types
- Integrations↔AW ownership conflict or reverse dependency
- Qualification owner unavailable without safe reuse
- Weak dict/Any contracts as only representation
- Change to frozen owner authority from prior accepted stages

P1-ARCH-R1 encountered **none** of the above.

---

## 27. Governance ≠ Execution (preserved)

Proposal (adaptation request) ≠ artifact ≠ qualification evidence ≠ Governance ALLOW ≠ Execution completion. AW orchestrates; Governance admits; Execution runs.

---

## 28. Ownership matrix (R1 closure — exactly-one semantic owner)

| Concern | Semantic owner |
| ------- | -------------- |
| A2 orchestration request semantics (`ScopedAdaptiveIntegrationExecutionRequest`) | **Autonomous Work** |
| `ScopedIntegrationAdaptationScope` | **Integrations** |
| `ScopedIntegrationAdaptationStrategy` | **Integrations** |
| `ScopedIntegrationAdaptationArtifact` | **Integrations** |
| Adaptation strategy composition | **Integrations** |
| `CapabilityQualificationSubject` | **Capability Qualification** |
| Qualification provider selection / composition | **Capability Qualification** |
| Governance admission | **Runtime / Governance** |
| Execution lifecycle | **Execution** |
| Credential material / use enforcement | **Credential domain** |
| Network / isolation enforcement | **Sandbox** |
| Evidence recording | **Evidence / observability consumers only** (no authority) |

No row uses shared, joint, or dual semantic ownership.

---

## 29. Dependency graph (R1 closure)

```text
AW
  → Integrations public adaptation contracts (ScopedIntegrationAdaptationScope, port, strategy SPI)
      → strategy implementations (Integrations providers / extensions)

AW
  → Capability Qualification public contracts (subject, request, provider)

Integrations adaptation artifact
  → deterministic qualification-subject projection
  → Capability Qualification

AW
  → canonical Governance / Execution boundary (existing ports)
```

**Forbidden reverse dependencies:**

- Integrations core → AW concrete orchestration implementation
- Capability Qualification core → AW concrete implementation
- Capability Qualification core → concrete Integration provider implementation
- Execution → AW-specific provider implementation

---

*End of AW-7C-P1-ARCH / P1-ARCH-R1 architecture lock.*

---

## AW-7C-P2 implementation record (READY FOR AUDIT)

| Item | Detail |
| ---- | ------ |
| **Baseline** | `9b89941ff695c7a01513e71d4e03fa7232883e98` |
| **Status** | READY FOR AUDIT — not CLOSED |
| **CQ** | `CapabilityQualificationSubject`, acquisition/adaptation projections, subject-oriented request/result/evidence/integrity/audit; `build_acquisition_qualification_request`; single `CapabilityQualificationService` |
| **Integrations** | `scoped_integration_adaptation.py`, `ScopedIntegrationAdaptationService`, scope/artifact fingerprints, strategy SPI |
| **AW** | `scoped_adaptive_integration.py` contracts + `WorkerScopedAdaptiveIntegrationOrchestrationService` → `QUALIFICATION_PENDING` |
| **Tests** | `test_aw_7c_p2_qualification_subject.py`, `test_scoped_integration_adaptation_service.py`, `test_scoped_adaptive_integration_service.py`, `test_aw_7c_p2_architecture_gates.py` |
| **Deferred** | P4 `qualify()` + Governance/Execution; CERT adversarial tenant global close |

## AW-7C-P2 carry-over hardening + AW-7C-P3 (READY FOR AUDIT)

| Item | Detail |
| ---- | ------ |
| **P2 baseline** | `d5e0b4b7dcfc0bd82d746f491c405cc73f8d73de` |
| **Status** | **READY FOR AUDIT** — not CLOSED (independent GitHub SHA audit required) |
| **Port ownership** | `WorkerScopedAdaptiveIntegrationOrchestrationPort.prepare(...)` (AW) ≠ `ScopedIntegrationAdaptationPort.adapt(...)` (Integrations); orchestration service consumes Integrations port only |
| **Target truth** | `ScopedIntegrationAdaptationTargetSource` + `SourceBackedScopedIntegrationAdaptationTargetResolver`; reference catalog `reference_scoped_integration_adaptation_target_source.py`; request-echo resolver removed in P4 |
| **Identity continuity** | Mandatory equality across request / scope / resolved target / artifact / artifact.scope (tenant, category, provider, resource_scope, candidate id/revision) |
| **Revision** | Exact `candidate_revision` continuity; **no** lexical `min_candidate_revision` / string ordering |
| **Operations** | `ScopedIntegrationAdaptationOperationId` extensible value object (no closed platform enum at SPI boundary) |
| **Reference strategy** | `intergrax/integrations/qualification/reference_scoped_integration_adaptation.py` — explicit DI only; **not** production default |
| **Replaceability proof** | Reference strategy A + alternate test strategy B; 0 → `STRATEGY_UNAVAILABLE`; >1 → `STRATEGY_AMBIGUOUS`; duplicate `strategy_id` → fail closed |
| **CQ legacy tests** | All direct `CapabilityQualificationRequest(` constructors migrated to `build_acquisition_qualification_request` / `build_subject_qualification_request` |
| **Tests (targeted)** | `test_scoped_integration_adaptation_service.py`, `test_scoped_adaptive_integration_service.py`, `test_aw_7c_p3_scoped_adaptive_integration.py`, `test_aw_7c_p3_architecture_gates.py`, CQ suites + migrated UCA consumers |
| **Remaining** | Independent SHA audit; CERT adversarial global close |

## AW-7C-P4 (READY FOR AUDIT)

| Item | Detail |
| ---- | ------ |
| **Baseline** | `7245bb1d6f2ce6b340cea5df43324672b4765974` |
| **Status** | **READY FOR AUDIT** — not CLOSED |
| **P3 carry-over** | Independent `ScopedIntegrationAdaptationTargetSource`; artifact ↔ resolved target continuity in adaptation service |
| **AW** | `WorkerScopedAdaptiveIntegrationExecutionCoordinator`, `scoped_adaptive_integration_execution.py` contracts (`ScopedAdaptiveIntegrationExecutionHandoff`, typed outcomes) |
| **CQ** | Revalidation before `CapabilityQualificationService.qualify()`; only `QUALIFIED` + lifecycle `ACCEPT` continues |
| **Governance / Execution** | `WorkerExecutionDispatchService` → `RootExecutionLaunchPort` → `CanonicalExecutionIntakePort` (no AW authority minting) |
| **Credential / Sandbox** | Execution-bound reference path: `ScopedCredentialBroker` after active `ExecutionId`; `validate_qualified_allowlist_attestation` on `SandboxSecurityCapabilities` |
| **Tests** | `test_aw_7c_p4_scoped_adaptive_integration_execution.py`, `test_aw_7c_p4_architecture_gates.py`, `test_aw_7c_p4_sandbox_attestation.py`, target-truth cases in `test_scoped_integration_adaptation_service.py` |
| **Next** | `AW-7C-CERT` — adversarial certification / global FRZ-TEN close |
| **P4 exact-SHA audit blockers (resolved in CERT)** | (A) `CapabilityQualificationDecision` bound on `ScopedAdaptiveIntegrationExecutionHandoff.accepted_qualification` + `validate_execution_bound_qualification_proof` at execution delegate; (B) `validate_handoff_credential_grant_identity` before `ScopedCredentialBroker.resolve_scoped`; (C) explicit `requested_operation`; (D) typed `ScopedAdaptiveIntegrationExecutionRuntimeEnvelope` (`CREDENTIAL_DENIED` ≠ `SANDBOX_SECURITY_UNSATISFIED`); (E) strong `ScopedIntegrationAdaptationArtifact` on operation port; (F) removed coordinator `_seen_idempotency_keys` — `execution_idempotency_key` is correlation intent only |

## AW-7C-CERT (READY FOR AUDIT)

| Item | Detail |
| ---- | ------ |
| **Baseline** | `6fda62ca792255e482895a574250af547cb18cba` |
| **Status** | **READY FOR AUDIT** — not CLOSED |
| **Integrated E2E** | `WorkerScopedAdaptiveIntegrationExecutionCoordinator` → CQ → `WorkerExecutionDispatchService` → root Governance → `ScopedAdaptiveIntegrationReferenceExecutionIntake` → execution-bound delegate → sandbox attestation → broker → adapted operation → `ScopedAdaptiveIntegrationExecutionRuntimeEnvelope` |
| **Idempotency** | No AW-local duplicate truth; no durable exactly-once claim; replay uses new canonical execution identity unless a future canonical execution owner supplies idempotency |
| **Tests** | `test_aw_7c_cert_scoped_adaptive_integration_execution.py` + P4/CQ/credential/sandbox targeted suites |
| **Tenant** | Local AW-7C chain adversarial cases in CERT tests; global `FRZ-TEN` PASS unchanged without checklist owner |
| **Remaining** | Independent GitHub SHA audit; roadmap/checklist global closure |

