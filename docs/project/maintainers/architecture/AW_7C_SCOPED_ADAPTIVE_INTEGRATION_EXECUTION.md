# AW-7C-P1-ARCH — Scoped Adaptive Integration Execution Architecture Lock

## 1. Status and scope

| Field | Value |
| ----- | ----- |
| **Task** | `AW-7C-P1-ARCH` (architecture / contract-boundary lock only) |
| **Parent** | `AW-7C` — Scoped Adaptive Integration Execution |
| **Program baseline** | `88e423109d7321e273a68fcc4be5e950dd29ffba` (`development`) |
| **Source** | Scenario #24 GAP-03; roadmap §3.0.1 |
| **Status** | **READY FOR AUDIT** — design lock only; **no A2 production implementation** |
| **Production / tests / contracts in P1** | **0** |
| **Prerequisites** | AW-7C-P0-3B + AW-7C-P0-3B-PHYSQ **CLOSED / accepted** (egress substrate evidence) |
| **Next implementation** | `AW-7C-P2` only after independent P1-ARCH audit |

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
| Adaptation strategy SPI (`ScopedIntegrationAdaptationStrategy`) | **Integrations contracts** | **Integrations** pure selection (`select_scoped_adaptation_strategy` pattern, mirror INT-CONFIG) | None |
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
  → ScopedAdaptiveIntegrationScope (immutable envelope ⊆ upstream authority)
  → ScopedIntegrationAdaptationPort.adapt (Integrations)
       → select exactly one ScopedIntegrationAdaptationStrategy
       → ScopedIntegrationAdaptationArtifact (not permission)
  → CapabilityQualificationRequest/Result (existing qualification owner)
       → QUALIFIED artifact evidence only
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
- `ScopedAdaptiveIntegrationScope` (see §7.2)
- Upstream acquisition decision refs / `WorkerCapabilityAcquisitionDecision` snapshot
- Execution correlation (`RunId`, task refs as already used by dispatch)
- Authority **reference** compatible with existing execution admission (not raw ParentExecutionAuthority minting by AW)

**MUST NOT carry:** raw secrets, provider clients, sandbox handles, callables, unstructured bags used as contracts (forbidden patterns listed in §17).

### 7.2 Adaptation scope — `ScopedAdaptiveIntegrationScope`

**Owner:** AW + Integrations shared immutable contract (P2); enforced at AW orchestration boundary and re-validated at Integrations port ingress.

**Minimum fields (typed):**

- `tenant_id`
- Integration identity (`integration_category`, `provider_id`, resource/integration ref)
- Permitted operations (bounded set)
- `NetworkEgressAllowlist` (or explicit empty/deny) — **reuse canonical type**
- `credential_use_grant_ref` / grant id binding (reference only)
- Resource scope string (tenant-scoped)
- `expires_at` / lifetime bound
- Candidate id, configuration/adaptation revision constraints

**Invariants:**

```text
adaptation_scope ⊆ upstream_capability_authority_scope
execution_scope ⊆ adaptation_scope
effective_network_scope = intersection(upstream_network, grant_target, artifact_declared_hosts)
effective_credential_scope = intersection(grant_scope, adaptation_scope) — never union widening
```

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

### 7.5 Qualification — reuse existing boundary

**Owner:** `intergrax/contracts/capability_qualification/` — `CapabilityQualificationProvider`, `CapabilityQualificationRequest`, `CapabilityQualificationResult`, `CapabilityQualificationEvidence`.

**Semantic distinction (mandatory):**

| State | Meaning |
| ----- | ------- |
| Artifact exists | Strategy returned `ScopedIntegrationAdaptationArtifact` |
| Artifact qualified | `CapabilityQualificationOutcome.QUALIFIED` with integrity-valid evidence |
| Artifact authorized | Governance/runtime admission ALLOW for concrete execution |
| Executed | `CanonicalExecutionIntakePort` completed successfully |

**Gap note:** P2 must define how `CapabilityQualificationRequest` carries adaptation artifact identity (new qualified-subject fields) without inventing `qualified=True` on the artifact itself. No adequate subject type exists on baseline — **tracked P2 contract work**, not a hidden boolean.

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

## 10. Failure-mode matrix (F1–F21)

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

---

## 11. Authority propagation

| Layer | What is carried | Widening allowed? |
| ----- | --------------- | ----------------- |
| Upstream acquisition / recovery authority | Max envelope for A2 candidacy | N/A |
| `ScopedAdaptiveIntegrationScope` | Adaptation envelope | **No** — ⊆ upstream |
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

**Owner:** Capability Qualification domain (`CapabilityQualificationProvider`).

Integrations/AW **submit** qualification requests; provider **records** truth; Governance **admits**; Execution **runs**.

If qualification provider unavailable → orchestration stops at F10 (no execution).

P4 wires artifact fingerprint + scope snapshot into `CapabilityQualificationRequest` subject (P2 typing).

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
| **AW-7C-P2** | Typed AW + Integrations contracts; `ScopedIntegrationAdaptationStrategy` SPI; pure AW orchestration + Integrations adaptation service; strategy selection; no provider/sandbox/execution side effects |
| **AW-7C-P3** | One reference adaptation strategy; structural replaceability; tenant continuity; scope narrowing proofs |
| **AW-7C-P4** | Qualification wiring; Governance/runtime admission; canonical Execution composition; broker + egress attestation; evidence correlation |
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

P1-ARCH encountered **none** of the above.

---

## 27. Governance ≠ Execution (preserved)

Proposal (adaptation request) ≠ artifact ≠ qualification evidence ≠ Governance ALLOW ≠ Execution completion. AW orchestrates; Governance admits; Execution runs.

---

*End of AW-7C-P1-ARCH architecture lock.*
