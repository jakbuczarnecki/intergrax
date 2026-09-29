# AW-7C A2 Scoped Adaptive Integration — Prerequisite Qualification

**Verdict (current program state, post–AW-7C-P0-3B-PHYSQ reconciliation):** AW-7C-P0-3B-PHYSQ **CLOSED / independently accepted** @ `deaafee1a7612824bbe06dd7f0d403c3e490124b`; AW-7C-P0-3B **CLOSED / prerequisite satisfied**; AW-7C **CURRENT** (implementation/certification open). Sections 1–13 below retain **historical** prerequisite audit unless a later section states **current accepted state**.

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `4dff9d136085fe46fbffc342d1bfc690ce7a8d85`

**Review HEAD (pre-P0-3A):** `73d023d0a34102ec35e7f01ceae56129ea72ff5e`

**Task:** AW-7C qualification-first audit — no A2 production implementation. **P0-1 egress contract remediation** (2026-09-07): typed host scope + fail-closed substrate matching. **P0-2 secret broker remediation** (2026-09-07): purpose-scoped grant/broker contracts + enforcement at resolution boundary. **P0-3 hosted provider inventory** (2026-09-07): canonical sandbox-host audit + `SandboxSecurityConfigurable` admission seam. **P0-3A E2B adapter** (2026-09-07): security-qualified `E2bSandboxHostBackend` + provider-state attestation; physical qualification pending operator credentials.

**P0-2 independent audit correction:** unenforced `max_uses` field removed. V1 bounded credential authority is time-bounded via `expires_at` only. Use-count restrictions require a future concurrency-safe lifecycle authority.

---

## AW-7C implementation architecture lock (current)

| Field | Value |
| ----- | ----- |
| **Stage** | AW-7C-P1-ARCH **CLOSED / architecture lock accepted through R1**; AW-7C-P1-ARCH-R1 **CLOSED / independently accepted** @ `fa5853ec009d015cb52c9de15dd7e283932a9b4d` |
| **Parent** | AW-7C **CURRENT** |
| **Prerequisites** | AW-7C-P0-3B + AW-7C-P0-3B-PHYSQ **CLOSED / accepted** |
| **Purpose** | Contract-boundary and ownership lock for A2 scoped adaptive integration execution — **no A2 production implementation** |
| **Canonical architecture** | [`AW_7C_SCOPED_ADAPTIVE_INTEGRATION_EXECUTION.md`](../architecture/AW_7C_SCOPED_ADAPTIVE_INTEGRATION_EXECUTION.md) |
| **A2 execution service on baseline** | **None** (gap documented; P2+ implements) |
| **Next work** | **AW-7C-P2** + **AW-7C-P3** — **READY FOR AUDIT** (integrated hardening + reference strategy on `development`; independent SHA audit required) |
| **Global FRZ** | No PASS delta from P1-ARCH / R1 |

Sections 1–13 below retain **historical** prerequisite audit unless explicitly superseded above.

---

## AW-7C-P1-ARCH independent audit (blocked)

| Field | Value |
| ----- | ----- |
| **Verdict** | **BLOCKED** |
| **Exact audited SHA** | `e055a5bdb3c120b90dad546dc33cc0219824ced7` |
| **Reason 1** | Shared semantic owner for adaptation scope (`ScopedAdaptiveIntegrationScope` documented as AW + Integrations shared contract) — violates exactly-one owner (FRZ-OWN-01) |
| **Reason 2** | Qualification subject / `CapabilityQualificationRequest` carry model for A2 artifacts left to P2 — material architecture decision deferred |

## AW-7C-P1-ARCH-R1 independent audit (accepted)

| Field | Value |
| ----- | ----- |
| **Verdict** | **CLOSED / independently accepted** |
| **Exact audited SHA** | `fa5853ec009d015cb52c9de15dd7e283932a9b4d` |
| **Scope** | Docs-only architecture remediation — **no** production code, contracts, or tests |
| **Blocker A closure** | `ScopedIntegrationAdaptationScope` owned by **Integrations** (`intergrax/integrations/contracts/`); AW supplies immutable instance on orchestration request only |
| **Blocker B closure** | `CapabilityQualificationSubject` owned by **Capability Qualification**; deterministic projections from `ScopedIntegrationAdaptationArtifact` and from successful `CapabilityAcquisitionResult`; single UCA-4 mechanism; V1→subject migration direction locked |
| **Parent effect** | **AW-7C-P1-ARCH** = **CLOSED**; **AW-7C** = **CURRENT**; **AW-7C-P2** = **READY FOR AUDIT**; **AW-7C-P3** = **READY FOR AUDIT**; **AW-7C-P4** = **NEXT / NOT ENTERED** |
| **Global FRZ** | **new global FRZ PASS = 0**; **new FRZ-TEN PASS = 0** |
| **R1 gates (mechanical)** | 38 passed; 4 passed — architecture-doc gate only; not code/behavior proof of P2+ |

## AW-7C-P3 scoped qualification evidence (READY FOR AUDIT)

| Field | Value |
| ----- | ----- |
| **Status** | **READY FOR AUDIT** — not CLOSED |
| **Scoped evidence** | Replaceability, resolver negatives, scope narrowing, tenant-local adversarial rejection, pure path to `QUALIFICATION_PENDING` without `qualify()` / Governance / Execution |
| **FRZ (scoped; global statuses OPEN)** | FRZ-OWN-01..03, FRZ-CTR-01..06, FRZ-TYP-01..04, FRZ-TYP-06, FRZ-PLG-01..02, FRZ-RPL-01..02, FRZ-GOV-09, FRZ-TRC-10; FRZ-TEN-01, FRZ-TEN-02, FRZ-TEN-07, FRZ-TEN-10, FRZ-TEN-11, FRZ-TEN-12 |
| **P3 carry-over (P4)** | `ScopedIntegrationAdaptationTargetSource` + reference catalog; request-echo resolver removed |
| **Deferred** | CERT global tenant close |

## AW-7C-P4 scoped qualification + execution evidence (READY FOR COMBINED INDEPENDENT ACCEPTANCE)

| Field | Value |
| ----- | ----- |
| **Status** | **READY FOR COMBINED INDEPENDENT ACCEPTANCE** — P4 exact-SHA audit blockers remediated in **AW-7C-CERT** |
| **Scoped evidence** | CQ `CapabilityQualificationDecision` carried on handoff; execution-bound `validate_execution_bound_qualification_proof`; Governance negatives; integrated reference intake E2E |
| **FRZ (scoped; global statuses OPEN)** | FRZ-EXE-01..07, FRZ-GOV-01..05/07/09, FRZ-SEC-02/03/05/07, FRZ-TRC-01/03/04/06/10, FRZ-TEN-01/02/07/10/11/12 |
| **Deferred** | Global FRZ-TEN close |

## AW-7C-CERT adversarial qualification (READY FOR AUDIT)

| Field | Value |
| ----- | ----- |
| **Status** | **READY FOR AUDIT** — not CLOSED |
| **Baseline** | `6fda62ca792255e482895a574250af547cb18cba` |
| **Proof binding** | `accepted_qualification` + `validate_execution_bound_qualification_proof` before credential/sandbox/operation |
| **Credential** | `validate_handoff_credential_grant_identity`; broker sees canonical `ExecutionId` via grant factory |
| **Operation** | `requested_operation` on P4 request/handoff; must ⊆ `permitted_operations`; credential scope uses same operation value |
| **Failure typing** | `ScopedAdaptiveIntegrationExecutionRuntimeEnvelope` — sandbox ≠ credential |
| **Idempotency** | No process-local duplicate authority in coordinator |
| **Tests** | `test_aw_7c_cert_scoped_adaptive_integration_execution.py` |

---

## 1. Verdict

```text
AW-7C: BLOCKED BY PREREQUISITE
```

AW-7 remains **IN PROGRESS**. AW-7C is **not** READY FOR IMPLEMENTATION.

Critical blockers:

1. **Host-scoped egress allowlist** — typed contract + fail-closed substrate matching **implemented** (P0-1); `SandboxSecurityConfigurable` admission seam **implemented** (P0-3); E2B provider adapter **implemented** (P0-3A); **physical** exact-host enforcement **not yet executed** in this session (credential unavailable).
2. **Purpose-scoped secret brokering** — **implemented** (P0-2): `ScopedCredentialBroker` + `CredentialUseGrant` enforce tenant, execution, operation, integration, and target host scope at resolution; legacy `SecretsStoreCredentialResolver` tenant-only path preserved for P1.7.
3. **Allowlist egress physical qualification** — contract tests prove fail-closed resolver behavior; no test proves non-approved destination **physically denied** at enforced substrate under host-scoped policy (provider qualification blocked).

---

## 2. Prerequisite matrix

| AW-7C prerequisite | Verdict |
| --- | --- |
| runtime egress enforcement | **PARTIALLY REMEDIATED** (typed deny + allowlist contract; deny proof retained; allowlist requires substrate evidence) |
| host allowlist enforcement | **PARTIALLY REMEDIATED** (contract + fail-closed resolver + E2B adapter; physical provider proof **PENDING CREDENTIAL**) |
| fail-closed substrate | **PASS** (deny + allowlist modes) |
| opaque secret storage | **PASS** |
| purpose-scoped secret brokering | **PASS** (P0-2 contract + broker enforcement; admission port pluggable) |
| tenant secret isolation | **PASS** |
| integration runtime owner | **PASS** |
| governance/HITL owner | **PASS** |
| runtime enforcement evidence | **PARTIALLY REMEDIATED** (`network_egress_allowlist_enforced`, `enforced_network_hosts`; no physical provider qualification) |
| public execution boundary | **PASS** |
| Nexus isolation | **PASS** (production AW paths; doc debt flagged) |

---

## 3. Nexus boundary (frozen invariant)

| Check | Result |
| --- | --- |
| Nexus public contract | **NO** |
| Nexus role | internal Execution Engine mechanism |
| AW production import of `intergrax.nexus` / `intergrax.runtime.nexus` | **NO** (grep + new architecture gate) |
| AW contract exposure of Nexus types | **NO** |
| Public execution surface | `CanonicalExecutionIntakePort` → `ExecutionRuntime`; `WorkerExecutionDispatchService` (AW-5A) |

**Architecture debt (doc only, not AW-7C blockers):**

- `intergrax/contracts/autonomous_work/execution_authority.py` trust-boundary docstring lists `→ ParentExecutionAuthority → Nexus` as if Nexus were a public AW seam.
- `docs/project/architecture/AUTONOMOUS_WORK.md` pairs Unified Execution Runtime with Nexus in dependency tables; acceptable as internal mechanism reference but must not become AW import contract.

Correct dependency direction for future A2:

```text
AW / Integration adapter
  → canonical Execution public boundary
  → Execution Engine
  → internal Nexus implementation
```

---

## 4. Public execution boundary

| Concern | Canonical public owner | Public contract | Nexus involved internally? |
| --- | --- | --- | --- |
| execution admission | Runtime/Governance | `RootExecutionAuthorityAdmissionPort`, `WorkerExecutionAdmissionService` | possible beneath `ExecutionRuntime` |
| execution dispatch | Autonomous Work (orchestration only) | `WorkerExecutionDispatchService`, `CanonicalExecutionIntakePort` | no direct AW import |
| runtime authority | Runtime/Governance | `ParentExecutionAuthority`, `bind_active_execution_authority` | internal minting |
| execution identity | Unified Execution Runtime | `RunId`, `AttemptId`, `ExecutionId` (`intergrax/contracts/execution_identity.py`) | internal |
| tool/integration execution | Tools / Integrations registry | `ToolRegistry`, integration provider contracts, `TenantConnectionIntegrationFactoryRegistry` | via runtime tool invoker internally |

No ARCHITECTURE BLOCKER on public execution boundary — consumers need not import Nexus.

---

## 5. Egress qualification

### Canonical owner

| Layer | Owner |
| --- | --- |
| Policy shape | `CodeCraftProfile.network_egress` (`deny` \| `allowlist`) + `network_egress_allowlist` (`NetworkEgressHost`) |
| Substrate resolution | `intergrax/runtime/codecraft/substrate.py` |
| Security evidence | `SandboxSecurityCapable.security_capabilities()` → `SandboxSecurityCapabilities` (`network_egress_deny_enforced`, `network_egress_allowlist_enforced`, `enforced_network_hosts`) |
| Typed host scope | `intergrax/runtime/sandbox/network_egress.py` |
| Operation-level network surface | `SandboxSession` (`browser_fetch` allowlist) |
| Integration-layer HTTP allowlist | `AllowlistHttpClient` (not sandbox-enforced) |

### Egress owner table

| Requirement | Owner | Implemented? | Runtime enforced? | Evidence |
| --- | --- | ---: | ---: | --- |
| deny all network | Sandbox / CodeCraft substrate | yes | **partial** | `network_egress_deny_enforced`; local session ties deny to absence of `browser_fetch` |
| allow exact hosts | Sandbox / CodeCraft substrate | **yes (contract)** | **fail-closed without proof** | `network_egress_allowlist_enforced` + `enforced_network_hosts`; local sandbox never attests allowlist |
| block non-approved host | Hosted provider substrate | **contract only** | **no physical proof** | Resolver rejects superset/missing proof; no enforceable hosted provider in repo |
| DNS/IP bypass protection | partial (web URL policy) | partial | partial | `test_web_url_intake.py` blocks localhost/IP at app URL policy; not sandbox egress |
| sandbox/provider proof | SandboxSecurityCapable | yes (deny + allowlist fields) | yes (deny); allowlist **contract-only** | `test_aw_7b_gate.py`, `test_network_egress_allowlist_substrate.py` |
| fail closed if enforcement unavailable | CodeCraft substrate | yes | yes | `network_egress_requirement_unsatisfied` / `network_egress_allowlist_requirement_unsatisfied` |

### Key substrate gap (post P0-1)

Allowlist **contract and fail-closed resolver matching** are implemented. Remaining gap: no hosted provider in repo attests **physical** exact-host enforcement on the same execution channel as A2 generated code (`test_physical_allowlist_qualification_blocked`).

`SandboxSecurityCapabilities.enforced_network_hosts` is substrate enforcement evidence — providers must not echo requested profile scope.

### Egress verdict

**PARTIALLY REMEDIATED** — typed host scope + mode-aware fail-closed substrate matching implemented; **physical** exact host allowlist enforcement at a real provider substrate **still blocked**.

---

## 6. Secret qualification

### Canonical store

| Component | Path |
| --- | --- |
| Store contract | `intergrax/integrations/contracts/secrets_store.py` (`SecretsStore`) |
| Opaque ref | `CredentialRef` (`intergrax/integrations/contracts/credential.py`) |
| Resolver | `SecretsStoreCredentialResolver` |
| Tenant rehydration | `TenantConnectionRehydrator`, tenant connection factories |

### Secret owner table

| Requirement | Canonical owner | Implemented? | A2-ready? |
| --- | --- | ---: | ---: |
| opaque secret reference | `CredentialRef` | yes | yes |
| secret material not in AW contract | AW contracts | yes | yes |
| tenant-scoped resolution | `SecretsStoreCredentialResolver._assert_tenant_scope` | yes | yes |
| purpose/operation scoped resolution | `ScopedCredentialBroker` + `CredentialUseGrant.operation` | **yes** | **yes** |
| host/integration binding | `CredentialUseGrant` + `CredentialUseScope` + `NetworkEgressAllowlist` | **yes** | **yes** |
| bounded lifetime | `CredentialUseGrant.expires_at` + `TimeProvider` | **yes** | **yes** |
| no environment inheritance | scoped broker path | **yes** (no env fallback) | **yes** for scoped path |
| audit/correlation without secret exposure | `CredentialUseEvidence` via broker result | **yes** | **yes** |

### Secret broker answer

> Does the current public runtime provide purpose-scoped, tenant-bound, execution-bound secret brokering suitable for generated A2 code?

**YES (contract path).** `ScopedCredentialBroker` validates `CredentialUseGrant` against `CredentialUseScope` (tenant, execution, operation, integration, target host scope, expiry), requires `CredentialScopeAdmissionPort` ALLOW, then resolves via `SecretsStoreCredentialResolver`. Legacy P1.7 tenant-only resolution remains for durable integration flows that do not opt into scoped grants.

### Secret verdict

**PASS (P0-2)** — persistence ≠ brokering; scoped path enforced at resolution boundary.

---

## 7. Other owners

| Owner | Canonical surface | A2-ready? |
| --- | --- | --- |
| Integration runtime | Provider integrations, `TenantConnectionIntegrationFactoryRegistry`, `intergrax/runtime/vendor_knowledge/` | yes (durable path); ephemeral A2 adapter path **not implemented** |
| Governance | `MeaningfulSideEffectAuthorizationBoundary.authorize_and_execute` | yes |
| HITL | `GovernedContinuationGrantCoordinator`, CodeCraft HITL notes | yes |
| Sandbox | `SandboxSecurityCapable`, `resolve_craft_sandbox` | deny + allowlist contract (physical allowlist **blocked**) |
| Execution public boundary | `CanonicalExecutionIntakePort` / `ExecutionRuntime` | yes |

A2 input eligibility (AW-7A) is **defined** in contracts:

- `CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE`
- `WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION`
- `WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE`

No A2 execution port or service exists yet (by design for this task).

---

## 8. Contract tests / runtime evidence

| Requirement | Status |
| --- | --- |
| Contract tests for HTTP adapter against local/mock endpoint | partial — `AllowlistHttpClient` exists; no A2 adapter contract suite |
| Physical deny of non-approved network destination at substrate | **blocked** — no enforceable hosted provider; contract tests only |
| Runtime evidence fields for A2 | **partial** — allowlist scope fingerprint + enforced host evidence fields; secret scope correlation via `CredentialUseEvidence` |

Existing evidence (AW-7B / CodeCraft): `CraftSubstrateCapabilities` (`provider_id`, `resolved_tier`, `network_egress_enforced`, `network_egress_deny_enforced`, `network_egress_allowlist_enforced`, `enforced_network_hosts`).

---

## 9. Qualification tests

| Suite | Result |
| --- | --- |
| `tests/unit/runtime/codecraft/test_aw_7b_gate.py` | passed |
| `tests/unit/runtime/codecraft/test_network_egress_allowlist_substrate.py` | 10 passed, 1 skipped (physical provider blocked) |
| `tests/unit/runtime/sandbox/test_network_egress_contract.py` | passed |
| `tests/unit/codecraft/test_profile_network_egress.py` | passed |
| `tests/unit/integrations/credentials/test_p1_7_credential_ref.py` | passed |
| `tests/unit/integrations/credentials/test_scoped_credential_broker.py` | passed (P0-2) |
| `tests/unit/integrations/credentials/test_credential_domain_architecture_gates.py` | passed (P0-2) |
| `tests/unit/autonomous_work/test_worker_execution_dispatch_architecture_gates.py` | passed |
| `tests/unit/autonomous_work/test_ephemeral_capability_execution_architecture_gates.py` | passed |
| `tests/unit/autonomous_work/test_worker_capability_acquisition_architecture_gates.py` | passed |
| `tests/unit/runtime/sandbox/test_sandbox_security_configurable_conformance.py` | passed (P0-3) |
| `tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py` | **1 failed** (pre-existing: `execution_environment_authority_unavailable` in `test_valid_sandbox_reaches_provider`; unrelated to AW-7C gate scope) |

**Batch command:**

```powershell
uv run pytest tests/unit/runtime/codecraft/test_aw_7b_gate.py `
  tests/unit/integrations/credentials/test_p1_7_credential_ref.py `
  tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py `
  tests/unit/autonomous_work/test_worker_execution_dispatch_architecture_gates.py `
  tests/unit/autonomous_work/test_ephemeral_capability_execution_architecture_gates.py `
  tests/unit/autonomous_work/test_worker_capability_acquisition_architecture_gates.py `
  tests/unit/autonomous_work/test_aw_7c_prerequisite_architecture_gates.py -q
```

**Counts:** 82 collected → 81 passed, 1 failed (pre-existing P0-SAFETY-7 unrelated failure).

---

## 10. Recommended prerequisite hardening (platform tasks, not AW-7C)

1. **Sandbox / CodeCraft substrate** — **DONE (contract P0-1)** typed host scope + fail-closed matching; **DONE (P0-3 seam)** `SandboxSecurityConfigurable` + pre-admission fail-closed for allowlist profiles; **remaining:** provider-specific adapter hardening + physical qualification (E2B preferred).
2. **Integration / credential domain** — **DONE (P0-2)** purpose-scoped secret broker: tenant + credential_ref + integration identity + operation/purpose + allowed target scope; enforce at resolution; bounded lifetime.
3. **Qualification tests** — at least one test where approved host succeeds and unapproved host is **physically denied** at enforced substrate (not metadata-only).

Do **not** implement `WorkerSecretBroker`, `AWSecretStore`, or host filtering inside AW core.

---

## 11. Status

```text
AW-7C SECRET BROKER PREREQUISITE: PASSED / independently verified
AW-7C EGRESS PREREQUISITE: IMPLEMENTED / PHYSICAL QUALIFICATION HARNESS READY — WAITING FOR REAL PROVIDER EXECUTION
AW-7C P0-3A: IMPLEMENTED / PHYSICAL QUALIFICATION HARNESS READY — WAITING FOR REAL PROVIDER EXECUTION
AW-7C P0-3A-02A: IMPLEMENTED / HARNESS READY — WAITING FOR REAL PROVIDER EXECUTION
AW-7C P0-3A-03A: IMPLEMENTED / HARNESS READY — WAITING FOR REAL PROVIDER EXECUTION
AW-7C: BLOCKED BY PREREQUISITE
AW-7:  IN PROGRESS
```

---

## 12. P0-3 — Hosted provider inventory (physical allowlist qualification)

**P0-3 verdict:**

```text
AW-7C P0-3: IMPLEMENTATION REQUIRED — e2b (primary), modal, daytona
```

**Qualified provider:** `NONE` (no in-repo adapter satisfies configurable + attestable + physical enforcement).

### Provider matrix

| Provider | Real SDK/API adapter? | Session creation supports egress policy? | Exact host allowlist? | Same exec channel? | Can attest? | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **E2B** | **NO** — `HttpSandboxHostBackend` + generic `POST /sessions` (`intergrax/integrations/_shared/p7/factories.py`) | **NO** — no `network` / `allowOut` payload | **NO** in repo | **YES** (would be `HostedSandboxSession.execute` → `backend.exec`) | **NO** — not `SandboxSecurityCapable` | **IMPLEMENTATION REQUIRED** — external API documents `network.allowOut` / `network.denyOut` domain egress ([E2B network docs](https://docs.e2b.dev/network/internet-access)) |
| **Modal** | **NO** — same HTTP placeholder | **NO** | **NO** in repo | **YES** | **NO** | **IMPLEMENTATION REQUIRED** — external SDK documents `outbound_domain_allowlist` / `outbound_cidr_allowlist` ([Modal sandbox networking](https://modal.com/docs/guide/sandbox-networking)) |
| **Daytona** | **NO** — same HTTP placeholder | **NO** | **NO** in repo | **YES** | **NO** | **IMPLEMENTATION REQUIRED** — external API documents `domainAllowList` / `networkAllowList` ([Daytona network limits](https://www.daytona.io/docs/en/network-limits/)) |
| **Generic HTTP** (`HttpSandboxHostBackend`) | HTTP shim only | **NO** | **NO** — must not attest allowlist | **YES** | **NO** | **UNSUPPORTED** — contract tests only |
| **Other** | — | — | — | — | — | none registered |

### Canonical contract (post P0-3 seam)

| Surface | Status |
| --- | --- |
| `SandboxHostBackend` | unchanged — legacy `create_session()` preserved |
| `SandboxSecurityRequirements` + `NetworkEgressAllowlist` | **PASS** (P0-1) |
| `SandboxSecurityConfigurable.create_session_with_security()` | **added** (P0-3) — required for allowlist admission |
| `SandboxSecurityCapable.security_capabilities()` | **PASS** — attestation seam; `enforced_network_hosts` must not echo request |
| Breaking changes | **none** for legacy providers; allowlist profiles fail closed without `SandboxSecurityConfigurable` |

Configuration flow (target):

```text
CodeCraftProfile
  → SandboxSecurityRequirements
  → HostedSandboxSession.open(security_requirements=…)
  → SandboxSecurityConfigurable.create_session_with_security()
  → [missing] provider SDK/API session configuration
  → [missing] provider runtime + attestation
```

### External capability notes (not in-repo proof)

| Provider | Documented egress API | Exact-host semantics | Redirect / DNS caveats |
| --- | --- | --- | --- |
| E2B | `network.allowOut` domains + `denyOut` default-deny | domain filter on HTTP:80 (Host) and TLS:443 (SNI); wildcards supported externally — Intergrax V1 is exact-host only | redirect destination evaluated by provider policy; QUIC/HTTP3 not domain-filtered per vendor docs |
| Modal | `outbound_domain_allowlist` (TLS:443 SNI) + `outbound_cidr_allowlist` | domain allowlist beta; non-TLS blocked unless CIDR allowlisted | runtime `updateNetworkPolicy` replaces policy atomically |
| Daytona | `domainAllowList` XOR `networkAllowList` (CIDR) at create | domain list for web ports; CIDR list IPv4-only | runtime `POST /sandbox/{id}/network-settings` behind feature flag |

### Physical test

```text
NOT RUN — NO QUALIFYING IN-REPO PROVIDER
```

Planned qualification (next task, one canonical provider — E2B preferred):

| Check | Plan |
| --- | --- |
| Approved endpoint A | deterministic public HTTPS endpoint (e.g. `https://httpbin.org/get` or vendor test infra) |
| Unapproved endpoint B | second deterministic public HTTPS endpoint |
| Execution channel | `HostedSandboxSession.execute("run_python", …)` network call inside sandbox |
| Redirect escape | allowed A returning redirect to B must be denied by substrate |
| Markers | `@pytest.mark.integration`, `@pytest.mark.external`, `@pytest.mark.sandbox_provider` |

Contract placeholder: `test_physical_allowlist_qualification_blocked` remains **skipped**.

### Fail-closed (verified)

| Scenario | Result |
| --- | --- |
| Unsupported provider (`HttpSandboxHostBackend`, plain `SandboxHostBackend`) + allowlist profile | **denied** — `network_egress_allowlist_requirement_unsatisfied`; `create_session()` not called |
| Policy rejection at provider | not exercised — no real adapter |
| Missing attestation (`SandboxSecurityCapable` absent or `network_egress_allowlist_enforced != True`) | **denied** |
| Scope mismatch (`enforced ⊄ requested`) | **denied** |

### Nexus (P0-3)

```text
Nexus touched: NO
Nexus public contract: NO
Sandbox → Nexus: NO
AW → Nexus: NO
```

### P0-3 tests

```powershell
uv run pytest tests/unit/runtime/codecraft/test_network_egress_allowlist_substrate.py `
  tests/unit/runtime/sandbox/test_sandbox_security_configurable_conformance.py `
  tests/unit/runtime/sandbox/test_network_egress_contract.py `
  tests/unit/runtime/codecraft/test_aw_7b_gate.py -q
```

**Counts:** 50 passed, 1 skipped (`test_physical_allowlist_qualification_blocked`).

### P0-3 files

```text
intergrax/runtime/sandbox/contracts.py
intergrax/runtime/sandbox/hosted_session.py
intergrax/runtime/sandbox/hosted_resolver.py
intergrax/runtime/codecraft/substrate.py
tests/unit/runtime/sandbox/test_sandbox_security_configurable_conformance.py
tests/unit/runtime/codecraft/test_network_egress_allowlist_substrate.py
docs/project/maintainers/qualification/AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md
```

---

## 13. P0-3A — E2B exact-host allowlist adapter

**P0-3A verdict:**

```text
AW-7C P0-3A: IMPLEMENTED / PHYSICAL QUALIFICATION HARNESS READY — WAITING FOR REAL PROVIDER EXECUTION
```

**P0-3A-01 (2026-09-07):** SDK `exit_code` mapping corrected — numeric zero preserved at E2B transport boundary (`SdkE2bSandboxApiClient` / legacy payload helper). Physical qualification remains pending if credentials unavailable.

**SDK/API source of truth:** `e2b` Python SDK (installed for dev session; optional extra `integrations-e2b`); control plane `POST https://api.e2b.app/sandboxes` with `network.allowOut` / `network.denyOut`; attestation via `GET /sandboxes/{sandboxID}` → `network.allowOut` / `network.denyOut` (SDK: `sandbox.get_info().network`).

**Mapping (V1 qualified scope):**

| Intergrax | E2B |
| --- | --- |
| `NetworkEgressHost` exact `https` host | `network.allowOut` domain entry (`hostname` only) |
| default deny | `network.denyOut: ["0.0.0.0/0"]` at create time |
| unsupported | `http`, non-443 ports, wildcards, CIDR in Intergrax authority |
| scheme semantics | E2B filters TLS:443 (SNI) and HTTP:80 (Host); adapter documents HTTPS:443-only qualification |

**Attestation:** provider `get_info().network` only — **no request echo**. Session-bound evidence via `SandboxSessionSecurityEvidenceProvider.session_security_capabilities(session_id)`.

**Physical test command:**

```powershell
uv run pytest tests/integration/providers/sandbox_host/e2b/ -q
```

**Harness (P0-3A-02A):** `tests/integration/providers/sandbox_host/e2b/qualification/` — causal-proof runner with C0 control, C3 qualified, and redirect-escape phases. Immutable evidence models; `SandboxNetworkProbe` abstraction; cleanup guaranteed via `try/finally` per phase.

**Attestation correlation (P0-3A-03A):** `ProviderAttestationCorrelation` evaluates `requested_scope == attested_scope == observed execution` via immutable `ProviderAttestationCorrelationEvidence`. Provider metadata alone is not qualification proof — correlation requires attestation **and** runtime probe evidence. Evaluator is provider-neutral (no E2B branching). Status remains **IMPLEMENTED / HARNESS READY** — not physically qualified without real provider execution.

**Physical session result:** harness tests **skip safely** when `E2B_API_KEY` / `INTERGRAX_E2B_API_KEY` unavailable — **not PASS**. Real provider execution required for egress qualification verdict.

**Unit tests:** `tests/unit/integrations/providers/sandbox_host/e2b/` — 18 passed.

**Known limitations:** domain filter is routing control per E2B docs (shared CDN/SNI caveats); UDP/QUIC not domain-filtered; DNS rebinding not verified; E2B may inject `8.8.8.8` DNS helper IP in raw `allowOut` (excluded from canonical enforced host evidence).

**Nexus:** untouched — qualification harness has architecture gate `test_no_nexus_dependency`.

---

## 14. P0-3B — Provider-neutral reference substrate physical qualification (2026-09-28)

**Architecture decision:** [`ADR_AW_7C_PROVIDER_NEUTRAL_PHYSICAL_SANDBOX_QUALIFICATION_BOUNDARY.md`](../architecture/ADR_AW_7C_PROVIDER_NEUTRAL_PHYSICAL_SANDBOX_QUALIFICATION_BOUNDARY.md)

**Corrected model:**

| Gate | Meaning |
|------|---------|
| **Capability qualification (AW-7C)** | Platform contract + **physical** kernel enforcement via qualification-only Linux reference substrate (`tests/integration/runtime/sandbox/reference_substrate/`). Traverses `HostedSandboxSession` → `ReferenceSandboxBackend` → network namespace + nftables/iptables. |
| **Provider qualification (PROD-Q)** | Concrete E2B / Modal / Daytona production correctness — **mandatory before production activation** of that provider. |

**E2B physical provider qualification:** **DEFERRED TO PROD-Q / NOT ESTABLISHED** for AW-7C closure. Historical P0-3A E2B harness attempts and skips remain valid historical evidence — not deleted.

**Reference substrate physical qualification:** harness + ADR **implemented** @ baseline `a59744517b92847f55def1db22826d17d89ee155`.

**Historical pre-PHYSQ state:** P0-3B was **BLOCKED — LOCAL PHYSICAL QUALIFICATION ENVIRONMENT UNAVAILABLE** (operator WSL2/Linux with root/CAP_NET_ADMIN required; session environment had no usable Linux distro with `python3`). **No mock PASS.** Independent exact-GitHub-SHA audit was required after physical run.

**Current accepted state (post–independent PHYSQ audit):**

```text
AW-7C-P0-3B-PHYSQ: CLOSED / independently accepted @ deaafee1a7612824bbe06dd7f0d403c3e490124b
AW-7C-P0-3B: CLOSED / prerequisite satisfied
AW-7C: CURRENT
EBH-3: PLANNED / NOT ENTERED
EBH-4: PLANNED / NOT ENTERED
```

See § **AW-7C-P0-3B-PHYSQ** for machine evidence, execution reference, and SHA-256.

**Shared harness extraction:** provider-neutral modules under `tests/integration/providers/sandbox_host/qualification/` (models, probes, runner, attestation correlation). E2B integration tests consume the shared harness unchanged semantically.

**Threat-model boundary:** reference proof establishes kernel egress enforcement and redirect blocking on owned topology (`allowed.test` / `denied.test`). It does **not** claim public DNS rebinding, CDN churn, or external SaaS isolation — those remain provider qualification scope.

**Tenant isolation audit (reference substrate only):** **N/A — WITH EVIDENCE** — synthetic qualification IDs only; no tenant provider selection, credentials, or persistence.

**Historical AW-7C-P0-3B verdict (pre-PHYSQ):**

```text
AW-7C-P0-3B: BLOCKED — LOCAL PHYSICAL QUALIFICATION ENVIRONMENT UNAVAILABLE (harness READY FOR AUDIT)
AW-7C: BLOCKED BY PREREQUISITE (physical capability proof pending operator Linux/WSL2 run)
```

**Current accepted state:** see block above under **Current accepted state (post–independent PHYSQ audit)**.

**Production Python changes:** **0** (qualification + documentation only).

---

## AW-7C-P0-3B-PHYSQ — Provider-Neutral Physical Qualification — CLOSED / independently accepted

**Stage:** AW-7C-P0-3B-PHYSQ-DOCKER-RUN  
**Parent:** AW-7C-P0-3B  
**Program parent:** AW-7C  
**Next mandatory program parent after AW-7C:** EBH-3 (not entered)

**Qualified code baseline:** `d44a238dbe0d23dd9b864297175f54e663cfee3d`

**Execution environment:** Docker Desktop 4.38.0 (181591), Linux engine 27.5.1, privileged disposable `ubuntu:24.04` container (qualification execution environment only — not a runtime provider).

| Component | Value |
|-----------|--------|
| Container image | `ubuntu:24.04` |
| Kernel (container view) | `6.18.33.2-microsoft-standard-WSL2` |
| uid | `0` (root) |
| Firewall substrate | nftables v1.0.9 |
| Python | 3.12.3 |
| uv | 0.12.20 |
| `UV_PROJECT_ENVIRONMENT` | `/opt/intergrax-venv` (container-local; repo `.venv` untouched) |
| `UV_CACHE_DIR` | `/opt/uv-cache` |

**Execution path (unchanged):** `QualificationRunner` → `QualificationSandboxProvider` → `HostedSandboxSession` → `SandboxHostBackend` / `ReferenceSandboxBackend` → Linux network namespace → nftables → sandbox process. **Direct host-side substitute proof:** NO.

**Command #1 (dedicated PHYSQ):**

```text
uv run pytest tests/integration/runtime/sandbox/reference_substrate/test_reference_physical_egress_qualification.py -p no:xdist -q -rs
```

**Result #1:** `12 passed in 124.67s` — 0 failed, 0 skipped, 0 xfailed.

**Command #2 (full reference substrate):**

```text
uv run pytest tests/integration/runtime/sandbox/reference_substrate/ -p no:xdist -q -rs
```

**Result #2:** `80 passed in 125.46s` — 0 failed, 0 skipped.

### Causal proof (machine evidence)

| Phase | Outcome |
|-------|---------|
| **Control** | `allowed.test:18080` reachable; `denied.test:18081` reachable; `baseline_valid=true` |
| **Qualified** | allowed reachable; denied **not** reachable |
| **Redirect** | attempted; `escaped=false` |
| **Attestation** | `network_egress_allowlist_enforced=true`; kernel-derived effective allowlist includes `http://allowed.test:18080`, excludes `http://denied.test:18081`; `ProviderAttestationCorrelation` → `PASS` |

**Kernel evidence source:** `ip netns exec <netns> nft -n list ruleset` (reference substrate firewall readback — request echo not used as evidence).

**Cleanup:** all `cleanup_phases` records `destroyed=true`, `error=null`. Post-suite residual checks: no `igx-qual-*` netns; no qualification veth; no `/etc/netns/igx-qual-*`; no `10.200.42.3/32` harness residue; no listeners on `18080`/`18081`.

**Machine evidence (this run only):**

| Field | Value |
|-------|--------|
| `execution_reference` | `reference-physical-egress-causal-proof:057165d0-14a6-4758-915c-b8bc8cc44920` |
| `timestamp_utc` | `2026-09-29T08:26:05+00:00` |
| `scenario_id` | `reference-physical-egress-causal-proof` |
| `provider_identity` | `reference-substrate-qualification` |
| Generated path (session) | `.tmp/session/reference-physical-egress-qualification/reference-physical-egress-causal-proof-057165d0-14a6-4758-915c-b8bc8cc44920.json` |
| Committed copy | `docs/project/maintainers/qualification/AW_7C_P0_3B_PHYSQ_EVIDENCE.json` |
| SHA-256 (exact generated file) | `05be0b3000aee7f30e70c7fa8f8ee2083a3621a50f32d8e28b6ee553b7b60f39` |
| Secret scan | clean (no token/credential/operator path) |

**Threat model (scoped):**

- **Establishes:** real Linux netns execution; kernel-level egress enforcement; allowed endpoint reachability; denied endpoint blocked; redirect escape blocked; kernel attestation ↔ effective scope ↔ observed connectivity; harness cleanup on qualification substrate.
- **Does NOT establish:** E2B/Modal/Daytona production correctness; cloud IAM; hostile kernel escape; public DNS rebinding/CDN; production-provider cleanup guarantees (PROD-Q).

**Tenant Isolation Audit:** **N/A — WITH EVIDENCE** — synthetic qualification tenant/session IDs only; no production tenant semantics, credentials, or provider selection changes.

**FRZ scoped contribution (evidence candidate only):** FRZ-SEC-05, FRZ-SEC-07, FRZ-REG-02, FRZ-REG-03, FRZ-REG-06, FRZ-REG-08. **new global FRZ PASS = 0**. FRZ-SEC-06, FRZ-PRD-02, FRZ-PRD-05 remain **OPEN**.

**physical PASS candidate = YES**  
**independent audit = ACCEPTED** @ `deaafee1a7612824bbe06dd7f0d403c3e490124b`

**AW-7C-P0-3B-PHYSQ verdict:**

```text
AW-7C-P0-3B-PHYSQ: CLOSED / independently accepted @ deaafee1a7612824bbe06dd7f0d403c3e490124b
AW-7C-P0-3B: CLOSED / prerequisite satisfied
AW-7C: CURRENT
EBH-3: PLANNED / NOT ENTERED
EBH-4: PLANNED / NOT ENTERED
```

**Production Python changes (this stage):** **0**. **Platform contracts:** **0**.
