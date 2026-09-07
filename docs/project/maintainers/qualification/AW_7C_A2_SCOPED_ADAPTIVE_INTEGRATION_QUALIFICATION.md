# AW-7C A2 Scoped Adaptive Integration — Prerequisite Qualification

**Verdict:** BLOCKED BY PREREQUISITE

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `aac75b8e8dfc82192595eaacd0a561249f47f171`

**Review HEAD (pre-docs):** `aac75b8e8dfc82192595eaacd0a561249f47f171`

**Task:** AW-7C qualification-first audit — no A2 production implementation. **P0-1 egress contract remediation** (2026-09-07): typed host scope + fail-closed substrate matching; physical provider qualification still blocked. **P0-2 secret broker remediation** (2026-09-07): purpose-scoped grant/broker contracts + enforcement at resolution boundary.

**P0-2 independent audit correction:** unenforced `max_uses` field removed. V1 bounded credential authority is time-bounded via `expires_at` only. Use-count restrictions require a future concurrency-safe lifecycle authority.

---

## 1. Verdict

```text
AW-7C: BLOCKED BY PREREQUISITE
```

AW-7 remains **IN PROGRESS**. AW-7C is **not** READY FOR IMPLEMENTATION.

Critical blockers:

1. **Host-scoped egress allowlist** — typed contract + fail-closed substrate matching **implemented** (P0-1); **physical** exact-host enforcement at a real hosted provider **still blocked** (no qualifying substrate in repo).
2. **Purpose-scoped secret brokering** — **implemented** (P0-2): `ScopedCredentialBroker` + `CredentialUseGrant` enforce tenant, execution, operation, integration, and target host scope at resolution; legacy `SecretsStoreCredentialResolver` tenant-only path preserved for P1.7.
3. **Allowlist egress physical qualification** — contract tests prove fail-closed resolver behavior; no test proves non-approved destination **physically denied** at enforced substrate under host-scoped policy (provider qualification blocked).

---

## 2. Prerequisite matrix

| AW-7C prerequisite | Verdict |
| --- | --- |
| runtime egress enforcement | **PARTIALLY REMEDIATED** (typed deny + allowlist contract; deny proof retained; allowlist requires substrate evidence) |
| host allowlist enforcement | **PARTIALLY REMEDIATED** (contract + fail-closed resolver; physical provider proof **BLOCKED**) |
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
| `tests/unit/autonomous_work/test_aw_7c_prerequisite_architecture_gates.py` | 4 passed (new) |
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

1. **Sandbox / CodeCraft substrate** — **DONE (contract P0-1)** typed host scope + fail-closed matching; **remaining:** real hosted provider with physical exact-host enforcement qualification.
2. **Integration / credential domain** — **DONE (P0-2)** purpose-scoped secret broker: tenant + credential_ref + integration identity + operation/purpose + allowed target scope; enforce at resolution; bounded lifetime.
3. **Qualification tests** — at least one test where approved host succeeds and unapproved host is **physically denied** at enforced substrate (not metadata-only).

Do **not** implement `WorkerSecretBroker`, `AWSecretStore`, or host filtering inside AW core.

---

## 11. Status

```text
AW-7C SECRET BROKER PREREQUISITE: IMPLEMENTED / AWAITING INDEPENDENT RE-AUDIT
AW-7C EGRESS PREREQUISITE: PARTIALLY REMEDIATED / PROVIDER QUALIFICATION BLOCKED
AW-7C: BLOCKED BY PREREQUISITE
AW-7:  IN PROGRESS
```
