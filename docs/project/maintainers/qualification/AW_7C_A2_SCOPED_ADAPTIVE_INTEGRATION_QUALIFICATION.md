# AW-7C A2 Scoped Adaptive Integration — Prerequisite Qualification

**Verdict:** BLOCKED BY PREREQUISITE

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `2104e4b463bd3d2c9b108037825d72bfde3339e8`

**Review HEAD (pre-docs):** `2104e4b463bd3d2c9b108037825d72bfde3339e8`

**Task:** AW-7C qualification-first audit — no A2 production implementation.

---

## 1. Verdict

```text
AW-7C: BLOCKED BY PREREQUISITE
```

AW-7 remains **IN PROGRESS**. AW-7C is **not** READY FOR IMPLEMENTATION.

Critical blockers:

1. **Host-scoped egress allowlist** — `CodeCraftProfile.network_egress` includes `"allowlist"` but substrate enforcement implements **deny-only** proof; no canonical sandbox/provider path enforces exact host allowlists for generated execution.
2. **Purpose-scoped secret brokering** — `SecretsStore` + `SecretsStoreCredentialResolver` resolve opaque refs with **tenant scope only**; `operation`, `execution_id`, host/integration binding, and bounded lifetime are not enforced at resolution time.
3. **Allowlist egress tests** — no qualification test proves a non-approved destination is **physically denied** at sandbox/substrate level under host-scoped policy (only deny-all / `browser_fetch` operation gating exists).

---

## 2. Prerequisite matrix

| AW-7C prerequisite | Verdict |
| --- | --- |
| runtime egress enforcement | **BLOCKED** (deny-only; no host allowlist substrate proof) |
| host allowlist enforcement | **BLOCKED** |
| fail-closed substrate | **PASS** (deny mode + isolation tier) |
| opaque secret storage | **PASS** |
| purpose-scoped secret brokering | **BLOCKED** |
| tenant secret isolation | **PASS** |
| integration runtime owner | **PASS** |
| governance/HITL owner | **PASS** |
| runtime enforcement evidence | **BLOCKED** (no allowlist/secret-scope correlation evidence) |
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
| Policy shape | `CodeCraftProfile.network_egress` (`deny` \| `allowlist`) |
| Substrate resolution | `intergrax/runtime/codecraft/substrate.py` |
| Security evidence | `SandboxSecurityCapable.security_capabilities()` → `SandboxSecurityCapabilities` |
| Operation-level network surface | `SandboxSession` (`browser_fetch` allowlist) |
| Integration-layer HTTP allowlist | `AllowlistHttpClient` (not sandbox-enforced) |

### Egress owner table

| Requirement | Owner | Implemented? | Runtime enforced? | Evidence |
| --- | --- | ---: | ---: | --- |
| deny all network | Sandbox / CodeCraft substrate | yes | **partial** | `network_egress_deny_enforced`; local session ties deny to absence of `browser_fetch` |
| allow exact hosts | — | **no** | **no** | `allowlist` enum value exists; no host list field or substrate proof |
| block non-approved host | — | **no** | **no** | `AllowlistHttpClient` is integration-only; not wired to A2 sandbox path |
| DNS/IP bypass protection | partial (web URL policy) | partial | partial | `test_web_url_intake.py` blocks localhost/IP at app URL policy; not sandbox egress |
| sandbox/provider proof | SandboxSecurityCapable | yes (deny) | yes (deny) | `test_aw_7b_gate.py` |
| fail closed if enforcement unavailable | CodeCraft substrate | yes | yes | `network_egress_requirement_unsatisfied` |

### Key substrate gap

`substrate._egress_deny_proven()` returns `True` for any `network_egress != "deny"`, so **`allowlist` is treated as satisfied without proof**. `resolve_craft_sandbox()` only fail-closes on `network_egress == "deny"`.

`SandboxSecurityCapabilities` exposes only `network_egress_deny_enforced` — no host allowlist evidence field.

### Egress verdict

**BLOCKED** — binary deny/restricted-operation enforcement exists; **exact host allowlist enforcement at sandbox/provider substrate does not**.

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
| purpose/operation scoped resolution | `CredentialResolutionContext.operation` | **declared only** | **no** |
| host/integration binding | — | **no** | **no** |
| bounded lifetime | — | **no** | **no** |
| no environment inheritance | partial (profile policy) | partial | **no** for dynamic A2 |
| audit/correlation without secret exposure | `CredentialUseEvidence` | **type only** | **no** (not enforced in resolver) |

### Secret broker answer

> Does the current public runtime provide purpose-scoped, tenant-bound, execution-bound secret brokering suitable for generated A2 code?

**NO.** Resolution is `store.get_secret(path)` after tenant check. `operation` and `execution_id` on `CredentialResolutionContext` are not authorization gates.

### Secret verdict

**BLOCKED** — persistence ≠ brokering.

---

## 7. Other owners

| Owner | Canonical surface | A2-ready? |
| --- | --- | --- |
| Integration runtime | Provider integrations, `TenantConnectionIntegrationFactoryRegistry`, `intergrax/runtime/vendor_knowledge/` | yes (durable path); ephemeral A2 adapter path **not implemented** |
| Governance | `MeaningfulSideEffectAuthorizationBoundary.authorize_and_execute` | yes |
| HITL | `GovernedContinuationGrantCoordinator`, CodeCraft HITL notes | yes |
| Sandbox | `SandboxSecurityCapable`, `resolve_craft_sandbox` | deny-only |
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
| Physical deny of non-approved network destination at substrate | **missing** for host allowlist |
| Runtime evidence fields for A2 | **missing** — AW-7B evidence covers substrate deny/isolation only; no `allowed_hosts`, secret scope correlation, or purpose-bound credential evidence |

Existing evidence (AW-7B / CodeCraft): `CraftSubstrateCapabilities` (`provider_id`, `resolved_tier`, `network_egress_enforced`).

---

## 9. Qualification tests

| Suite | Result |
| --- | --- |
| `tests/unit/runtime/codecraft/test_aw_7b_gate.py` | 28 passed |
| `tests/unit/integrations/credentials/test_p1_7_credential_ref.py` | passed (in batch) |
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

1. **Sandbox / CodeCraft substrate** — implement and prove `network_egress=allowlist` with typed host scope, substrate evidence field(s), and fail-closed when provider cannot enforce.
2. **Integration / credential domain** — purpose-scoped secret broker: tenant + credential_ref + integration identity + operation/purpose + allowed target scope; enforce at resolution; bounded lifetime.
3. **Qualification tests** — at least one test where approved host succeeds and unapproved host is **physically denied** at enforced substrate (not metadata-only).

Do **not** implement `WorkerSecretBroker`, `AWSecretStore`, or host filtering inside AW core.

---

## 11. Status

```text
AW-7C: BLOCKED BY PREREQUISITE
AW-7:  IN PROGRESS
```
