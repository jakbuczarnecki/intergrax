# AW-7C A2 Scoped Adaptive Integration — Prerequisite Qualification

**Verdict:** BLOCKED BY PREREQUISITE

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `4dff9d136085fe46fbffc342d1bfc690ce7a8d85`

**Review HEAD (pre-P0-3A):** `73d023d0a34102ec35e7f01ceae56129ea72ff5e`

**Task:** AW-7C qualification-first audit — no A2 production implementation. **P0-1 egress contract remediation** (2026-09-07): typed host scope + fail-closed substrate matching. **P0-2 secret broker remediation** (2026-09-07): purpose-scoped grant/broker contracts + enforcement at resolution boundary. **P0-3 hosted provider inventory** (2026-09-07): canonical sandbox-host audit + `SandboxSecurityConfigurable` admission seam. **P0-3A E2B adapter** (2026-09-07): security-qualified `E2bSandboxHostBackend` + provider-state attestation; physical qualification pending operator credentials.

**P0-2 independent audit correction:** unenforced `max_uses` field removed. V1 bounded credential authority is time-bounded via `expires_at` only. Use-count restrictions require a future concurrency-safe lifecycle authority.

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
AW-7C EGRESS PREREQUISITE: IMPLEMENTED / PHYSICAL QUALIFICATION BLOCKED (E2B adapter in-repo; credential not available in qualification session)
AW-7C P0-3A: IMPLEMENTED / PHYSICAL QUALIFICATION BLOCKED
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
AW-7C P0-3A: IMPLEMENTED / PHYSICAL QUALIFICATION BLOCKED
```

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
uv run pytest tests/integration/providers/sandbox_host/e2b/test_e2b_physical_egress_qualification.py `
  -m "integration and network and sandbox_provider" -vv
```

**Physical session result:** `SKIPPED` — `E2B_API_KEY` / `INTERGRAX_E2B_API_KEY` unavailable in qualification session.

**Unit tests:** `tests/unit/integrations/providers/sandbox_host/e2b/` — 18 passed.

**Known limitations:** domain filter is routing control per E2B docs (shared CDN/SNI caveats); UDP/QUIC not domain-filtered; DNS rebinding not verified; E2B may inject `8.8.8.8` DNS helper IP in raw `allowOut` (excluded from canonical enforced host evidence).

**Nexus:** untouched.
