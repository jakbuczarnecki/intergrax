# SECURITY_BOUNDARIES - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** SECURITY_BOUNDARIES
- **Owning architecture/program:** TIER3_APPLICATION_ENVIRONMENT · UNIFIED_EXECUTION_RUNTIME (cross-layer security composition)
- **Tier(s):** Tier-3 `intergrax/applications/_shared/` (identity, security wiring, admin routes); Tier-1 `intergrax/runtime/security/` (encryption, audit trail); Tier-0 `intergrax/integrations/contracts/` (identity provider)
- **audited_sha:** `6005514aee4f8c39bb15554876f69d83a3459f8d`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 3 CRITICAL / 3 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`
- **Scope in:**
  - authentication source resolution vs middleware/dependency enforcement
  - Agent Platform admin authorization vs verified identity
  - restricted/confidential payload encryption fail-closed semantics
  - security profile toggle qualification vs wired enforcement
  - critical action signing evidence vs production effect paths
  - security audit trail durability and multi-region qualification claims
  - positive controls: ApplicationSecurityProfile, security wiring, assembly validation, TenantSecurityMiddleware, IdentityProviderBackend neutrality, SecretsStorePayloadEncryptor target shape
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - duplicating IDENTITY_TRUST-01/02 principal binding or delegated scope narrowing
  - duplicating TOOLS caller-controlled allowed-tool narrowing, MODALITY local-media exfiltration, or CODE_CRAFT security findings
  - claiming every security control is broken
- **Prior audit reference(s):** [`IDENTITY_TRUST`](IDENTITY_TRUST.md) (finding 01 - principal propagation); [`POLICY_GOVERNANCE`](POLICY_GOVERNANCE.md); [`PLATFORM_EXTENSIBILITY`](PLATFORM_EXTENSIBILITY.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `5497229615aaa646486e5175a6679b41fbb65a6f`

## Scope / ownership mapping

| Concept | Canonical ownership |
|---------|---------------------|
| Audit unit (Protocol v2 layer code) | **SECURITY_BOUNDARIES** |
| Tier-3 identity/auth/admin composition | **TIER3_APPLICATION_ENVIRONMENT** |
| Runtime encryption / audit authority | **UNIFIED_EXECUTION_RUNTIME** |
| Verified principal → execution identity | **IDENTITY_TRUST** - **IDT-FIX-A** (cross-link; do not duplicate) |
| Authorization policy engine | **POLICY_GOVERNANCE** (reuse; no second engine) |
| Per-layer report | `docs/audit_results/2026-08-18/SECURITY_BOUNDARIES.md` |
| Target invariants (Tier-3) | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` - [Protocol v2 security boundaries target invariants (2026-08-18)](#protocol-v2-security-boundaries-target-invariants-2026-08-18) |
| Target invariants (runtime) | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` - [Protocol v2 security runtime target invariants (2026-08-18)](#protocol-v2-security-runtime-target-invariants-2026-08-18) |

## Executive summary

**Verdict: FAIL.** Three accepted CRITICAL and three accepted HIGH findings show that configurable API-key authentication can pass startup yet leave routes unauthenticated; verified OIDC identity is reduced to boolean presence and treated as Agent Platform admin authorization without role/scope/tenant/app/environment binding; required encryption can silently downgrade to Base64 envelope encoding when SecretsStore resolution fails; security profile toggles can report enabled without proven enforcement middleware; critical action signing can bootstrap locally with a development fallback secret without binding production effects; and in-process multi-list audit simulation can qualify as immutable multi-region security audit trail. Positive controls: ApplicationSecurityProfile + security wiring + assembly validation are real platform mechanisms; TenantSecurityMiddleware blocks missing tenant_id; IdentityProviderBackend and SecretsStorePayloadEncryptor preserve provider-neutral target shapes; IDENTITY_TRUST-01/02 retain ownership of principal propagation and delegated scope narrowing. Remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 3 CRITICAL / 3 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-SECURITY_BOUNDARIES-01 (SEC-BND-01)

- **Severity:** CRITICAL
- **Category:** AUTHENTICATION BOUNDARY / IMPLEMENTATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-AUTHORITY-BOUNDARY-INTEGRITY
- **Claim falsified:** One resolved authentication source materializes one canonical credential authority consumed by all middleware and route dependencies; required configured credentials that cannot be materialized fail startup.
- **Observation:** `IdentityProfile` allows configurable `api_key_env`. `wire_application_identity()` checks the configured env name and therefore startup may successfully validate e.g. `MY_PRODUCT_API_KEY`. Actual harness auth resolution ignores `profile.api_key_env`: `resolve_harness_api_key()` always reads `INTERGRAX_HARNESS_API_KEY`. `apply_harness_auth_middleware()` adds middleware only if that hard-coded env key or an identity provider exists. `require_harness_api_key()` also uses the hard-coded default resolver and permits requests when it is absent. Therefore `require_api_key=True`, `api_key_env=CUSTOM_KEY`, `CUSTOM_KEY` configured, `INTERGRAX_HARNESS_API_KEY` absent, no IdP can pass startup but result in no auth middleware and permissive route dependency.
- **Location:**
  - `intergrax/applications/contracts/environment_profile/sub_profiles.py` - `IdentityProfile.api_key_env`
  - `intergrax/applications/_shared/identity_wiring.py` - `wire_application_identity()`
  - `intergrax/applications/_shared/harness_auth.py` - `resolve_harness_api_key()`, `apply_harness_auth_middleware()`, `require_harness_api_key()`
- **Impact:** Complete authentication bypass despite an explicitly required API key.
- **Confidence:** CONFIRMED

### AUDIT-20260818-SECURITY_BOUNDARIES-02 (SEC-BND-02)

- **Severity:** CRITICAL
- **Category:** AUTHORIZATION BOUNDARY / PRIVILEGE ESCALATION
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-AUTHORITY-BOUNDARY-INTEGRITY
- **Claim falsified:** Authentication and authorization remain distinct; Agent Platform admin operations require explicit authorization bound to operation, application, environment, tenant, and principal - not merely successful token verification.
- **Observation:** `IdentityProviderBackend.verify_token()` returns `IdentityUser` including `user_id`, `tenant_id`, and metadata. `is_harness_identity_token_valid()` discards that principal and returns only `bool(user.user_id)`. `require_agent_platform_admin_auth()` accepts any successfully verified OIDC token as Agent Platform admin authorization. No proven check of admin role, authorization scope, tenant ownership, application/environment ownership, or operation capability. Agent Platform admin routes include install/bind/config/build/activate/rollback control-plane operations.
- **Location:**
  - `intergrax/integrations/contracts/identity_provider.py` - `IdentityProviderBackend.verify_token()`
  - `intergrax/applications/_shared/harness_auth.py` - `is_harness_identity_token_valid()`
  - `intergrax/applications/_shared/agent_platform_admin_routes.py` - `require_agent_platform_admin_auth()`
- **Impact:** Any verified OIDC subject can invoke control-plane admin operations without proven admin authorization.
- **Confidence:** CONFIRMED

### AUDIT-20260818-SECURITY_BOUNDARIES-03 (SEC-BND-03)

- **Severity:** CRITICAL
- **Category:** DATA PROTECTION / FAIL-OPEN ENCRYPTION
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-DATA-PROTECTION-INTEGRITY
- **Claim falsified:** When cryptographic protection is required, secure backend resolution failure blocks or fails startup; no silent downgrade to non-cryptographic transforms labeled as encryption.
- **Observation:** `resolve_restricted_payload_encryptor()` attempts to resolve configured SecretsStore but catches any `Exception` and returns `HarnessEnvelopeEncryptor`. `SecurityWiringOptions` still treats `secrets_store` as configured. Encryption enforcement therefore permits RESTRICTED/CONFIDENTIAL data and can pass it to `HarnessEnvelopeEncryptor`. `HarnessEnvelopeEncryptor` labels output as ciphertext/encryption envelope but uses `base64.b64encode(plaintext)`. Base64 is reversible encoding, not encryption. A production SecretsStore failure can therefore silently downgrade protection instead of failing closed.
- **Location:**
  - `intergrax/applications/_shared/security_wiring.py` - `resolve_restricted_payload_encryptor()`, `SecurityWiringOptions`
  - `intergrax/applications/_shared/application_security_wiring.py`
  - `intergrax/applications/_shared/security_runtime_bridge.py`
  - `intergrax/applications/_shared/security_assembly_resolver.py`
  - `intergrax/runtime/security/encryption_policy.py`
  - `intergrax/runtime/security/encryption_middleware.py`
  - `intergrax/runtime/security/encryption_transform.py` - `HarnessEnvelopeEncryptor`
- **Impact:** RESTRICTED/CONFIDENTIAL payloads may be stored or transmitted with reversible encoding while assembly reports encryption configured.
- **Confidence:** CONFIRMED

### AUDIT-20260818-SECURITY_BOUNDARIES-04 (SEC-BND-04)

- **Severity:** HIGH
- **Category:** PAPER SECURITY CONTROL / IMPLEMENTATION-ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-DEFENSE-QUALIFICATION-INTEGRITY
- **Claim falsified:** Security capabilities use explicit qualification state (DISABLED / ENFORCED-PROVEN / UNAVAILABLE-REQUIRED); enabled toggles require proven enforcement points mechanically verified at assembly.
- **Observation:** `ApplicationSecurityProfile` defaults `retrieval_poisoning_defense_enabled=True`. `resolve_security_wiring_options` carries the flag. But `_enabled_middleware_names()` does not add a retrieval-poisoning middleware; `register_application_security_hooks()` does not mount one; security assembly validation only checks flag/profile equality; no corresponding enforced middleware is proven. Thus `enabled=True` can pass assembly without an implementing mechanism.
- **Location:**
  - `intergrax/applications/_shared/security_wiring.py` - `_enabled_middleware_names()`, `resolve_security_wiring_options`
  - `intergrax/applications/_shared/application_security_wiring.py` - `register_application_security_hooks()`
  - `intergrax/applications/_shared/security_assembly_resolver.py`
- **Impact:** Product/STRICT assembly can report active retrieval-poisoning defense without runtime enforcement.
- **Confidence:** CONFIRMED

### AUDIT-20260818-SECURITY_BOUNDARIES-05 (SEC-BND-05)

- **Severity:** HIGH
- **Category:** CRITICAL ACTION INTEGRITY / PAPER CONTROL
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-DEFENSE-QUALIFICATION-INTEGRITY
- **Claim falsified:** When product signing is enabled, missing configured secret fails closed; no development fallback secret; real critical operations require exact signed action evidence immediately before effect.
- **Observation:** `critical_action_signing_enabled` on product hosts resolves only a bootstrap `CriticalActionPayload`, signs it, and verifies the same signature locally. No production consumer path binding the signature to actual critical actions was established. When signing secret env is absent, PRODUCT wiring falls back to literal `harness-dev-signing-key` and may still report signing enabled.
- **Location:**
  - `intergrax/applications/_shared/critical_action_signing_wiring.py`
  - `intergrax/applications/_shared/security_wiring.py`
  - `intergrax/applications/_shared/security_assembly_resolver.py`
- **Impact:** Critical control-plane or governance-adjacent operations may proceed without bound signed action evidence; dev signing key on product hosts.
- **Confidence:** CONFIRMED

### AUDIT-20260818-SECURITY_BOUNDARIES-06 (SEC-BND-06)

- **Severity:** HIGH
- **Category:** AUDIT EVIDENCE / FALSE MULTI-REGION QUALIFICATION
- **Status at publication:** ACCEPTED
- **Remediation block:** SEC-AUDIT-AUTHORITY-INTEGRITY
- **Claim falsified:** Multi-region qualified audit trail requires independently durable replicas and explicit replication/tamper evidence - not process-local list duplication inside one host.
- **Observation:** `SecurityAuditTrail` is explicitly in-memory. `MultiRegionSecurityAuditTrail` represents regions as multiple Python lists inside one process. `append()` copies one generated entry ID into every list. `verify_replication()` compares those process-local lists and can return `replicated=True`, `immutable=True`. `security_audit_trail_wiring` labels this an immutable multi-region security audit trail for product configuration. This does not prove independent regions, durability, WORM, failure-domain separation, restart survival, or tamper resistance.
- **Location:**
  - `intergrax/applications/_shared/security_audit_trail_wiring.py`
  - `intergrax/runtime/security/security_audit_trail.py` - `SecurityAuditTrail`
  - `intergrax/runtime/security/multi_region_audit_trail.py` - `MultiRegionSecurityAuditTrail`
- **Impact:** Product configuration can claim immutable multi-region audit evidence while storing only in-process simulation.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| `ApplicationSecurityProfile` + security wiring + assembly validation are real platform mechanisms | NOT falsified |
| `TenantSecurityMiddleware` blocks missing `tenant_id` | NOT falsified |
| `IdentityProviderBackend` is provider-neutral and should be reused | NOT falsified |
| `SecretsStorePayloadEncryptor` is valid provider-neutral target shape; SEC-BND-03 concerns fallback semantics | NOT falsified |
| IDENTITY_TRUST-01 owns verified-principal → tenant/user execution identity binding | NOT falsified - cross-link only |
| IDENTITY_TRUST-02 owns delegated permission scope narrowing | NOT falsified - cross-link only |
| TOOLS caller-controlled allowed-tool narrowing remains owned by TOOLS | NOT falsified - do not duplicate |
| MODALITY local-media exfiltration finding remains owned by MODALITY | NOT falsified - do not duplicate |
| CODE_CRAFT security findings remain owned by CODE_CRAFT | NOT falsified - do not duplicate |
| No claim that every security control is broken | NOT falsified |

## Duplicate ownership / cross-links

| Existing finding / domain | Relationship |
|---------------------------|--------------|
| **IDENTITY_TRUST-01 / IDT-FIX-A** | Verified principal propagation - cross-link; SEC-BND-02 adds admin authorization boundary, not duplicate principal spine |
| **IDENTITY_TRUST-02 / IDT-FIX-B** | Delegated scope narrowing - owned by IDENTITY_TRUST; do not duplicate |
| **POLICY_GOVERNANCE** | Reuse Governance/identity authority for admin authorization - no second permission engine |
| **TOOLS** | Caller-controlled allowed-tool narrowing - owned by TOOLS |
| **MODALITY** | Local-media exfiltration - owned by MODALITY |
| **CODE_CRAFT** | Session/HITL/isolation findings - owned by CODE_CRAFT |

## Root-cause remediation grouping

### SEC-AUTHORITY-BOUNDARY-INTEGRITY - one authentication authority and explicit admin authorization

**Priority:** P0  
**Findings:** SEC-BND-01, SEC-BND-02  
**Owner:** TIER3_APPLICATION_ENVIRONMENT (plan) · cross-layer with POLICY_GOVERNANCE  

One resolved authenticated principal and one authorization authority. No configurable API-key bypass. No authentication==admin equivalence. Admin authorization bound to exact tenant/app/environment/action. Cross-link **IDT-FIX-A** and **POLICY_GOVERNANCE**.

### SEC-DATA-PROTECTION-INTEGRITY - encryption fail-closed

**Priority:** P0  
**Findings:** SEC-BND-03  
**Owner:** UNIFIED_EXECUTION_RUNTIME  

Required encryption fails closed. No Base64-as-encryption semantics. No production downgrade when SecretsStore resolution fails. Preserve SecretsStorePayloadEncryptor provider-neutral architecture.

### SEC-DEFENSE-QUALIFICATION-INTEGRITY - proven security toggles and action signing

**Priority:** P1  
**Findings:** SEC-BND-04, SEC-BND-05  
**Owner:** TIER3_APPLICATION_ENVIRONMENT · runtime enforcement cross-link UNIFIED_EXECUTION_RUNTIME  

Security toggles/signing are considered active only when real enforcement paths are wired and verified. No paper controls. No development signing key on product hosts. Signing supplements Governance authorization; does not replace it.

### SEC-AUDIT-AUTHORITY-INTEGRITY - durable audit authority vs in-memory simulation

**Priority:** P1  
**Findings:** SEC-BND-06  
**Owner:** UNIFIED_EXECUTION_RUNTIME  

Separate lab/in-memory proof implementation from production audit authority. Define `ImmutableSecurityAuditTrail` persistence port. Multi-region qualification requires independently durable replicas and explicit replication/tamper evidence. No specific cloud vendor required.

## Architecture / plan sync state

| Doc | Section | Status |
|-----|---------|--------|
| `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | Protocol v2 security boundaries target invariants | SYNCED |
| `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | Protocol v2 security runtime target invariants | SYNCED |
| `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` | SEC-AUTHORITY-BOUNDARY-INTEGRITY, SEC-DEFENSE-QUALIFICATION-INTEGRITY | SYNCED |
| `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` | SEC-DATA-PROTECTION-INTEGRITY, SEC-AUDIT-AUTHORITY-INTEGRITY | SYNCED |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `6005514aee4f8c39bb15554876f69d83a3459f8d`; current `development` HEAD was not re-audited beyond persistence sync.
- Remediation not performed in this task.
- Historical AUDIT-IDEAL Done rows for signing and multi-region audit trail remain historical facts; Protocol v2 findings reopen current qualification honesty.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-SECURITY_BOUNDARIES-01` … `06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
