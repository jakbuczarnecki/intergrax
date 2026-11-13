# Security Architecture Satellite — D1–D8 surfaces and responsibility matrix

| Field | Value |
| --- | --- |
| **Document ID** | PDC-D6-SEC-SAT-01 |
| **Kind** | Satellite (consolidation and cross-reference only) |
| **Requirements** | PDC-D6-R021 (threat model covers D1–D8 surfaces), PDC-D6-R023 (encryption / residency responsibility matrix) |
| **Batch / ticket** | B-D6 / T-IGX-036-P |
| **Decision record** | [`ADR-PDC-06`](../ADR_PDC_06_RUNTIME_SECURITY_HARDENING.md) |
| **Authority** | None. This page owns no policy, identity, trust, signature or audit authority. Owners are named per row; where this page and an owner document differ, the owner document wins. |

## 1. Purpose and boundaries

This satellite gives one place to read how the runtime security controls fit the eight PDC domain surfaces, who owns each control, which test proves it, and what is still open. It does not replace:

- the Execution Engine threat model [`EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL`](../EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL.md) (EE-B3-A; certification record: [`EE_B3_A_EXECUTION_SECURITY_BOUNDARY_THREAT_MODEL_CERTIFICATION`](../../qualification/EE_B3_A_EXECUTION_SECURITY_BOUNDARY_THREAT_MODEL_CERTIFICATION.md)) — trust boundaries, principal model, authority, tenant isolation, child execution, tool authorization, retry and recovery;
- [`GOVERNED_EXECUTION`](../../../architecture/GOVERNED_EXECUTION.md) — governance plane, root admission, meaningful side effects, human review;
- [`INTEGRATIONS`](../../../architecture/INTEGRATIONS.md) § purpose-scoped credential brokering — `CredentialRef`, `ScopedCredentialBroker`, secrets store;
- the sandbox qualification boundary [`ADR_AW_7C_PROVIDER_NEUTRAL_PHYSICAL_SANDBOX_QUALIFICATION_BOUNDARY`](../ADR_AW_7C_PROVIDER_NEUTRAL_PHYSICAL_SANDBOX_QUALIFICATION_BOUNDARY.md) — physical sandbox and network egress scope;
- the execution identity model [`EXECUTION_IDENTITY_AUTHORITY_MODEL`](../EXECUTION_IDENTITY_AUTHORITY_MODEL.md).

> Link note: several hub documents under `docs/project/architecture/` link to `IDENTITY_TRUST.md`, which does not exist in the repository. Identity and trust for this satellite are covered by `EXECUTION_IDENTITY_AUTHORITY_MODEL` above and by the agent package trust mechanism in section 4. Repairing the hub links is outside the B-D6 write set and is reported to the integration owner.

## 2. Security principles applied by B-D6

1. **Fail closed.** A missing or malformed security configuration raises `SecurityConfigurationError` (typed, `item` + `reason_code`, never a secret value) instead of being defaulted.
2. **Defaults protect.** The retrieval-poisoning defense is on unless a profile says `False` explicitly.
3. **One path per defense.** The tool-input injection guard runs inside the canonical `ToolInjectionDefenseMiddleware`; there is no second tool-input path.
4. **Reuse the owner.** Plugin provenance reuses the agent package trust coordinator and the Ed25519 verifier; the audit trail uses the canonical evidence persistence port; secrets use the credential broker.
5. **Late resolution.** Secret material is read at the use site and never stored in profiles, wiring results, evidence, logs or reprs.
6. **Labs are explicit.** Lab conveniences (ephemeral signing key, demo attestation seed, unsigned plugins) exist only behind a named opt-in and are refused or ignored in strict (product) profiles.

## 3. D1–D8 surface coverage

"Strict" means `ApplicationProfile.PRODUCT`. Status values: **enforced** (control exists and has a test), **delivered** (control added or corrected by B-D6), **owner-elsewhere** (control belongs to another domain batch), **open** (known gap, see section 6).

| Surface | Security-relevant assets and threats | Control and owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| **D1 Agent lifecycle** (packages, upgrade, retire) | Tampered or unsigned agent and plugin packages; revoked publishers; privilege growth on upgrade | `AgentPackageTrustPolicy` + `AgentPackageTrustCoordinator` + Ed25519 attestation verifier (agent distribution); plugin entry points reuse it via `PluginProvenanceAdmission`; `EffectiveDelegationAuthority` narrowing for child agents | `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py`, `test_xd06_delegation_widening.py` | delivered (plugins), enforced (agents) |
| **D2 Context and memory** | Retrieval poisoning; tenant data in provider prompts; retention of restricted data | Poisoning filter on `rag.retrieve` (default on); retention classes in `retention_policy`; provider-egress sensitivity policy is owned by batch B-D2 (`intergrax/context/policy/egress.py`) and is **not** duplicated here | `test_d6_poisoning_default.py`, `test_xd06_provider_boundary.py` | delivered (poisoning default), owner-elsewhere (egress sensitivity) |
| **D3 Model and provider adapters** | Credentials sent to the wrong provider; cross-tenant prompts; guardrail bypass | `ScopedCredentialBroker` checks tenant, execution, operation, integration and egress target before the secrets store is read; tool allow-list in `ToolInjectionDefenseMiddleware`; optional vendor guardrail profile | `test_xd06_provider_boundary.py`, `test_d6_secrets.py` | enforced |
| **D4 Hosting** | Unauthenticated callers; weak or missing signing secret on product hosts; demo seeds in product wiring | Product host requires the configured signing secret (no literal fallback); `assert_strict_security_configuration` rejects missing items; demo attestation seed refused in strict profiles | `test_d6_signing_fail_closed.py`, `test_d6_security_config_fail_closed.py`, `test_d6_demo_seed_gate.py` | delivered |
| **D5 Evaluation** | Secrets or tenant data inside evaluation datasets and results | No dedicated control owned by this domain; evaluation outputs fall under the secret-safe validation primitives (`core/security/secret_safety`) and the static raw-secret gate for source | `test_d6_secrets.py` (source gate) | owner-elsewhere |
| **D6 Runtime security** | Dev signing key fallback; fail-open poisoning default; dead tool-input guard; volatile audit trail; raw secrets in source or logs; unsigned plugins; demo seed reachable | See section 4 | `tests/qualification/pdc_x/d6/` | delivered |
| **D7 Background and recurrence** | Recurring work running with stale authority or a revoked credential; secrets in dead-letter entries | Controls are owned by batch B-D7 (recurrence and dead-letter handling). B-D6 contributes only the proof that the credential broker re-admits every attempt, so a credential revoked mid-retry is denied for any caller, including background work | `test_xd06_credential_revocation.py` | owner-elsewhere |
| **D8 Artifacts** | Cross-tenant artifact reads; unbounded retention; secrets in artifacts | Tenant-scoped evidence persistence; retention classes; governed read port is owned by batch B-D8 | `test_d6_audit_trail_durable.py` (tenant scope) | owner-elsewhere |

## 4. Controls delivered by B-D6

| Requirement | Control | Owner module | Proof |
| --- | --- | --- | --- |
| PDC-D6-R008 | No literal signing fallback; missing secret raises; lab opt-in is `LabEphemeralSigningOptIn` (random per-process key) and is rejected on product profiles | `applications/_shared/critical_action_signing_wiring.py` | `test_d6_signing_fail_closed.py` |
| PDC-D2-R008, PDC-D6-R014 | `retrieval_poisoning_defense_enabled_or_default`: absent profile or absent toggle means enabled; only literal `False` disables | `core/security/security_configuration.py`, `tools/providers/rag/service.py`, `applications/_shared/security_assembly_resolver.py` | `test_d6_poisoning_default.py` |
| PDC-D6-R013 | `assert_tool_input_safe` runs for every argument value in `ToolInjectionDefenseMiddleware` | `applications/_shared/application_security_wiring.py` | `test_d6_tool_input_defense_path.py` |
| PDC-D6-R020 | `EvidencePlaneSecurityAuditTrail` appends canonical `RuntimeEvent`s through `EvidencePersistencePort` (append-only, tenant-scoped); wiring requires the port and at least two regions | `runtime/security/security_audit_trail.py`, `applications/_shared/security_audit_trail_wiring.py` | `test_d6_audit_trail_durable.py` |
| PDC-D6-R006, R007 | Late resolution through the broker; static raw-secret gate (`core/security/raw_secret_gate.py`) over `intergrax/`, `agents/`, `applications/` product source; caplog and evidence-store leak scan | `core/security/` | `test_d6_secrets.py` |
| PDC-D6-R016 | `PluginProvenanceAdmission` verifies publisher attestations with the agent package trust coordinator before an entry-point target is imported; unsigned plugins are rejected when the policy forbids unsigned packages (default) | `core/plugins/admission.py`, `core/catalog_bootstrap.py`, `runtime/security/defense_plugin_loader.py` | `test_d6_plugin_provenance.py` |
| PDC-D6-R004, R009, R015 | Adversarial coverage of existing controls (delegation narrowing, revocation mid-retry, provider and tenant boundary) | contracts and integrations owners | `test_xd06_*.py` |
| PDC-D6-R024 | `validate_strict_security_configuration` / `assert_strict_security_configuration` report every missing or malformed item as a typed error | `applications/_shared/security_assembly_resolver.py` | `test_d6_security_config_fail_closed.py` |
| PDC-D6-R027 | Demo attestation seed resolved only through `resolve_demo_attestation_seed(strict_profile=False)` (lazy import); strict profile raises; strict assembly refuses host signing without an explicit key | `runtime/execution_evidence/attestor.py`, `security_assembly_resolver.py` | `test_d6_demo_seed_gate.py` |

## 5. Encryption and residency responsibility matrix (PDC-D6-R023)

The platform enforces **policy gates** and **fail-closed behavior**. It does not implement storage encryption, key custody or data residency itself; those are delivered by the chosen storage and secrets backends and by the operator. `HarnessEnvelopeEncryptor` is a lab-only non-cryptographic transform and is never a protection claim.

| Concern | Platform (Intergrax) | Application host / composition root | Operator / deployer | Provider or backend |
| --- | --- | --- | --- | --- |
| Data classification of payloads | Defines `DataClassification`; marks which classes require encryption | Tags payloads and tool outputs | Chooses classification policy for the tenant | — |
| Encryption enforcement decision | `evaluate_encryption_enforcement` denies RESTRICTED or CONFIDENTIAL data when no secrets store is configured on a strict host | Enables enforcement and wires a secrets store | Runs the strict profile in production | — |
| Encryption at rest | Never claims it; storage ports are backend-neutral | Selects the durable backend (SQLite, document store, vector store) | Enables disk, volume or database encryption | Backend implements encryption at rest |
| Encryption in transit | Egress scope limits where credentials may be used (`NetworkEgressAllowlist`) | Configures TLS endpoints | Terminates TLS, rotates certificates | Provider enforces TLS |
| Key custody and rotation | Resolves secrets late through `SecretsStore`; signing secrets come from environment configuration, never literals | Supplies secret references (`CredentialRef`), not values | Owns the KMS/HSM or secrets manager and rotation | Secrets backend stores and versions keys |
| Signing keys (critical actions, host attestation) | Fails closed when missing in strict profiles; demo seed refused | Passes configuration | Provisions production keys | KMS/HSM may replace the local signer through DI |
| Data residency | Audit trail requires at least two configured regions and records them; region replication of the evidence store itself is **not** performed by the platform | Chooses where each store is deployed | Guarantees residency of stores and backups | Backend guarantees physical location |
| Retention and deletion | Retention classes per classification (`retention_policy`); evidence plane is append-only (no update or delete port) | Applies retention jobs where a domain owns them | Legal hold and deletion procedures | Backend purge and backup expiry |
| Tenant isolation | Tenant scope on every persistence and broker call | Passes the authenticated tenant | Provisions tenant data boundaries | — |
| Audit integrity | Append-only canonical evidence; tenant-scoped reads | Provides the `EvidencePersistencePort` | Retains and exports evidence | Backend durability |

## 6. Gap closures (Amendment 1) and remaining limits

Closed by Amendment 1 of T-IGX-036-P:

1. **Host attestation fallback (R027).** `resolve_host_attestation_sealer_from_env` no longer falls back to the public PoC seed: with `INTERGRAX_EBE_HOST_SIGNING_KEY` unset it raises `SecurityConfigurationError` unless the caller passes the named `lab_demo_seed_opt_in=True`. `attestation_runtime_bridge.build_boundary_event_buffer` fails closed on strict profiles (product profile or `ExecutionMode.STRICT`), refuses the opt-in there, and applies it only on non-strict (lab and proof) profiles. The seed is a lazy module attribute of `host_attestation.py`, not an import-time global; `runtime/attestation/__init__.py` does not name it.
2. **Entry-point discovery gating.** `register_plugins_with_report(provenance=..., require_provenance=...)` checks provenance before any target is imported; `bootstrap_catalogs` passes both through.
3. **Plan-context poisoning default.** `run_rag_context` keeps the poisoning filter on when `production_mode` is true and the security profile is absent; non-strict behavior is unchanged.

Remaining limits:

1. The attested digest is bound by the publisher signature; installed files are not re-hashed (the installed distribution version is checked).
2. Composition roots on strict hosts must call `assert_strict_security_configuration` and pass `plugin_provenance` with `require_plugin_provenance=True`; the library does not force them to.
3. `MultiRegionSecurityAuditTrail` tracks replication of entry identifiers at bootstrap; physical replication of the evidence store is a backend and operator concern (section 5).
4. Hosts that relied on the removed signing fallback or on the in-memory audit trail must now supply the signing secret and an evidence persistence port.
