# INTEGRATIONS — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** INTEGRATIONS
- **Constituent domains:** INTEGRATIONS (IntegrationProfile · catalog · registry v2 metadata · PlatformIntegrationContract)
- **Tier(s):** Tier-0 `intergrax/integrations/` · Tier-1 `intergrax/runtime/integrations/`
- **audited_sha:** `f15813cf5d2ffbd29f11a22daa1906a07e6ce23d`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 2 HIGH / 3 MEDIUM / 0 LOW
- **Operator decision:** all 5 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/INTEGRATIONS.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/INTEGRATIONS.md`
- **Scope in:**
  - `IntegrationProfile` typed category slots and `resolve_from_profile` / `resolve` factory path
  - `IntegrationBinding` manifest · plugin · slug · pre-built instance normalization
  - `validate_integration_ref` / `resolve_ref_to_slug` profile validation semantics
  - Open catalog `register_integration` / `get_entry` / `IntegrationEntry` metadata
  - `IntegrationStatus` (STABLE / BETA / DEPRECATED) on catalog entries
  - `IntegrationManifest` frozen extra-forbid authoring path
  - `PlatformIntegrationContract` identity fields and `derive_platform_integration_id`
  - Registry v2 `(provider_id, category)` metadata validation (INTEGRATIONS-3A)
  - Integration / Tool / Skill / Agent responsibility split (positive control)
  - Historical INTEGRATIONS-1A/1B/2A–2E delivery facts (positive control — not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second Integration runtime/registry design
  - universal production qualification of full **194**-slug catalog scale
  - INTEGRATIONS-3B explicit registry-backed runtime binding (remains Planned)
  - full ToolRuntime / Governed Execution re-audit beyond Integration wiring touchpoints
- **Prior audit reference(s):** Protocol v2 [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md) (PBA-FIX-B/C on INTEGRATIONS arch/plan); INTEGRATIONS-1A/1B/2A–2E **Done**; INTEGRATIONS-3A additive registry v2 **In progress**; INTEGRATIONS-3B **Planned**
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Five accepted findings (2 HIGH, 3 MEDIUM) show pre-built integration instances bypass canonical slug/catalog contract and identity validation; provider lifecycle status has no runtime eligibility effect; profile validation can accept manifest/plugin bindings whose slug is not registered until later `resolve()` fails; catalog registration drops `requires_local_container` metadata; and `PlatformIntegrationContract` accepts contradictory `integration_id` / `provider_id` / `integration_kind` without enforcing canonical identity. Positive controls: Integration / Tool / Skill / Agent split remains sound; host owns provider selection via `IntegrationProfile`; slug/catalog resolution validates category membership for factory paths; `IntegrationProfile` rejects unknown structural fields; `IntegrationManifest` is frozen extra-forbid; registry v2 validates category contract, integration class, integration kind, config class, and disabled-by-default posture; architecture honestly documents registry v2 as additive metadata and INTEGRATIONS-3B runtime authority as Planned; broad catalog scale is not claimed as universal production qualification; no finding requires a second Integration runtime/registry.

## Verdict

**FAIL** — 0 CRITICAL / 2 HIGH / 3 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-INTEGRATIONS-01

- **Severity:** HIGH
- **Category:** CONTRACT / AUTHORITY BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** INTEGRATIONS-RUNTIME-BINDING-INTEGRITY
- **Claim falsified:** Dependency-injected pre-built instances satisfy the same canonical category-specific integration contract and provider identity as catalog-created providers.
- **Substance:** `IntegrationProfile` slots accept pre-built instances through `IntegrationBinding`. `IntegrationBinding.instance` is typed `Any`. `normalize_integration_binding()` treats essentially any non-string/non-manifest/non-plugin object as an integration instance via `is_integration_instance()`. `validate_integration_ref()` immediately accepts bindings containing an instance without validating category contract or `PlatformIntegrationContract` identity. `resolve_from_profile()` returns the instance directly. Pre-built instances therefore bypass canonical slug/catalog checks: category membership, expected category contract, provider/integration identity, and factory construction contract.
- **Evidence:**
  - `intergrax/integrations/core/binding.py` — `instance: Any`; `from_instance()`
  - `intergrax/integrations/core/ref.py` — `is_integration_instance()`, `normalize_integration_binding()`, `validate_integration_ref()` early return on instance
  - `intergrax/integrations/registry/factory.py` — `resolve_from_profile()` direct instance return
- **Confidence:** HIGH — direct code path; instance path skips catalog validation entirely.
- **Target invariant:** Dependency-injected/pre-built instances are allowed only if they satisfy the exact canonical category-specific integration contract for that profile slot. Validate provider/category/integration identity as appropriate. Do not remove dependency injection merely to solve validation. Do not introduce a second integration abstraction.

### AUDIT-20260818-INTEGRATIONS-02

- **Severity:** HIGH
- **Category:** LIFECYCLE / PRODUCTION ROUTING DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** INTEGRATIONS-RUNTIME-BINDING-INTEGRITY
- **Claim falsified:** Provider lifecycle state (`STABLE` / `BETA` / `DEPRECATED`) has explicit runtime/bootstrap qualification semantics; production binding does not treat all statuses as equivalent.
- **Substance:** `IntegrationEntry` exposes status: STABLE / BETA / DEPRECATED. Runtime `resolve()` loads the entry, validates category, and calls its factory, but does not evaluate `entry.status`. Lifecycle status currently has no effect on runtime provider eligibility.
- **Evidence:**
  - `intergrax/integrations/contracts/base.py` — `IntegrationStatus`, `IntegrationEntry.status`
  - `intergrax/integrations/registry/factory.py` — `resolve()` no status gate
  - `intergrax/integrations/registry/catalog.py` — status stored on registration
- **Confidence:** HIGH — status field present but unused in resolution path.
- **Target invariant:** Provider lifecycle state must have explicit runtime/bootstrap semantics. At minimum canonical production binding must not treat STABLE, BETA, and DEPRECATED as automatically equivalent. Define a typed host/provider qualification policy: stable default eligibility; beta explicit opt-in where desired; deprecated fail-closed for new production bindings or explicitly authorized compatibility posture. Do not silently infer exact product policy in persistence; record target invariant and remediation ownership.

### AUDIT-20260818-INTEGRATIONS-03

- **Severity:** MEDIUM
- **Category:** CONFIGURATION VALIDATION GAP / FAIL-LATE
- **Status at publication:** ACCEPTED
- **Remediation block:** INTEGRATIONS-RUNTIME-BINDING-INTEGRITY
- **Claim falsified:** Explicit host configuration has a deterministic declared → admitted/registered → resolvable lifecycle; host readiness proves every selected integration is resolvable by the active runtime authority.
- **Substance:** `resolve_ref_to_slug()` catches `UnknownIntegrationError` and returns the manifest slug when an `IntegrationManifest` declares categories. Profile validation can accept a manifest/plugin binding whose slug is not currently registered in the runtime catalog. Later `resolve()` calls `get_entry(resolved_slug)` and fails.
- **Evidence:**
  - `intergrax/integrations/core/ref.py` — `resolve_ref_to_slug()` manifest fallback on `UnknownIntegrationError`
  - `intergrax/integrations/core/ref.py` — `validate_integration_ref()` calls `resolve_ref_to_slug` only for non-instance bindings
  - `intergrax/integrations/registry/factory.py` — `resolve()` → `get_entry(resolved_slug)`
- **Confidence:** HIGH — manifest-with-categories bypasses catalog admission check at validation time.
- **Target invariant:** Explicit host configuration should have a deterministic declared → admitted/registered → resolvable lifecycle. Before a host is considered ready, every selected integration must be proven resolvable by the active runtime authority. Plugin declaration before registration may remain a supported authoring phase, but production/startup validation must fail fast before serving work.

### AUDIT-20260818-INTEGRATIONS-04

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION DEFECT / METADATA INTEGRITY
- **Status at publication:** ACCEPTED
- **Remediation block:** INTEGRATIONS-CONTRACT-METADATA-INTEGRITY
- **Claim falsified:** Catalog normalization preserves all contractually meaningful provider metadata exactly, except explicitly documented normalization fields.
- **Substance:** `IntegrationEntry` contains `requires_local_container: bool`. `IntegrationEntry.metadata` preserves it. But `register_integration()` reconstructs `IntegrationEntry` while normalizing the slug and copies slug, categories, factory, status, env_prefix, description — it does NOT copy `requires_local_container`, so an incoming `True` becomes the dataclass default `False` after registration.
- **Evidence:**
  - `intergrax/integrations/contracts/base.py` — `IntegrationEntry.requires_local_container`, `metadata` property
  - `intergrax/integrations/registry/catalog.py` — `register_integration()` reconstruction omits field
  - `intergrax/integrations/registry/plugin_register.py` — sets `requires_local_container` from manifest at registration
- **Confidence:** HIGH — field dropped in single normalization path.
- **Target invariant:** Catalog normalization must preserve deployment/security metadata exactly. Add conformance proof that registration round-trips deployment/security metadata. Do not overstate current runtime impact; finding is metadata-integrity defect, not proven active security bypass.

### AUDIT-20260818-INTEGRATIONS-05

- **Severity:** MEDIUM
- **Category:** IDENTITY CONTRACT GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** INTEGRATIONS-CONTRACT-METADATA-INTEGRITY
- **Claim falsified:** `PlatformIntegrationContract` has one canonical integration identity truth; registry v2 `(provider_id, category)` identity and the base integration contract cannot disagree.
- **Substance:** Architecture defines canonical integration identity: `integration_id = provider_id:integration_kind`. `derive_platform_integration_id()` and `for_provider()` implement this. But `PlatformIntegrationContract` itself accepts independent `integration_id`, `provider_id`, and `integration_kind` without enforcing their consistency. A directly constructed contract can contain contradictory canonical identity fields.
- **Evidence:**
  - `intergrax/runtime/integrations/contracts.py` — `derive_platform_integration_id()`, `PlatformIntegrationContract` fields, `for_provider()` derivation
  - `intergrax/runtime/integrations/registry_v2.py` — `(provider_id, category)` registry identity
- **Confidence:** HIGH — direct construction path has no equality validator.
- **Target invariant:** Either derive `integration_id` from `provider_id` + `integration_kind` or validate strict equality at construction. Registry v2 identity and the base integration contract must not disagree.

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Integration / Tool / Skill / Agent responsibility split | NOT falsified — sound separation |
| Host owns provider selection via `IntegrationProfile` | NOT falsified |
| Slug/catalog runtime resolution validates category membership (factory path) | NOT falsified for factory path; instance path is finding 01 |
| `IntegrationProfile` rejects unknown structural fields | NOT falsified |
| `IntegrationManifest` frozen extra-forbid | NOT falsified |
| Registry v2 validates category contract, integration class, integration kind, config class, disabled-by-default | NOT falsified |
| Registry v2 additive metadata today; INTEGRATIONS-3B runtime authority Planned | NOT falsified — architecture honest |
| Broad catalog scale ≠ universal production qualification | NOT falsified |
| No second Integration runtime/registry required | NOT falsified |

## Root-cause remediation grouping

### INTEGRATIONS-RUNTIME-BINDING-INTEGRITY — typed pre-built instances, lifecycle eligibility, startup resolvability

**Findings:** `AUDIT-20260818-INTEGRATIONS-01`, `AUDIT-20260818-INTEGRATIONS-02`, `AUDIT-20260818-INTEGRATIONS-03`

One fail-closed host provider binding boundary covering instance contract validation, lifecycle qualification policy, and fail-fast host readiness resolvability. Coordinate with planned INTEGRATIONS-3B rather than building a competing runtime resolver.

### INTEGRATIONS-CONTRACT-METADATA-INTEGRITY — lossless catalog metadata and canonical integration identity

**Findings:** `AUDIT-20260818-INTEGRATIONS-04`, `AUDIT-20260818-INTEGRATIONS-05`

Catalog registration round-trip for deployment/security metadata and single canonical identity on `PlatformIntegrationContract`.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `f15813cf5d2ffbd29f11a22daa1906a07e6ce23d`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification at full catalog scale.
- Remediation not performed in this task.
- Historical INTEGRATIONS-1A/1B/2A–2E **Done** rows and INTEGRATIONS-3A additive registry facts remain valid — not rewritten.

## Open questions / blocked items

- 02: exact product policy for BETA opt-in vs DEPRECATED compatibility — operator decision deferred to remediation (record invariant, not product policy).
- 03: whether plugin authoring-phase deferral remains supported after startup validation lands — coordinate with INTEGRATIONS-3B.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 5 (`AUDIT-20260818-INTEGRATIONS-01` … `AUDIT-20260818-INTEGRATIONS-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
