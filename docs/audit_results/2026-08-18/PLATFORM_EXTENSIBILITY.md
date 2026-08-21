# PLATFORM_EXTENSIBILITY — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** PLATFORM_EXTENSIBILITY
- **Owning architecture/program:** PLATFORM_PLUGINS
- **Tier(s):** Tier-0 `intergrax/core/plugins/` — package contract, discovery, qualification, admission; Tier-3 `intergrax/applications/_shared/` wiring; Policy Rule plugin loader
- **audited_sha:** `70c947c889f40222e5efb191241bdd8fa9035b17`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/PLATFORM_PLUGINS.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/PLATFORM_PLUGINS.md`
- **Scope in:**
  - `PluginQualificationResult.production_allowed` and `evaluate_package_production_admission()`
  - `PlatformPluginPackageQualificationBundle` package-level indexing and `lookup_for_entry_point()`
  - `evaluate_external_package_entry_point_production_admission()` manifest/capability binding
  - Tier-3 `ApplicationEnvironmentProfile` / `environment_wiring` qualification bundle handoff
  - Policy Rule/Definition plugin loader `require_production_admission` enforcement
  - installed manifest resolution and `PlatformPluginManifest` / `CapabilityDescriptor` consumption
  - shared entry-point discovery cache lifecycle (`iter_entry_point_specs`, `reset_entry_point_spec_cache_for_tests`)
  - historical PLATFORM-PLUGIN-1..9 Done/CLOSED delivery as positive controls
  - PROVIDER-QUAL ongoing track relationship (cross-link, no duplication)
- **Scope out:**
  - remediation implementation
  - source/test/CI/script/packaging changes
  - creating a competing `PLATFORM_EXTENSIBILITY` architecture subsystem (canonical ownership remains **PLATFORM_PLUGINS**)
  - rewriting historical PLATFORM-PLUGIN closeout rows
  - reopening PROVIDER-QUAL ongoing work
- **Prior audit reference(s):** [`POLICY_GOVERNANCE`](POLICY_GOVERNANCE.md); [`INTEGRATIONS`](INTEGRATIONS.md); [`TOOLS`](TOOLS.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Scope / ownership mapping

| Concept | Canonical name |
|---------|----------------|
| Audit unit (Protocol v2 layer code) | **PLATFORM_EXTENSIBILITY** |
| Architecture / program ownership | **PLATFORM_PLUGINS** |
| Per-layer report | `docs/audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md` |
| Target invariants | `docs/project/architecture/PLATFORM_PLUGINS.md` — [Protocol v2 platform extensibility target invariants (2026-08-18)](#protocol-v2-platform-extensibility-target-invariants-2026-08-18) |
| Planned remediation | `docs/project/maintainers/plans/PLATFORM_PLUGINS.md` — Protocol v2 remediation blocks |

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that production qualification can be asserted by enum status without evidence-derived authority; production-admission enforcement is implemented for Policy plugins but not uniformly across other public PEP domains; package-level qualification applies to every entry point from a distribution without capability/domain binding; production admission checks package identity and compatibility but not exact manifest capability/entry-point binding; invalid manifests collapse to the same `compatibility=None` state as absent manifests; and entry-point discovery cache semantics are incidental with no explicit process-lifetime policy. Positive controls: Platform Plugin remains package-level coordination, not a universal executable wrapper; domain runtime contracts remain authoritative; scan-only `EntryPointSpec` supports pre-load admission; Policy production admission executes before plugin target loading; manifests reject duplicate capability descriptors; third-party plugins remain explicitly trusted-in-process; discovery ≠ production qualification; DO-NOT-UNIFY decisions and historical PLATFORM-PLUGIN-1..9 delivery facts remain valid. Remediation is **PLANNED**, not implemented. Findings harden PLATFORM_PLUGINS coordination — no second Tools/Skills/RAG/Policy/Integration runtime.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-01

- **Severity:** HIGH
- **Category:** QUALIFICATION AUTHORITY / SELF-ATTESTATION
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-QUALIFICATION-AUTHORITY-INTEGRITY
- **Claim falsified:** Production qualification status is derived/validated against a versioned qualification policy and required evidence set; a caller cannot create authoritative production qualification merely by setting an enum.
- **Observation:** `PluginQualificationResult.production_allowed` returns true solely when `status == PRODUCTION_QUALIFIED`. The contract does not require any evidence kind or non-empty evidence for that status. `evaluate_package_production_admission()` trusts `production_allowed` after compatibility checks without validating evidence against a qualification policy.
- **Location:**
  - `intergrax/core/plugins/platform_qualification.py` — `PluginQualificationResult.production_allowed`, `evaluate_package_production_admission()`
- **Reproduction:** Construct `PluginQualificationResult` with `status=PRODUCTION_QUALIFIED` and `evidence=()`; pass through `evaluate_package_production_admission()` with compatible platform metadata; observe admission with reason `"production-qualified evidence present"`.
- **Impact:** Self-attested production qualification undermines trust model in §18; reuse `intergrax.core.qualification` — no second qualification engine.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-02

- **Severity:** HIGH
- **Category:** ARCHITECTURE / PARTIAL PRODUCTION ENFORCEMENT
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-ADMISSION-COVERAGE-INTEGRITY
- **Claim falsified:** Each public external plugin domain consumes one shared pre-registration production-admission contract in strict/product profiles; no domain bypasses the common boundary while retaining its own loader/registry.
- **Observation:** Canonical Tier-3 wiring accepts `PlatformPluginPackageQualificationBundle` via `platform_plugin_package_qualifications` in `environment_wiring.py` and passes it to policy wiring. Repository production-admission enforcement (`require_production_admission`) is implemented for Policy Rule/Policy Definition plugin loading in `plugin_loader.py`. No equivalent common Platform Plugin production-admission enforcement was found across all other public extension domains (Tools, Skills, Integrations, Context, RAG EP groups, etc.).
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — qualification bundle parameter
  - `intergrax/applications/_shared/policy_wiring.py` — `require_production_admission`, `package_qualification_lookup`
  - `intergrax/runtime/policy/rules/plugin_loader.py` — `_production_admission_rejections()`
- **Reproduction:** Enable strict execution mode; compare Policy plugin load path (production admission enforced) with other domain loaders that discover external entry points without shared admission helper.
- **Impact:** Architecture claim that external plugins require qualification before production reliance is only partially enforced; strict/product profiles may load unqualified extensions in non-Policy domains.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-03

- **Severity:** HIGH
- **Category:** QUALIFICATION SCOPE / OVER-BROAD AUTHORITY
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-QUALIFICATION-AUTHORITY-INTEGRITY
- **Claim falsified:** Package qualification may be a prerequisite, but capability/domain admission is bound separately where the domain requires it (distribution identity + domain + exact capability/EP + qualification policy + evidence).
- **Observation:** Qualification model declares PACKAGE / CAPABILITY / DOMAIN levels. `PlatformPluginPackageQualificationBundle` accepts PACKAGE-level subjects only and indexes qualification by exact distribution name/version. `lookup_for_entry_point()` returns that package qualification for every entry point from the same distribution. One package-level production qualification can therefore apply to multiple different capabilities/domains from the same multi-capability package without capability-scoped evidence.
- **Location:**
  - `intergrax/core/plugins/platform_qualification.py` — `PlatformPluginPackageQualificationBundle`, `lookup_for_entry_point()`, `_validate_package_qualification_entry()`
  - `intergrax/core/plugins/package_contract.py` — multi-capability `PlatformPluginManifest`
- **Reproduction:** Qualify package `foo@1.0` at PACKAGE level for one capability; load a different entry point from the same distribution via `lookup_for_entry_point()`; observe same production qualification returned.
- **Impact:** Over-broad qualification authority across capabilities within one distribution.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-04

- **Severity:** HIGH
- **Category:** MANIFEST / QUALIFICATION BINDING
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-QUALIFICATION-AUTHORITY-INTEGRITY
- **Claim falsified:** Production admission binds exact distribution + manifest identity/hash + exact capability descriptor/EP + qualification result; a package cannot gain production admission for an undeclared/unqualified capability merely because another capability in the same distribution qualified.
- **Observation:** `CapabilityDescriptor` contains domain, entry_point_group, entry_point_name, and capability_ids. `evaluate_external_package_entry_point_production_admission()` checks package identity, qualification status, and platform compatibility from installed manifest. It does not verify that the `EntryPointSpec` being admitted is declared by the qualified `PlatformPluginManifest` capability set. Manifest is consumed for `platform_compatibility`, not exact capability admission binding.
- **Location:**
  - `intergrax/core/plugins/platform_qualification.py` — `evaluate_external_package_entry_point_production_admission()`, `resolve_installed_distribution_platform_compatibility()`
  - `intergrax/core/plugins/package_contract.py` — `CapabilityDescriptor`, `PlatformPluginManifest`
- **Reproduction:** Admit entry point whose group/name is absent from manifest capabilities but package has PRODUCTION_QUALIFIED PACKAGE qualification and compatible platform metadata; observe admission without manifest capability match.
- **Impact:** Undeclared capabilities in a qualified distribution may receive production admission.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-05

- **Severity:** MEDIUM
- **Category:** DIAGNOSTICS / EVIDENCE QUALITY
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-LIFECYCLE-EVIDENCE-INTEGRITY
- **Claim falsified:** Manifest resolution distinguishes VALID / ABSENT / INVALID / UNREADABLE with safe reason codes; operator/audit evidence preserves true failure cause.
- **Observation:** `_try_parse_platform_plugin_manifest_from_distribution()` catches `PlatformPluginManifestValidationError` and returns `None`. Missing manifest and invalid manifest therefore collapse to the same downstream `compatibility=None` state in `resolve_installed_distribution_platform_compatibility()`. Admission remains fail-closed, but diagnostics lose the true cause.
- **Location:**
  - `intergrax/core/plugins/platform_qualification.py` — `_try_parse_platform_plugin_manifest_from_distribution()`, `resolve_installed_distribution_platform_compatibility()`
- **Reproduction:** Install distribution with invalid `[tool.intergrax.plugin]` manifest; resolve compatibility; observe `None` indistinguishable from absent manifest.
- **Impact:** Operator and audit evidence cannot distinguish absent vs invalid manifest failures.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_EXTENSIBILITY-06

- **Severity:** MEDIUM
- **Category:** LIFECYCLE / DISCOVERY CONSISTENCY
- **Status at publication:** ACCEPTED
- **Remediation block:** PLATFORM-EXTENSIBILITY-LIFECYCLE-EVIDENCE-INTEGRITY
- **Claim falsified:** Installed plugin lifecycle/discovery cache semantics are explicit policy (immutable process lifetime requiring restart, or controlled versioned rediscovery), not incidental cache behavior.
- **Observation:** Shared entry-point discovery caches specs per group in process. After first scan, `iter_entry_point_specs()` returns the cached snapshot. The only explicit cache reset API is `reset_entry_point_spec_cache_for_tests()`, named/documented for tests/dev bootstrap. No production lifecycle contract states whether installed plugin mutation requires host restart or whether controlled rediscovery is supported.
- **Location:**
  - `intergrax/core/plugins/discovery.py` — `_EP_SPECS_CACHE`, `iter_entry_point_specs()`, `reset_entry_point_spec_cache_for_tests()`
- **Reproduction:** Scan entry-point group; install new plugin package without restart or cache reset; observe stale cached spec set on subsequent scans.
- **Impact:** Undefined lifecycle semantics for production hosts; incidental cache behavior may hide plugin set changes.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Platform Plugin remains package-level coordination, not universal executable wrapper | NOT falsified |
| Domain runtime contracts remain authoritative | NOT falsified |
| No second Tools/Skills/RAG/Policy/Integration runtime should be introduced | NOT falsified |
| Scan-only `EntryPointSpec` exists and supports pre-load admission | NOT falsified |
| Policy production admission currently executes before plugin target loading | NOT falsified |
| Manifests are frozen/extra-forbid and reject duplicate capability descriptors | NOT falsified |
| Package qualification identity matches package name/version | NOT falsified |
| Third-party plugins remain explicitly trusted-in-process; no sandbox/signing claim | NOT falsified |
| Discovery ≠ production qualification remains the correct principle | NOT falsified |
| Existing DO-NOT-UNIFY decisions remain valid | NOT falsified |
| Historical PLATFORM-PLUGIN-1..9 Done/CLOSED facts remain delivery facts | NOT falsified |
| Protocol-v2 findings are residual architecture/implementation gaps, not a reason to erase program history | NOT falsified |

## Historical PLATFORM-PLUGIN delivery vs Protocol-v2 residual gaps

Historical **Done/CLOSED** PLATFORM-PLUGIN-1..9 rows (taxonomy freeze, author contract, shared discovery primitives, config/secrets matrix, lifecycle vocabulary, qualification contracts, dual-mode E2E proof, program closeout) remain valid delivery facts — package coordination, PEP/IP/HCE/IEP/NE taxonomy, partial discovery harmonization, and Policy-path production gates were delivered as claimed. The six accepted Protocol-v2 findings document **residual qualification authority, admission coverage, capability binding, manifest diagnostics, and discovery lifecycle** gaps at `audited_sha`. Remediation hardens existing PLATFORM_PLUGINS coordination; it does **not** reopen closed PLATFORM-PLUGIN program rows, introduce a universal `PlatformPlugin.execute()`, or unify domain loaders into one global runtime.

## PROVIDER-QUAL relationship

**PROVIDER-QUAL** extends PLUGIN-7 for **provider-scoped** evidence via `intergrax/core/qualification/` — orthogonal to package-level PEP admission gaps documented here. Protocol-v2 remediation reuses core qualification (PEXT-01) and cross-links PROVIDER-QUAL where provider evidence applies; it does **not** duplicate or overwrite PROVIDER-QUAL-1..3 ongoing rows. Package/capability admission integrity (PEXT-03/04) and cross-domain admission coverage (PEXT-02) remain PLATFORM_PLUGINS scope; provider binding and evidence persistence remain PROVIDER-QUAL scope.

## Root-cause remediation grouping

### PLATFORM-EXTENSIBILITY-QUALIFICATION-AUTHORITY-INTEGRITY — evidence-derived qualification and exact admission binding

**Priority:** P0  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-01`, `03`, `04`

Production qualification becomes evidence-derived against a versioned qualification policy (reuse `intergrax.core.qualification`; no second engine). Package qualification may remain a prerequisite, but capability/domain admission binds distribution identity + domain + exact capability/EP + policy + evidence. Production admission binds exact distribution + manifest identity/hash + capability descriptor/EP + qualification result.

### PLATFORM-EXTENSIBILITY-ADMISSION-COVERAGE-INTEGRITY — shared production-admission boundary across PEP domains

**Priority:** P0/P1  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-02`

All supported public PEP domains consume the common pre-registration production-admission contract in strict/product profiles while retaining domain loaders/contracts/registries. No global runtime plugin loader.

### PLATFORM-EXTENSIBILITY-LIFECYCLE-EVIDENCE-INTEGRITY — manifest diagnostics and discovery lifecycle policy

**Priority:** P1/P2  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-05`, `06`

Typed manifest resolution result (VALID / ABSENT / INVALID / UNREADABLE) with safe reason codes. Explicit process-lifetime policy: installed plugin set immutable for process lifetime (restart required) **or** controlled versioned rediscovery/cache invalidation — not incidental cache behavior.

## Cross-links to existing remediation

| Existing block | Relationship |
|----------------|--------------|
| **POLICY_GOVERNANCE** / Policy plugin loader | Reference implementation for `require_production_admission` — extend pattern, do not duplicate Policy runtime |
| **INTEGRATIONS** / **TOOLS** / **SKILLS** | Domain loaders retain authority; consume shared admission boundary only |
| **PROVIDER-QUAL** | Provider-scoped evidence track — cross-link for qualification engine reuse, do not overwrite ongoing rows |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `70c947c889f40222e5efb191241bdd8fa9035b17`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests and CI gates are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical PLATFORM-PLUGIN closeout rows remain valid delivery facts — not rewritten.
- Audit unit **PLATFORM_EXTENSIBILITY** maps to architecture/program **PLATFORM_PLUGINS** — no competing subsystem created.

## Open questions / blocked items

- PEXT-02: exact domain enumeration for admission coverage rollout order — deferred to remediation design.
- PEXT-06: policy choice A (immutable process lifetime) vs B (controlled rediscovery) — deferred to operator/architecture decision in remediation.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-PLATFORM_EXTENSIBILITY-01` … `AUDIT-20260818-PLATFORM_EXTENSIBILITY-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
