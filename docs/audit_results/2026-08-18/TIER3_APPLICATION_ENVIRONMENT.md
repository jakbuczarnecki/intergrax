# TIER3_APPLICATION_ENVIRONMENT — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** TIER3_APPLICATION_ENVIRONMENT
- **Tier(s):** Tier-3 `intergrax/applications/` — manifest, environment profile, `wire_application_environment()`, `ApplicationEnvironmentWiring`, `EnvironmentSnapshot`, host composition boundary
- **audited_sha:** `6ed70b6f3231a1514876244872b441c02cde788d`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
- **Scope in:**
  - `ApplicationManifest`, `ApplicationEnvironmentProfile`, `AgentBinding[]`
  - `wire_application_environment()` canonical composition entry
  - `ApplicationEnvironmentWiring` frozen output contract
  - `EnvironmentSnapshot` deploy/intake materialization
  - environment conformance validators (`EnvironmentSkillToolConsistencyCheck`, `ProfileInvariantValidator`)
  - `RuntimeEventBus` wiring through `ApplicationBuildContext`
  - integration-profile resolution (`integration_profile` / `env.integration_profile` / `manifest.integration_profile`)
  - sandbox session materialization in composition path
  - historical Tier-3 **Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - second Tier-3 composition subsystem invention
  - rewriting historical Done rows or universal production qualification claims
- **Prior audit reference(s):** [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) TL-FIX-C/D; [`INTERFACE_TASK_INTAKE`](INTERFACE_TASK_INTAKE.md) ITI-FIX-*; [`IDENTITY_TRUST`](IDENTITY_TRUST.md) IDT-FIX-A
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that canonical Tier-3 composition can proceed after fail-open conformance validation; silently mint a local `RuntimeEventBus` when none is supplied; materialize sandbox sessions with synthetic `harness/bootstrap` tenant/task identity; resolve conflicting integration profiles through truthiness precedence without mismatch detection; emit `EnvironmentSnapshot` evidence that proves configuration identity but not execution binding; and expose platform-significant wiring artifacts (including `policy_bundle`) as `Any`. Positive controls: Tier-3 remains a platform adopter, not a second runtime; composition vs Application Hosting boundary remains sound; manifest/registry/capability-graph assembly validation remains fail-closed; historical TL-FIX-C/D, ITI-FIX-*, and IDT-FIX-A remain correctly **PLANNED**; maturity remains **A4/I3/P3/E3** unless independently re-assessed. Remediation is **PLANNED**, not implemented. Findings harden the existing composition root — no second Tier-3 subsystem required.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-01

- **Severity:** HIGH
- **Category:** CONFIGURATION INTEGRITY / FAIL-OPEN CONFORMANCE
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-COMPOSITION-AUTHORITY-INTEGRITY
- **Claim falsified:** Product/STRICT Tier-3 composition fails closed on required profile/roster invariant violations; diagnostic validation is explicitly distinct from authoritative composition conformance.
- **Observation:** `wire_application_environment()` defaults `conformance_check=True`. Within the canonical conformance block it executes `EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_roster(...)` and `ProfileInvariantValidator(fail_on_violation=False).validate(env)` and discards both returned violation lists. Both validators support fail-closed behavior when `fail_on_violation=True`. Current detected invariants include RAG-enabled requires vector-store integration, websearch enabled requires corresponding tool capability, and agent tool/skill requirements fit environment profiles. Canonical Tier-3 composition can therefore continue after these validators detect semantic violations.
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — `wire_application_environment()`, conformance block
  - `intergrax/applications/_shared/environment_conformance.py` — `EnvironmentSkillToolConsistencyCheck`, `ProfileInvariantValidator`
- **Reproduction:** Wire a profile with RAG enabled but no vector-store integration (or other known invariant violation); observe composition succeeds while validators would fail closed with `fail_on_violation=True`.
- **Impact:** Product/STRICT hosts can compose with semantically invalid environment posture; conformance appears authoritative but is advisory-only.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-02

- **Severity:** HIGH
- **Category:** OBSERVABILITY / GOVERNANCE SPINE SPLIT
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-RUNTIME-SCOPE-INTEGRITY
- **Claim falsified:** One authoritative `RuntimeEventBus` per runtime composition; Tier-3 canonical wiring does not silently create an isolated bus when the runtime owner already has a spine.
- **Observation:** `wire_application_environment()` accepts `runtime_event_bus: RuntimeEventBus | None = None` and passes into `ApplicationBuildContext`: `runtime_event_bus or RuntimeEventBus()`. Absence of an explicitly supplied bus silently creates a fresh local `RuntimeEventBus`. For hosts with an existing Nexus/runtime/Observability event spine this permits Tier-3 build/runtime artifacts to use a different bus without configuration failure.
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — `wire_application_environment()`, `ApplicationBuildContext` bus resolution
- **Reproduction:** Call `wire_application_environment()` without supplying `runtime_event_bus` while a host runtime already owns a canonical bus; observe a new local bus is created and used for Tier-3 wiring.
- **Impact:** Observability/governance events from Tier-3 composition may not reach the authoritative runtime spine; split-brain event routing without explicit lab policy.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-03

- **Severity:** HIGH
- **Category:** TENANT / EXECUTION SCOPE INTEGRITY
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-RUNTIME-SCOPE-INTEGRITY
- **Claim falsified:** Task-scoped sandbox session is materialized from canonical runtime execution identity (tenant + TaskId / required scope); no reusable production sandbox session carries synthetic `harness/bootstrap` ownership.
- **Observation:** When no sandbox session is already present and an integration profile is available, `wire_application_environment()` resolves hosted sandbox session with `tenant_id="harness"` and `task_id="bootstrap"`. The resulting `hosted_session` is passed directly into `build_application_tool_wiring()` as the sandbox session. These identities are constants rather than trusted tenant/Task execution identity.
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — sandbox session resolution in composition path
- **Reproduction:** Wire an environment without an existing sandbox session but with integration profile enabled; inspect resolved sandbox session tenant/task identity — observe constant `harness` / `bootstrap`.
- **Impact:** Sandbox isolation scope may not match actual task execution identity; cross-task or cross-tenant sandbox ownership ambiguity in production-adjacent paths.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-04

- **Severity:** HIGH
- **Category:** CONFIGURATION AUTHORITY / COMPOSITION DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-COMPOSITION-AUTHORITY-INTEGRITY
- **Claim falsified:** One canonical effective integration-profile authority; conflicting authoritative inputs fail explicitly rather than silently resolve through truthiness precedence.
- **Observation:** Integration posture may come from three sources: explicit `integration_profile` argument, `env.integration_profile`, and `manifest.integration_profile`. Canonical resolution is `integration_profile or env.integration_profile or manifest.integration_profile`. There is no mismatch detection or typed merge authority when two public configuration sources disagree. `ApplicationManifest` exposes `integration_profile` and defaults it to a lab profile. Effective integration configuration affects tools, RAG, Memory, sandbox, integration health, and provider wiring.
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — integration profile resolution
  - `intergrax/applications/contracts/manifest.py` — `ApplicationManifest.integration_profile`
  - `intergrax/applications/contracts/environment_profile/root.py` — profile integration field
- **Reproduction:** Supply conflicting integration profiles via manifest and explicit argument; observe silent precedence without mismatch error.
- **Impact:** Effective integration posture can drift from author intent; product vs lab profile confusion without explicit contract.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-05

- **Severity:** MEDIUM
- **Category:** SNAPSHOT PROVENANCE / EXECUTION IDENTITY
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-SNAPSHOT-PROVENANCE-INTEGRITY
- **Claim falsified:** For Task intake, canonical evidence proves Task/Run identity ↔ exact `EnvironmentSnapshot`; deploy configuration snapshot and execution binding evidence have explicit semantics.
- **Observation:** `EnvironmentSnapshot` contains application/profile/manifest/graph/org/roster identity and capture metadata but no tenant/task/run execution identity. The contract is used both for deploy and Task-intake materialization and is placed into Task metadata. It therefore proves configuration identity but does not itself prove the execution scope to which that snapshot was bound.
- **Location:**
  - `intergrax/applications/contracts/environment_snapshot.py` — `EnvironmentSnapshot` fields
  - `intergrax/applications/_shared/environment_wiring.py` — snapshot materialization on intake/deploy
- **Reproduction:** Materialize snapshot on Task intake; inspect snapshot fields and Task metadata — observe configuration digests without tenant/task/run binding evidence.
- **Impact:** Audit and replay cannot prove which execution scope consumed a given snapshot without external inference.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-06

- **Severity:** MEDIUM
- **Category:** CONTRACT QUALITY / TYPE BOUNDARY
- **Status at publication:** ACCEPTED
- **Remediation block:** T3-COMPOSITION-AUTHORITY-INTEGRITY
- **Claim falsified:** Platform-significant canonical wiring artifacts use concrete types, Protocols, or typed unions/generics; policy/governance artifacts are not typed as arbitrary `Any`.
- **Observation:** Canonical `ApplicationEnvironmentWiring` exposes `policy_bundle: Any`. The central wiring entry also accepts several platform-significant values through `Any`, including settings/integration_profile/sandbox_session/document_store/boundary_event_buffer. This weakens a composition API whose purpose is to be the typed canonical boundary between Tier-3 applications and platform mechanisms.
- **Location:**
  - `intergrax/applications/_shared/environment_wiring.py` — `ApplicationEnvironmentWiring`, `wire_application_environment()` parameters
- **Reproduction:** Inspect `ApplicationEnvironmentWiring` and wiring entry signatures — observe `Any` on policy and other platform-significant fields.
- **Impact:** Composition boundary loses static contract guarantees; policy/governance drift harder to detect at compile/type-check time.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Tier-3 composition vs Application Hosting boundary remains sound | NOT falsified |
| Tier-3 remains a platform adopter, not a second runtime | NOT falsified |
| `ApplicationManifest` uses `extra=forbid` and validates major identity/routing fields | NOT falsified |
| At most one enabled default `AgentBinding` is enforced | NOT falsified |
| `ApplicationEnvironmentWiring` is frozen | NOT falsified |
| Registry assembly validation remains fail-closed | NOT falsified |
| Capability-graph assembly validation remains fail-closed | NOT falsified |
| Manifest package closure remains fail-closed | NOT falsified |
| Environment/profile nested-bundle model remains canonical | NOT falsified |
| Historical TL-FIX-C / TL-FIX-D remain correctly **PLANNED** | NOT falsified |
| ITI-FIX-A / ITI-FIX-D remain correctly **PLANNED** | NOT falsified |
| IDT-FIX-A remains correctly **PLANNED** | NOT falsified |
| Historical Tier-3 Done rows remain delivery facts, not universal production qualification | NOT falsified |
| Current maturity remains A4/I3/P3/E3 unless independently re-assessed | NOT falsified |
| Findings harden existing composition root; no second Tier-3 subsystem required | NOT falsified |

## Historical Tier-3 delivery vs Protocol-v2 residual defects

Historical **L3 / Done** plan rows and AUDIT-IDEAL closeout labels remain valid harness delivery facts — manifest/profile bundles, `wire_application_environment()`, `EnvironmentSnapshot`, `UnifiedTaskRunner` convergence on reference hosts, and production gate scripts were delivered as claimed. The six accepted Protocol-v2 findings document **residual composition-authority, runtime-scope, snapshot-provenance, and typed-boundary gaps** at `audited_sha`. Remediation hardens the existing composition root; it does **not** reopen closed historical Done rows, claim universal production qualification, or require a second Tier-3 subsystem.

## Root-cause remediation grouping

### T3-COMPOSITION-AUTHORITY-INTEGRITY — typed configuration authority and blocking conformance

**Findings:** `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-01`, `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-04`, `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-06`

Canonical Tier-3 composition has one typed configuration authority and blocking conformance semantics where required. Cross-link **TL-FIX-C/D** where ownership overlaps; do not duplicate boundary remediation.

### T3-RUNTIME-SCOPE-INTEGRITY — canonical event spine and execution-scoped sandbox identity

**Findings:** `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-02`, `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-03`

Tier-3 reuses the canonical event spine and execution-scoped sandbox identity rather than silently minting isolated/synthetic runtime authorities. Cross-link [`OBSERVABILITY_EVIDENCE`](../../project/architecture/OBSERVABILITY_EVIDENCE.md) and [`IDENTITY_TRUST`](../../project/architecture/IDENTITY_TRUST.md) / **IDT-FIX-A** rather than create parallel event or identity subsystems.

### T3-SNAPSHOT-PROVENANCE-INTEGRITY — Task/Run ↔ EnvironmentSnapshot binding

**Findings:** `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-05`

Configuration snapshot evidence is explicitly bound to the Task/Run execution that consumed it. Reuse canonical execution identity; binding may be a separate typed artifact rather than overloading `EnvironmentSnapshot` with every execution field.

## Cross-links to existing remediation

| Existing block | Relationship |
|----------------|--------------|
| **TL-FIX-C** / **TL-FIX-D** | Remain **PLANNED** — product-neutral contracts and public composition API; coordinate with T3-COMPOSITION-AUTHORITY-INTEGRITY where overlap exists |
| **ITI-FIX-A** / **ITI-FIX-D** | Remain **PLANNED** — intake normalization and streaming parity; orthogonal to composition authority except shared Tier-3 ownership |
| **IDT-FIX-A** | Remain **PLANNED** — canonical execution identity spine; T3-RUNTIME-SCOPE-INTEGRITY and T3-SNAPSHOT-PROVENANCE-INTEGRITY cross-link rather than duplicate |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `6ed70b6f3231a1514876244872b441c02cde788d`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical Tier-3 **Done** plan rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- Finding 04: manifest as default/reference vs explicit override merge semantics — deferred to remediation design.
- Finding 05: separate binding artifact vs snapshot field extension — deferred to remediation; reuse canonical execution identity.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-01` … `AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED. **TL-FIX-C/D**, **ITI-FIX-***, and **IDT-FIX-A** remain **ACCEPTED / PLANNED**.
