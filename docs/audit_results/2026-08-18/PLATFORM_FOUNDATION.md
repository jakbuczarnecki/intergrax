# PLATFORM_FOUNDATION — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** PLATFORM_FOUNDATION
- **Tier(s):** Tier-0 Platform Foundation · tier topology · gate spine · foundation proof runners
- **audited_sha:** `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 0 MEDIUM / 1 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-19
- **Architecture doc(s):**
  - `docs/project/architecture/PLATFORM_FOUNDATION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md`
- **Scope in:**
  - four-tier topology and dependency direction as documented
  - tier-boundary enforcement posture and proof completeness
  - foundation proof runners (`intergrax doctor --ci`, umbrella gates)
  - CI integration-path coverage for shared `development`
  - documented vs actual harness PR gate contract
  - legacy `DeploymentTier` contract cleanliness
- **Scope out:**
  - remediation implementation
  - unrelated domain internals (runtime, agents, applications)
  - complete security audit
  - re-audit of other Protocol v2 layers
- **Prior audit reference(s):** PF-TIER-ENFORCEMENT snapshot `4c92e0a` (plan §6.1ax); Protocol v2 [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) (related TL-FIX-A themes)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `60eff55ca7105cc8d277201c95785b4c037e3bd9`

## Executive summary

**Verdict: FAIL.** Six accepted findings (5 HIGH, 1 LOW) show incomplete authoritative package→tier enforcement (revalidating open PF-TIER-ENFORCEMENT / TL-FIX-A), fail-open script resolution in `intergrax doctor --ci`, basename-only path resolution and short-circuit execution in the umbrella AUDIT-IDEAL gate, missing automated protection on direct pushes to shared `development`, plan↔CI drift on required harness PR gates, and a deprecated unused `DeploymentTier.PRODUCT` alias. Positive controls: canonical four-tier model, explicit dependency rules, dedicated tier guard scripts, and an existing `scripts/ci/script_paths.py` canonical-path registry not yet consumed by foundation proof runners. PF-02 and PF-03 are proof-runner integrity defects — not reasons to redesign the four-tier model.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 0 MEDIUM / 1 LOW

## Findings

### AUDIT-20260818-PLATFORM_FOUNDATION-01

**No authoritative executable package→tier classification or complete semantic forbidden dependency matrix**

- **Severity:** HIGH
- **Category:** ARCHITECTURE DEFECT / PROOF
- **Status at publication:** ACCEPTED
- **Remediation block:** TL-FIX-A (§6.1ax PF-TIER-ENFORCEMENT)
- **Claim falsified:** One authoritative fail-closed package→tier mechanism mechanically enforces the complete forbidden Tier dependency matrix on the integration path.
- **Observation:** Intergrax documents strict Tier-0 → Tier-1 → Tier-2 → Tier-3 dependency direction, but enforcement remains fragmented across guards that manually enumerate `SCAN_ROOTS`, duplicate classification knowledge, use regex/text matching rather than one complete semantic dependency classifier, and do not prove the full forbidden matrix. This revalidates the already-open PF-TIER-ENFORCEMENT / TL-FIX-A problem recorded in plan §6.1ax.
- **Location:**
  - `scripts/check_no_upward_application_imports.py` — `SCAN_ROOTS`, `FORBIDDEN` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `scripts/maintenance/check_intergrax_no_applications_imports.py` — duplicate `SCAN_ROOTS` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `scripts/maintenance/check_agents_no_tier3_imports.py` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.1ax @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. `git show f21d5c3dc417907acb50d597642d3892e704bd47:scripts/check_no_upward_application_imports.py` — inspect manually enumerated `SCAN_ROOTS` and regex `FORBIDDEN`.
  2. Compare with plan §6.1ax deliverables A–F requiring authoritative classifier, full matrix, fail-closed discovery, semantic analysis, tests, and CI wiring.
- **Impact:** New or misclassified production packages or unsupported upward dependency forms can escape mechanical proof.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_FOUNDATION-02

**`intergrax doctor --ci` resolves scripts by basename and treats missing required checks as PASS-like skip**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / PROOF
- **Status at publication:** ACCEPTED
- **Remediation block:** PF-PROOF-INTEGRITY
- **Claim falsified:** Foundation CI proof via `intergrax doctor --ci` is fail-closed for required checks and resolves each declared script through canonical repository paths.
- **Observation:** `run_doctor` resolves every declared check as `root / "scripts" / script_name` while canonical files live under paths such as `scripts/maintenance/...` or `scripts/gates/...` (see `scripts/ci/script_paths.py`). `_run_script()` returns `(True, "skip missing …")` when the resolved path is not a file, so a missing health guard contributes a PASS-like result instead of failing closed.
- **Location:**
  - `intergrax/cli/doctor.py` — `_run_script()`, `scripts` list, `root / "scripts" / script_name` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `scripts/ci/script_paths.py` — `SCRIPT_PATHS` canonical registry (not consumed by doctor) @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. `git show f21d5c3dc417907acb50d597642d3892e704bd47:intergrax/cli/doctor.py` — `_run_script` skip-missing branch; basename-only resolution in `run_doctor`.
  2. Compare declared names (e.g. `check_agents_no_tier3_imports.py`) with `SCRIPT_PATHS` entries mapping to `maintenance/check_agents_no_tier3_imports.py`.
- **Impact:** Doctor CI can report success while required foundation guards were never executed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_FOUNDATION-03

**Umbrella AUDIT-IDEAL gate resolves basenames incorrectly and short-circuits remaining checks**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / TEST GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** PF-PROOF-INTEGRITY
- **Claim falsified:** Umbrella foundation gates resolve canonical script paths and execute the complete intended check set, collecting failure state without short-circuiting execution.
- **Observation:** `scripts/gates/check_audit_ideal_gates.py` resolves each script as `REPO_ROOT / "scripts" / script` despite canonical subdirectory paths. The main loop uses `exit_code = exit_code or _run(script, *extra)`, which stops invoking subsequent checks after the first non-zero result, so later checks in the declared set may never run.
- **Location:**
  - `scripts/gates/check_audit_ideal_gates.py` — `_run()`, `REPO_ROOT / "scripts" / script`, `exit_code = exit_code or _run(...)` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `scripts/ci/script_paths.py` — canonical path map @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. `git show f21d5c3dc417907acb50d597642d3892e704bd47:scripts/gates/check_audit_ideal_gates.py` — basename resolution and `exit_code or` short-circuit loop.
  2. Note doctor invokes this gate as `gates/check_audit_ideal_gates.py` while internal `_run` still prefixes only `scripts/`.
- **Impact:** Umbrella gate may silently skip checks and under-report failure breadth.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_FOUNDATION-04

**Shared `development` integration branch lacks push-triggered regression protection**

- **Severity:** HIGH
- **Category:** RELIABILITY / CI / PROOF
- **Status at publication:** ACCEPTED
- **Remediation block:** TL-FIX-A (§6.1ax PF-TIER-ENFORCEMENT deliverable F)
- **Claim falsified:** The active shared integration path receives appropriate automated regression/tier protection.
- **Observation:** Shared `development` is the active integration branch, but `.github/workflows/unit-tests.yml` `push` trigger covers `main` only, not direct pushes to `development`. At audited SHA the pinned tree had no combined commit statuses evidencing branch protection on that path.
- **Location:**
  - `.github/workflows/unit-tests.yml` — `push.branches: [main]` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. `git show f21d5c3dc417907acb50d597642d3892e704bd47:.github/workflows/unit-tests.yml` — inspect `on.push.branches`.
  2. Compare with plan §6.1ax deliverable F and TL-FIX-A integration-path requirement.
- **Impact:** Direct commits to `development` may bypass the same automated gate suite relied on for `main`/PR qualification.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_FOUNDATION-05

**Documented harness PR gate contract exceeds actual CI smoke wiring**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** PF-PROOF-INTEGRITY
- **Claim falsified:** Documentation and actual required CI enforcement describe the same gate contract for harness PR qualification.
- **Observation:** Platform Foundation plan §6.1 lists a wide verify set on every harness PR, including all three tier guards (`check_agents_no_tier3_imports.py`, `check_intergrax_no_applications_imports.py`, `check_no_upward_application_imports.py`) and `uv run intergrax doctor --ci`. Audited PR/smoke workflow `ci-smoke` tier-boundary step runs only `check_harness_no_getattr.py` and `check_agents_no_tier3_imports.py`; `intergrax doctor --ci` runs in the `gate-governance-wiring` job conditioned on schedule or `workflow_dispatch` full profile — not the default PR smoke path.
- **Location:**
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.1 verify block @ `f21d5c3dc417907acb50d597642d3892e704bd47`
  - `.github/workflows/unit-tests.yml` — `ci-smoke` tier step; `gate-governance-wiring` job `if` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. Compare plan §6.1 declared PR verify list with `ci-smoke` and `gate-governance-wiring` job conditions in workflow YAML at audited SHA.
- **Impact:** Operators and contributors may believe tier/doctor proof runs on every harness PR when smoke CI executes a subset.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PLATFORM_FOUNDATION-06

**Deprecated `DeploymentTier.PRODUCT` alias remains without evidenced production consumer**

- **Severity:** LOW
- **Category:** LEGACY / CONTRACT CLEANLINESS
- **Status at publication:** ACCEPTED
- **Remediation block:** §6.1ax PF-TIER-ENFORCEMENT deliverable H (subordinate)
- **Claim falsified:** Tier metadata contracts contain only active, non-deprecated production labels.
- **Observation:** `DeploymentTier.PRODUCT` / `TIER_PRODUCT` remains as deprecated alias for `AGENT`. Audited evidence found no active production dependency requiring the alias; cleanup is deferred to implementation-time revalidation.
- **Location:**
  - `intergrax/agent_kit/tiers.py` — `DeploymentTier.PRODUCT`, `TIER_PRODUCT` @ `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Reproduction:**
  1. `git show f21d5c3dc417907acb50d597642d3892e704bd47:intergrax/agent_kit/tiers.py` — inspect `PRODUCT` alias and exports.
- **Impact:** Low — contract noise and potential confusion; no evidenced production breakage at audited SHA.
- **Confidence:** CONFIRMED

## Falsification log (negative results)

1. **Four-tier model invalid** — not falsified; topology and dependency direction remain documented and architecturally sound.
2. **No tier guards exist** — not falsified; dedicated scripts exist under `scripts/` and `scripts/maintenance/`.
3. **Canonical script path registry absent** — not falsified; `scripts/ci/script_paths.py` exists but is not wired into foundation proof runners (PF-02/PF-03 concern is consumption, not absence of registry).
4. **PF-02/PF-03 require tier model redesign** — not falsified; defects are proof-runner path resolution and execution semantics only.

## Prior-audit comparison

Revalidates and extends PF-TIER-ENFORCEMENT (`4c92e0a`, plan §6.1ax) and overlaps TL-FIX-A themes from [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) with foundation-layer focus on proof runners and CI/docs gate-contract parity. First canonical Protocol v2 `PLATFORM_FOUNDATION` layer snapshot at `f21d5c3dc417907acb50d597642d3892e704bd47`.

## Provider / backend abstraction

`NOT APPLICABLE — PLATFORM_FOUNDATION scope is tier topology, enforcement posture, and foundation proof runners; no material external provider/backend substitution boundary in this layer.`

## Positive controls

1. **Canonical four-tier model** — Tier-0..3 topology and strict dependency direction documented in `docs/project/architecture/PLATFORM_FOUNDATION.md` @ audited SHA.
2. **Dedicated tier import guards** — `check_no_upward_application_imports.py`, `check_intergrax_no_applications_imports.py`, `check_agents_no_tier3_imports.py`.
3. **Canonical script path registry** — `scripts/ci/script_paths.py` `SCRIPT_PATHS` maps basenames to `scripts/<subdir>/...` paths (foundation runners should consume this or equivalent).
4. **Open enforcement qualification** — plan §6.1ax and architecture hub already record enforcement as incomplete (`CONDITIONALLY SOUND — ENFORCEMENT REMEDIATION REQUIRED`).

**FAIL qualification:** verdict means enforcement and foundation proof integrity are not closed — **not** that the four-tier architecture is invalid.

## Root-cause remediation grouping

Planning only — **not implemented** by this persistence task.

### TL-FIX-A / PF-TIER-ENFORCEMENT — tier enforcement and integration-path protection

**Findings:** 01, 04

Reuse existing §6.1ax PF-TIER-ENFORCEMENT and TL-FIX-A; do **not** create competing remediation architecture.

### PF-PROOF-INTEGRITY — foundation proof and gate-contract parity

**Findings:** 02, 03, 05

Canonical script-path resolution for foundation proof runners; fail-closed required check resolution; umbrella gates execute full intended check set without short-circuit omission; CI/docs gate-contract parity.

### Legacy cleanup (subordinate)

**Finding:** 06

Assess/remove `DeploymentTier.PRODUCT` if revalidation confirms no real consumer — attached to §6.1ax deliverable H.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `f21d5c3dc417907acb50d597642d3892e704bd47`; current `development` HEAD was not re-audited.
- PF-01 does not claim every current import is invalid — only that mechanical proof is incomplete.
- PF-02/PF-03 do not claim the four-tier model is wrong — only that proof runners can report false confidence.
- PF-04 does not prescribe a specific GitHub branch strategy — only that the active integration path must receive appropriate automated protection.
- PF-06 does not claim confirmed production dependency on `PRODUCT` alias.
- Remediation not performed in this task.

## Open questions / blocked items

- Exact GitHub workflow design for `development` protection left to implementation (PF-04) while preserving architectural flexibility.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-19
- **Accepted findings:** all 6 (`AUDIT-20260818-PLATFORM_FOUNDATION-01` … `AUDIT-20260818-PLATFORM_FOUNDATION-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
