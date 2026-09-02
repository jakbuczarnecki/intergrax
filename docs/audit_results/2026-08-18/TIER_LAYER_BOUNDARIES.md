# TIER_LAYER_BOUNDARIES - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** TIER_LAYER_BOUNDARIES
- **Tier(s):** cross-domain Tier-0 / Tier-1 / Tier-2 / Tier-3 boundary audit
- **audited_sha:** `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Status:** COMPLETE
- **Auditor:** OpenAI ChatGPT / GPT-5.6 Sol - independent auditor
- **Verdict:** FAIL
- **Architecture doc(s):**
  - `docs/project/architecture/PLATFORM_FOUNDATION.md`
  - `docs/project/architecture/AGENT_DISTRIBUTION.md`
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md`
  - `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md`
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
- **Scope in:**
  - canonical Tier ownership and dependency direction
  - enforcement/proof completeness
  - concrete-agent ownership
  - application profile neutrality
  - public/private application composition boundaries
  - static/dynamic contract enforcement at consumer boundaries
- **Scope out:**
  - remediation
  - unrelated runtime behavior
  - complete security audit
  - complete application production qualification
  - re-audit of STRATEGIC_HARNESS_MODEL findings
  - unrelated provider/tool/RAG internals
- **Prior audit reference(s):** prior PF-TIER-ENFORCEMENT enforcement audit (`4c92e0a`) remains historical evidence; Protocol v2.1 findings below are the canonical AUDIT-2 observations for this layer.
- **post_sync_sha:** `a5d6f83d0ea274dec269377a9ce1cc4421b1bd12`
- **Exact audit-start time:** not captured; date-level UTC precision preserved.

## Executive summary

**Verdict: FAIL.** Intergrax has an explicit Tier-0 → Tier-1 → Tier-2 → Tier-3 ownership model and multiple boundary guards, but mechanical proof of tier compliance is incomplete: enforcement is fragmented across manually enumerated scanners, does not cover the full forbidden dependency matrix or all production consumer boundaries, and CI smoke invokes only a subset of tier guards on `main`/PR paths. Five accepted findings (2 HIGH, 3 MEDIUM) additionally show colliding EchoAgent production identity across Tier-1 and Tier-2, LKW-specific vocabulary in generic Tier-3 platform contracts, Tier-3 hosts mutating private `DefaultRunService` state, and static getattr discipline that omits top-level `applications/`. FAIL means enforcement and ownership are not universally guaranteed - not that the tier concept is invalid or all layers are mixed.

## Verdict

**FAIL**

## Findings

### AUDIT-20260818-TIER_LAYER_BOUNDARIES-01

**Incomplete mechanical enforcement of Tier-0..3 boundaries**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** One authoritative fail-closed package→tier mechanism mechanically enforces the complete forbidden Tier dependency matrix on the integration path relied on for development/PR qualification.
- **Observation:** Intergrax documents strict Tier-0 → Tier-1 → Tier-2 → Tier-3 dependency direction, but enforcement is split across guards that manually enumerate `SCAN_ROOTS`, duplicate classification knowledge, focus primarily on upward Tier-3/application imports, use regex/text matching rather than one complete semantic dependency classifier, and do not prove the full forbidden matrix. `PLATFORM_FOUNDATION` plan §6.1ax (PF-TIER-ENFORCEMENT) already records authoritative package→tier classification, complete matrix enforcement, fail-closed unclassified packages, semantic analysis, deterministic guard tests, and CI alignment as open/PLANNED. At audited SHA, `.github/workflows/unit-tests.yml` push trigger is `main` (not shared `development`); PR/smoke tier-boundary invocation runs `check_harness_no_getattr.py` and `check_agents_no_tier3_imports.py` only - not `check_no_upward_application_imports.py` or `check_intergrax_no_applications_imports.py` - and does not constitute the canonical enforcement described by architecture/plan.
- **Location:**
  - `scripts/check_no_upward_application_imports.py:L15-L24,L26-L28` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `scripts/maintenance/check_intergrax_no_applications_imports.py:L68-L82` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `scripts/maintenance/check_dependency_ownership.py:L125-L143` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa` (explicit manual allowlist - not package→tier classifier)
  - `.github/workflows/unit-tests.yml:L17,L100-L108` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.1ax @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Reproduction:**
  1. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:scripts/check_no_upward_application_imports.py` - inspect manually enumerated `SCAN_ROOTS` (L15-L24) and regex `FORBIDDEN` (L26-L28).
  2. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:scripts/maintenance/check_intergrax_no_applications_imports.py` - duplicate `SCAN_ROOTS` list (L73-L82).
  3. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:.github/workflows/unit-tests.yml` - `push.branches: [main]` (L17); smoke tier step (L100-L108) omits upward-application guards.
  4. Compare with plan §6.1ax deliverables A–F requiring authoritative classifier, full matrix, fail-closed discovery, semantic analysis, tests, and CI wiring.
- **Impact:** A new or misclassified production package or unsupported upward dependency form can escape mechanical proof because enforcement knowledge is partial/manual.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER_LAYER_BOUNDARIES-02

**Two production implementations own the same EchoAgent identity across Tier-1 and Tier-2**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Each production `(contract_id, agent_version)` has one canonical concrete implementation authority; Tier-1 framework packages do not silently duplicate reusable Tier-2 agents under the same production identity.
- **Observation:** Two concrete `EchoAgent` implementations exist: `agents/echo/echo_agent.py` (Tier-2 reusable agent package per `agents/echo/pyproject.toml`) and `intergrax/agents/echo/echo_agent.py` (shipped inside core wheel via `pyproject.toml` `packages = ["intergrax"]`). Both declare `contract_id = "echo"`, `agent_version = "1.0.0"`, `lifecycle_state = PRODUCTION`, `production_eligible = True`. Implementations are not behaviorally identical: the Tier-1 copy constructs additional `RuntimeConfig` / `SessionManager` / `InMemorySessionStorage` in `build_context` (L111-L121) while the Tier-2 copy uses the reference harness builder without the same local construction.
- **Location:**
  - `agents/echo/pyproject.toml:L299-L301` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `agents/echo/echo_agent.py:L62-L66,L85-L86` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `intergrax/agents/echo/echo_agent.py:L63-L67,L85-L86,L111-L121` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `pyproject.toml:L499-L500` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `docs/project/architecture/AGENT_DISTRIBUTION.md` §4, §9 @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Reproduction:**
  1. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:agents/echo/echo_agent.py` - `contract_id`, `agent_version`, `lifecycle_state=PRODUCTION`, `production_eligible=True`.
  2. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:intergrax/agents/echo/echo_agent.py` - same identity fields; additional session/runtime construction in `build_context`.
  3. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:agents/echo/pyproject.toml` - Tier-2 workspace member with `contract_id = "echo"`.
  4. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:pyproject.toml` - core wheel packages only `intergrax`.
- **Impact:** One contract identity/version has multiple implementation authorities and potentially different runtime/session semantics depending on import/package path.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER_LAYER_BOUNDARIES-03

**LKW-specific deployment vocabulary leaks into the generic Tier-3 platform contract**

- **Severity:** MEDIUM
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Generic Tier-3 platform environment/profile contracts remain product-neutral; product-specific deployment vocabulary belongs to application-owned or typed extension configuration.
- **Observation:** `HostDeploymentProfile` in `intergrax/applications/contracts/environment_profile/sub_profiles.py` contains product-specific fields: `lkw_hybrid_daemon_enabled`, `lkw_daemon_bind_host`, `lkw_daemon_port`, `business_agents_deploy_enabled`. This makes generic Tier-3 platform configuration know Local Knowledge Workspace and specific product deployment vocabulary.
- **Location:**
  - `intergrax/applications/contracts/environment_profile/sub_profiles.py:L401-L409` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Reproduction:**
  1. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:intergrax/applications/contracts/environment_profile/sub_profiles.py`
  2. Inspect `HostDeploymentProfile` (L401-L409): LKW-prefixed and business-agent deploy fields on generic platform contract.
- **Impact:** Generic platform contract is not fully product-neutral; product concerns can accumulate in shared platform configuration.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER_LAYER_BOUNDARIES-04

**Tier-3 applications mutate private DefaultRunService execution state**

- **Severity:** MEDIUM
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Tier-3 application hosts compose lower layers through public typed contracts; direct mutation of private platform service fields is not a supported composition API.
- **Observation:** `DefaultRunService` stores `self._execution_adapter` in `__init__` (L37) with no public rebinding API. `applications/legal_application/host/factory.py` and `applications/dispute_sim_application/host/factory.py` switch queue execution by directly assigning `run_service._execution_adapter = queue_wiring.execution_adapter` (L110 and L89 respectively). `wire_optional_queue_execution` requires an already-created `DefaultRunService` to construct `QueuedNexusExecutionAdapter` (L34, L65-L67), creating a composition cycle resolved only by consumer reach-in.
- **Location:**
  - `intergrax/fastapi_core/runs/default_service.py:L31-L37` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `intergrax/applications/_shared/queue_worker_wiring.py:L29-L37,L65-L67` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `applications/legal_application/host/factory.py:L110` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `applications/dispute_sim_application/host/factory.py:L89` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Reproduction:**
  1. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:intergrax/fastapi_core/runs/default_service.py` - private `_execution_adapter` assignment only in constructor.
  2. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:intergrax/applications/_shared/queue_worker_wiring.py` - `run_service` parameter required for `QueuedNexusExecutionAdapter`.
  3. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:applications/legal_application/host/factory.py` - `run_service._execution_adapter = ...` (L110).
  4. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:applications/dispute_sim_application/host/factory.py` - same pattern (L89).
- **Impact:** Tier-3 hosts are coupled to a private implementation field of a lower platform layer; internal refactoring can break application composition and ownership is not enforced through a stable public contract.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TIER_LAYER_BOUNDARIES-05

**Static contract guard does not cover top-level Tier-3 applications**

- **Severity:** MEDIUM
- **Category:** TEST GAP
- **Status at publication:** ACCEPTED
- **Claim falsified:** Static/dynamic-boundary proof mechanisms cover material production consumer boundaries including Tier-3 `applications/`, or have explicit typed exception ownership.
- **Observation:** `scripts/maintenance/check_harness_no_getattr.py` scans `intergrax`, `agents`, `tests`, `scripts`, `testing_support` (L12-L18) but not top-level `applications/`. Tier-3 is a major consumer boundary. The canonical application authoring guide (`applications/USAGE.md`) contains a getattr-based wiring example for `integration_profile` resolution (L159). The finding is not that every getattr is prohibited - it is that the guard cannot prove intended discipline at a major production consumer layer.
- **Location:**
  - `scripts/maintenance/check_harness_no_getattr.py:L12-L18` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `applications/USAGE.md:L159` @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` checklist L138 @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa` (`wire_application_environment() - no getattr on manifest?`)
- **Reproduction:**
  1. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:scripts/maintenance/check_harness_no_getattr.py` - `SCAN_ROOTS` omits `applications`.
  2. `git show d8d10bb5099d003eb9495674c28e0f6e6762dbfa:applications/USAGE.md` - getattr example at L159.
  3. Confirm no `applications` root in scan list - Tier-3 consumer code evades this guard unless separately controlled.
- **Impact:** Dynamic/reflection-based application boundary behavior can evade the current guard unless separately controlled.
- **Confidence:** CONFIRMED

## Positive / non-falsified evidence

The audit did **not** falsify the existence or documentation of the tier model:

1. **Canonical Tier-0 → Tier-1 → Tier-2 → Tier-3 model** is explicit in `docs/project/architecture/PLATFORM_FOUNDATION.md` Tier Mapping and Dependency Direction sections @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`.
2. **Strict forbidden upward dependency direction** is documented (higher tiers import lower only; Tier-0/1/2 MUST NOT import applications).
3. **Dedicated tier-boundary guard scripts exist** - `check_no_upward_application_imports.py`, `check_intergrax_no_applications_imports.py`, `check_agents_no_tier3_imports.py` (invoked in CI smoke).
4. **Application runtime graph** documents explicit forbidden-edge semantics (`docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md` referenced from agent packaging).
5. **Root Tier-2 agents** are separate workspace dependency projects (`agents/echo/pyproject.toml` declares reusable Tier-2 package).
6. **Tier-3 authoring docs** state application business logic belongs in Tier-2 agents and generic infrastructure belongs to platform domains (`applications/USAGE.md` L60, L1062-L1063).
7. **Application environment architecture** states Tier-3 is a platform adopter (`TIER3_APPLICATION_ENVIRONMENT.md` governance note @ `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`).

**FAIL qualification:** verdict means enforcement/ownership is not universally guaranteed - **not** that all layers are mixed or the architecture concept is invalid.

## Prior-audit revalidation note

Prior PF-TIER-ENFORCEMENT audit (`4c92e0a08f92341f559408c234d213a8ac482d76`, verdict `CONDITIONALLY SOUND - ENFORCEMENT REMEDIATION REQUIRED`) remains **historical evidence** in `PLATFORM_FOUNDATION` plan §6.1ax. Its findings (FND-01..FND-09) are **not** copied into this snapshot. Protocol v2.1 findings `AUDIT-20260818-TIER_LAYER_BOUNDARIES-01`..`05` are the canonical AUDIT-2 observations for the TIER_LAYER_BOUNDARIES layer at `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`. Historical verdicts and statuses remain historical.

## Root-cause remediation grouping

Planning only - **not implemented** by this persistence task.

### TL-FIX-A - Executable tier ownership

**Findings:** 01, 05

Authoritative package→tier classification; full forbidden dependency matrix; fail-closed unknown production packages; semantic enforcement; CI integration-path enforcement; static-contract coverage for Tier-3 application consumer code; consolidate/retire duplicated old guards only after canonical replacement exists.

### TL-FIX-B - Single agent ownership

**Finding:** 02

One canonical implementation authority for `echo@1.0.0`; concrete Tier-2 agent lives in Tier-2 ownership; Tier-1 framework does not carry a colliding concrete production agent; if a core reference agent is required, it must have explicit distinct identity/contract semantics.

### TL-FIX-C - Product-neutral Tier-3 platform

**Finding:** 03

Remove LKW/product-specific vocabulary from generic Tier-3 platform contracts; product-specific deployment configuration belongs to application or typed product-owned extension over generic platform capability.

### TL-FIX-D - Public application composition contract

**Finding:** 04

Tier-3 composes execution through public typed platform contracts; eliminate direct mutation of `DefaultRunService._execution_adapter`; resolve composition cycle through explicit platform-owned API/factory/binding model.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`; current `development` HEAD was not re-audited.
- No claim that no boundary guards exist, that all current imports are invalid, or that every prior PF-TIER finding remains unchanged.
- Finding 02 does not claim Echo necessarily causes runtime outage today or that every concrete agent is duplicated.
- Finding 03 does not claim security bypass, outage, or that every LKW helper is production-reachable.
- Finding 04 is not reduced to style-only debt.
- Finding 05 does not claim every getattr is prohibited.
- Remediation not performed; STRATEGIC_HARNESS_MODEL layer untouched.

## Operator acceptance

- **Date:** 2026-08-18
- **Accepted findings:** all 5 IDs (`AUDIT-20260818-TIER_LAYER_BOUNDARIES-01` … `AUDIT-20260818-TIER_LAYER_BOUNDARIES-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
