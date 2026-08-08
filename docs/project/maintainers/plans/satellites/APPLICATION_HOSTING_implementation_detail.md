# APPLICATION_HOSTING — implementation detail

**Parent plan:** [`APPLICATION_HOSTING.md`](../APPLICATION_HOSTING.md)
**Architecture:** [`../../../architecture/APPLICATION_HOSTING.md`](../../../architecture/APPLICATION_HOSTING.md)
**Architecture detail:** [`../../../architecture/satellites/APPLICATION_HOSTING_extended_depth.md`](../../../architecture/satellites/APPLICATION_HOSTING_extended_depth.md)

> Load only the section for the selected `APP-HOST-*` task. This document contains implementation detail, code targets, acceptance criteria, and verification guidance. The parent plan remains the queue/status authority.

---

# APP-HOST-0 — Architecture and governance

## APP-HOST-0D — Tier-3/LKW cross-plan ownership correction

**Type:** Documentation-only
**Depends on:** 0A–0C
**Status:** **Done** (2026-07-13)
**Goal:** Make platform ownership unambiguous before code starts.

Required changes:

```text
docs/project/architecture/intergrax_runtime_architecture.md
  - register APPLICATION_HOSTING as domain pair 23

docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md
  - add related-domain reference
  - state that Tier-3 application definitions may be wrapped by Application Hosting

docs/project/architecture/satellites/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md
  - add author-surface distinction:
      ApplicationHost.on_hook = execution reactions
      HostedApplicationHooks = process lifecycle reactions
  - reference future .hosting(...) facade without declaring it implemented

docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md
  - add cross-plan ownership note
  - no copy of APP-HOST backlog

docs/project/technical/applications/local_workspace_application/ARCHITECTURE.md
  - state LKW is first adopter/proof only

docs/project/technical/applications/local_workspace_application/IMPLEMENTATION_PLAN.md
  - reframe LKW.6B as adoption/proof
  - block generic hosting work on platform APP-HOST foundation
```

Validation:

```text
- no generic hosting contract is assigned to LKW
- domain architecture/plan pair links resolve
- runtime architecture domain count/index updated
- Tier-3 and hosting ownership do not overlap
- no code/tests changed
- git diff --check
```

Commit suggestion:

```text
docs(hosting): register platform hosting ownership
```

---

---

# APP-HOST-W1 — Complete Public Hosting Foundation

**Status:** **Done** (2026-07-14)

**Closes:** APP-HOST-1A.2, APP-HOST-1B, APP-HOST-1C, APP-HOST-1D, APP-HOST-1E, APP-HOST-1F, APP-HOST-3A (contract vocabulary only).

**Delivered code:**

```text
intergrax/hosting/__init__.py
intergrax/hosting/services.py
intergrax/hosting/contracts/{identity,profile,public_data,context,lifecycle,hooks,components,policies,events}.py
tests/unit/hosting/
```

**Next wave:** APP-HOST-W3 — Process Control and Supervision (4A..4E, 5A..5C).

# APP-HOST-W2 — Complete Hosting Engine

**Status:** **Done** (2026-07-14)

**Closes:** APP-HOST-2A, APP-HOST-2B (foundation profile composition), APP-HOST-2C, APP-HOST-2D, APP-HOST-2E, APP-HOST-2F, APP-HOST-3B, APP-HOST-3C.

**Delivered code:**

```text
intergrax/hosting/engine/
intergrax/hosting/eventing.py
intergrax/hosting/errors.py
tests/unit/hosting/engine/
```

**Next wave:** APP-HOST-W3 — Process Control and Supervision (not started).

# APP-HOST-1 — Public authoring contracts

## APP-HOST-1A.1 — Hosted Application Profile Core

**Status:** Done (2026-07-13)

**Architecture:** hub §4–5; satellite §21–22
**Depends on:** APP-HOST-0D
**Goal:** First implementation slice after governance — hosting package skeleton and profile identity core only.

**First task after APP-HOST-0D.**

Target code:

```text
intergrax/hosting/__init__.py
intergrax/hosting/contracts/__init__.py
intergrax/hosting/contracts/identity.py
intergrax/hosting/contracts/profile.py
```

Delivered public API:

```text
HostedApplicationIdentity
HostedApplicationProfile
HostedApplicationProfilePublicView
HOSTED_APPLICATION_PROFILE_SPEC_VERSION
HostedApplicationProfile.identity
HostedApplicationProfile.public_view()
HostedApplicationProfile.profile_digest()
```

In scope:

```text
hosting package skeleton
application identity contract
application_id validation
spec_version
runtime-only application factory reference
stable application factory descriptor/id
safe metadata
safe deterministic public projection
deterministic profile digest
```

Explicitly out of scope:

```text
HostedApplicationEngine
HostedApplicationContext
hooks
components
policies
events
plugins
interaction composition
OS adapters
supervisor
LKW adoption
```

Scope rules:

- use frozen/immutable typed models where practical,
- no OS-specific classes in the root profile,
- no engine/supervisor implementation,
- no direct `NexusLoop` field,
- application factory may be represented by a runtime-only callable excluded from public serialization,
- public view/schema must never serialize secrets or arbitrary callable internals,
- **do not introduce untyped `Any` placeholders** for contracts owned by later rows.

Acceptance:

```text
minimal profile core creates successfully
invalid application id rejected
safe public view deterministic
spec_version present
profile digest deterministic for equivalent safe config
runtime-only factory excluded from digest or represented by stable id
no second ApplicationEnvironmentProfile introduced
```

Tests:

```text
tests/unit/hosting/test_hosted_application_profile_core.py
tests/unit/hosting/test_hosted_application_profile_core_schema.py
```

## APP-HOST-1A.2 — Foundation HostedApplicationProfile Composition Root
**Status:** **Done** (2026-07-14)


**Architecture:** hub §4–5; satellite §21–22
**Depends on:** APP-HOST-1A.1, APP-HOST-1B, APP-HOST-1C, APP-HOST-1D, APP-HOST-1E, APP-HOST-3A
**Goal:** Complete the foundation `HostedApplicationProfile` composition root required before the engine foundation, using typed contracts from foundation rows.

Allowed typed composition fields:

```text
application identity and factory core from APP-HOST-1A.1
HostedApplicationHooks registrations
HostedApplicationComponent registrations
hosting policy references
typed hosting event subscriptions from APP-HOST-3A
safe metadata
safe deterministic public projection
deterministic digest
```

Clarifications:

```text
HostedApplicationContext is instance-scoped and is not a HostedApplicationProfile field.
InteractionProfile extends the same canonical HostedApplicationProfile in APP-HOST-6A.
The plugin field and plugin contract extend the same canonical HostedApplicationProfile in APP-HOST-9E.
```

The split changes implementation granularity, not architecture ownership — **`HostedApplicationProfile` remains the one canonical hosting composition root.**

Explicitly out of scope:

```text
HostedApplicationEngine
HostedApplicationSupervisor
OS adapters
LKW adoption
InteractionProfile (APP-HOST-6A)
plugins (APP-HOST-9E)
HostedApplicationContext as a profile field
```

Acceptance:

```text
foundation profile composes from typed contract fields only
duplicate component ids rejected
safe public view deterministic for foundation profile
profile digest deterministic for equivalent safe config
no untyped placeholders for deferred contracts
no InteractionProfile or plugin fields on foundation profile
```

Tests:

```text
tests/unit/hosting/test_hosted_application_profile.py
tests/unit/hosting/test_hosted_application_profile_schema.py
```

## APP-HOST-1B — HostedApplicationContext
**Status:** **Done** (2026-07-14)


**Architecture:** satellite §23
**Goal:** Safe instance-scoped context for extensions.


Delivered API:

```python
run_hosted_application(profile: HostedApplicationProfile) -> HostedApplicationSupervisorResult
```

Target:

```text
intergrax/hosting/contracts/context.py
intergrax/hosting/services.py
```

Required:

```text
application_id
instance_id
profile safe reference
resolved paths
clock/logger/event publisher interfaces
shutdown token/request interface
scoped typed service registry
lifecycle snapshot provider
```

Acceptance:

- typed service registration/resolution,
- duplicate service policy explicit,
- context public view redacted,
- no global registry,
- no FastAPI/Nexus/OS requirement,
- closed/terminal context rejects unsafe late mutation.

## APP-HOST-1C — HostedApplicationHooks
**Status:** **Done** (2026-07-14)


**Architecture:** satellite §25
**Goal:** One coherent hook registration contract.

Target:

```text
intergrax/hosting/contracts/hooks.py
```

Required hook points:

```text
before_start
before_ready
before_stop
after_start
after_ready
after_stop
on_failure
```

Required metadata:

```text
hook_id
priority
timeout
blocking/observer semantics
source/plugin id
```

Acceptance:

- deterministic ordering,
- duplicate ids rejected,
- async and safe sync adaptation policy explicit,
- no generic engine-loop callback,
- no application execution/Nexus hook overlap.

## APP-HOST-1D — HostedApplicationComponent
**Status:** **Done** (2026-07-14)


**Architecture:** satellite §26
**Goal:** Cohesive component lifecycle/health contract.

Target:

```text
intergrax/hosting/contracts/components.py
```

Required:

```text
component_id
start(context)
stop(context)
health(context)
required
dependencies
timeout/failure metadata
```

Acceptance:

- runtime-checkable or ABC contract consistent with repo style,
- dependency metadata typed,
- safe health model,
- no forced micro-interface fragmentation,
- active interaction sources can implement the contract later.

## APP-HOST-1E — Policies
**Status:** **Done** (2026-07-14)


**Architecture:** satellite §29
**Goal:** Typed immutable policy models and presets only.

Target:

```text
intergrax/hosting/contracts/policies.py
```

Models:

```text
LifecyclePolicy
ShutdownPolicy
RestartPolicy
ComponentFailurePolicy
HookFailurePolicy
InstancePolicy
```

Acceptance:

- safe defaults,
- bounded timeouts/backoff,
- invalid combinations rejected,
- deterministic serialization/public view,
- custom decision callable represented as runtime extension, not serialized code.

## APP-HOST-1F — Contract exports and compatibility gates
**Status:** **Done** (2026-07-14)


Required:

```text
public package exports
schema snapshot/contract tests
import boundary tests
architecture naming checks
no provider/OS SDK imports in contracts
```

---

# APP-HOST-2 — Engine lifecycle foundation

## APP-HOST-2A — Lifecycle state machine

**Status:** **Done** (2026-07-14)

**Architecture:** satellite §24
**Target:** `intergrax/hosting/engine/lifecycle.py`

Implement:

```text
CREATED → STARTING → READY → STOPPING → STOPPED
                         ↘ FAILED
```

Acceptance:

- full valid/invalid transition matrix,
- terminal instance semantics,
- accepting-work predicate separated from state,
- transition timestamps/reason metadata,
- thread/async-safe serialization,
- no restart transition inside one engine instance.

## APP-HOST-2B — Composition validation

**Status:** **Done** (2026-07-14) — foundation profile composition

Resolve profile + plugin contributions into an immutable `HostedApplicationDefinition`.

Acceptance:

- duplicate/conflicting ids rejected,
- component dependency graph validated,
- explicit override precedence,
- stable composition diagnostics/digest,
- no lifecycle side effects during composition.

## APP-HOST-2C — Hook coordinator

**Status:** **Done** (2026-07-14)

Implement deterministic blocking/observer execution.

Acceptance:

```text
priority/source/declaration ordering
bounded timeout
blocking failure preserves original error
observer failure diagnostic only by default
failure injection for every hook point
no re-entrant lifecycle transition from hook
```

## APP-HOST-2D — Component coordinator

**Status:** **Done** (2026-07-14)

Implement:

```text
DAG resolution
topological start
bounded concurrency for independent components
startup rollback
reverse stop order
required/optional failure semantics
health polling/snapshot integration
```

Acceptance includes partial-start rollback and multiple secondary cleanup failure preservation.

## APP-HOST-2E — Health/readiness

**Status:** **Done** (2026-07-14)

Target:

```text
intergrax/hosting/engine/health.py
```

Acceptance:

- required enabled unhealthy blocks readiness,
- optional unhealthy degrades diagnostics only,
- disabled optional is not failure,
- runtime/instance/shutdown predicates included,
- public snapshot safe and deterministic,
- readiness changes publish events once per transition.

## APP-HOST-2F — HostedApplicationEngine

**Status:** **Done** (2026-07-14)

Compose lifecycle, hooks, components, runtime factory, health, events, instance lease interface, and shutdown control.

Initial engine may use test doubles for instance guard/event publisher until APP-HOST-3/4 concrete bridges land, but contracts must be real.

Acceptance:

```text
successful start to READY
startup failure rollback to FAILED
new work rejected outside READY
clean shutdown to STOPPED
stop idempotency
original failure preserved
runtime treated as opaque contract
no Task/Nexus imports in engine package
```

---

# APP-HOST-3 — Events and diagnostics

## APP-HOST-3A — Event contracts
**Status:** **Done** (2026-07-14) — contracts only; spine bridge remains APP-HOST-3B.


Implement versioned typed event envelope/families from architecture §28.

Acceptance:

- event ids/timestamps injectable for tests,
- schema ids stable,
- payloads safe/redacted,
- lifecycle/component/instance/restart/hook/plugin families covered,
- no private event bus implementation.

## APP-HOST-3B — Existing spine bridge

**Status:** **Done** (2026-07-14)

Investigate and use existing Intergrax runtime event/observability contracts. If no suitable application-level publisher exists, add the smallest shared bridge in the owning observability/runtime domain and cross-plan it.

Forbidden:

```text
hosting_events.sqlite
HostingEventBus
new private exporter stack
```

## APP-HOST-3C — Diagnostics

**Status:** **Done** (2026-07-14)

Implement safe snapshots and typed failure records.

Acceptance:

- stable public view,
- current/last failure summary,
- component/hook/plugin composition summary,
- instance/profile digest,
- no secrets/raw stack traces.

## APP-HOST-3D — Metrics

Add hosting metrics through existing metrics abstractions. Avoid high-cardinality labels such as raw instance ids on global counters unless policy allows.

---

# APP-HOST-4 — Instance ownership and graceful control

## APP-HOST-4A — InstanceGuard contracts/reference lock

Implement platform-neutral contracts and a portable safe file-lock reference.

Acceptance:

- second owner rejected,
- lease metadata safe,
- release idempotent,
- ownership verification,
- crash/stale simulation,
- no application-specific code.

## APP-HOST-4B — Stale recovery/path safety

Acceptance:

```text
controlled run directory
symlink/path traversal defense
owner process missing detection
corrupted metadata handling
explicit stale recovery event
no unsafe lock stealing
```

## APP-HOST-4C — Control coordinator

Implement typed shutdown/restart requests and coalescing.

Acceptance:

- repeated shutdown idempotent,
- strongest deadline/reason policy deterministic,
- health probes unaffected,
- restart request reaches supervisor boundary without application self-exec.

## APP-HOST-4D — Graceful shutdown policy

Implement drain/cancel/flush phases with injectable active-work and flush interfaces.

Acceptance:

- bounded total shutdown,
- phase timing diagnostics,
- timeout leads to classified result,
- cleanup continues after failure,
- intake stops before runtime/components.

## APP-HOST-4E — Foreground signal adapter

Implement portable SIGINT/SIGTERM/console-control bridge where supported, with platform-specific imports isolated.

Acceptance:

- signal becomes one typed shutdown request,
- handler installation/restoration tested,
- duplicate signals coalesced,
- core engine imports no native OS SDK.

---

# APP-HOST-5 — Supervisor and restart

## APP-HOST-5A — Exit classification

Implement terminal result/failure taxonomy.

Acceptance:

- clean stop,
- config error,
- instance conflict,
- startup/runtime/supervisor failures,
- restart requested,
- forced termination,
- retryable classification explicit.

## APP-HOST-5B — Restart policy evaluator

Use fake clock/random source.

Acceptance:

- never/on_failure/always,
- max attempts/window,
- exponential bounded backoff,
- jitter deterministic in tests,
- stable-window reset,
- interruptible wait.

## APP-HOST-5C — Supervisor

Implement reference supervisor over opaque engine factory.

Import-boundary test MUST prove no Task/Nexus/agent/tool dependencies.

Acceptance:

```text
start engine
receive classified terminal result
restart when allowed
new instance id
attempt exhaustion
shutdown interrupts backoff
profile source/digest preserved
```

## APP-HOST-5D — Configuration preservation

Add immutable profile source/resolution contract and digest evidence across restarts.

## APP-HOST-5E — Process proof harness

Provide a generic test/proof app independent from LKW to verify process-level start, lock, signal, stop, restart, and request/health probe if applicable.

---

# APP-HOST-6 — Interaction composition

## APP-HOST-6A — InteractionProfile

Design a compact facade only after inspecting existing platform interaction and Tier-3 settings.

Acceptance:

- simple HTTP/MCP/intake declarations,
- no duplicate interaction model,
- safe disabled defaults,
- custom surface extension reference,
- stable public schema.

## APP-HOST-6B — Composition bridge

Bridge profile to existing application factory/wiring:

```text
verifier
adapter
InteractionIntakeService
TaskExecutor
router/MCP
component health/lifecycle
```

Do not move Task execution into hosting.

## APP-HOST-6C — Active source component

Define/reuse one component adapter pattern for long-lived sources. Parser and active source lifecycle remain separate.

## APP-HOST-6D — Example/security

Implement one lab/local custom source example, not Slack-specific production work. Prove no unauthenticated surface is enabled implicitly.

---

# APP-HOST-7 — OS adapters

## APP-HOST-7A — OS services contract

Define selected services needed by core:

```text
paths
instance guard factory
signal bridge
process identity
optional service descriptor generator
```

Keep public author surface unified.

## APP-HOST-7B/C/D — Platform adapters

Each OS adapter must:

- isolate native imports,
- implement shared contract tests where possible,
- provide deterministic path/identity behavior,
- document unsupported capabilities,
- avoid installer side effects in unit tests.

## APP-HOST-7E — Descriptors

Generate service-manager descriptors/commands as data/artifacts. Do not automatically install without explicit operator action.

## APP-HOST-7F — Packaging ownership

Decide whether installers/auto-update belong to Application Hosting, DX/packaging, or a future cross-layer desktop feature. Record ADR if ownership changes.

---

# APP-HOST-8 — LKW adoption and proof

## Prerequisite gate

At minimum closed:

```text
APP-HOST-1A.1, APP-HOST-1B..1F, APP-HOST-1A.2
APP-HOST-W2 Done (2A..2F, 3B, 3C)
APP-HOST-3A..3C
APP-HOST-4A,4C,4D,4E
APP-HOST-5A..5C
APP-HOST-9A
```

**APP-HOST-9A (`run_hosted_application(profile)`) MUST precede APP-HOST-8A** — LKW proof validates one-profile authoring.

Initial LKW adoption does **not** require `InteractionProfile` (APP-HOST-6A). LKW may host existing interaction/runtime surfaces through the application runtime adapter until APP-HOST-6A closes.

Equivalent consolidated delivery is acceptable only if gates remain independently testable.

## APP-HOST-8A — LKW hosted profile — **Done** (2026-07-16)

**Delivered:** profile builder; private FastAPI/Uvicorn HostedApplicationRuntime adapter; existing LKW lifecycle retained temporarily; existing Uvicorn entrypoint retained; live adoption proof not started.

Initial LKW adoption does **not** require `InteractionProfile`. LKW may host its existing interaction/runtime surfaces through the application runtime adapter. `InteractionProfile` adoption follows **APP-HOST-6A**; an interaction profile in LKW is allowed only when APP-HOST-6A is closed.

LKW may initially define only:

```text
application factory/runtime adapter
LKW-specific hooks
LKW-specific components
hosting presets and metadata
```

Generic engine, supervisor, control, and OS infrastructure remain platform-owned. No generic engine/supervisor/OS implementation under LKW.

## APP-HOST-8B — Lifecycle migration — **Done** (2026-07-16)

Hosted LKW work acceptance projects `HostedApplicationReadinessService` via `_HostedLocalWorkspaceReadiness`. Hosted `runtime.ready()` is limited to Uvicorn/FastAPI startup (no platform READY cycle). Direct Uvicorn `LocalWorkspaceHostLifecycle` remains compatible.

## APP-HOST-8C — Foreground/single-instance proof — **Done** (2026-07-16)

Foreground LKW hosted entrypoint delivered; canonical LKW hosted profile now has one required boundary component; canonical profile now has one blocking before_ready hook; real hosted process reached READY; real local.workspace.index request succeeded; second real process was rejected as INSTANCE_CONFLICT; first process remained READY.

Proof delivered:

```text
one profile
start to READY
hook evidence
component health
second instance rejected
real LKW task succeeds
```

## APP-HOST-8D — Stop/restart proof — **Done** (2026-07-16)

Public foreground LKW process stopped through the platform signal bridge; foreground shutdown produced CLEAN_STOP; a replacement process reached READY in the same instance scope; instance lock release was proven. Typed restart request stopped the first hosted engine gracefully; first attempt released its lease and closed its context; supervisor created a second engine with a new instance_id; profile and definition digests remained unchanged; second hosted instance reached READY; real local.workspace.index succeeded after restart; final typed shutdown produced CLEAN_STOP; final lock reacquisition succeeded.

Proof delivered:

```text
graceful stop
lock released
stopped events
supervisor restart
new instance id
same profile digest
real LKW task succeeds after restart
```

## APP-HOST-8E — Receipt/reviewer path — **Done**

Accepted path:

`	ext
accepted live hosting tests
→ JUnit evidence
→ ProofReceipt
→ ProofReceiptStore
→ DocumentStore
→ MongoDB
→ Mongo Express inspection
`

Markdown is not the source of truth. JUnit is not the source of truth. MongoDB ProofReceipt is the source of truth.

One-command reviewer runner: applications/local_workspace_application/scripts/run-lkw-hosting-proof.bat.

---

# APP-HOST-9 — Developer experience

## APP-HOST-9A — Runner facade


**Status:** Done (2026-07-14)

**Depends on:** APP-HOST-1A.2, APP-HOST-1F, APP-HOST-2F minimum, APP-HOST-4 minimum foundation, APP-HOST-5C minimum
**Blocks:** APP-HOST-8A (LKW hosted-profile adoption)

Target:

```python
run_hosted_application(profile)
```

Must support standard foreground execution without requiring engine/supervisor assembly. Required before first adopter proof so LKW validates one-profile authoring.

## APP-HOST-9B — HarnessApplication integration

Add `.hosting(...)` or an equivalent accepted facade without turning `ApplicationEnvironmentProfile` into a second hosting schema.

## APP-HOST-9C — Scaffold

New application scaffolds may offer optional hosted profile/runner generation. Default non-hosted applications remain supported.

## APP-HOST-9D — Author guide

Required sections:

```text
5-minute minimal app
hooks
components
interaction surfaces
health/readiness
shutdown/restart
OS posture
custom plugin
troubleshooting
```

## APP-HOST-9E — Conformance kit

Provide reusable contract tests for custom components/plugins/OS adapters.

## APP-HOST-9F — Stable release

Declare spec/API version, compatibility policy, migration guide, and production maturity evidence.

---

# Cross-cutting validation commands

Actual paths must be used as implementation lands.

Suggested gate families:

```bash
uv run pytest tests/unit/hosting -q
uv run pytest tests/integration/hosting -q
uv run pytest tests/process/hosting -q
python scripts/maintenance/check_harness_adr.py
python scripts/maintenance/check_architecture_plan_pairs.py  # if present
uv run intergrax doctor --ci
git diff --check
```

Boundary searches:

```bash
rg "NexusLoop|from agents|from intergrax.runtime.task|ToolRegistry" intergrax/hosting/supervisor intergrax/hosting/os
rg "win32|ctypes|fcntl|systemd|launchd" intergrax/hosting --glob '!os/**'
rg "HostedApplicationEngine|HostedApplicationSupervisor" applications/
```

Expected final boundary:

```text
platform packages own generic types
products own profiles/components/hooks only
```

---

# Reporting template for every implementation row

```text
Status: Done / Blocked

Task:
- APP-HOST-...

Files changed:
- exact list

Contract/behavior delivered:
- exact public API and semantics

Architecture fidelity:
- cited architecture sections
- invariants preserved

Tests:
- exact commands
- exact results

Boundary validation:
- platform ownership
- no cognition/runtime leakage
- no OS leakage
- no private event bus

Documentation:
- architecture/plan rows updated
- next task

Commit:
- full SHA
- exact message
```

No row may be marked Done when its directly affected regression suite is red.

# APP-HOST-W3 corrective pass � Process control and supervision hardening

**Status:** **Done** (2026-07-14)

**Closes:** APP-HOST-4A..4E, APP-HOST-5A..5C, APP-HOST-W3 corrective pass.

Delivered corrections:

```text
HostedApplicationInstanceAcquisitionResult + guard port alignment
linearizable file-lock acquisition under native lock
truthful lease is_valid/release + INSTANCE_RELEASED only after verified release
HostedApplicationEffectiveControlRequest through wait_until_requested � stop � shutdown
HostedApplicationGlobalShutdownBudget across before_stop/intake/drain/cancel/flush/component/runtime/observer/lease/terminal phases
wrapped exit classification (INSTANCE_CONFLICT, configuration errors)
deterministic restart backoff + stable-window reset via ready_duration_seconds
supervisor restart evaluation after engine failures without bypassing cleanup verification
regression suites: instance (12), shutdown (7), supervisor (7) W3 tests
```

**APP-HOST-8A - Done** (2026-07-16): LKW `build_local_workspace_hosted_profile()` + private FastAPI/Uvicorn `HostedApplicationRuntime` adapter; existing LKW lifecycle and uvicorn entrypoint retained; live adoption proof not started.

**APP-HOST-8B - Done** (2026-07-16): hosted LKW work acceptance projects `HostedApplicationReadinessService`; hosted runtime readiness is Uvicorn/FastAPI startup only; direct Uvicorn lifecycle remains compatible.

**APP-HOST-8C - Done** (2026-07-16): foreground LKW hosted entrypoint; required boundary component + blocking before_ready; real READY + local.workspace.index; second process INSTANCE_CONFLICT; first process remained READY.

**APP-HOST-8D - Done** (2026-07-16): public foreground CLEAN_STOP + lock release; typed supervisor restart with new instance_id; same profile/definition digests; real local.workspace.index after restart; final CLEAN_STOP + lock reacquisition. **APP-HOST-8E - Done** (2026-07-16): accepted live hosting tests → JUnit evidence → ProofReceipt → ProofReceiptStore → DocumentStore → MongoDB → Mongo Express inspection. Markdown/JUnit are not the source of truth; MongoDB ProofReceipt is the source of truth. APP-HOST-8A through APP-HOST-8E complete.

**APP-HOST-9A - Done** (2026-07-14): `run_hosted_application(profile) -> HostedApplicationSupervisorResult` in `intergrax/hosting/runner.py`.

