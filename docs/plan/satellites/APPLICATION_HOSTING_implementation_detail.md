# APPLICATION_HOSTING — implementation detail

**Parent plan:** [`APPLICATION_HOSTING.md`](../APPLICATION_HOSTING.md)  
**Architecture:** [`../../architecture/APPLICATION_HOSTING.md`](../../architecture/APPLICATION_HOSTING.md)  
**Architecture detail:** [`../../architecture/satellites/APPLICATION_HOSTING_extended_depth.md`](../../architecture/satellites/APPLICATION_HOSTING_extended_depth.md)

> Load only the section for the selected `APP-HOST-*` task. This document contains implementation detail, code targets, acceptance criteria, and verification guidance. The parent plan remains the queue/status authority.

---

# APP-HOST-0 — Architecture and governance

## APP-HOST-0D — Tier-3/LKW cross-plan ownership correction

**Type:** Documentation-only  
**Depends on:** 0A–0C  
**Goal:** Make platform ownership unambiguous before code starts.

Required changes:

```text
docs/intergrax_runtime_architecture.md
  - register APPLICATION_HOSTING as domain pair 23

docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md
  - add related-domain reference
  - state that Tier-3 application definitions may be wrapped by Application Hosting

docs/architecture/satellites/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md
  - add author-surface distinction:
      ApplicationHost.on_hook = execution reactions
      HostedApplicationHooks = process lifecycle reactions
  - reference future .hosting(...) facade without declaring it implemented

docs/plan/TIER3_APPLICATION_ENVIRONMENT.md
  - add cross-plan ownership note
  - no copy of APP-HOST backlog

applications/local_workspace_application/docs/ARCHITECTURE.md
  - state LKW is first adopter/proof only

applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md
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

# APP-HOST-1 — Public authoring contracts

## APP-HOST-1A — HostedApplicationProfile

**Architecture:** hub §4–5; satellite §21–22  
**Goal:** Introduce the single versioned author composition root without engine behavior.

Target code:

```text
intergrax/hosting/__init__.py
intergrax/hosting/contracts/profile.py
intergrax/hosting/contracts/identity.py
```

Minimum model:

```text
application_id
application_factory reference/descriptor
spec_version
instance policy
lifecycle policy
shutdown policy
restart policy
interaction profile reference/placeholder
components
hooks
event subscriptions
plugins
safe metadata
```

Scope rules:

- use frozen/immutable typed models where practical,
- no OS-specific classes in the root profile,
- no engine/supervisor implementation,
- no direct `NexusLoop` field,
- application factory may be represented by a runtime-only callable excluded from public serialization,
- public view/schema must never serialize secrets or arbitrary callable internals,
- support standard defaults/presets without requiring every field.

Acceptance:

```text
minimal profile creates successfully
invalid application id rejected
duplicate component/plugin ids rejected
safe public view deterministic
spec_version present
profile digest deterministic for equivalent safe config
runtime-only factory excluded from digest or represented by stable id
no second ApplicationEnvironmentProfile introduced
```

Tests:

```text
tests/unit/hosting/test_hosted_application_profile.py
tests/unit/hosting/test_hosted_application_profile_schema.py
```

## APP-HOST-1B — HostedApplicationContext

**Architecture:** satellite §23  
**Goal:** Safe instance-scoped context for extensions.

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

Resolve profile + plugin contributions into an immutable `HostedApplicationDefinition`.

Acceptance:

- duplicate/conflicting ids rejected,
- component dependency graph validated,
- explicit override precedence,
- stable composition diagnostics/digest,
- no lifecycle side effects during composition.

## APP-HOST-2C — Hook coordinator

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

Implement versioned typed event envelope/families from architecture §28.

Acceptance:

- event ids/timestamps injectable for tests,
- schema ids stable,
- payloads safe/redacted,
- lifecycle/component/instance/restart/hook/plugin families covered,
- no private event bus implementation.

## APP-HOST-3B — Existing spine bridge

Investigate and use existing Intergrax runtime event/observability contracts. If no suitable application-level publisher exists, add the smallest shared bridge in the owning observability/runtime domain and cross-plan it.

Forbidden:

```text
hosting_events.sqlite
HostingEventBus
new private exporter stack
```

## APP-HOST-3C — Diagnostics

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
APP-HOST-1A..1F
APP-HOST-2A..2F
APP-HOST-3A..3C
APP-HOST-4A,4C,4D,4E
APP-HOST-5A..5C
```

Equivalent consolidated delivery is acceptable only if gates remain independently testable.

## APP-HOST-8A — LKW hosted profile

LKW defines only:

```text
application factory/runtime adapter
LKW-specific hooks
LKW-specific components
interaction profile
hosting presets/metadata
```

No generic engine/supervisor/OS implementation under LKW.

## APP-HOST-8B — Lifecycle migration

Replace or adapt LKW.6A local lifecycle/readiness with the platform engine while preserving:

- `/health` compatibility,
- readiness semantics,
- shared task executor,
- existing proof paths.

Temporary compatibility adapters must have an explicit removal plan.

## APP-HOST-8C — Foreground/single-instance proof

Proof:

```text
one profile
start to READY
hook evidence
component health
second instance rejected
real LKW task succeeds
```

## APP-HOST-8D — Stop/restart proof

Proof:

```text
graceful stop
lock released
stopped events
supervisor restart
new instance id
same profile digest
real LKW task succeeds after restart
```

## APP-HOST-8E — Receipt/reviewer path

Persist a structured ProofReceipt through the platform store with hosting evidence. Update reviewer docs only after live PASS.

---

# APP-HOST-9 — Developer experience

## APP-HOST-9A — Runner facade

Target:

```python
run_hosted_application(profile)
```

Must support standard foreground execution without requiring engine/supervisor assembly.

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