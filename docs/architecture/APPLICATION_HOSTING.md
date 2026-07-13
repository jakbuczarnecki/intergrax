# Application Hosting

**Status:** Canonical architecture — accepted for implementation planning
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/APPLICATION_HOSTING.md`](../plan/APPLICATION_HOSTING.md)
**ADR:** [`ADR-HOST-001`](../adr/entries/2026-07-13/ADR-HOST-001.md)
**Extended detail:** [`satellites/APPLICATION_HOSTING_extended_depth.md`](satellites/APPLICATION_HOSTING_extended_depth.md)
**Primary integration:** [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md)
**First adopter/proof:** `applications/local_workspace_application/`
**Architecture governance:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) — Application Hosting is the canonical example of platform ownership, architecture before adoption, deployment transparency, and first-adopter proof through LKW (`PLATFORM-INV-001`, `PLATFORM-INV-002`, `PLATFORM-INV-004`, `PLATFORM-INV-006`, `PLATFORM-INV-007`).

---

## Cursor read scope

Read this hub first. Load the extended-depth satellite only for contract, lifecycle, event, policy, supervisor, OS-adapter, or author-DX implementation.

```text
Default implementation context:
1. docs/architecture/APPLICATION_HOSTING.md
2. docs/plan/APPLICATION_HOSTING.md
3. at most one matching satellite
4. owning code/tests for the selected APP-HOST row
```

Do not read the full Tier-3 canon unless the selected task changes Tier-3 public composition contracts.

---

## 1. Purpose

Application Hosting is the platform domain that turns any configured Intergrax Tier-3 application into a continuously running, managed, observable, extensible application instance.

```text
Tier-3 application definition
        │
        ▼
HostedApplicationProfile
        │
        ▼
HostedApplicationEngine
        │
        ├── lifecycle
        ├── hooks
        ├── components
        ├── health/readiness
        ├── events
        ├── graceful shutdown
        └── interaction composition
        │
        ▼
HostedApplicationSupervisor
        │
        ├── instance guard
        ├── signals
        ├── exit classification
        ├── restart policy
        └── OS adapter
```

The domain answers:

> How does an Intergrax application instance live, remain ready, accept extensions, stop safely, restart, and integrate with its operating environment?

It does not answer:

- how agents reason,
- how Nexus orchestrates,
- how application capabilities are defined,
- how business tasks are executed,
- how a product implements domain behavior.

---

## 2. Ownership boundary

### Application Hosting owns

- `HostedApplicationProfile`,
- `HostedApplicationContext`,
- `HostedApplicationEngine`,
- `HostedApplicationComponent`,
- `HostedApplicationHooks`,
- hosting lifecycle events,
- component health/readiness aggregation,
- instance guards,
- signal translation,
- graceful shutdown coordination,
- restart policies and supervision,
- OS hosting adapters,
- a simple interaction-composition facade,
- hosting authoring and diagnostics.

### Tier-3 Application Environment owns

- `ApplicationManifest`,
- `ApplicationEnvironmentProfile`,
- `HarnessApplication`,
- `ApplicationHost.on_hook`,
- application task preparation,
- application interaction/capability declaration,
- `UnifiedTaskRunner`,
- product host composition.

### Runtime/agents own

- `Task`, `TaskResult`,
- `NexusLoop`,
- agent cognition,
- tool/skill/integration execution,
- policy, checkpoints, trace, and task lifecycle.

### Operating-system adapters own

- native lock primitives,
- signal/control translation,
- service-manager metadata,
- user-session process conventions,
- platform-specific run/data path resolution.

---

## 3. Normative invariants

- **HOST-INV-01:** Application Hosting is platform-owned; products adopt it.
- **HOST-INV-02:** Hosting never performs cognition or orchestration.
- **HOST-INV-03:** Supervisor code has no dependency on `Task`, `NexusLoop`, agents, tools, or product capabilities.
- **HOST-INV-04:** Application code contains no standard Windows/Linux/macOS branching.
- **HOST-INV-05:** `HostedApplicationProfile` is the primary public composition surface.
- **HOST-INV-06:** Advanced extension points are typed and reachable from that profile.
- **HOST-INV-07:** Hosting uses the existing Intergrax event/observability spine.
- **HOST-INV-08:** Hosting hooks are lifecycle callbacks, not private execution loops.
- **HOST-INV-09:** Required unhealthy components block readiness; optional components do not.
- **HOST-INV-10:** LKW is a proof workload, never the owner of generic hosting contracts.
- **HOST-INV-11:** `ApplicationHost.on_hook` and `HostedApplicationHooks` are distinct and MUST NOT be merged.
- **HOST-INV-12:** Restart creates a new application instance/process lifecycle; the hosted application does not `exec` itself.

**`HOST-INV-*` vs `PLATFORM-INV-*`:** domain-local `HOST-INV-*` rules remain authoritative for hosting contracts and runtime behavior. Platform `PLATFORM-INV-*` identifiers in [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) govern why this domain exists, who owns generic hosting, and how applications adopt it — do not replace `HOST-INV-*` with `PLATFORM-INV-*` in this document.

---

## 4. Developer mental model

### Level 1 — convention only

```python
from intergrax.hosting import HostedApplicationProfile, run_hosted_application

profile = HostedApplicationProfile(
    application_id="my_application",
    application_factory=create_application,
)

run_hosted_application(profile)
```

Expected defaults:

- one application instance,
- lifecycle state machine,
- standard hosting events,
- liveness/readiness,
- signal handling,
- graceful shutdown,
- safe instance guard,
- standard logs/diagnostics,
- no OS-specific application code.

### Level 2 — hooks and components

```python
profile = HostedApplicationProfile(
    application_id="my_application",
    application_factory=create_application,
    hooks=HostedApplicationHooks(
        before_ready=[warm_local_index],
        after_stop=[flush_local_state],
    ),
    components=[MyBackgroundComponent()],
)
```

### Level 3 — policies, plugins, and custom adapters

```python
profile = HostedApplicationProfile(
    ...,
    plugins=[CustomDesktopBridgePlugin()],
    restart=RestartPolicy.on_failure(max_attempts=3),
    shutdown=ShutdownPolicy.drain_then_cancel(timeout_seconds=30),
)
```

The platform may have many internal services. Authors should not need to assemble them manually.

---

## 5. Public composition model

`HostedApplicationProfile` is the hosting composition root. It references an existing application factory/definition and groups hosting concerns:

```text
HostedApplicationProfile
├── application identity/factory
├── instance policy
├── lifecycle/shutdown policy
├── restart policy
├── interaction profile
├── components
├── hooks
├── event subscriptions
├── plugins
└── OS/deployment posture
```

It is not a second `ApplicationEnvironmentProfile`. Harness configuration remains owned by Tier-3.

Public authoring should satisfy:

```text
80%  profile + run_hosted_application()
15%  profile + hooks/components
 5%  plugins/custom policies/OS adapters
```

---

## 6. Engine model

`HostedApplicationEngine` coordinates one application instance:

1. resolve profile and safe defaults,
2. create `HostedApplicationContext`,
3. acquire instance ownership,
4. emit starting event,
5. run blocking startup hooks,
6. start components in dependency order,
7. build/start the Tier-3 application host,
8. verify required component health,
9. transition to ready,
10. accept work until shutdown is requested,
11. stop intake and new work,
12. drain/cancel according to policy,
13. stop components in reverse order,
14. close application resources,
15. release instance ownership,
16. emit terminal event and return classified exit.

The engine manages lifecycle. It does not execute application tasks itself.

---

## 7. Lifecycle

Canonical lifecycle:

```text
CREATED → STARTING → READY → STOPPING → STOPPED
                └──────────────→ FAILED
```

`accepts_new_work` is true only in `READY` when all required readiness conditions hold.

Blocking lifecycle hooks may prevent a transition. Non-blocking observers cannot silently change lifecycle state.

Liveness, readiness, and component health are separate concepts:

- liveness: the process/engine loop is alive,
- readiness: the application may accept new work,
- component health: one component's current status.

Full transition and failure semantics are in the extended-depth satellite.

---

## 8. Extension model

The public extension model is intentionally small.

### `HostedApplicationHooks`

Lifecycle callbacks such as:

```text
before_start
before_ready
before_stop
after_start
after_ready
after_stop
on_failure
```

Blocking semantics apply only to explicitly blocking hooks.

### `HostedApplicationComponent`

A cohesive hosted component contract:

```python
class HostedApplicationComponent(Protocol):
    async def start(self, context: HostedApplicationContext) -> None: ...
    async def stop(self, context: HostedApplicationContext) -> None: ...
    async def health(self, context: HostedApplicationContext) -> ComponentHealth: ...
```

The platform MUST NOT force ordinary authors to implement separate startable/stoppable/health/readiness contracts for one component.

### Typed hosting events

Events inform observers and observability infrastructure. They are not orchestration commands.

### Policies

Policies configure decisions such as restart, shutdown, readiness criticality, and hook failure handling. Authors do not replace the engine loop.

### Plugins

A plugin may contribute components, hooks, event subscriptions, interaction surfaces, config schema, or policies through one registration object.

---

## 9. Event model

Hosting publishes typed events through the platform event/observability spine.

Minimum event families:

```text
application.starting
application.started
application.ready
application.stopping
application.stopped
application.failed

component.starting
component.started
component.healthy
component.unhealthy
component.stopping
component.stopped
component.failed

instance.acquired
instance.rejected
instance.recovered

restart.requested
restart.scheduled
restart.started
restart.exhausted
```

Events carry at minimum:

```text
schema_id/version
event_id
timestamp
application_id
instance_id
lifecycle state
severity
safe payload
correlation/causation metadata when available
```

Secrets and raw exception internals must be redacted before public/export surfaces.

---

## 10. Interaction composition

Hosting provides an author-friendly `InteractionProfile` facade over existing platform mechanisms:

```text
InteractionAdapter
InboundRequestVerifier
InteractionIntakeService
TaskExecutor
HTTP router
active interaction source runtime
component health/lifecycle
```

Simple declaration example:

```python
interaction=InteractionProfile(
    http=True,
    mcp=True,
    intake_surface="lab_json",
)
```

Advanced authors may add a custom interaction surface/plugin. Hosting performs lifecycle, health, event, and shutdown wiring. The existing canonical `InboundInteraction → Task → application executor` path remains unchanged.

---

## 11. Instance, signals, shutdown, and supervision

### Instance ownership

`InstanceGuard` prevents unsafe duplicate ownership for the configured scope. Implementations may use lock files or native primitives, but application code sees one contract.

### Signals

OS adapters translate native signals/control events into platform shutdown requests.

### Graceful shutdown

Normative order:

```text
mark STOPPING
→ reject new work
→ stop active interaction sources
→ drain bounded active execution
→ cancel after timeout
→ flush traces/events/checkpoints
→ stop components in reverse order
→ close Tier-3 host/resources
→ release instance guard
→ mark STOPPED
```

### Supervision

`HostedApplicationSupervisor` owns process-level restart policy, backoff, attempt limits, and exit classification. It treats the application as an opaque hosted process/runtime contract.

---

## 12. OS boundary

The core engine is OS-neutral.

```text
intergrax.hosting
intergrax.hosting.os.windows
intergrax.hosting.os.linux
intergrax.hosting.os.macos
```

OS adapters may provide:

- instance-lock implementation,
- signal translation,
- run/data path conventions,
- service-manager descriptor generation,
- user-session/service identity metadata.

Full installer/service registration may be implemented in later packaging phases. It is not required for the engine foundation.

---

## 13. LKW adoption boundary

LKW is the first adopter and proof workload:

```text
LocalWorkspace hosted profile
→ HostedApplicationEngine
→ existing LKW FastAPI/MCP/Nexus host
```

The platform implementation must be complete enough to test independently before LKW adoption.

LKW proof should demonstrate:

- one-profile authoring,
- custom hook execution,
- custom component health,
- ready state,
- real interaction/task execution,
- duplicate-instance rejection,
- graceful stop,
- restart with preserved configuration,
- successful work after restart,
- no OS-specific LKW code.

Generic contracts and engine code MUST NOT live under `applications/local_workspace_application/`.

---

## 14. Anti-patterns

| ID | Anti-pattern | Correct |
|----|--------------|---------|
| HOST-AP-01 | Product-owned generic daemon framework | Platform `APPLICATION_HOSTING` domain |
| HOST-AP-02 | Application directly handles WinAPI/systemd/launchd | OS adapter selected by platform |
| HOST-AP-03 | Supervisor calls Nexus/tasks/tools | Supervisor treats app as opaque runtime |
| HOST-AP-04 | New private hosting event bus | Existing Intergrax event/observability spine |
| HOST-AP-05 | Dozens of mandatory micro-contracts | One profile + cohesive hooks/components/plugins |
| HOST-AP-06 | Hosting hooks implement orchestration loops | Hooks react only at lifecycle boundaries |
| HOST-AP-07 | Restart via `os.exec` inside app | Supervisor starts a new instance/process lifecycle |
| HOST-AP-08 | Optional component blocks readiness by default | Only required components block readiness |
| HOST-AP-09 | Hosting profile duplicates harness environment | Hosting wraps existing Tier-3 application definition |
| HOST-AP-10 | LKW proof considered platform contract test | Independent platform tests plus LKW adoption proof |

---

## 15. Related documents

| Document | Relationship |
|----------|--------------|
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | Application composition and execution owner |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | Hosting event export, health telemetry, diagnostics |
| [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | Failure classification, retry/backoff principles |
| [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | Authoring facade, scaffolding, guides |
| [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | Deployment posture boundary; not local supervisor ownership |
| [`plan/APPLICATION_HOSTING.md`](../plan/APPLICATION_HOSTING.md) | Implementation truth and fidelity matrix |
| [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) | Platform evolution and adoption governance |

---

## 16. Maturity target

Application Hosting is complete only when all four dimensions are explicit:

- **Architecture:** contracts, ownership, invariants, extension and OS boundaries frozen;
- **Implementation:** engine, events, component coordination, instance/shutdown/supervisor paths implemented;
- **Product proof:** LKW demonstrates real always-on adoption without owning generic mechanics;
- **Ecosystem:** author guide, scaffold/facade, custom component/plugin example, and cross-platform posture documented.

Detailed normative contracts and scenarios are in [`satellites/APPLICATION_HOSTING_extended_depth.md`](satellites/APPLICATION_HOSTING_extended_depth.md).
