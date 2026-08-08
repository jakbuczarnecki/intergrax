# APPLICATION_HOSTING — extended depth

**Parent hub:** [`APPLICATION_HOSTING.md`](../APPLICATION_HOSTING.md)  
**Plan:** [`../../plan/APPLICATION_HOSTING.md`](../../maintainers/plans/APPLICATION_HOSTING.md)  
**ADR:** [`ADR-HOST-001`](../../technical/adr/entries/2026-07-13/ADR-HOST-001.md)

> Load this satellite only for `APP-HOST-*` contract, engine, event, component, policy, supervisor, OS-adapter, author-DX, or LKW-adoption work.

---

# 20. Domain model and terminology

## 20.1 Core terms

| Term | Meaning |
|------|---------|
| **Tier-3 application** | Existing Intergrax application composition: manifest, environment profile, task executor, product surfaces, runtime resources. |
| **Hosted application** | One Tier-3 application definition wrapped in platform hosting configuration and lifecycle. |
| **Application instance** | One concrete lifecycle execution identified by `instance_id`. A restart creates a new instance identity. |
| **Hosting engine** | In-process lifecycle coordinator for one hosted application instance. |
| **Supervisor** | Process/runtime-level owner of restart decisions, backoff, exit classification, and re-instantiation. |
| **Hosted component** | Lifecycle-managed extension such as a socket listener, cache warmer, local database bridge, interaction source, or watcher. |
| **Hosting hook** | Author callback at a hosting lifecycle boundary. Not an application execution/Nexus hook. |
| **Hosting event** | Typed observation published through the existing event/observability spine. |
| **Instance guard** | Exclusive-ownership mechanism for the configured application scope. |
| **OS adapter** | Native implementation of locks, signals, paths, or service-manager integration. |
| **Hosting plugin** | One discoverable package contributing components, hooks, subscriptions, surfaces, policies, and config schema. |

## 20.2 Namespaces

Expected implementation ownership:

```text
intergrax/hosting/
├── contracts/
├── engine/
├── events/
├── components/
├── instance/
├── policies/
├── supervisor/
├── interactions/
├── plugins/
└── os/
```

The final package structure may be flatter during early waves, but public contracts MUST remain under platform ownership.

## 20.3 Separation from existing `ApplicationHost`

`ApplicationHost.on_hook()` is an application-execution extension surface invoked around Nexus/task boundaries.

`HostedApplicationHooks` is a process-hosting extension surface invoked around application instance lifecycle boundaries.

```text
ApplicationHost.on_hook
  before/after task intake, execution, tool, result, etc.

HostedApplicationHooks
  before_start, before_ready, before_stop, on_failure, etc.
```

One MUST NOT call the other implicitly. A developer may configure both through higher-level authoring, but they remain separate typed mechanisms.

---

# 21. Hosted application definition

## 21.1 Composition root

Canonical target:

```python
@dataclass(frozen=True)
class HostedApplicationProfile:
    application_id: str
    application_factory: HostedApplicationFactory
    instance: InstancePolicy = InstancePolicy.standard()
    lifecycle: LifecyclePolicy = LifecyclePolicy.standard()
    shutdown: ShutdownPolicy = ShutdownPolicy.standard()
    restart: RestartPolicy = RestartPolicy.on_failure()
    interaction: InteractionProfile | None = None
    components: tuple[HostedApplicationComponent, ...] = ()
    hooks: HostedApplicationHooks = HostedApplicationHooks()
    event_subscriptions: tuple[HostingEventSubscription, ...] = ()
    plugins: tuple[HostedApplicationPlugin, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
```

Exact implementation types may evolve, but the author-facing composition MUST remain centralized.

## 21.2 Application factory

The factory yields an application runtime handle, not necessarily only a FastAPI object.

Conceptual contract:

```python
class HostedApplicationFactory(Protocol):
    async def create(self, context: HostedApplicationContext) -> HostedApplicationRuntime: ...
```

A compatibility adapter may wrap synchronous factories and existing FastAPI factories.

## 21.3 Runtime handle

The hosting engine needs an opaque runtime contract:

```python
class HostedApplicationRuntime(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def health(self) -> ComponentHealth: ...
```

Alternative integration through an async context manager is acceptable. The runtime handle MAY expose typed services to the context, but the hosting engine MUST NOT depend on Nexus internals.

## 21.4 Profile validation

Validation MUST reject:

- empty or unstable `application_id`,
- duplicate component ids,
- conflicting plugins/surfaces,
- restart policy without a supervisor mode capable of restart,
- required component configured as disabled,
- duplicate exclusive instance scopes,
- secrets embedded in public metadata,
- cyclic component dependencies.

Validation SHOULD happen before instance ownership is acquired when possible.

---

# 22. Author experience and progressive disclosure

## 22.1 Minimal API

```python
run_hosted_application(
    HostedApplicationProfile(
        application_id="research_desktop",
        application_factory=create_research_desktop,
    )
)
```

This activates standard platform behavior. Authors do not manually create the engine, event publisher, health registry, signal coordinator, or instance lock.

## 22.2 Fluent facade integration

Target Tier-3 authoring may support:

```python
app = (
    HarnessApplication("research_desktop")
    .environment(profile)
    .agents(...)
    .hosting(
        HostedApplicationProfile.standard(
            components=[LocalIndexComponent()],
        )
    )
    .build()
)
```

The exact fluent API is implemented only after the core contracts stabilize.

## 22.3 Advanced access

Advanced authors may:

- register custom components,
- add blocking/non-blocking hooks,
- subscribe to typed events,
- override restart/shutdown policies,
- contribute plugins,
- provide custom OS adapters in controlled environments,
- contribute interaction surfaces.

They MUST NOT replace the core lifecycle loop through a generic callback.

## 22.4 Discoverability requirements

Every public extension must be discoverable from one of:

```text
HostedApplicationProfile
HostedApplicationPlugin
HostedApplicationContext
```

No public workflow may require authors to know internal registries spread across unrelated packages.

---

# 23. Hosted application context

## 23.1 Purpose

`HostedApplicationContext` is the stable, safe context passed to hooks, components, policies, and plugins.

Target fields/services:

```text
application_id
instance_id
profile
lifecycle state snapshot
process identity
resolved data_home
run directory
event publisher
health registry
service registry
shutdown request/token
structured logger
clock
safe configuration view
```

## 23.2 Typed services

Application/plugin services should be exposed as typed resources rather than arbitrary string-key dictionaries where possible.

Conceptual access:

```python
context.services.require(LocalWorkspaceTaskExecutor)
context.services.optional(MongoDBDocumentStoreIntegration)
```

The service registry MUST NOT be an ungoverned global locator. Services are scoped to the hosted instance and registered during composition.

## 23.3 Security

The context MUST NOT expose:

- raw process environment dumps,
- credential-bearing URIs in public views,
- mutable engine internals,
- arbitrary OS handles without an explicit adapter contract,
- cross-instance global mutable state.

---

# 24. Lifecycle state machine

## 24.1 States

```text
CREATED
STARTING
READY
STOPPING
STOPPED
FAILED
```

Optional `RESTART_PENDING` belongs to supervisor state, not application engine lifecycle.

## 24.2 Valid transitions

```text
CREATED  → STARTING
STARTING → READY | STOPPING | FAILED
READY    → STOPPING | FAILED
STOPPING → STOPPED | FAILED
STOPPED  → terminal for this instance
FAILED   → terminal for this instance
```

A restart creates a new `HostedApplicationEngine` instance with a new `instance_id`.

## 24.3 Work acceptance

```text
accepts_new_work = state == READY
                   and required_components_healthy
                   and runtime_ready
                   and not shutdown_requested
```

The engine exposes this state to application surfaces through a narrow readiness/acceptance service.

## 24.4 Startup sequence

Normative order:

```text
1. validate profile
2. create context
3. acquire instance guard
4. transition CREATED → STARTING
5. publish application.starting
6. execute before_start hooks
7. resolve plugin contributions
8. start required infrastructure components in dependency order
9. create/start application runtime
10. start remaining components/interactions
11. execute before_ready hooks
12. evaluate required health/readiness
13. transition STARTING → READY
14. publish application.ready
15. execute non-blocking after_ready observers
```

A phase may be split internally, but externally observable ordering MUST be deterministic.

## 24.5 Shutdown sequence

```text
1. receive shutdown request
2. transition to STOPPING
3. publish application.stopping
4. reject new application work
5. stop active interaction sources
6. execute before_stop hooks
7. drain active work until timeout
8. cancel remaining work according to policy
9. flush events/traces/checkpoints
10. stop components in reverse dependency order
11. stop application runtime
12. execute after_stop hooks/observers
13. release instance guard
14. transition STOPPING → STOPPED
15. publish application.stopped
```

If shutdown fails, resources MUST still receive best-effort cleanup. Final state and exit classification must preserve failure evidence.

## 24.6 Startup failure

A blocking startup failure:

- marks the failing phase/component,
- emits a safe failure event,
- runs compensation cleanup for already-started components,
- releases instance ownership,
- transitions to `FAILED`,
- returns a classified terminal result to the supervisor.

---

# 25. Hooks

## 25.1 Hook groups

Blocking hooks:

```text
before_start
before_ready
before_stop
```

Observer hooks:

```text
after_start
after_ready
after_stop
on_failure
```

The final API may expose a typed enum/registration methods, but semantics are fixed.

## 25.2 Ordering

Hooks are ordered by:

1. explicit priority,
2. plugin registration order,
3. declaration order within one source.

Ordering MUST be deterministic and included in diagnostics.

## 25.3 Timeouts

Every blocking hook has a bounded timeout from profile/policy defaults. Timeout is treated as hook failure.

Observer timeout/failure is reported through diagnostics and hosting events. It does not normally roll back a completed transition.

## 25.4 Failure handling

| Hook | Default failure result |
|------|------------------------|
| `before_start` | startup fails |
| `before_ready` | readiness blocked; startup fails or remains non-ready according to explicit policy |
| `before_stop` | shutdown continues with failure recorded |
| `after_*` | diagnostic failure; completed state is not reverted |
| `on_failure` | secondary diagnostic only; original failure preserved |

## 25.5 Anti-pattern

Hooks MUST NOT implement a permanent loop, process watchdog, task orchestration, or message-consumer runtime. Long-lived behavior is a component.

---

# 26. Components

## 26.1 Contract

```python
class HostedApplicationComponent(Protocol):
    @property
    def component_id(self) -> str: ...

    async def start(self, context: HostedApplicationContext) -> None: ...
    async def stop(self, context: HostedApplicationContext) -> None: ...
    async def health(self, context: HostedApplicationContext) -> ComponentHealth: ...
```

Optional metadata may include:

```text
required
dependencies
start_timeout
stop_timeout
health_interval
failure_policy
```

## 26.2 Why one cohesive contract

Authors should not need to implement separate `Startable`, `Stoppable`, `Closable`, `ReadinessContributor`, and `HealthContributor` interfaces for one resource.

Internally the engine may adapt components to narrower services.

## 26.3 Dependency graph

Components form a DAG. The engine:

- validates missing/cyclic dependencies,
- starts in topological order,
- stops in reverse topological order,
- may start independent components concurrently within configured limits.

## 26.4 Required vs optional

Required component:

- startup failure blocks readiness,
- unhealthy state makes application not ready,
- runtime failure follows configured required-component failure policy.

Optional component:

- failure is visible in component health/events,
- does not block core readiness by default,
- may be retried independently if policy allows.

## 26.5 Active interaction sources

Long-lived interaction sources are components. Payload parsing still uses existing platform `InteractionAdapter` contracts.

Example:

```text
SlackSocketSourceComponent
  owns connection loop/lifecycle/health
  delegates payload normalization to SlackInteractionAdapter
  delegates Task execution to configured TaskExecutor
```

This preserves parser/runtime separation.

---

# 27. Health and readiness

## 27.1 Component health model

Minimum fields:

```text
component_id
enabled
required
state
healthy
ready
detail_code
safe_message
last_transition_at
last_check_at
```

Raw exceptions and secrets are internal diagnostics only.

## 27.2 Aggregate liveness

Liveness means the engine/supervisor process loop is responsive. It does not imply application readiness.

## 27.3 Aggregate readiness

Readiness requires:

- engine state `READY`,
- application runtime ready,
- instance ownership valid,
- all enabled required components healthy/ready,
- no active shutdown request,
- accepting-work guard enabled.

## 27.4 Degradation

Optional component failure produces degraded diagnostics, not a separate mandatory top-level lifecycle state.

A future aggregate health score may integrate with `APP-OPS-*`, but initial implementation uses deterministic required/optional rules.

---

# 28. Hosting events

## 28.1 Event envelope

Hosting events use the platform event spine and conform to typed/versioned envelope rules.

Conceptual model:

```python
class HostedApplicationEvent(BaseModel):
    schema_id: str
    schema_version: str
    event_id: str
    event_type: str
    occurred_at: datetime
    application_id: str
    instance_id: str
    lifecycle_state: str
    severity: str
    correlation_id: str | None
    causation_id: str | None
    payload: dict[str, JSONValue]
```

## 28.2 Required event types

Lifecycle:

```text
hosting.application.starting
hosting.application.started
hosting.application.ready
hosting.application.stopping
hosting.application.stopped
hosting.application.failed
```

Components:

```text
hosting.component.starting
hosting.component.started
hosting.component.health_changed
hosting.component.stopping
hosting.component.stopped
hosting.component.failed
```

Instance:

```text
hosting.instance.acquired
hosting.instance.rejected
hosting.instance.stale_recovered
hosting.instance.released
```

Supervisor:

```text
hosting.restart.requested
hosting.restart.scheduled
hosting.restart.started
hosting.restart.exhausted
```

Hooks/plugins:

```text
hosting.hook.started
hosting.hook.completed
hosting.hook.failed
hosting.plugin.loaded
hosting.plugin.failed
```

## 28.3 Subscription semantics

Event subscribers are non-blocking by default. A lifecycle gate must use a hook or required component, not an event subscriber.

Delivery should support existing observability/event storage/export mechanisms. Hosting MUST NOT introduce a private durable event store.

## 28.4 Redaction

Public event payloads exclude:

- passwords/tokens,
- raw credential URIs,
- complete environment variables,
- arbitrary application payload content,
- stack traces unless policy explicitly allows internal-only diagnostics.

---

# 29. Policies

## 29.1 Lifecycle policy

Controls:

```text
hook timeouts
component startup concurrency
startup failure compensation
readiness stabilization window
```

## 29.2 Shutdown policy

Target model:

```python
ShutdownPolicy(
    strategy="drain_then_cancel",
    drain_timeout_seconds=30,
    cancel_timeout_seconds=5,
    flush_timeout_seconds=10,
)
```

Supported strategies initially:

```text
drain_then_cancel
cancel_immediately
wait_until_complete (explicitly bounded)
```

Unbounded shutdown is forbidden.

## 29.3 Restart policy

Target presets:

```text
never
on_failure
always
custom classifier
```

Fields:

```text
max_attempts
attempt_window
initial_backoff
max_backoff
multiplier
jitter
reset_after_stable_seconds
```

## 29.4 Component failure policy

Options may include:

```text
fail_host
mark_not_ready
mark_degraded
restart_component
request_process_restart
ignore_with_diagnostic
```

Defaults derive from `required` status.

## 29.5 Custom policy hooks

Custom decision callbacks receive immutable safe context and typed failure/exit records. They MUST NOT receive engine internals or execute restart directly.

---

# 30. Instance ownership

## 30.1 Instance identity

Every engine lifecycle has:

```text
application_id
instance_id
process_id
started_at
host/user identity
profile digest
```

`instance_id` changes on restart. `application_id` and profile digest remain stable for equivalent configuration.

## 30.2 Instance scope

Default scope:

```text
one instance per application_id per OS user session/machine posture
```

Deployment profiles may opt into a different explicit scope.

## 30.3 Instance guard contract

Conceptual contract:

```python
class InstanceGuard(Protocol):
    async def acquire(self, identity: InstanceIdentity) -> InstanceLease: ...
```

`InstanceLease` supports:

```text
ownership verification
heartbeat/metadata update where required
release
safe public view
```

## 30.4 Stale ownership

Lock-file implementations must distinguish:

- active owner,
- stale owner/process missing,
- corrupted metadata,
- inaccessible lock,
- ownership mismatch.

Stale recovery emits an explicit event and preserves diagnostics.

## 30.5 Safety

Lock paths use controlled run directories. Symlink/path traversal attacks must be prevented. Ownership metadata contains no secrets.

---

# 31. Signal and control translation

## 31.1 Platform signal contract

Core engine receives typed requests:

```text
ShutdownRequested(reason, deadline)
RestartRequested(reason)
HealthProbeRequested
```

It does not process native signals directly.

## 31.2 OS adapter responsibilities

Windows adapter may translate:

```text
CTRL_C_EVENT
CTRL_BREAK_EVENT
service stop/shutdown controls
```

POSIX adapter may translate:

```text
SIGINT
SIGTERM
SIGHUP (policy-defined reload/restart request)
```

## 31.3 Idempotency

Repeated shutdown signals coalesce into one shutdown sequence. A stronger second signal may shorten the deadline according to policy, but MUST NOT start a second cleanup flow.

---

# 32. Supervisor

## 32.1 Boundary

The supervisor knows:

```text
profile reference/factory
process/runtime launch contract
exit result
restart policy
backoff clock
instance metadata
```

The supervisor does not know:

```text
Task
TaskResult
NexusLoop
agents
tools
skills
capabilities
application business state
```

## 32.2 Exit classification

Minimum classifications:

```text
clean_stop
startup_failure
runtime_failure
forced_termination
instance_conflict
configuration_error
restart_requested
supervisor_error
```

## 32.3 Restart sequence

```text
receive terminal result or restart request
→ classify exit
→ evaluate policy
→ publish restart.requested/scheduled
→ wait interruptible backoff
→ create new instance_id/context/engine
→ start new hosted instance
→ reset attempt counter after stable window
```

The old engine instance is never reused after terminal state.

## 32.4 Watchdog

Initial supervisor may remain in-process for a foreground runner, but architecture must allow a parent-process supervisor. Production service-manager integrations may delegate crash restart to the OS while preserving platform policy metadata.

## 32.5 Configuration preservation

Restart uses an immutable/resolvable profile source and records profile digest. Runtime mutation of the source profile is forbidden.

---

# 33. Interaction profile facade

## 33.1 Goal

Application authors configure interaction surfaces in one place without learning every internal adapter/verifier/router/component contract.

## 33.2 Target API

```python
InteractionProfile(
    http=HttpSurface(enabled=True),
    mcp=McpSurface(enabled=True),
    intake=IntakeSurface(
        enabled=True,
        surface="lab_json",
    ),
    custom_surfaces=(DesktopIpcSurface(...),),
)
```

Convenience form may accept booleans/ids and expand to typed profiles.

## 33.3 Wiring result

The hosting/Tier-3 bridge resolves:

```text
transport/source component
→ verifier
→ InteractionAdapter
→ InteractionIntakeService
→ application TaskExecutor
→ reply/result adapter where applicable
→ component lifecycle/health/events
```

## 33.4 Custom surface contract

A custom active surface is normally contributed as a hosting component/plugin plus an existing or custom `InteractionAdapter`.

The custom surface MUST NOT create a parallel task/execution model.

## 33.5 Disabled-by-default security

Network/public interaction surfaces remain disabled unless explicitly configured or included by a trusted preset. Hosting convenience MUST NOT expose unauthenticated production endpoints silently.

---

# 34. Plugins

## 34.1 Plugin contribution model

Conceptual contract:

```python
class HostedApplicationPlugin(Protocol):
    @property
    def plugin_id(self) -> str: ...
    def contribute(self, context: HostingCompositionContext) -> HostingContributions: ...
```

Contributions may include:

```text
components
hooks
event subscriptions
interaction surfaces
policy defaults
config schema
health metadata
```

## 34.2 Composition rules

- duplicate ids are rejected,
- conflicts are deterministic and diagnostic,
- application-declared explicit configuration overrides plugin defaults,
- plugins cannot mutate the final profile after validation,
- plugin contributions appear in profile/composition diagnostics.

## 34.3 Example future plugins

```text
SlackHostingPlugin
TrayBridgeHostingPlugin
FileWatcherHostingPlugin
ObservabilityHostingPlugin
AutoUpdateHostingPlugin
```

These are examples, not commitments for the foundation wave.

---

# 35. OS adapter boundary

## 35.1 Core adapter

Target OS services interface may expose:

```text
resolve_paths(application_id)
create_instance_guard(...)
install_signal_bridge(...)
process_identity()
service_descriptor(...)
```

Avoid one giant platform adapter if narrower cohesive adapters prove clearer internally. Public authors still see one selected OS posture through the profile/runner.

## 35.2 Windows

Potential capabilities:

- named mutex or safe file lock,
- service control translation,
- scheduled-task/user-login metadata,
- `%LOCALAPPDATA%` data/run paths,
- Windows Event Log bridge in later observability work.

## 35.3 Linux

Potential capabilities:

- advisory file lock,
- SIGTERM/SIGINT translation,
- XDG state/runtime paths,
- systemd user/service unit generation.

## 35.4 macOS

Potential capabilities:

- advisory file lock,
- POSIX signal translation,
- Application Support/runtime paths,
- launchd plist generation.

## 35.5 Installer boundary

Generating descriptors and commands may belong to Application Hosting. Full installers/updaters may belong to packaging/DX features. The plan must keep this distinction explicit.

---

# 36. Observability and diagnostics

## 36.1 Standard metrics

At minimum:

```text
hosting_start_total
hosting_start_failures_total
hosting_ready_duration_seconds
hosting_shutdown_duration_seconds
hosting_restart_total
hosting_restart_exhausted_total
hosting_instance_conflict_total
hosting_component_health_changes_total
hosting_hook_failures_total
```

Metric names may adapt to repository conventions.

## 36.2 Structured diagnostics

`HostedApplicationEngine.public_view()` or equivalent diagnostic snapshot includes:

```text
application_id
instance_id
profile digest
lifecycle state
accepts_new_work
component health summaries
active shutdown/restart reason
supervisor attempt state
safe OS posture
```

No secrets or raw stack traces.

## 36.3 Proof receipts

Platform and product live proofs may persist `ProofReceipt` documents describing hosting scenarios. Proof receipts do not replace normal hosting events/metrics.

---

# 37. Security and trust

## 37.1 User identity

A local hosted process runs under an explicit OS user/service identity. Hosting does not silently elevate privileges.

## 37.2 Secrets

Secrets remain in existing configuration/provider mechanisms. Hosting profiles store references or safe config, not raw credentials in public serialization/events.

## 37.3 Local endpoints

Default local endpoints bind loopback unless explicitly configured. Remote exposure requires Tier-3 identity/security configuration.

## 37.4 Plugins

Plugins are trusted code extensions and follow Intergrax extension governance. Unknown/untrusted runtime code is not loaded automatically from writable directories.

## 37.5 Instance metadata

Run-directory and lease metadata are permission-restricted and validate ownership before stale recovery.

---

# 38. Failure taxonomy

Minimum failure categories:

```text
profile_validation
instance_conflict
instance_guard_failure
hook_failure
component_start_failure
component_runtime_failure
application_runtime_failure
readiness_failure
shutdown_timeout
component_stop_failure
event_publish_failure
supervisor_failure
os_adapter_failure
```

Failures carry:

```text
failure_id
category
phase
application_id
instance_id
component/hook id when applicable
retryable
safe_message
internal_cause reference
```

Original failures are preserved when secondary cleanup/reporting failures occur.

---

# 39. Concurrency and thread safety

- Lifecycle transitions are serialized.
- Shutdown/restart requests are idempotent and coalesced.
- Component registry becomes immutable after composition/start begins.
- Event observers cannot mutate engine state directly.
- Health checks may run concurrently but publish snapshots atomically.
- Hooks execute according to deterministic ordering and configured concurrency (default sequential for blocking hooks).
- Supervisor never starts a replacement before the old instance has released exclusive ownership, unless explicit blue/green scope is introduced in a future deployment posture.

---

# 40. Testing model

## 40.1 Contract tests

- profile validation,
- context safety/public view,
- lifecycle transition matrix,
- hook ordering/timeouts/failure semantics,
- component dependency/order/rollback,
- health/readiness aggregation,
- event schema/order/redaction,
- instance guard behavior,
- shutdown policy,
- restart policy and backoff,
- supervisor domain independence,
- OS adapter contract tests.

## 40.2 Process tests

- start foreground hosted application,
- readiness becomes true,
- second instance rejected,
- signal requests graceful stop,
- lock released after stop,
- restart creates new instance id,
- configuration/profile digest preserved,
- work succeeds after restart.

## 40.3 Product proof

LKW adoption demonstrates a real Intergrax application, but independent platform tests remain mandatory.

## 40.4 Failure injection

Inject failures at every startup/shutdown phase and verify:

- deterministic cleanup,
- terminal state,
- emitted events,
- exit classification,
- instance lease release,
- no duplicate lifecycle sequences.

---

# 41. LKW adoption scenario

## 41.1 Profile

Conceptual LKW adoption:

```python
LOCAL_WORKSPACE_HOSTING = HostedApplicationProfile(
    application_id="local_workspace",
    application_factory=create_local_workspace_runtime,
    interaction=InteractionProfile(
        http=True,
        mcp=True,
        intake_surface="lab_json",
    ),
    hooks=HostedApplicationHooks(
        before_ready=[verify_local_workspace_runtime],
    ),
    components=[LocalWorkspaceRuntimeComponent()],
)
```

Generic engine, supervisor, instance guard, hooks/events, and OS adapter remain outside LKW.

## 41.2 Live proof sequence

```text
start hosted LKW
→ verify application.ready event
→ verify custom before_ready hook evidence
→ verify component health
→ reject second instance
→ execute real LKW interaction/task
→ request graceful shutdown
→ verify stopping/stopped events and lock release
→ supervisor restart
→ verify new instance id, same profile digest
→ execute real LKW interaction/task again
```

## 41.3 Reviewer evidence

Reviewer-visible output should include:

```text
application_id
first_instance_id
second_instance_rejected
ready_before_request
request_result
shutdown_result
restart_attempt
second_instance_id
profile_digest_preserved
request_after_restart_result
platform_hosting_engine=true
lkw_owned_hosting_engine=false
```

A structured ProofReceipt may be persisted after live proof completion.

---

# 42. Cross-domain integration

## 42.1 Tier-3

Tier-3 supplies application factory/runtime and Task execution surfaces. Application Hosting supplies process lifecycle.

## 42.2 Observability

Hosting publishes typed events, metrics, and safe diagnostic snapshots through existing observability contracts.

## 42.3 Reliability

Restart/backoff and failure classification reuse reliability principles but remain process-hosting mechanics.

## 42.4 Developer experience

DX owns guides, scaffold commands, templates, and ergonomic fluent integration once contracts stabilize.

## 42.5 Elastic capacity/deployment

Application Hosting primarily owns local/single-node process hosting. Cluster scheduling, replicas, Helm/Kubernetes, and capacity provisioning remain deployment/scaling concerns.

## 42.6 Integrations/plugins

Provider SDKs belong in integrations/plugins, not hosting core. Hosting components may consume platform integrations through typed services.

---

# 43. Public API compatibility

- Contracts are versioned before ecosystem release.
- Early `APP-HOST-1/2` implementation may remain experimental.
- Public profile serialization must include `spec_version` before stable release.
- Deprecations require migration paths and diagnostics.
- Internal engine decomposition may change without forcing authors to rewire low-level services.

---

# 44. Architecture acceptance checklist

Before implementation of the engine foundation:

```text
[ ] ADR-HOST-001 accepted
[ ] domain pair registered
[ ] public author mental model agreed
[ ] ownership boundary with Tier-3 frozen
[ ] ApplicationHost vs HostedApplicationHooks distinction explicit
[ ] lifecycle transition/failure semantics mapped
[ ] component contract and required/optional rules mapped
[ ] event ownership/spine integration mapped
[ ] instance/signal/shutdown/supervisor boundaries mapped
[ ] OS adapter and installer boundary mapped
[ ] LKW adoption separated from platform ownership
[ ] plan fidelity matrix maps every normative section
```

---

# 45. Final synthesis

```text
Application author
  declares one HostedApplicationProfile
        │
        ▼
Hosting composition
  validates profile + plugins + components + policies
        │
        ▼
HostedApplicationEngine
  lifecycle + hooks + components + readiness + events
        │
        ▼
Existing Tier-3 application runtime
  surfaces + TaskExecutor + Nexus + agents/tools/integrations
        │
        ▼
HostedApplicationSupervisor
  exit classification + restart/backoff + new instance
        │
        ▼
OS adapter
  lock + signals + paths + service-manager posture
```

The platform complexity is real, but it is hidden behind one coherent authoring surface. Products configure and extend hosting; they do not rebuild it.