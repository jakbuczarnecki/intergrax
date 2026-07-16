# Application Hosting — Implementation Plan

**Architecture (1:1):** [`architecture/APPLICATION_HOSTING.md`](../architecture/APPLICATION_HOSTING.md)  
**Architecture detail:** [`architecture/satellites/APPLICATION_HOSTING_extended_depth.md`](../architecture/satellites/APPLICATION_HOSTING_extended_depth.md)  
**Plan detail:** [`satellites/APPLICATION_HOSTING_implementation_detail.md`](satellites/APPLICATION_HOSTING_implementation_detail.md)  
**ADR:** [`ADR-HOST-001`](../adr/entries/2026-07-13/ADR-HOST-001.md)  
**First adopter/proof:** `applications/local_workspace_application/`

**Status:** Public hosting foundation and single-instance engine foundation complete. Process control, real instance ownership and supervision not started. APP-HOST-W2 closed (2026-07-14).

---

## Cursor read scope

For one implementation task read only:

```text
1. docs/architecture/APPLICATION_HOSTING.md
2. this plan hub
3. the selected APP-HOST row in the implementation-detail satellite
4. directly affected code/tests
```

Read the extended architecture satellite only when the selected row cites a specific normative section.

Do not implement LKW adoption before the required platform foundation rows are closed.

---

## 1. Objective

Deliver a platform-owned, reusable Application Hosting subsystem that can run any Intergrax Tier-3 application continuously with:

- one declarative author profile,
- lifecycle and readiness,
- typed hooks and events,
- lifecycle-managed components,
- safe single-instance mechanics,
- signal translation and graceful shutdown,
- restart supervision,
- OS-neutral core with explicit adapters,
- simple interaction composition,
- developer tooling and documentation.

LKW is the first product adoption and proof. Generic code MUST be implemented under platform namespaces and `APP-HOST-*` plan identifiers.

**Governance prerequisite** ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)): APP-HOST implementation follows platform-domain ownership. LKW adoption begins only after the required platform implementation slices are complete. Delivery order remains **platform implementation → LKW adoption → LKW live proof** — do not move LKW proof tasks before platform foundation rows.

---

## 2. Delivery principles

1. **Platform first, product proof second.**
2. **One public composition root.** Internal decomposition must not leak into ordinary authoring.
3. **Small extension surface.** Hooks, components, events, policies, plugins.
4. **No private event bus.** Reuse platform event/observability infrastructure.
5. **OS-neutral engine.** Native behavior behind explicit adapters.
6. **No second runtime.** Hosting wraps existing Tier-3 runtime and task execution.
7. **Green gates per row.** No wave is closed only by product proof.
8. **No broad speculative framework.** Implement smallest contract slice that satisfies the accepted architecture.
9. **Backward compatibility.** Existing FastAPI/Tier-3 applications continue to run without adopting hosting.
10. **LKW code never owns generic engine mechanics.**

---

## 3. Implementation waves

### APP-HOST-0 — Architecture and governance

| ID | Task | Status |
|----|------|--------|
| APP-HOST-0A | ADR and dedicated domain registration | **Done** |
| APP-HOST-0B | Architecture hub + extended-depth canon | **Done** |
| APP-HOST-0C | Implementation plan + fidelity matrix | **Done** |
| APP-HOST-0D | Tier-3/LKW cross-plan ownership correction | **Done** |

APP-HOST-0 closes with **0D Done**. Runtime implementation began at **APP-HOST-1A.1** (profile core).

### APP-HOST-1 — Public authoring contracts

| ID | Task | Status |
|----|------|--------|
| APP-HOST-1A.1 | Hosted Application Profile Core | **Done** |
| APP-HOST-1A.2 | Foundation `HostedApplicationProfile` composition root (contracts available before engine foundation) | **Done** |
| APP-HOST-1B | `HostedApplicationContext` and scoped typed service registry | **Done** |
| APP-HOST-1C | `HostedApplicationHooks` and hook registration model | **Done** |
| APP-HOST-1D | `HostedApplicationComponent` and component metadata contract | **Done** |
| APP-HOST-1E | Hosting policies and standard presets | **Done** |
| APP-HOST-1F | Contract package exports, schemas, compatibility gates | **Done** |

**Functional wave:** **APP-HOST-W1 — Complete Public Hosting Foundation — Done** (closes 1A.2, 1B, 1C, 1D, 1E, 1F, 3A contract vocabulary).

### APP-HOST-2 — Engine lifecycle foundation

| ID | Task | Status |
|----|------|--------|
| APP-HOST-2A | Lifecycle state machine and transition validation | **Done** |
| APP-HOST-2B | Profile composition/validation and immutable runtime definition | **Done** (foundation profile composition; plugin contribution/override semantics remain APP-HOST-9E) |
| APP-HOST-2C | Hook coordinator with ordering, timeouts, and failure semantics | **Done** |
| APP-HOST-2D | Component dependency graph and start/rollback/stop ordering | **Done** |
| APP-HOST-2E | Aggregate health/readiness and accepting-work guard | **Done** |
| APP-HOST-2F | `HostedApplicationEngine` startup/shutdown orchestration | **Done** |

### APP-HOST-3 — Events and diagnostics

| ID | Task | Status |
|----|------|--------|
| APP-HOST-3A | Typed/versioned hosting event contracts | **Done** |
| APP-HOST-3B | Event publication through existing Intergrax event spine | **Done** |
| APP-HOST-3C | Safe diagnostic/public snapshots and failure records | **Done** |
| APP-HOST-3D | Hosting metrics and observability integration | Planned |

### APP-HOST-4 — Instance ownership and graceful control

| ID | Task | Status |
|----|------|--------|
| APP-HOST-4A | `InstanceGuard`/`InstanceLease` contracts and file-lock reference implementation | **Done** |
| APP-HOST-4B | Stale ownership recovery and run-directory safety | **Done** |
| APP-HOST-4C | Typed shutdown/restart requests and idempotent control coordinator | **Done** |
| APP-HOST-4D | Drain/cancel/flush shutdown policy implementation | **Done** |
| APP-HOST-4E | Signal bridge contract and portable foreground signal adapter | **Done** |

### APP-HOST-5 — Supervisor and restart

| ID | Task | Status |
|----|------|--------|
| APP-HOST-5A | Exit/failure classification contracts | **Done** |
| APP-HOST-5B | Restart policy evaluator and deterministic backoff | **Done** |
| APP-HOST-5C | `HostedApplicationSupervisor` reference implementation | **Done** |
| APP-HOST-5D | Restart configuration/profile-digest preservation | Planned |
| APP-HOST-5E | Crash/restart/process-level proof harness | Planned |

### APP-HOST-6 — Interaction composition facade

| ID | Task | Status |
|----|------|--------|
| APP-HOST-6A | `InteractionProfile` public facade over existing platform intake | Planned |
| APP-HOST-6B | HTTP/MCP/intake surface composition bridge | Planned |
| APP-HOST-6C | Active interaction source component contract/adapter | Planned |
| APP-HOST-6D | Custom interaction surface/plugin example and security defaults | Planned |

### APP-HOST-7 — OS adapters and packaging posture

| ID | Task | Status |
|----|------|--------|
| APP-HOST-7A | OS services contract and automatic adapter selection | Planned |
| APP-HOST-7B | Windows paths/lock/signal adapter | Planned |
| APP-HOST-7C | Linux/XDG/file-lock/signal adapter | Planned |
| APP-HOST-7D | macOS paths/file-lock/signal adapter | Planned |
| APP-HOST-7E | Service-manager descriptor generation boundary | Planned |
| APP-HOST-7F | Packaging/install ownership decision and DX handoff | Planned |

### APP-HOST-8 — LKW adoption and live proof

| ID | Task | Status |
|----|------|--------|
| APP-HOST-8A | Define LKW hosted profile using platform contracts only | **Done** — `build_local_workspace_hosted_profile()` + private FastAPI/Uvicorn runtime adapter; no live `run_hosted_application()` yet |
| APP-HOST-8B | Migrate LKW.6A lifecycle/readiness to platform engine integration | **Planned — next** |
| APP-HOST-8C | LKW foreground hosted runner and single-instance proof | Blocked |
| APP-HOST-8D | Graceful stop + restart + request-after-restart live proof | Blocked |
| APP-HOST-8E | Structured ProofReceipt and reviewer documentation | Blocked |

`LKW.6B` is reframed as adoption/proof work. It must not implement generic Application Hosting internals.

### APP-HOST-9 — Developer experience and ecosystem

| ID | Task | Status |
|----|------|--------|
| APP-HOST-9A | `run_hosted_application(profile)` author facade | **Done** |
| APP-HOST-9B | `HarnessApplication.hosting(...)` integration | Planned |
| APP-HOST-9C | New-application scaffold/template support | Planned |
| APP-HOST-9D | Application creation/hosting author guide | Planned |
| APP-HOST-9E | Plugin/component examples and conformance kit; extends canonical `HostedApplicationProfile` with plugin field and plugin contract | Planned |
| APP-HOST-9F | Migration guide and stable API declaration | Planned |

---

## 4. Recommended execution order

The waves are not a license to implement every row at once. Recommended first sequence:

```text
APP-HOST-0D                    [Done]
→ APP-HOST-1A.1                [Done]
→ APP-HOST-W1                  [Done — public hosting foundation]
→ APP-HOST-W2                  [Done — engine foundation]
→ APP-HOST-W3                  [Done — process control and supervision]
  → APP-HOST-4A..4E
  → APP-HOST-5A..5C
→ APP-HOST-9A                  [Done — author facade]
→ APP-HOST-8A                  [Done — LKW hosted profile + runtime adapter]
→ APP-HOST-8B...               [next — lifecycle/readiness migration]
```

Required ordering before first adopter proof:

```text
platform contracts (W1 Done: 1A.1..1E, 3A, 1A.2, 1F)
→ engine foundation (2A..2F)
→ instance/shutdown minimum (4A/4C/4D/4E)
→ supervisor minimum (5A/5B/5C)
→ run_hosted_application(profile)   (APP-HOST-9A)
→ LKW adoption                        (APP-HOST-8A)
→ LKW live proof                      (APP-HOST-8C..8E)
```

Rationale:

- public contracts and lifecycle semantics precede the engine,
- typed hosting events (APP-HOST-3A) are designed before the foundation composition root (APP-HOST-1A.2) and engine behavior become opaque,
- APP-HOST-1A.2 is the foundation `HostedApplicationProfile` composition root — only contracts available before engine foundation; `HostedApplicationContext` is instance-scoped and not a profile field,
- contract package exports and compatibility gates (APP-HOST-1F) follow the foundation composition root,
- component coordination precedes LKW adoption,
- instance/shutdown/supervisor mechanics precede restart proof,
- **`run_hosted_application(profile)` (APP-HOST-9A) precedes LKW adoption (APP-HOST-8A)** — depends on APP-HOST-1A.2, 1F, 2F, 4 minimum foundation, and 5C,
- `InteractionProfile` extends the same canonical profile in APP-HOST-6A — not required for initial LKW proof; LKW may host existing interaction/runtime surfaces through the application runtime adapter,
- plugin field and plugin contract extend the same canonical profile in APP-HOST-9E,
- OS-specific service installation (APP-HOST-7) is not required for initial platform proof.

---

## 5. Code ownership target

Expected platform code:

```text
intergrax/hosting/
  contracts.py or contracts/
  context.py
  engine.py or engine/
  hooks.py
  components.py
  events.py or events/
  health.py
  instance.py or instance/
  control.py
  policies.py
  supervisor.py
  interactions.py
  plugins.py
  os/
```

Tier-3 bridges:

```text
intergrax/applications/_shared/hosting_wiring.py
intergrax/harness/app.py              # facade integration later
intergrax/applications/contracts/...  # only explicit references/bridges
```

First adopter:

```text
applications/local_workspace_application/hosting/
```

LKW files may define only LKW-specific profile, hooks, components, and proof runners.

---

## 6. Gates

Every `APP-HOST-*` implementation row requires:

```text
- focused unit/contract tests,
- architecture/plan fidelity update,
- no product-owned generic hosting code,
- no Task/Nexus dependency in supervisor/OS packages,
- no private hosting event bus,
- git diff --check,
- relevant repository governance checks,
- public API/schema tests when contracts change.
```

### Foundation gate

Before APP-HOST-2F closes:

```text
profile/context/hooks/components/policies defined
lifecycle transition matrix green
hook/component failure injection green
event contracts available
required/optional readiness behavior green
startup rollback and reverse shutdown ordering green
```

### Supervisor gate

Before APP-HOST-5 closes:

```text
clean stop vs failure classified
restart policy deterministic under fake clock
attempt exhaustion verified
new instance id created
profile digest preserved
supervisor imports no application/runtime cognition modules
```

### LKW proof gate

Before APP-HOST-8 closes:

```text
platform contract/engine suites green
LKW contains no generic engine/supervisor/OS code
foreground hosted start
READY state verified
real LKW request succeeds
single-instance rejection verified
graceful stop releases ownership
supervisor restart creates new instance identity
real LKW request succeeds after restart
structured receipt recorded
reviewer steps reproducible
```

Initial LKW proof does **not** require `InteractionProfile` (APP-HOST-6A) — LKW hosts existing interaction/runtime surfaces through the application runtime adapter. Interaction facade adoption may follow APP-HOST-6A.

Service-manager installation, OS adapter registration, and reboot survival are **not** required unless APP-HOST-7 is closed — they remain later operator/packaging targets.

---

## 7. Cross-domain plan ownership

| Domain | Required coordination |
|--------|-----------------------|
| `TIER3_APPLICATION_ENVIRONMENT` | application factory/runtime bridge; `HarnessApplication.hosting(...)`; distinction from `ApplicationHost.on_hook` |
| `OBSERVABILITY` | event/metric export and redaction integration |
| `RELIABILITY_FAILURE_AND_HITL` | failure taxonomy/backoff alignment; no ownership transfer |
| `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | scaffolding, guides, templates, examples |
| `ELASTIC_CAPACITY_AND_SCALING` | deployment boundary; cluster scheduling remains outside local hosting core |
| `INTEGRATIONS` | provider SDK/plugin ownership; no SDK leakage into hosting core |

Concrete domain-specific implementation rows must be added to the owning plan when work begins. This hosting plan coordinates but does not silently override other domain architecture.

---

## 8. Documentation deliverables

Required before stable release:

```text
docs/architecture/APPLICATION_HOSTING.md
docs/architecture/satellites/APPLICATION_HOSTING_extended_depth.md
docs/plan/APPLICATION_HOSTING.md
docs/plan/satellites/APPLICATION_HOSTING_implementation_detail.md
docs/guides/APPLICATION_HOSTING_GUIDE.md or equivalent accepted guide
applications/USAGE.md hosting section
guides/APPLICATION_CREATION_GUIDE.md hosting step
OS packaging/operator guides
LKW reviewer proof updates
```

---

## 9. Fidelity matrix

| Architecture section | Plan owner | Primary code target | Verification artifact | Status |
|----------------------|------------|---------------------|-----------------------|--------|
| Hub §1–3 purpose/ownership/invariants | APP-HOST-0 | documentation/governance | doc/index checks | Done |
| Hub §4–5 authoring/profile | APP-HOST-1, 9 | `intergrax/hosting/contracts*`, facade | schema/DX tests | **Contracts Done (W1); facade pending** |
| Hub §6–7 engine/lifecycle | APP-HOST-2 | `intergrax/hosting/engine*` | transition/failure tests | Planned |
| Hub §8 hooks/components/policies/plugins | APP-HOST-1,2,9 | contracts/coordinators/plugins | contract/order tests | **Contracts Done (W1); runtime/coordinators pending** |
| Hub §9 events | APP-HOST-3 | `intergrax/hosting/events*` | schema/order/redaction tests | **Contracts Done (W1); event spine bridge pending** |
| Hub §10 interactions | APP-HOST-6 | hosting/Tier-3 bridge | surface composition tests | Planned |
| Hub §11 instance/control/supervisor | APP-HOST-4,5 | instance/control/supervisor | process/restart tests | Planned |
| Hub §12 OS boundary | APP-HOST-7 | `intergrax/hosting/os/` | adapter contract tests | Planned |
| Hub §13 LKW adoption | APP-HOST-8 | LKW hosted profile/proof | live proof + receipt | Blocked |
| Satellite §20–23 model/context | APP-HOST-1 | contracts/context | validation/public-view tests | **Done (W1)** |
| Satellite §24 lifecycle | APP-HOST-2 | state machine/engine | transition matrix | Planned |
| Satellite §25 hooks | APP-HOST-1C,2C | hooks/coordinator | ordering/timeout/failure tests | **Contracts Done (W1)** |
| Satellite §26 components | APP-HOST-1D,2D | components/coordinator | DAG/rollback tests | **Contracts Done (W1)** |
| Satellite §27 health | APP-HOST-2E | health/readiness | aggregate health tests | Planned |
| Satellite §28 events | APP-HOST-3 | event contracts/publisher | event spine tests | **Contracts Done (W1)** |
| Satellite §29 policies | APP-HOST-1E,4D,5B | policies | preset/decision tests | **Contracts Done (W1)** |
| Satellite §30 instance | APP-HOST-4A/4B | instance guard | conflict/stale recovery tests | Done |
| Satellite §31 control | APP-HOST-4C/4E | signal/control bridge | idempotency tests | Planned |
| Satellite §32 supervisor | APP-HOST-5 | supervisor | restart/backoff proof | Planned |
| Satellite §33 interactions | APP-HOST-6 | interaction facade | existing intake reuse tests | Planned |
| Satellite §34 plugins | APP-HOST-9E | plugins | contribution/conflict tests | Planned |
| Satellite §35 OS | APP-HOST-7 | OS adapters | platform-specific tests | Planned |
| Satellite §36 observability | APP-HOST-3D | observability bridge | metric/event integration | Planned |
| Satellite §37–39 security/failure/concurrency | APP-HOST-1..5 | cross-cutting | security/failure/concurrency tests | Planned |
| Satellite §40 testing | all | tests/proof harness | gate suites | Planned |
| Satellite §41 LKW | APP-HOST-8 | LKW adoption | live reviewer proof | Blocked |
| Satellite §42–43 cross-domain/API | APP-HOST-0D,9F | docs/bridges | consistency/API compatibility | Planned |

No architecture section may remain without a plan owner before the domain is declared implementation-complete.

---

## 10. Current next task

```text
APP-HOST-8B — Migrate LKW.6A lifecycle/readiness to platform engine integration
```

**APP-HOST-8A — Define LKW hosted profile using platform contracts only — Done** (2026-07-16). LKW-owned `build_local_workspace_hosted_profile()` in `applications/local_workspace_application/hosting/`; private FastAPI/Uvicorn `HostedApplicationRuntime` adapter; settings/bind snapshot captured once; existing `LocalWorkspaceHostLifecycle` and `uvicorn … host.main:app` entrypoint retained. Live adoption proof and lifecycle migration remain APP-HOST-8B+.

**APP-HOST-9A — `run_hosted_application(profile)` author facade — Done** (2026-07-14). Public synchronous foreground runner: `run_hosted_application(profile) -> HostedApplicationSupervisorResult` in `intergrax/hosting/runner.py`; resolves profile once before side effects; assembles reference paths, clocks, logger, process identity, event publisher, control coordinator, signal adapter, instance guard, engine factory, and supervisor internally.

**APP-HOST-W3 — Process Control and Supervision — Done** (2026-07-14; corrective pass). Closes APP-HOST-4A..4E and APP-HOST-5A..5C.

W3 corrective pass hardened: linearizable file-lock acquisition (`HostedApplicationInstanceAcquisitionResult`), truthful lease release/events, effective STOP/RESTART control propagation, one global monotonic shutdown budget across all phases, wrapped exit classification, deterministic restart backoff with stable-window reset, and supervisor restart flow without bypassing cleanup evaluation.

W3 supervisor is **in-process**; W3 file lock and signal bridge are **portable reference implementations**. Full OS adapter/service-manager posture remains **APP-HOST-7**. Profile-source preservation contract remains **APP-HOST-5D**. Process proof harness remains **APP-HOST-5E**.

**APP-HOST-W3** delivered:

```text
APP-HOST-4A — InstanceGuard / InstanceLease reference implementation — Done
APP-HOST-4B — stale ownership recovery and run-directory safety — Done
APP-HOST-4C — typed shutdown/restart requests and control coordinator — Done
APP-HOST-4D — drain/cancel/flush shutdown policy implementation — Done
APP-HOST-4E — signal bridge and portable foreground signal adapter — Done
APP-HOST-5A — exit/failure classification contracts — Done
APP-HOST-5B — restart policy evaluator and deterministic backoff — Done
APP-HOST-5C — HostedApplicationSupervisor reference implementation — Done
```

**APP-HOST-W2 — Complete Hosting Engine — Done** (2026-07-14). Closes APP-HOST-2A..2F, APP-HOST-3B, APP-HOST-3C.

APP-HOST-2B note: foundation profile resolution is complete; real plugin contribution/override semantics remain APP-HOST-9E.

Target code:

```text
intergrax/hosting/engine/
intergrax/hosting/eventing.py
intergrax/hosting/errors.py
tests/unit/hosting/engine/
```

APP-HOST-0D (Tier-3/LKW cross-plan ownership correction) is **Done**. APP-HOST-1A.1 (Hosted Application Profile Core) is **Done**. **APP-HOST-W1** (public hosting foundation) is **Done**.

### APP-HOST-W1 delivered (2026-07-14)

Target code:

```text
intergrax/hosting/__init__.py
intergrax/hosting/services.py
intergrax/hosting/contracts/__init__.py
intergrax/hosting/contracts/identity.py
intergrax/hosting/contracts/profile.py
intergrax/hosting/contracts/public_data.py
intergrax/hosting/contracts/context.py
intergrax/hosting/contracts/lifecycle.py
intergrax/hosting/contracts/hooks.py
intergrax/hosting/contracts/components.py
intergrax/hosting/contracts/policies.py
intergrax/hosting/contracts/events.py
tests/unit/hosting/
```

Public API (from `intergrax.hosting`):

```text
HostedApplicationIdentity
HostedApplicationProfile / HostedApplicationProfilePublicView
HOSTED_APPLICATION_PROFILE_SPEC_VERSION
HostedApplicationContext / HostedApplicationContextPublicView
HostedApplicationPaths / HostedApplicationProcessIdentity
HostedApplicationLifecycleState / HostedApplicationLifecycleSnapshot
HostedApplicationServiceRegistry
HostedApplicationHook / HostedApplicationHooks / HostedApplicationHookPoint / HostedApplicationHookMode
HostedApplicationComponent / HostedApplicationComponentRegistration
HostedApplicationComponentHealth / HostedApplicationComponentState
LifecyclePolicy / ShutdownPolicy / RestartPolicy
ComponentFailurePolicy / HookFailurePolicy / InstancePolicy
HostedApplicationEvent / HostedApplicationEventType / HostedApplicationEventSubscription
HostedApplicationClock / HostedApplicationLogger / HostedApplicationEventPublisher
HostedApplicationShutdownCoordinator
```

W1 is contracts-only: no engine, supervisor, hook/component execution, event bus, or OS adapters.

APP-HOST-1A.1 (profile core) remains backward compatible:

```text
HostedApplicationProfile(application_id=..., application_factory=...)
```

APP-HOST-1A.2 defines the foundation `HostedApplicationProfile` composition root after APP-HOST-1B..1E and APP-HOST-3A exist. It may include only contracts available before the engine foundation:

```text
identity and application factory core
instance/lifecycle/shutdown/restart policies
HostedApplicationHooks
HostedApplicationComponent registrations
typed hosting event subscriptions (after APP-HOST-3A)
safe metadata/public view/digest
```

Explicitly out of scope for APP-HOST-1A.2:

```text
context references — HostedApplicationContext is instance-scoped, not a profile field (APP-HOST-1B)
InteractionProfile — extends the same canonical profile in APP-HOST-6A
plugins — plugin field and plugin contract extend the same canonical profile in APP-HOST-9E
HostedApplicationEngine
OS adapters
supervisor
LKW adoption
```

APP-HOST-9A (`run_hosted_application(profile)`) depends on at least APP-HOST-1A.2, APP-HOST-1F, APP-HOST-2F, APP-HOST-4 minimum foundation (4A/4C/4D/4E), and APP-HOST-5C, and must close before APP-HOST-8A.