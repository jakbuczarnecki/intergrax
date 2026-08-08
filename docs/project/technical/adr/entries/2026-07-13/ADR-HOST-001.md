# ADR-HOST-001: Application Hosting as a Dedicated Platform Domain

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and implementation planning only |
| **Date** | 2026-07-13 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/APPLICATION_HOSTING.md`](../../../../architecture/APPLICATION_HOSTING.md) · [`plan/APPLICATION_HOSTING.md`](../../../../maintainers/plans/APPLICATION_HOSTING.md) · [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) |

## Context

Intergrax Tier-3 applications already define what an application is and how it executes:

- `ApplicationManifest`,
- `ApplicationEnvironmentProfile`,
- `HarnessApplication`,
- `UnifiedTaskRunner`,
- `ApplicationHost.on_hook`,
- interaction intake and task normalization,
- FastAPI/MCP product surfaces.

The platform does not yet define a reusable, author-facing standard for keeping an arbitrary Tier-3 application running continuously as a managed local or server process. The missing capability includes:

- process lifecycle,
- lifecycle hooks and typed events,
- component start/stop/health coordination,
- readiness and liveness aggregation,
- single-instance protection,
- signal handling and graceful shutdown,
- restart supervision,
- operating-system adapters,
- a simple declarative authoring surface,
- reusable interaction-surface composition.

Local Workspace Application exposed this gap while evolving toward an always-on personal Agent OS. Implementing the mechanism inside LKW would create product-owned infrastructure that future applications would have to copy.

The new capability is larger than a narrow LKW feature and has a coherent implementation owner. It also introduces public contracts and runtime primitives, so it requires its own architecture/plan pair rather than informal additions to product documentation.

## Alternatives considered

### 1. Implement an LKW-specific daemon

Rejected. It would couple generic process hosting to one product and encourage future `legal daemon`, `coding daemon`, and `research daemon` copies.

### 2. Extend only `TIER3_APPLICATION_ENVIRONMENT`

Rejected as sole ownership. Tier-3 defines application composition and execution. Continuous process hosting has its own lifecycle, supervisor, OS boundary, events, policies, and verification surface. Folding all of it into the already frozen Tier-3 canon would blur responsibilities and significantly increase authoring/document context.

### 3. Document Application Hosting as a multi-layer feature

Rejected as primary ownership. Feature documents coordinate multiple owning domains; Application Hosting has a coherent contract/runtime owner and therefore qualifies as a domain pair. Cross-domain integration rows may still be added to Tier-3, observability, reliability, and DX plans.

### 4. Create a dedicated `APPLICATION_HOSTING` domain

Accepted.

## Decision

1. Introduce a dedicated platform domain pair:

   ```text
   docs/project/architecture/APPLICATION_HOSTING.md
   docs/project/maintainers/plans/APPLICATION_HOSTING.md
   ```

   with extended detail in matching architecture and plan satellites.

2. Application Hosting is **100% platform-owned**. LKW is the first adopter and proof workload, not the owner of generic hosting contracts or runtime mechanics.

3. Application Hosting wraps a Tier-3 application instance. It does not replace:

   - `ApplicationManifest`,
   - `ApplicationEnvironmentProfile`,
   - `ApplicationHost.on_hook`,
   - `UnifiedTaskRunner`,
   - `NexusLoop`,
   - the existing interaction model.

4. Preserve a single Tier-3 application composition root. `HostedApplicationProfile` is a hosting profile for a built/configured application, not a competing harness environment root.

5. The primary author experience is declarative and centralized:

   ```python
   profile = HostedApplicationProfile(
       application_id="my_application",
       application_factory=create_application,
   )

   run_hosted_application(profile)
   ```

6. The platform may use multiple internal services, but the public extension model is intentionally small:

   - `HostedApplicationHooks`,
   - `HostedApplicationComponent`,
   - typed hosting events/subscriptions,
   - hosting policies,
   - optional hosting plugins.

7. `ApplicationHost.on_hook` and `HostedApplicationHooks` remain different mechanisms:

   - `ApplicationHost.on_hook` reacts to Nexus/application-execution boundaries;
   - `HostedApplicationHooks` reacts to process-hosting lifecycle boundaries.

8. Hosting lifecycle events must use the existing Intergrax event/observability spine. No private hosting-only event bus is allowed.

9. The hosting engine remains OS-neutral. Windows, Linux/systemd, and macOS/launchd integrations implement explicit adapter contracts outside application code.

10. The supervisor owns process restart, backoff, exit classification, and single-instance mechanics. It must not depend on `Task`, `NexusLoop`, agents, tools, or product capabilities.

11. Convention over configuration is normative. Standard signal handling, health/readiness, lifecycle events, graceful shutdown, and safe single-instance defaults are enabled without requiring authors to wire many contracts manually.

12. LKW adoption may begin only after the platform architecture and implementation plan are accepted. Generic implementation work uses `APP-HOST-*` identifiers; LKW identifiers cover adoption and proof only.

## Consequences

### Positive

- Any Intergrax application can adopt always-on hosting without copying LKW code.
- Hosting contracts, runtime, supervisor, OS adapters, events, and authoring DX receive clear ownership.
- Authors receive one composition profile with progressive extension depth.
- Application hosting becomes testable independently from any one product or operating system.
- LKW can demonstrate real platform value while remaining a normal Tier-3 application.

### Negative

- A new domain pair and governance surface must be maintained.
- Tier-3, observability, reliability, and developer-experience plans require explicit cross-plan coordination.
- Temporary overlap exists with LKW.6A lifecycle types until platform hosting contracts are implemented and LKW migrates.
- OS service installation remains a separate complexity from the platform-neutral hosting engine.

## Compliance and invariants

- **HOST-INV-01:** Application Hosting is platform-owned; product applications may only adopt or extend it.
- **HOST-INV-02:** Hosting does not perform agent cognition or Nexus orchestration.
- **HOST-INV-03:** The supervisor does not depend on `Task`, `NexusLoop`, agents, tools, skills, or capabilities.
- **HOST-INV-04:** Application code contains no OS branching for standard hosting behavior.
- **HOST-INV-05:** One declarative profile is the primary author composition surface.
- **HOST-INV-06:** Public extension points remain small, typed, and discoverable from the hosting profile.
- **HOST-INV-07:** Hosting events use the platform event/observability spine.
- **HOST-INV-08:** Hosting hooks are lifecycle reactions, not orchestration loops.
- **HOST-INV-09:** Disabled or optional components do not block readiness; unhealthy required components do.
- **HOST-INV-10:** LKW proof cannot substitute for platform contract and engine tests.

## Implementation notes

- Architecture hub: `docs/project/architecture/APPLICATION_HOSTING.md`
- Architecture detail: `docs/project/architecture/satellites/APPLICATION_HOSTING_extended_depth.md`
- Plan hub: `docs/project/maintainers/plans/APPLICATION_HOSTING.md`
- Plan detail: `docs/project/maintainers/plans/satellites/APPLICATION_HOSTING_implementation_detail.md`
- Expected code root: `intergrax/hosting/`
- Tier-3 integration remains under `intergrax/applications/` and `intergrax/harness/`.
- First adopter/proof: `applications/local_workspace_application/`.
- No production implementation is authorized by this ADR alone; implementation proceeds through accepted `APP-HOST-*` plan rows.