# ADR-SEC-002: Third-Party Isolation and External Execution Boundary

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture only — assessment stage) |
| **Date** | 2026-09-07 |
| **Deciders** | Platform architecture (Capability Catalog Stage 12) |
| **Related** | [`PLATFORM_PLUGINS.md`](../../../architecture/PLATFORM_PLUGINS.md) · [`CAPABILITY_CATALOG_AND_DISCOVERY.md`](../../../architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md) · [`PLATFORM_PLUGIN_PRODUCTION_AUDIT.md`](../../../maintainers/plans/PLATFORM_PLUGIN_PRODUCTION_AUDIT.md) · [`AUTONOMOUS_WORK.md`](../../../architecture/AUTONOMOUS_WORK.md) · [`CODE_CRAFT.md`](../../../architecture/CODE_CRAFT.md) · [ADR-SEC-001](../2026-06-19/ADR-SEC-001.md) · [ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md) |

## Context

Capability Catalog Stages 1–11 delivered federated discovery, governance narrowing, AW integration, bootstrap evidence, and a read-only Marketplace product surface. Stage 11 increases pressure to host third-party capabilities from public and enterprise-private sources.

The platform as-built Platform Plugin model is **trusted in-process Python** (`PlatformPluginTrustModel.TRUSTED_IN_PROCESS` only). Production audit confirms: no sandbox, no signing verification, no process isolation. This is truthful current posture — not a defect to paper over.

Stage 12 is an **assessment and architecture decision** stage. It must freeze threat model vocabulary, authority boundaries, go/no-go criteria for future isolation providers, and explicit rejection of premature universal execution engines — **without** shipping sandbox runtime, remote execution, or mandatory bootstrap changes.

**User goal:** Platform assesses trusted in-process limitations and defines a governed path toward stronger third-party isolation when required.

## Current state

| Area | As-built fact |
|------|---------------|
| Plugin trust model | `PlatformPluginTrustModel.TRUSTED_IN_PROCESS` only — contract forbids sandbox/signing enum values |
| Default bootstrap | `wire_application_environment()` remains canonical Tier-3 composition; no isolation provider default |
| Capability Catalog | Read-only federating consumer; discovery does not mutate registries |
| Marketplace | Read-only product layer; listing ≠ execution permission |
| CodeCraft / AW generated code | Separate sandbox substrate path (CodeCraft); not Platform Plugin containment |
| Isolation provider interface | **Does not exist** in platform layer |

### Existing guarantees (after Platform Plugin production audit)

- Typed manifests and manifest secret rejection
- Plugin admission and explicit opt-in discovery (`INTERGRAX_DISCOVER_PLUGINS`)
- Platform compatibility checks (fail-closed for external packages without evidence)
- Production qualification gates (`require_production_qualification`)
- Scoped DI / `ToolWiringContext` per registration
- Tool invocation governance (access, scope, declarative policy)
- Domain-owned lifecycle and registry materialization
- Immutable bootstrap / evidence contracts where present (Stage 10)

### Explicitly NOT guaranteed

- Malicious Python containment
- Filesystem, process, network, or memory boundary isolation
- Syscall filtering
- Secret exfiltration prevention by sandbox substrate
- Cryptographic package authenticity (signing not implemented)
- Tenant safety against hostile plugin code holding mutable global state
- Resource quotas enforced by isolation substrate
- Manifest validation protecting host from arbitrary code execution at import time

**Package loading vs runtime execution:** Entry-point enumeration scans metadata; `load_entry_point_value()` invokes `EntryPoint.load()` which **imports executable third-party module code into the host process** before domain qualification gates run on the loaded target. Manifest validation does **not** prevent malicious import side effects. Remote Tool execution alone does **not** solve unsafe plugin import if the package was already imported into the trusted host.

## Threat model

### Two isolation planes (must not collapse)

| Plane | Question | Current state |
|-------|----------|---------------|
| **Package isolation / loading** | Can host inspect/admit package metadata without importing executable third-party code? | **Partial** — metadata scan exists; import occurs on load path |
| **Execution isolation** | Where/how does admitted code execute? | **Trusted in-process** for Platform Plugins; CodeCraft has separate sandbox tiers for generated code only |

### Risk classes

**A. Plugin loading/import risk** — Python package imported into host during bootstrap or plugin load.

**B. Runtime capability execution risk** — Tool/Agent operation after bootstrap.

If future goal is **untrusted third-party code**, architecture must ensure untrusted package code is **not imported into the trusted host**. A remote Tool executor is insufficient when plugin modules already execute in-process at import.

### Trust and execution posture vocabulary

**Code contract (today):** `PlatformPluginTrustModel.TRUSTED_IN_PROCESS` in `intergrax/core/plugins/platform_qualification.py`.

**Documentation-only execution trust classes** (future decision vocabulary — no new enum until ≥2 producers/consumers):

| Class | Meaning |
|-------|---------|
| `TRUSTED_IN_PROCESS` | Host accepts in-process execution under current model |
| `ISOLATION_REQUIRED` | Effective execution requires stronger boundary; no in-process fallback |
| `REMOTE_ONLY` | Execution must occur outside host process |
| `UNSPECIFIED` | Posture not declared; host policy decides |

**Execution posture metadata** (future catalog field — descriptive only):

Examples: `TRUSTED_IN_PROCESS`, `HOST_ISOLATION_REQUIRED`, `REMOTE_EXECUTION_REQUIRED`, `UNKNOWN`.

```text
execution_posture metadata ≠ execution enforcement proof
catalog discovery ≠ sandbox decision authority
marketplace source ≠ trust authority
plugin provenance ≠ safe-to-run proof
```

Capability Catalog **MAY** tag execution posture metadata in a future stage. Capability Catalog **MUST NOT** enforce isolation, choose isolation provider, launch sandbox, or mutate registries.

### Supply chain / signing assessment

| Capability | Status |
|------------|--------|
| Package signing | **Missing** |
| Publisher signing | **Missing** |
| Artifact digest verification | **Partial** (wheel/install paths vary; not platform-wide admission gate) |
| Transparency / provenance attestations | **Missing** |
| Revocation | **Missing** |
| Trusted roots | **Missing** |

Signing proves provenance/integrity, **not** behavior safety. Signed package ≠ safe package.

### Multi-tenancy

Plugin classes and catalog metadata are process-global. Third-party plugins with mutable global tenant caches create cross-tenant contamination risk. Stronger isolation may be required for hostile tenants — container isolation does not automatically solve all tenant semantics (secrets, shared caches, domain wiring).

### Generated code (LLM)

LLM-generated code is **untrusted code** unless governance explicitly classifies otherwise. Never equivalent to reviewed first-party Python. AW-7B / CodeCraft path requires hardened isolation when profile demands it; anti-downgrade semantics are documented in CodeCraft canon.

### Tool side effects

```text
sandbox ≠ authorization
```

Isolated execution does not replace Tool permission, side-effect policy, budgets, or HITL. Isolation is placement/security; governance remains authoritative.

### Network, filesystem, secrets, resources (future prerequisites)

Future isolation providers should support host-defined: deny-all / allowlist / explicit egress network policy; no filesystem / read-only / scoped writable workspace; scoped secret capability references (not full host environment); CPU, memory, wall time, process count, output size limits; deterministic cancellation/termination.

Stage 12 records requirements only — no implementation.

## Decision

### 1. V1 remains trusted in-process by default

- **Stronger isolation is NOT introduced** as default runtime in Stage 12.
- **No change** to `wire_application_environment()` default behavior.
- **No new mandatory** host profile fields or isolation provider defaults.
- **No new runtime dependencies** (Docker, K8s, gVisor, Firecracker, remote worker SDKs).

### 2. Isolation decision authority

Future isolation decisions are owned by:

```text
host/application security profile
+ governance policy
+ package/plugin qualification evidence
+ capability execution requirements
↓
Isolation Decision (host/governance owned)
↓
domain execution authority
```

**Not:**

```text
CapabilityCatalogEntry.execution_posture
↓
automatically select remote executor
```

Catalog metadata may **inform** decisions only.

### 3. Domain-specific execution remains authoritative

| Domain | Execution authority |
|--------|---------------------|
| Tool | `RuntimeToolInvoker` / Tool domain execution |
| Agent | Agent runtime / `RuntimeRevision` / Nexus path |
| Skill | Declarative resolution into Tool/runtime requirements — **Skill is not directly executable** |

Never define `isolate_skill_execution(skill)`. Skills may carry requirements that result in isolated Tool/Agent execution.

Isolation wraps domain execution boundaries; it does **not** unify Tool/Agent/Skill/Integration semantics.

### 4. Future isolation provider boundary (conceptual — not implemented)

A future provider **may** be responsible for: prepare execution boundary; execute isolated workload; return typed result/evidence; terminate/cleanup boundary.

A future provider **must not** own: capability discovery; `ToolRegistry` / `SkillRegistry` / `AgentRegistry`; Agent Distribution; governance rules; marketplace; billing; domain semantics.

Provider implementation requires **explicit follow-up approval** after prerequisites below are satisfied.

### 5. Future effective posture resolution (governed — not implemented in Stage 12)

```text
requested posture + source trust + host policy + operation risk
↓
effective execution posture
```

Reuse existing governance architecture ([ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md)); do not introduce a new policy engine in Stage 12.

### 6. Fail-closed future rules (hard contract — not wired yet)

When effective posture requires isolation and no qualifying provider exists:

```text
FAIL CLOSED — no implicit fallback to trusted in-process
```

Do not silently downgrade `REMOTE_REQUIRED` → in-process, including dev mode, unless host profile **explicitly** configures permissive override (future product decision).

STRICT hosts must not execute capabilities requiring isolation without provider evidence.

### 7. Capability Catalog and Marketplace boundaries (frozen)

- Catalog discovery **MUST NOT** enforce isolation.
- Marketplace listing (`CATALOG_AVAILABLE`) **≠** permission to execute (`HOST_AVAILABLE`).
- Marketplace origin **≠** execution trust.

### 8. AW-7B prerequisites (linkage only)

Autonomous Work may use stronger sandbox for generated/adaptive code only when: execution boundary exists; policy can require it; evidence is captured; no arbitrary fallback; resource limits exist; artifact provenance exists. See [`AUTONOMOUS_WORK.md`](../../../architecture/AUTONOMOUS_WORK.md) and [`CODE_CRAFT.md`](../../../architecture/CODE_CRAFT.md).

### 9. Future isolation levels (taxonomy only)

| Level | Description |
|-------|-------------|
| L0 | Trusted in-process |
| L1 | Dedicated process |
| L2 | Container / sandbox |
| L3 | Remote execution boundary |

Documentation-only until consumers exist. No `IsolationLevelProviderRegistry` in Stage 12.

## Rejected alternatives

### Universal sandbox / execution engine

```text
Capability Catalog → UniversalSandboxEngine → execute arbitrary capability
```

**Rejected:** duplicates domain runtime; collapses Tool/Agent/Skill semantics; wrong authority; speculative abstraction.

### Mandatory subprocess isolation for all plugins now

**Rejected:** unnecessary breaking change; no transport contract; no resource/secret model; no package-loading solution; no evidence semantics; performance regression.

### Marketplace trust = execution trust

```text
official marketplace → trusted in-process automatically
```

**Rejected:** marketplace origin alone is insufficient security authority.

### Signature only = safe

**Rejected:** signing proves provenance/integrity, not behavior safety.

## Go / no-go criteria

| Condition | In-process allowed? | Strong isolation required? |
|-----------|---------------------|----------------------------|
| First-party reviewed code | Yes (host policy) | No (unless capability risk demands) |
| Enterprise-controlled private plugin | Possibly (policy-dependent) | Policy-dependent |
| Signed approved vendor | Policy-dependent | Policy-dependent |
| Unknown public publisher | No (default posture) | Yes when execution required |
| Untrusted arbitrary package | No | Yes |
| Capability requires filesystem/network secrets | Higher isolation | Yes |
| Multi-tenant hostile boundary | No default | Yes |
| LLM-generated executable code | No default | Yes (CodeCraft / AW path) |

No absolute guarantee beyond documented facts. Host profile and governance remain authoritative.

## Go-live prerequisites for future isolation provider

Implementation may proceed only when all are frozen:

1. Trust/posture contract (shared vocabulary, effective vs requested)
2. Host policy ownership
3. Provider selection authority (not catalog)
4. Package-loading model (import vs out-of-process admission)
5. Tool/Agent transport contract
6. Secret scoping
7. Filesystem policy
8. Network policy
9. Resource limits
10. Cancellation/timeout semantics
11. Result serialization
12. Audit evidence schema
13. Failure semantics vocabulary
14. Compatibility / fallback rules (fail-closed default)
15. Production qualification tests (independent of normal plugin qualification)

### Remote execution additional prerequisites

Authenticated transport; request integrity; replay protection/idempotency where relevant; capability identity/version digest; tenant identity; policy decision reference; result integrity; retry semantics; observability correlation.

### Future failure semantics (vocabulary)

`ISOLATION_UNAVAILABLE`, `ISOLATION_POLICY_DENIED`, `ISOLATION_START_FAILED`, `EXECUTION_TIMEOUT`, `EXECUTION_CANCELLED`, `REMOTE_TRANSPORT_FAILED`, `REMOTE_RESULT_INVALID` — document only until implementation approved.

### Future provider qualification axes

Containment; resource enforcement; secret boundary; network boundary; cancellation; cleanup; evidence; failure injection.

### Future evidence fields

Isolation/provider identity; posture requested/effective; execution environment identifier; resource limits; package/artifact digest; result status; timeout/cancel; policy decision reference.

```text
execution evidence ≠ runtime authority
```

## Migration / compatibility

- Stage 12 introduces **no** schema changes to `CapabilityCatalogEntry`, `CapabilityProvenance`, or `MarketplaceCapabilityListing`.
- Default bootstrap and `PlatformPluginTrustModel` unchanged.
- Stage 13+ metering and stronger isolation implementation remain separate programs.

## Security consequences

- Operators must treat third-party plugin installation as **deploying code into the host process**.
- Public Marketplace growth increases discovery surface; it does **not** reduce execution trust requirements.
- Future isolation without package-loading plane redesign cannot safely host arbitrary untrusted Python packages.

## Open future decisions

- Whether `execution_posture` becomes a typed catalog field (requires ≥2 real producers/consumers).
- Which isolation substrate(s) qualify for enterprise multi-tenant (process vs container vs remote).
- Signing and revocation product requirements.
- Whether dev profiles may opt into permissive downgrade (explicit configuration only).

## Compliance

- Tier boundaries preserved — no Tier-0 universal execution engine
- Capability Catalog remains read-only federator
- Domain registries and Nexus execution paths unchanged
- [`PLATFORM_PLUGINS.md`](../../../architecture/PLATFORM_PLUGINS.md), [`CAPABILITY_CATALOG_AND_DISCOVERY.md`](../../../architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md), and plan Stage 12 row updated

## Implementation notes

- Stage 12 deliverable: this ADR + documentation updates + traceability tests only
- Verification: `python scripts/maintenance/check_harness_adr.py`; `pytest tests/unit/architecture/test_stage12_isolation_decision.py -q`
- Regression: capability catalog, marketplace, AW Stage 9, platform plugin contract suites
