# ADR-SFX-001: SCENARIO-FACTORY-X placement and ownership (Option B)

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (T-IGX-037-C, SFX-P2-01; **not Accepted** - acceptance is OD-SFX-06, a separate operator review) |
| **Date** | 2026-11-12 |
| **Deciders** | Operator (OD-SFX-01, OD-SFX-06); C-ECAP-01 and ME-1 are confirmed in the OD-SFX-06 review |
| **Code reference SHA** | `eadb1c2116fe` (no file under `intergrax/`, `agents/`, `applications/`, `scripts/` or `tests/` differs between this SHA and the T-IGX-037-C branch base) |
| **Related** | [`T-IGX-037-C`](../../../../../../governance/tickets/T-IGX-037-C.md) · SFX master plan R1 §3, §6, §7.1-§7.3, §7.8 (`governance/tickets/plans/SFX-MASTER-PLAN.md`, published by T-IGX-037-A) · [`ADR-SEF-001`](../2026-11-05/ADR-SEF-001.md) · [`ADR-SFX-005`](ADR-SFX-005.md) · [`ADR-SFX-006`](ADR-SFX-006.md) · [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) · [`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](../../../../maintainers/qualification/INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) · [`ECAP`](../../../../maintainers/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) |

## Context

SCENARIO-FACTORY-X (SFX) is the post-freeze product layer that turns a business problem into a discoverable, configurable, executable, testable and demonstrable scenario (OD-SFX-01). It has two front doors, Scenario Studio and an IDE toolkit (OD-SFX-02, OD-SFX-03), and it must be built only from existing platform mechanisms: no second engine, governance authority, ToolRuntime, registry or audit mechanism.

This ADR decides where SFX code lives, who owns each SFX subsystem, and which existing platform owner each subsystem must reuse. It also records three open review items from the master plan:

- **SFX-GAP-09:** the dual `ToolRegistry` ownership review (MISSING_EVIDENCE in R1 §6).
- **C-ECAP-01:** whether ECAP-§16 applies to a Tier-2 product module.
- **ME-1:** the post-freeze change class of a new `applications/<x>` host.

Hard boundaries that constrain every option: `intergrax/` must not import `agents/` or `applications/`; `agents/` must not import `applications/`; `applications/` may import both.

Alternatives considered:

| Option | Summary | Disposition and rationale |
|--------|---------|---------------------------|
| **A - Core platform domain** | `intergrax/scenario_factory/` as a Tier-0/Tier-1 domain | **Rejected.** It would create a new platform domain and responsibility inside the frozen core, which triggers ECAP-§16 (`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md:1111`) and very likely Class C under the touch rule (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:176`). It contradicts OD-SFX-01 ("SFX is the product layer") and PLATFORM-INV-007 Simplicity First: the factory's business behaviour does not need platform authority. |
| **B - Tier-2 factory core + Tier-3 host + agent-env toolkit** | `agents/scenario_factory` (single semantic implementation), `applications/scenario_studio_application` (Studio API, `sfx` CLI, MCP surface, SPA), agent-env skills/rules/agents | **Selected.** See Decision. |
| **C - External product** | A separate repository or service that calls Intergrax over HTTP/MCP | **Rejected.** It loses the in-repo tier boundary checks and the shared Tier-2 schema source (parity rule PC-1), needs a second API/auth/persistence stack, and turns Intergrax Runtime into a remote dependency instead of the only execution environment (OD-SFX-01). Reuse of Collaborative Work, decision bindings and evidence would degrade to a vendor-style integration. |
| **D - IDE-only toolkit** | agent-env skills and CLI only, no Studio and no service layer | **Rejected as a standalone option; kept as a subset of B.** It cannot serve business users (OD-SFX-02), and the offline CLI is limited to L0/L1 on local files (ADR-SFX-006), so shared-store writes and L2 runs would have no authenticated door. In B, the IDE toolkit is one front door over the same Tier-2 service. |

## Decision

Adopt **Option B**.

1. **Tier-2 `agents/scenario_factory/` owns all SFX semantics.** This covers the `sfx.<name>.v1` contracts and JSON Schemas, the canonicalizer, the service layer (the single semantic implementation for every door), the Discovery and Specification agents (STAGING, run through NexusLoop), the resolver, blueprint/profile composer, integration setup, `EngineeringHandoffPort`, scenario qualification and security gates, reporter/ROI, the catalog sidecar ([`ADR-SFX-005`](ADR-SFX-005.md)), and persistence adapters onto existing owners.
2. **Tier-3 `applications/scenario_studio_application/` is the only composition root for SFX doors** (PLATFORM-INV-008). It holds the Studio API v1 on `create_app(ApiConfig)` (`intergrax/fastapi_core/app_factory.py:29`), the `sfx` CLI, the MCP surface on `compose_application_mcp_server` (`intergrax/applications/_shared/mcp_application_server.py:45`) with an explicit allowlist, and the SPA under `web/` (ADR-SFX-003). Doors are adapters: they parse transport and call the Tier-2 service, with no validation or semantics of their own.
3. **agent-env** holds the IDE toolkit (skills, rules, agents) as an adapter over the CLI (offline) and MCP (live), generated through the existing SSOT mechanism; there is no second rule source.
4. **The platform is reuse-only.** SFX adds no Tier-0/Tier-1 code. Every platform need maps to an existing owner, as listed in the table below.
5. **Generic-capability stop rule (PLATFORM-INV-001, PLATFORM-INV-011).** If SFX work reveals a reusable, non-scenario-specific capability, implementation stops and the capability is routed to its platform domain owner. SFX does not host it "temporarily".

### No-duplicate-mechanism table

Code anchors are `path:line` at `eadb1c2116fe`.

| SFX need | Existing owner (reuse) | Forbidden duplicate | Principles |
|----------|------------------------|---------------------|------------|
| Scenario execution | `execute_scenario_task` (`intergrax/applications/_shared/scenario_runtime_baseline.py:569`) → `NexusLoop` (`intergrax/runtime/nexus/nexus_loop.py:215`) → `PolicyEngine` (`intergrax/runtime/policy/policy_engine.py:46`) → `ToolRuntime` (`intergrax/runtime/nexus/tools/tool_runtime.py:134`) | SFX engine, loop, planner runtime or tool invoker | PLATFORM-INV-003, PLATFORM-INV-010, PLATFORM-INV-011 |
| Tool availability | Runtime `ToolRegistry` (`intergrax/tools/registry/runtime.py:28`) read through `ToolRegistryRead` (`intergrax/tools/registry/read.py:14`) | SFX tool registry; any use of the deprecated `intergrax/tools/tools_base.py:110` registry | PLATFORM-INV-003, PLATFORM-INV-010; "global registry with mutable ownership" is forbidden (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:317`) |
| Capability resolution and gap acquisition | `FederatedCapabilityCatalog` (`intergrax/capability_catalog/federation.py:74`); UCA Governed Capability Fulfillment | SFX capability index or acquisition flow | PLATFORM-INV-001, PLATFORM-INV-010 |
| Agent binding | `AgentRegistry` (`intergrax/runtime/registry/agent_registry.py:39`) | SFX agent registry | PLATFORM-INV-003 |
| Authorization and approval | `PolicyEngine`; `CollaborativeDecisionBindingRepository` (`intergrax/collaborative_work/repository.py:1414`) | SFX approval store (the in-memory `ApprovalStorePort` is avoided, SFX-GAP-08) | PLATFORM-INV-003, PLATFORM-INV-010 |
| Durable scenario artifacts | `WorkArtifact` / `WorkArtifactVersion` (`intergrax/contracts/collaborative_work.py:1238`, `:1283`); `ArtifactContentStore` (`intergrax/collaborative_work/content_storage.py:105`) | SFX store, SFX state engine | PLATFORM-INV-010 |
| Sessions and resume | `SessionManager` (`intergrax/runtime/nexus/session/session_manager.py:44`) | SFX session store | PLATFORM-INV-010 |
| Scaffold jobs and schedules | `TaskCheckpoint` (`intergrax/runtime/long_running/models.py:19`); `LongRunningScheduler` (`intergrax/runtime/long_running/scheduler.py:66`) | SFX job queue or retry owner (a new retry owner is Class C, `INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:135`) | PLATFORM-INV-013, PLATFORM-INV-010 |
| Run facts, audit, telemetry | `EvidencePersistencePort` (`intergrax/contracts/execution_evidence/persistence_port.py:23`); observability spine | Second audit or event mechanism | PLATFORM-INV-014, PLATFORM-INV-010 |
| Evaluation | `EvalRunner` (`intergrax/eval/eval_runner.py:21`) | SFX evaluation engine | PLATFORM-INV-010 |
| Deterministic build | `intergrax/scaffold/cli.py` (`new-agent`, `new-application`, ...); `scripts/proof/init_scenario_implementation.py` | SFX code generator | PLATFORM-INV-003, PLATFORM-INV-010 |
| Agentic build | SEF-X Engineering Execution Authority ([`ADR-SEF-001`](../2026-11-05/ADR-SEF-001.md)), only after GATE-SEF-CONSUME ([`ADR-SFX-006`](ADR-SFX-006.md)) | SFX engineering FSM, workspace lease or coding backend | PLATFORM-INV-003 |
| Lifecycle and conformance | `scripts/proof/scenario_lifecycle.py`, `scripts/proof/scenario_architecture_conformance.py` (mapping only, R1 §7.9) | SFX conformance authority | PLATFORM-INV-005 |
| Scenario catalog | Sidecar metadata modelled on `AgentCatalogEntry` (`intergrax/agent_distribution/catalog.py:99`), persisted in Collaborative Work ([`ADR-SFX-005`](ADR-SFX-005.md)) | Scenario registry; manifest change; `EnvironmentExtensions` use | PLATFORM-INV-009, PLATFORM-INV-003 |
| HTTP host and MCP | `create_app` (`intergrax/fastapi_core/app_factory.py:29`); `compose_application_mcp_server` (`intergrax/applications/_shared/mcp_application_server.py:45`) | SFX web framework or MCP server stack | PLATFORM-INV-008, PLATFORM-INV-004 |
| IDE rules and skills | `agent-env/` + `scripts/governance/sync_agent_env.py` | Second rule SSOT under `.claude/` or `.cursor/` | PLATFORM-INV-003 |
| Hooks | Existing hook points only | Private engines inside hooks | PLATFORM-INV-015 |

Other principles: PLATFORM-INV-002 Architecture Before Adoption (these ADRs precede any SFX code, which waits for ARCH-FREEZE CLOSED); PLATFORM-INV-005 Proof Follows Implementation (no performance or quality value without a run artifact); PLATFORM-INV-006 Deployment Transparency and PLATFORM-INV-012 Applications Remain Semantically Portable (generated scenarios are ordinary Tier-3 packages and carry no SFX runtime dependency).

### Dual `ToolRegistry` ownership review (SFX-GAP-09)

**Verdict: VERIFIED.** ToolRuntime uses only the runtime registry `intergrax.tools.registry.runtime.ToolRegistry`. The legacy `intergrax.tools.tools_base.ToolRegistry` is deprecated, has no production importer, and is guarded by a DUP-X gate.

Import trace at `eadb1c2116fe`:

1. `ToolRuntime` (`intergrax/runtime/nexus/tools/tool_runtime.py:134`) imports no registry class. It reaches tools through three paths: `run_tools_context` (`:269`), `invoke_catalog_tool_ids` (`:287`) and `RuntimeToolGateway.for_state` (`:327`).
2. All three paths resolve the registry structurally from `state.context.config.tool_invoker` (`intergrax/runtime/nexus/tools/plan_context_invocation.py:326`, `:344`; `intergrax/runtime/nexus/tools/catalog_dispatch.py:111`; `intergrax/runtime/nexus/tools/tool_gateway.py:106`) through `resolve_tool_registry` (`intergrax/runtime/nexus/tools/catalog_dispatch.py:57-62`). That function returns `ToolInvokerProtocol.registry`, typed `ToolRegistryRead` (`intergrax/runtime/nexus/tools/tool_invoker_protocol.py:21`).
3. `ToolRegistryRead.get` returns `RegisteredTool` from `intergrax.tools.registry.runtime` (`intergrax/tools/registry/read.py:11`, `:19`). The canonical invoker `RuntimeToolInvoker` takes `registry: ToolRegistryRead` (`intergrax/runtime/nexus/tools/invoker.py:202`, `:307`).
4. The planner paths import the runtime class directly or through the package re-export (`intergrax/tools/registry/__init__.py:8`): for example `intergrax/runtime/nexus/tools/atomic_planner_round.py:42`, `planner_bootstrap.py:11` and `tool_planning_service.py:24`.
5. The legacy registry (`intergrax/tools/tools_base.py:110`) stores `ToolBase` objects, has no `has` or `activation_metadata`, and `get` returns `ToolBase` (`:125`). It therefore cannot satisfy `ToolRegistryRead`. Its constructor emits a deprecation warning (`:117`).
6. No `.py` file under `intergrax/`, `agents/` or `applications/` other than `tools_base.py` imports `tools_base`. This is enforced by `tests/qualification/dup_x/test_dup_x_tools_base_production_import_gate.py:38` and `tests/qualification/dup_x/test_dup_x_closed_world_integrity.py:60`, and recorded as DUPX-CAND-01 `N_A_WITH_EVIDENCE` (`tests/qualification/dup_x/_findings_register.py:125-128`).

**Consequence for SFX:** SFX code and generated packages bind tools only through `ToolProfile` → runtime `ToolRegistry` → `RuntimeToolInvoker`. The SFX conformance rule (SFX-P9-02) must forbid imports of `intergrax.tools.tools_base` in generated packages, matching the DUP-X gate. SFX-GAP-09 moves from MISSING_EVIDENCE to VERIFIED, with no platform action needed.

### C-ECAP-01 - ECAP-§16 applicability to a Tier-2 product module

ECAP-§16 requires a protocol update for, among other triggers, a "new platform domain or responsibility" and a "new public contract or extension mechanism" (`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md:1109-1112`).

**Recommendation: ECAP-§16 does not apply to SFX under Option B, and no ECAP revision is required.**

- SFX is a Tier-2 domain module and a Tier-3 host, not a platform domain.
- Its `sfx.*.v1` contracts are product-owned, and they extend no platform extension surface.
- SFX adds no `FRZ-*` criterion (R1 A2: 225 / 26 unchanged), no infrastructure/provider class and no reopen trigger.
- ECAP is procedural, not a semantic authority (`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md:3-4`, ECAP-§1.4).
- SFX remains subject to ECAP application through post-freeze evolution (ECAP-§1.3; ECAP-§2.D D1-D6 per wave, `:172-185`).
- The closest precedent is PDC-X: the §16 domain-section question was decided inside the stage (PDC-P3-D04, "no new ECAP domain sections"), and coverage mapped onto existing FRZ families (`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md:26`). MCP-SERVER-X did the same with incremental evidence (`:24`).
- Following that precedent, an optional one-line campaign note ("incremental evidence on existing families, no `FRZ-SFX-*`") may be added at SFX closure (OD-SFX-11). That note is not a §16 revision.

**ECAP-§16 becomes applicable** if a later decision makes SFX introduce a platform-owned public contract or extension mechanism. Examples are the [`ADR-SFX-005`](ADR-SFX-005.md) manifest exception path, or moving any SFX mechanism into `intergrax/`.

**Who decides:** the operator, in the OD-SFX-06 review of this ADR. ECAP-§1.4 routes semantic conflicts to "STOP - ARCHITECTURE DECISION REQUIRED", so neither SFX nor an implementer session may settle it.

### ME-1 - post-freeze class of a new `applications/<x>`

**Finding: Class A by definition, subject to per-wave classification of the actual diff.**

- Class A, Safe extension, explicitly lists "New domain module attached via contracts" (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:53`, `:61`), with the model "existing contract → new implementation → composition registration" (`:67`).
- The Extension Point Registry allows "Domain module" and "Composition root binding" at Class A (`:191-192`), and "Production composition roots" at Class A for a new allowed binding (`:173`).
- A wholly new contract without new ownership or authority is Class A or B (`:147`).
- This applies to both `agents/scenario_factory` and `applications/scenario_studio_application`, provided that neither modifies a frozen surface (touch rule `:176`) nor introduces new ownership, authority or an alternate execution/governance path (`:148`). Either of those would be Class C.
- The fail-safe rule applies: ambiguity defaults to Class C (`:44`). Classification is ECAP-§2.D step D1 on each wave's actual diff, so this ADR cannot pre-classify future diffs.
- Class A minimum gates apply (`:359`). Independent audit is only "recommended" for high-risk Class A extensions (`:414`). This ADR proposes making it **mandatory** for SFX waves that touch auth, persistence or security gates.
- Not covered by the class: IIAC ACT-N11 product-path re-binds (C-IIAC-01) and ARCH-FREEZE inventory interactions (C-AF-01) are separate obligations.

## Consequences

### Positive

- One semantic implementation serves Studio, CLI, MCP and the IDE (parity PC-1..PC-6, ADR-SFX-004). Tier boundaries make door-local semantics structurally impossible.
- No frozen-core change: SFX is a Class A domain module plus host, and the platform keeps sole ownership of execution, governance, tools, persistence and evidence.
- The ToolRegistry ambiguity is closed with evidence and an existing gate.

### Negative

- Tier-2 placement requires Tier-2 tests and boundary scans for a large product module. Tier-3 doors must stay thin, which needs a conformance check (no validation logic in doors).
- SFX inherits the open qualification states of its owners: Collaborative Work is REQUALIFICATION_REQUIRED, MCP is not CLOSED, and the long-running PDC work is open. SFX claims cannot exceed them.
- New product paths under `agents/` and `applications/` trigger IIAC ACT-N11 re-binds for every product wave (C-IIAC-01).

## Compliance

- Tier boundaries preserved: SFX imports downward only, and `intergrax/` gains no SFX code.
- Policy and security: all SFX side effects go through PolicyEngine/ToolRuntime. SFX holds no L3 authority (ADR-SFX-006).
- This ADR stays **Proposed** until OD-SFX-06. It authorizes no code: implementation requires ARCH-FREEZE CLOSED and GATE-SFX-ENTRY.

## Implementation notes

- **Not in scope at Proposed:** any file under `agents/scenario_factory/` or `applications/scenario_studio_application/`, and any IIAC re-bind.
- **Revision triggers:** any SFX mechanism proposed for `intergrax/`; any door holding semantics; any Class B/C classification of an SFX wave; reappearance of a `tools_base` production importer.
- **Verification (when implemented):** boundary scans, the DUP-X `tools_base` gate, the SFX door-thinness conformance rule, and `check_harness_adr.py` after the index row is added.
