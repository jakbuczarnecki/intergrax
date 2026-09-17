# Harness Top-Tier Gap Audit — Integrax vs capability baseline

## 1. Audit scope

| Field | Value |
| --- | --- |
| **Task** | Harness Gap Audit — Integrax vs Top-Tier Harness Baseline |
| **Type** | As-built platform map, A–Z capability scorecard, enterprise gap analysis, remediation roadmap (docs only) |
| **Methodology** | Code-first evidence (`intergrax/`, `agents/`, `applications/`, `tests/`); qualification docs as secondary anchors; no runtime implementation |
| **START_HEAD** | `cfde2b8a497e9153838ae7ccc1c21bd0d049fc7c` (= `origin/development` at session open) |
| **AUDITED_HEAD / FINAL_HEAD** | `55db3f95f2af61166cedfdb4667fbac97cb075a5` (local `development`, **ahead 2** of `origin/development`) |
| **FINAL_ORIGIN_DEVELOPMENT** | `cfde2b8a497e9153838ae7ccc1c21bd0d049fc7c` |
| **TASK SHA (audit commit)** | *(set at commit — see git log for `docs(architecture): audit top-tier harness gaps`)* |
| **DIRECT PARENT (at AUDITED_HEAD)** | `8684e77a60af1bf96e3ea1e9fa080ff49c2e76ba` |
| **HEAD reconciliation** | Two local commits on governance/capability-catalog (`8684e77a6`, `55db3f95f`) after `cfde2b8a`; delegated-provider paths unchanged since P2.1 closeout `6fbccd65813bb8eb2e2056f0e636758ef00592d0`. **Uncommitted** working-tree edits under `intergrax/runtime/execution/` were excluded from classification (not part of AUDITED_HEAD). |
| **Companion roadmap** | [`HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md`](../../overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md) |
| **Prior as-built** | [`HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md`](../plans/HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md) |

---

## 2. Phase 2 — P2.1 frozen baseline verification

**Verdict: P2.1 = CLOSED / ENTERPRISE QUALIFIED (unchanged on AUDITED_HEAD).**

| Check | Evidence |
| --- | --- |
| No delegated-plane regression since closeout | `git diff 6fbccd658..55db3f95f -- '**/delegated*'` → empty |
| Contracts | `intergrax/contracts/delegated_execution_*.py` |
| Qualified provider | `intergrax/integrations/providers/delegated_execution/subprocess/provider.py` (`subprocess_delegated_execution`) |
| Qualification record | Roadmap §P2.1 @ `6fbccd65813bb8eb2e2056f0e636758ef00592d0`; tests `tests/unit/runtime/execution/test_delegated_execution_subprocess_provider_s2d.py` + S2C suite |

Optional **integration** work (remote/ACP providers) remains product backlog, not an open plane defect.

---

## 3. Platform map (code-backed)

| Domain | Canonical implementation | Main contracts | Composition owner | Runtime owner | Status |
| --- | --- | --- | --- | --- | --- |
| Execution Engine | `intergrax/runtime/execution/runtime.py`, `boundary.py`, `strategy_router.py` | `intergrax/contracts/execution_identity.py`, governed execution contracts | `ApplicationEnvironmentProfile` → host wiring | `ExecutionRuntime` / `ExecutionBoundary` | **ENTERPRISE QUALIFIED** |
| Governance / Decision | `intergrax/runtime/governance/`, `runtime/policy/` | `DecisionRequirementPolicy`, `meaningful_side_effect_policy` | Profile + policy bundles | Runtime policy engine + decision requirement evaluator | **STRONG** |
| ToolRuntime / Tools | `RuntimeToolGateway`, `catalog_dispatch`, `ToolRuntime`, `ToolRegistry` | `ToolRequest`/`ToolResponse`, `ToolExecutionRequest` | Tool profile + skill manifests + EP `intergrax.tools` | Nexus tool gateway + catalog invoker | **PARTIAL** |
| Plugins / composition | `intergrax/core/plugins/discovery.py`, domain EP groups | `package_contract`, admission, qualification hooks | Profile / package install | Entry-point load at composition (cached registry) | **PARTIAL** |
| Plugin trust | `platform_qualification.py`, admission | Manifest IO, conflict policy | Control plane / operator | Load-time admission | **PARTIAL** |
| Providers (delegated) | `DelegatedExecutionService` + provider adapters | `DelegatedExecutionProvider` | Execution + durability policy | Child execution via boundary | **ENTERPRISE QUALIFIED** |
| Model runtime | `intergrax/llm_adapters/` | Adapter contracts, routing | Profile model bundles | Adapter registry | **STRONG** |
| Agents / UAEP | `intergrax/agents/uaep.py` | `RuntimeExecutionContext` | Application host | Execution strategy (agentic) | **STRONG** |
| Multi-agent / Multiplayer | `intergrax/collaborative_work/` | Collaborative decision binding, evidence projection | Multiplayer host wiring | Collaborative work runtime | **STRONG** |
| Memory | `intergrax/memory/`, runtime task memory | Memory governance contracts | Profile memory stores | Governed recall/mutation paths | **STRONG** |
| Context Engineering | `intergrax/context/`, `runtime/nexus/context/context_engine.py` | Context providers, assembly policy | CE provider EPs | Nexus context engine + orchestrator | **PARTIAL** |
| RAG / search | `intergrax/rag/`, websearch | Retriever/reranker EPs | Profile RAG bundle | Tool + provider plugins | **STRONG** |
| Sessions / conversations | Session history providers, task metadata | Context/session contracts | Host | Multiple projection paths | **PARTIAL** |
| Persistence / checkpoint | `long_running/`, `agents/persistence/checkpoint_store.py` | `TaskCheckpointPersistence`, tree checkpoint | Host wiring | Split task vs agent stores | **FRAGMENTED** |
| Artifacts | Runtime replay / artifact DTOs | Artifact contracts | Storage providers | Tool + runtime spill paths | **PARTIAL** |
| Sandbox | `intergrax/runtime/sandbox/` | `SandboxProfile`, isolation authority | Profile sandbox | Pre-tool enforcement (P1.8) | **PARTIAL** |
| Security / secrets | `integrations/contracts/secrets_store.py` | Secret references, late resolution | Profile security bridge | Provider injection | **STRONG** |
| Scheduling / background | `long_running/store.py`, `CapacityScheduler` | Scheduler ledger, scheduled resume | Host / harness | SQLite schedule store + capacity | **PARTIAL** |
| HITL / human interaction | Declarative HITL bridges, debug HITL | HITL pause contracts | Governance | Tool + checkpoint resume | **STRONG** |
| Observability | Events, tracing, functional evidence wiring | `RuntimeEvent`, evidence spine | Profile observability | `runtime/observability` | **STRONG** |
| Evidence / audit | `intergrax/runtime/evidence/` | Evidence posture, functional evidence | Execution + multiplayer projection | Collectors/recorders | **STRONG** |
| Diagnostics | `intergrax/runtime/diagnostics/` | Diagnostic read models | Diagnostics kernel | Projections over facts | **ENTERPRISE QUALIFIED** |
| Runtime inspection | `applications/_shared/runtime_inspection/` | `RuntimeInspectionProvider` | Application host | Composed read providers | **PARTIAL** |
| Runtime control | Execution cancel/pause via boundary + delegated control | Control ports on execution/delegation | Governance | Execution runtime | **STRONG** |
| API / SDK / CLI | `fastapi_core/`, `intergrax/cli/` | Host adapters | Applications | In-process / HTTP | **PARTIAL** |
| MCP / ACP | Application adapters, ACP checkpoint enricher | Host execution semantics | Tier-3 apps | Must route via host task port | **PARTIAL** |
| Capability catalog | Marketplace + catalog tools | Catalog governance narrowing | Profile capability graph | Search/ranking plugins | **STRONG** |
| Configuration / profiles | `ApplicationEnvironmentProfile` | Environment sub-profiles | Tier-3 sole authority | Effective composition | **STRONG** |
| Recovery / continuation | NPSC-5E recovery plane, continuation persistence | Continuation contracts | Execution | Checkpoint + resume planners | **STRONG** |
| Resilience | `runtime/resilience/`, dependency boundary | Retry/timeout at dependency seam | Policy | Bounded bulkheads | **PARTIAL** |
| Developer experience | CLI, scaffold, extension guides | — | Docs + tooling | Local dev profiles | **PARTIAL** |

---

## 4. Baseline A–Z scorecard (summary)

| Cap | Status | Evidence (anchor) | Enterprise quality | Main gap | Priority |
| --- | --- | --- | --- | --- | --- |
| A Canonical execution | ENTERPRISE QUALIFIED | `boundary.py`, EE freeze qual docs, bypass arch tests | Yes | Consumer adoption only | P3 |
| B ToolRuntime | PARTIAL | `tool_gateway.py`, `catalog_dispatch.py`, `uaep_tool_gateway.py` | No single spine | Sandbox/runtime-bound side paths; not all catalog calls share identical governance middleware stack | **P1** |
| C Plugin runtime | PARTIAL | `core/plugins/discovery.py` | EP load + admission; global EP cache | No unified dynamic mount/unmount lifecycle across domains | P1 |
| D Plugin trust | PARTIAL | `admission.py`, `platform_qualification.py` | Manifest-level | Publisher provenance / activation governance not enterprise-closed | P1 |
| E Session persistence | FRAGMENTED | Task vs agent checkpoint stores | Durable pieces exist | One session truth model across restart | P1 |
| F Context engineering | PARTIAL | `context_engine.py`, `context/orchestrator.py` | Provider pipeline | Compaction/token path convergence; restart reconstruction | P2 |
| G Sandbox | PARTIAL | `sandbox/enforcement.py`, resolver | Contract + P1.8 | Full isolation qualification + policy evidence | P2 |
| H Background execution | PARTIAL | `UnifiedTaskRunner` (harness-only doc), `host_task.py` | UER entry exists | Product background UX + durability convergence | P2 |
| I Scheduling | PARTIAL | `long_running/store.py` | Durable SQLite schedules | Misfire/timezone/idempotency enterprise semantics | P2 |
| J Observability | STRONG | OBS R1 certification, runtime events | Correlation spine | Universal adoption on all side-effect paths | P2 |
| K Evidence | STRONG | `runtime/evidence/`, MP evidence adoption | Typed collectors | Tamper/version attribution not uniform everywhere | P2 |
| L Governance | STRONG | Policy engine, decision requirement, side-effect gates | Fail-closed pockets | Not every operation class has versioned policy snapshot | P2 |
| M Memory | STRONG | Governance on recall/mutation (recent commits) | Tenant + governance tests | Long-horizon procedural memory productization | P3 |
| N Agent / multi-agent | STRONG | UAEP, delegation, collaborative work | Agent ≠ execution | Dynamic topology governance still TARGET | P2 |
| O HITL | STRONG | Declarative HITL tool bridge | No implicit approval | Durability of approval artifacts across hosts | P2 |
| P Artifacts | PARTIAL | Replay DTOs, tool spill | Storage abstraction | Versioning/retention convergence | P2 |
| Q Secrets | STRONG | `secrets_store` contract | Late resolution | Rotation/audit uniformity | P3 |
| R RAG / search | STRONG | Hybrid retrieval EPs, tenant filters | Provider abstraction | HITL on low-confidence not universal | P3 |
| S Model runtime | STRONG | `llm_adapters` | Routing + streaming | Private cloud qual matrix | P3 |
| T Runtime inspection | PARTIAL | `RuntimeInspectionService` | Provider composition | Not all domains expose canonical read models | P1 |
| U Runtime control | STRONG | Execution + delegated control | Typed outcomes | Host transport parity | P2 |
| V Host/API/SDK/MCP | PARTIAL | `HostTaskExecutionPort`, fastapi adapters | Common identity goal | Residual host-specific shortcuts (auth bypass dev) | P1 |
| W DX | PARTIAL | CLI, guides | Good docs | Testing harness for plugins at scale | P3 |
| X Capability catalog | STRONG | Marketplace + governance evaluator | Trust narrowing | Runtime resolution DX | P3 |
| Y Resilience | PARTIAL | Dependency execution boundary | Some bulkheads | Circuit breaking not platform-unified | P2 |
| Z Enterprise security | STRONG | Tenant scope, sandbox authority | Layer gates | Dev auth bypass paths documented — must stay non-prod | P2 |

**Baseline A–Z:** complete (evidence-based classification; not numeric scoring).

---

## 5. Pluginability matrix (selected domains)

| Domain | Contract | Custom impl | Runtime selection | Core patch needed | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Execution strategies | YES | YES | YES | NO | ENTERPRISE |
| Delegated providers | YES | YES | YES | NO | ENTERPRISE |
| Tools / handlers | YES | YES | YES (EP) | NO | STRONG |
| Tool invocation patterns | YES | YES | YES (EP) | NO | STRONG |
| Skills | YES | YES | YES (EP) | NO | STRONG |
| Integrations | YES | YES | YES (EP) | NO | STRONG |
| Context providers | YES | YES | YES (EP) | NO | STRONG |
| Memory stores | YES | YES | YES (EP) | NO | STRONG |
| Platform plugin packages | YES | PARTIAL | PARTIAL | NO | PARTIAL |
| Dynamic runtime plugin mount | PARTIAL | PARTIAL | NO | YES (lifecycle) | PARTIAL |
| Sandbox providers | YES | YES | Profile | NO | PARTIAL |
| Runtime inspection providers | YES | YES | Host compose | NO | PARTIAL |

---

## 6. Governance coverage matrix

| Operation | Governance boundary | Fail closed | Policy versioned | Evidence | Gap |
| --- | --- | ---: | ---: | ---: | --- |
| Execution admission | Execution boundary + authority | YES | PARTIAL | YES | Pin policy revision to execution |
| Tool invoke | Tool access + declarative HITL | YES | PARTIAL | YES | Sandbox path shorter than catalog path |
| Provider delegate | Delegated service + correlation | YES | YES | YES | — |
| Delegation control | Durable correlation + capability | YES | YES | YES | — |
| Memory recall | Governance scope (recent) | YES | PARTIAL | PARTIAL | Cross-host uniform snapshots |
| Memory mutation | Governance inheritance | YES | PARTIAL | PARTIAL | — |
| Side effect | Meaningful side-effect authorization | YES | PARTIAL | YES | ADOPT on all integrations |
| Background work | Host task port | PARTIAL | PARTIAL | PARTIAL | Harness runner documented non-prod |
| Schedule | Ledger + fence | PARTIAL | NO | PARTIAL | Enterprise schedule semantics |
| Artifact write | Tool/storage policy | PARTIAL | NO | PARTIAL | Retention governance |
| Plugin activation | Admission + qualification | YES | PARTIAL | PARTIAL | Trust provenance |
| Topology change | Governance TARGET | PARTIAL | NO | PARTIAL | Dynamic orchestration open |

---

## 7. Durability matrix

| State | Durable | Provider-neutral | Restart recoverable | Source of truth | Gap |
| --- | ---: | ---: | ---: | --- | --- |
| Execution identity | YES | YES | YES | Execution store / authority | — |
| Delegation correlation | YES | YES | YES | `DelegatedInvocationCorrelationStore` | — |
| Session / task checkpoint | YES | PARTIAL | YES | Task checkpoint persistence | Agent vs task split |
| Memory | YES | YES | PARTIAL | Store provider | Recall reconstruction |
| Background task | PARTIAL | PARTIAL | PARTIAL | Host wiring | Converge with UER |
| Schedule | YES | PARTIAL | PARTIAL | SQLite schedule store | Misfire recovery |
| Approval / HITL | PARTIAL | PARTIAL | PARTIAL | Checkpoint + grants | Cross-transport durability |
| Artifacts | PARTIAL | PARTIAL | PARTIAL | Storage backend | Version lineage |
| Provider correlation | YES | YES | YES | Delegated store | — |
| Agent topology | PARTIAL | PARTIAL | PARTIAL | Collaborative bindings | MP durability qualified on PG |

---

## 8. Recovery matrix (semantics)

| Scenario | Execution | Delegation | Session | Tools | Gap |
| --- | --- | --- | --- | --- | --- |
| Process crash | Typed recovery plane (NPSC-5E) | Durable correlation + reattach | Checkpoint resume | In-flight tool ambiguous | Document per-host |
| Provider crash | Boundary outcomes | Status/read + no auto-retry | — | Provider errors typed | ADOPT |
| Host crash | Host task re-entry | Reattach provider | Token resume | — | Session SSOT |
| Network loss | Dependency boundary | Transport vs provider split (S2D) | — | Timeouts | STRONG on delegation |
| Partial persistence | Fail-closed correlation | REQUIRED durability mode | Stale writer gates | — | Tests exist |
| Corrupt state | Checkpoint revision gates | Integrity errors → PLATFORM_FAILURE | — | — | QUALIFY more hosts |
| Lost process-local | No synthetic binding (S2C2) | — | Rebuild from durable | EP cache reload | Plugin cache is rebuildable |

---

## 9. Side-effect audit (summary)

| Path | Fresh auth before effect | Evidence |
| --- | --- | --- |
| Catalog tool | HITL bridge + invoker execute | `catalog_dispatch.invoke_catalog_tool_request` |
| Sandbox tool | Access policy only in `BoundToolGateway._invoke_sandbox` | **Gap:** weaker than full catalog policy stack |
| Provider delegate | Execution-governed dispatch | Delegated service |
| Memory mutation | Governance gates | Recent memory governance tests |
| External integration | Provider admission contracts | `external_operations/provider.py` doc |

---

## 10. Bypass audit

| Finding | Severity | Evidence |
| --- | --- | --- |
| `BoundToolGateway` sandbox branch bypasses `RuntimeToolGateway` middleware stack | **P1** | `uaep_tool_gateway.py` |
| `invoke_runtime_bound_tool` shortcut | **P2** | `runtime_bound_catalog.py` |
| `UnifiedTaskRunner` — harness/scheduling only (documented) | **P3** (controlled) | `unified_task_runner.py` |
| Local dev auth bypass in harness routes | **P0** if enabled in prod | `harness_auth.py`, `agent_platform_admin_routes.py` |
| Execution bypass arch gates report **0** supported bypass on inventory | — | `test_platform_execution_unification_u5_final_zero_bypass.py`, post-freeze audit |

No **CRITICAL** alternate legal execution engine found on gated inventory.

---

## 11. Duplication audit

| Concern | Locations | Convergence note |
| --- | --- | --- |
| Checkpoint / resume | Task checkpoint, agent checkpoint, execution tree, decision checkpoint | **CONVERGE** session SSOT |
| Tool dispatch | ToolRuntime planner vs catalog_dispatch vs gateway | **CONVERGE** enterprise pipeline |
| Context assembly | `intergrax/context` vs Nexus `context_engine` | **CONVERGE** canonical pipeline owner |
| Retry / timeout | Resilience boundary, adapter retries, tool timeouts | Document ownership per layer |
| Policy evaluation | Runtime policy engine vs token optimization policy bypass | Audit bypass reasons |
| Plugin discovery | Per-domain EP groups + caches | **CONVERGE** trust + lifecycle |

---

## 12. Canonical ownership audit

| Concern | Owner | Gap? |
| --- | --- | --- |
| Identity | `identity_authority` / Execution | NO |
| Execution lifecycle | `ExecutionRuntime` | NO |
| Provider physical lifecycle | Delegated providers + integrations | NO |
| Policy | Runtime policy + governance | PARTIAL versioning |
| Tool execution | **Split:** Gateway vs catalog vs sandbox | **YES — architectural** |
| Context | **Split:** CE orchestrator vs Nexus context engine | **YES** |
| Memory | Memory domain + governance | NO |
| Artifacts | Storage + replay | PARTIAL |
| Evidence | Evidence plane + functional evidence | NO |
| Sessions | Host + context providers | FRAGMENTED |
| Scheduling | `long_running` + capacity | PARTIAL |
| Background work | Host task execution port | ADOPT |

---

## 13. Contract quality (phase 4)

Platform **generally operates on contracts** at tier boundaries (execution, delegation, tools catalog, skills EPs). Notable leaks:

- `RuntimeExecutionContext.tool_gateway: Optional[Any]` — stable agent ABI uses `Any` for gateway/emitter (`runtime_execution_context.py`).
- Entry-point plugin discovery uses module-level `_EP_SPECS_CACHE` (`discovery.py`) — replaceable but global mutable cache (documented isolation policies).
- Tier-3 harness local auth bypass — configuration-gated, must remain non-production.

No widespread vendor SDK imports in `intergrax/runtime/execution/` core.

---

## 14. Test inventory (phase 37, sample)

| Domain | Unit | Integration / boundary | Restart / fault | Arch gates |
| --- | --- | --- | --- | --- |
| Execution | ~98 files under `tests/unit/runtime/execution` | platform proofs, composite e2e | continuation, stale writer, cross-process | EE bypass, identity authority |
| Tools | ~38 files `tests/unit/runtime/nexus/tools` | `test_plugin8_dual_mode_tool_e2e` | partial | tool policy tests |
| Governance | ~22 files | GR-6 architecture tests | — | authority policy gate |
| Evidence | ~30 files | OBS universal spine e2e | — | — |
| Delegation | subprocess S2D + S2C suites | TCP/process | REQUIRED durability | P2.1 qual |

**Gap:** Few cross-host ToolRuntime enterprise qualification tests spanning sandbox + catalog + governance middleware in one suite.

---

## 15. Production qualification (phase 15)

| Subsystem | Production-qualified? | Basis |
| --- | --- | --- |
| Execution Engine | YES | Freeze + exhaustive gap audit |
| P2.1 delegated plane | YES | Closeout @ `6fbccd658` |
| Diagnostics | YES | Enterprise diagnostic kernel |
| ToolRuntime spine | NO | Multiple paths; lacks single closure qual |
| Plugin dynamic runtime | NO | EP load ≠ enterprise plugin lifecycle |
| Runtime Invariant Service | NO | Roadmap GAP |
| Session persistence SSOT | NO | Fragmented stores |

---

## 16. Enterprise strengths — DO NOT REBUILD

| Subsystem | Action |
| --- | --- |
| Execution Engine (UER) | **FROZEN / PRESERVE** — extend via boundary only |
| P2.1 delegated provider plane | **FROZEN / PRESERVE** |
| Recovery plane (NPSC-5E) | **FROZEN / PRESERVE** |
| Decision capability (hosted by execution) | **PRESERVE / ADOPT** |
| Diagnostics kernel | **PRESERVE / EXTEND** |
| Evidence plane + multiplayer projection | **PRESERVE / EXTEND** |
| Capability catalog governance narrowing | **PRESERVE** |
| Skills EP + manifest model | **PRESERVE / EXTEND** |
| ApplicationEnvironmentProfile authority | **PRESERVE** |

---

## 17. Work classification lists

### CONVERGE

- Tool invocation spine (gateway + catalog + sandbox + runtime-bound).
- Checkpoint/session persistence (task vs agent vs execution tree).
- Context pipeline (CE orchestrator vs Nexus context engine).
- Scheduling vs background vs host task entry.
- Host/API/MCP/ACP error and identity semantics.

### BUILD

- Runtime Invariant Service (shared runner, domain-owned rules).
- Enterprise plugin trust/provenance activation governance (beyond admission).
- Dynamic reversible runtime registration (scoped; per roadmap P3).
- Unified session/runtime conversation durability SSOT (product-level).
- Runtime inspection read models for all baseline T domains.

### QUALIFY

- Sandbox end-to-end isolation proof suite.
- ToolRuntime single-pipeline enterprise closure.
- Schedule misfire/restart semantics.
- Cross-domain scenario proofs (roadmap initiative).

### ADOPT

- Meaningful side-effect authorization on all integration paths.
- HostTaskExecutionPort for all production Tier-3 entry paths.
- Governance policy revision pinning on execution bind.
- Functional evidence on remaining tool side-effect paths.

---

## 18. Gap severity

### P0 — architecture / security blockers

- Production enablement of harness **local dev auth bypass** (configuration discipline; fail-closed in prod profiles).

### P1 — top-tier harness gaps

- ToolRuntime not a single enterprise invocation pipeline (sandbox/runtime-bound divergence).
- Fragmented session/checkpoint source of truth.
- Runtime Invariant Service missing (roadmap N).
- Runtime inspection incomplete vs baseline T.
- Host transport convergence (residual private semantics).

### P2 — enterprise hardening

- Context compaction/retention convergence.
- Background/scheduling durability semantics.
- Plugin trust/provenance beyond manifests.
- Policy version snapshots not universal.
- Resilience pattern unification.

### P3 — product / DX

- SDK/examples for plugin installation.
- Instruction skills.
- Marketplace recommendation polish.

---

## 19. Dependency graph (roadmap ordering)

```text
Runtime Invariant Service (RI)
    ↓
ToolRuntime Enterprise Closure (TR)
    ↓
Governance Everywhere ADOPT (GV)
    ↓
Session / Checkpoint SSOT (SESSION)
    ↓
Background + Scheduling convergence (BG/SCHED)
    ↓
Context + Compaction (CE)
    ↓
Plugin Trust + Dynamic Registration (PLUG)
    ↓
Host/API/MCP convergence (HOST)
    ↓
Cross-Domain Scenario Proofs (PROOF)
```

TR and RI can start in parallel; TR benefits from invariant runner for regression gates.

---

## 20. Final ordered roadmap (10–22 major workstreams)

| Order | Task ID | Name | User-level goal | Gap type | Depends on | Exit criterion |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | RI-01 | Runtime Invariant Service foundation | One runner executes domain invariant rules with diagnostic correlation | BUILD | — | Shared runner + 3 domain rule packs gated in CI |
| 2 | TR-01 | ToolRuntime enterprise closure | Every agent tool call uses one governed pipeline | **CLOSED / ENTERPRISE QUALIFIED** | RI-01 (recommended) | TR-01-RQ-FINAL; bypass count 0 |
| 3 | GV-01 | Governance adoption sweep | Side effects and tools always pass fresh auth + evidence | ADOPT | TR-01 | Matrix §6 gaps closed on canonical paths |
| 4 | SESSION-01 | Session/checkpoint SSOT | Restart does not lose session truth | **CLOSED / ENTERPRISE QUALIFIED** | — | `tests/qualification/session_01/`; C1R `b10000a87` |
| 5 | INSPECT-01 | Runtime inspection expansion | Operators read execution/tool/policy/memory uniformly | BUILD | SESSION-01 partial | Baseline T read models for core domains |
| 6 | HOST-01 | Host/API/MCP/ACP convergence | No transport-specific execution semantics | CONVERGE | TR-01, GV-01 | Arch tests: common identity/errors |
| 7 | BG-01 | Background execution convergence | Background work is first-class UER citizen | CONVERGE | SESSION-01 | Host task port only on prod paths |
| 8 | SCHED-01 | Scheduling enterprise semantics | Durable schedules survive restart/misfire | QUALIFY | BG-01 | Misfire/idempotency tests |
| 9 | CE-01 | Context pipeline convergence | One canonical context assembly owner | CONVERGE | — | CE vs Nexus ownership doc + code seam |
| 10 | CE-02 | Compaction + retention | Long runs without evidence loss | PARTIAL | CE-01 | Compaction proofs |
| 11 | PLUG-01 | Plugin trust & provenance | Install plugins with governance + audit | BUILD | GV-01 | Activation policy + audit trail |
| 12 | PLUG-02 | Dynamic registration (scoped) | Reversible runtime plugin lifecycle | BUILD | PLUG-01 | Mount/unmount qual tests |
| 13 | SBX-01 | Sandbox qualification | Isolation claims evidenced | QUALIFY | TR-01 | Boundary suite P0/P1 |
| 14 | ART-01 | Artifacts & spill convergence | Large outputs durable with lineage | CONVERGE | SESSION-01 | Version + retention policy |
| 15 | OBS-02 | Observability universal adoption | All side-effect paths emit spine events | ADOPT | TR-01 | OBS scenario proof extended |
| 16 | MEM-01 | Memory long-horizon hardening | Enterprise retention/recall governance | QUALIFY | GV-01 | Cross-tenant fault tests |
| 17 | MA-01 | Multi-agent topology governance | Dynamic topology proposals gated | BUILD | GV-01, PLUG-01 | Governance proof for topology change |
| 18 | RES-01 | Resilience convergence | Platform retry/circuit semantics documented | CONVERGE | — | Ownership matrix + tests |
| 19 | DX-01 | Developer experience closure | Plugin/local test harness | P3 | PLUG-02 | Documented golden path |
| 20 | PROOF-01 | Cross-domain scenario proofs | Prove harness invariants in realistic flows | QUALIFY | TR-01, SESSION-01, HOST-01 | Scenario catalog in CI |

---

## 21. Next recommended task

**TR-01 — ToolRuntime enterprise closure** — **CLOSED / ENTERPRISE QUALIFIED** (TR-01-RQ-FINAL, qualification SHA `94c0abde805f3da244bd1eb3e9d5362e0ec2fdcc`).

**SESSION-01 — Session/checkpoint SSOT** — **CLOSED / ENTERPRISE QUALIFIED** (SESSION-01-C1R, TASK SHA `b10000a87`).

**Next:** **INSPECT-01 — Runtime inspection expansion** (baseline read models for core domains).

Run **RI-01** in parallel if staffing allows (feeds CI gates for TR-01).

---

## 22. Architectural deviations

- Documented harness-only paths (`UnifiedTaskRunner`) — acceptable if gated from production profiles.
- Uncommitted execution-tree edits at audit time — reconcile before next execution qualification.

---

## 23. Document validation

Classifications cite paths under `intergrax/`, `tests/`, and qualification markdown in `docs/project/maintainers/qualification/` and roadmap §P2.1. No classification relies solely on README claims.

---

*Audit complete for AUDITED_HEAD `55db3f95f2af61166cedfdb4667fbac97cb075a5`. Commit SHA for this document: see git history.*
