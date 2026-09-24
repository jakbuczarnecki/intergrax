# Platform Enterprise Completion Roadmap

**Document role:** canonical maintainer program tracker for reaching a fully recertified enterprise platform state before scenario-focused development.

**Branch:** `development`

**Scenario gate:** scenario work is **blocked** until all mandatory stages in this roadmap are independently closed, the architecture freeze is certified, and the final enterprise certification is complete.

This document is a **program/closure SSOT**. It does not replace domain semantic authorities. Domain architecture, contracts and qualification records remain authoritative for their own semantics.

Primary companion sources:

- [Harness Architecture Evolution Roadmap](../../overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md)
- [Harness Top-Tier Gap Audit](../qualification/HARNESS_TOP_TIER_GAP_AUDIT.md)
- [Platform Plugin Enterprise Roadmap](PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md)
- [Platform Enterprise Freeze Acceptance Checklist](../qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- domain architecture / qualification records referenced by each stage

**Document roles (do not conflate):**

| Artifact | Role |
|---|---|
| This roadmap | Ordering, program status, stage closure SSOT |
| [Freeze Acceptance Checklist](../qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) | Global acceptance completeness control (`FRZ-*`); not a second semantic authority |
| Domain architecture / qualification records | Semantic authority for mechanisms, contracts and in-domain evidence |

DeepSeek/external audit findings may motivate work, but repository code, canonical contracts, architecture documents and independently audited GitHub commits are the authority for closure.

**Whole-program goal:** reach a **formally freezable enterprise architecture** with evidence-backed certainty—before scenario work—on boundaries, ownership, contracts, typing, pluginability, Governance, Execution, Observability, Traceability, Persistence, Compatibility, Production qualification and regression protection. After `ARCH-FREEZE`, fundamental architecture is frozen; scenarios must not be used to discover known fundamental architecture gaps.

---

## 1. Non-negotiable enterprise invariants

Every stage in this roadmap must preserve and revalidate these rules where applicable:

1. **Contracts over implementations** — consumers depend on platform-defined contracts, not concrete implementations.
2. **Exactly one owner per concern** — one semantic owner, one canonical contract, one sanctioned composition/selection owner.
3. **Hard layer boundaries** — no reverse dependencies, no hidden cross-layer construction and no bypass of canonical control points.
4. **Pluginability / replaceability** — externally supplied strategies/providers/implementations must be attachable through defined contracts where the mechanism is extensible.
5. **Strong typing** — no weak semantic boundaries built from `Any`, generic `object`, reflection, string dispatch, dynamic attribute probing or dict pseudo-contracts when typed contracts exist.
6. **No duplicate mechanisms** — no parallel factories, validators, dispatchers, authorities, execution paths, compatibility branches or shadow owners for the same responsibility.
7. **Fail closed** — missing authority, configuration, evidence or required dependency must not silently become permission or fallback.
8. **Evidence-backed closure** — a task is not CLOSED because an implementation report says so; closure requires independent audit of the exact GitHub SHA.
9. **Execution authority remains singular** — all executable platform work enters through the approved Execution boundary.
10. **Governance is not Execution** — Governance decides whether work/effects are permitted; Execution owns how admitted work runs.
11. **Proposal != Permission != Execution.**
12. **Child authority cannot exceed parent authority** — `child_authority ⊆ parent_authority`.
13. **Downstream scopes may only narrow upstream authority**, never widen it.
14. **Meaningful side effects require fresh governed authorization immediately before the effect** where policy requires it.
15. **Required causal/audit evidence must exist before meaningful work begins** where the boundary requires such evidence.
16. **Absence of HITL/human interaction never implies approval.**
17. **Runtime extensions cannot self-expand authority.**
18. **Observability/Diagnostics record or interpret truth; they do not mint execution truth.**
19. **Configured state != effective state.**
20. **No scenario transition before the final scenario gate is explicitly CLOSED.**
21. **No architecture freeze with unresolved architecture state** — `Partial`, `Deferred`, `Planned`, temporary compatibility seams, transitional authorities or known architecture debt must be either closed or explicitly classified outside the frozen platform scope with evidence.
22. **Frozen contracts must have an evolution policy** — public/stable contracts, persisted schemas, events and plugin/provider contracts must have explicit versioning, compatibility, migration and deprecation rules before `ARCH-FREEZE`.

---

## 2. Mandatory update protocol

This file must be referenced in every implementation/audit instruction that belongs to this enterprise-completion program.

### 2.0 Global freeze-acceptance protocol

**Before every task** (implementation or audit instruction generation), read:

1. `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`
2. `docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`

Then resolve:

- current stage ID and status;
- parent stage (if any);
- next mandatory ordered stage;
- applicable `FRZ-*` criteria for the task scope;
- known blockers;
- relevant domain semantic authorities (not the checklist).

**After every Cursor implementation:** status may advance only to `READY FOR AUDIT`. Cursor must not finally close a stage or a freeze criterion. Closure requires an **independent exact GitHub SHA audit**.

**After every independent closure**, update **atomically** (same maintenance action, no partial closure):

1. roadmap stage/status in §3;
2. roadmap evidence ledger in §5;
3. freeze acceptance checklist evidence for every `FRZ-*` criterion closed or advanced by that stage.

Forbidden state: roadmap stage `CLOSED` while corresponding freeze evidence was not updated.

### 2.0.1 Stage → freeze criteria linkage

From the current workflow position forward, every **parent-level** roadmap stage must declare which `FRZ-*` families it verifies or for which it supplies closure evidence. The checklist [Freeze Criteria Coverage Matrix](../qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md#freeze-criteria-coverage-matrix) is the completeness detector; an `FRZ` family without a stage owner means the program is incomplete.

Minimum planned linkage:

| Stage group | Primary `FRZ` families |
|---|---|
| EBH-2* | BND, OWN, CTR, TYP, PLG |
| HARNESS-* | HRN, EXE, GOV, OBS |
| GOV-X1 / GOV-X2 | GOV, EXE, TRC |
| EBH-3 | BND, OWN, CTR |
| EBH-4 | BND, CTR, EXE, GOV |
| CTRL-X | CTL, SEC, REL, OBS |
| STATE-X | STA, REC |
| TRACE-X | TRC, OBS, GOV |
| COMPAT-X | CMP |
| PROD-Q | PRD, SEC, REL |
| QUAL-X | REG |
| EBH-5 | PLG, RPL |
| EBH-6 | all applicable architecture families |
| EBH-7 | all enterprise families |
| ARCH-FREEZE | FRZ, DEBT, DOC + all remaining |

### 2.1 Required instruction header

Every Cursor/implementation instruction must state:

- this file path: `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`;
- current roadmap stage ID;
- parent stage, if any;
- whole-program goal;
- statement that the task must not expand beyond its scope or cross a layer boundary without an architecture decision.

### 2.2 Status lifecycle

Allowed statuses:

- `[x] CLOSED` — independently audited on exact GitHub SHA;
- `[ ] CURRENT` — active task;
- `[ ] READY FOR AUDIT` — implementation finished, independent GitHub audit still required;
- `[ ] BLOCKED` — waiting for child task / architecture decision / dependency;
- `[ ] QUEUED` — next ordered work;
- `[ ] PLANNED` — later mandatory work;
- `[ ] FINAL / MANDATORY` — terminal certification gate.

### 2.3 Who may close a step

Cursor/implementation agents **must not** declare a roadmap step finally CLOSED on their own authority.

They may report `READY FOR AUDIT` and propose the new status. Final closure requires:

1. exact implementation commit SHA;
2. independent code audit on GitHub at that SHA;
3. tests/evidence appropriate to the stage;
4. no unresolved blocker in the stage scope;
5. update of this roadmap with closure evidence.

### 2.4 Evidence rule

Every CLOSED row should eventually have an evidence anchor in §5:

- exact SHA,
- qualification/audit document when applicable,
- important child task IDs.

### 2.5 New blockers

If a new blocker is discovered:

- create a child task under the affected stage;
- parent remains BLOCKED / NOT CLOSED when the blocker breaks the parent invariant;
- do not skip forward;
- do not hide the finding as “out of scope” if it breaks the parent invariant;
- if a finding is proven not to break the current parent but is still a real enterprise/qualification debt, record it as an explicit mandatory roadmap item before the relevant final certification wave.

---

## 3. Canonical ordered roadmap

| Etap | Co robimy | Opis zadania | Status |
|---|---|---|---|
| EBH-2E | Final LLM Boundary Certification | Finalna certyfikacja całego LLM boundary: provider/profile/registry/routing/failover/usage/composition jako jeden spójny subsystem z canonical contracts, jednym ownerem concernów i realną replaceability. | [x] CLOSED |
| ADR3-IMP-03-R1 | Typed Declarative Dispatch & Contract Deduplication | Usunięcie realnych semantic duplicates w declarative dispatch i pozostawienie canonical typed contracts oraz jednego resolver/dispatch ownera. | [x] CLOSED |
| ADR3-IMP-03 / M4 | Final Typed Declarative Dispatch Certification | Certyfikacja M4 jako jednego spójnego typed flow: metadata/composition → identity → dispatch → catalog gateway → canonical result → batch aggregation. | [x] CLOSED |
| ADR3-IMP-04-R1 | Catalog Host Capability Contract Deduplication | Usunięcie drugiej definicji per-call `invoke(...)` z host-specific Protocolu. `ExecutionBoundDeclarativeToolInvoker` pozostaje jedynym ownerem invocation semantics, a host Protocol dodaje wyłącznie host capability. | [x] CLOSED |
| ADR3-IMP-04 | Consumer Migration & Final Bind Removal — final recertification | Finalny audit migracji konsumentów: zero narrow legacy Protocolu, zero `inspect.signature` compatibility dispatch, metadata na canonical contract, explicit identity, brak concrete coupling i exactly-one owner per-call invocation contractu. | [x] CLOSED |
| ADR3-IMP-05 | Qualification Gates | Mechaniczne architecture gates blokujące powrót narrow compatibility paths, reflection dispatch, duplicated ownership, concrete coupling, weak typing i alternate execution paths. | [x] CLOSED |
| HARNESS-01-R5-W3-R1-Q2 | Finalne zamknięcie Harness W3 | Finalna kwalifikacja Tools/WebSearch Nexus Dependency Inversion po zamknięciu ADR3: zero statycznego/dynamicznego/lazy Tools/WebSearch → Nexus resolution, typed provider-neutral seams, structural pluginability, poprawny ownership i brak bypassów. | [x] CLOSED |
| **EBH-2F** | Integrations & Hosting Boundary Hardening | Parent integrations/hosting pozostaje otwarty: niezależny audit (`8c89bdfa046bbf124ed838b64834bdc825bb8f57`) wykrył słabo typowany i dynamiczny publiczny integration-plugin boundary (`Any`, `Callable[..., Any]`, reflection `CONTRACT_SPECS`). | **[ ] BLOCKED** |
| **EBH-2F-R2** | Integration Plugin Contract Typing & Registration Purity | Plugin/reflection blocker naprawiony; independent audit (`7611cc8b352c29af38c964e70f675967d2e959cd`) wykrył weak typing w canonical `resolve` / `resolve_from_profile` i `IntegrationContractSpec`. Parent pozostaje BLOCKED do domknięcia EBH-2F-R2-R1. | **[ ] BLOCKED** |
| **EBH-2F-R2-R1** | Integration Resolution & Contract Spec Typing Closure | Strong typing całego flow contract spec → catalog factory → registration → resolution → `PlatformIntegrationContract`; independent audit (`8570c8440ddf75d1256f58ecb5182d9a1a441ec5`) wykrył słabszy pre-built binding path (`instance_for_category` akceptował generic `PlatformIntegrationContract` / arbitrary objects). Parent BLOCKED do domknięcia EBH-2F-R2-R1-R1. | **[ ] BLOCKED** |
| **EBH-2F-R2-R1-R1** | Pre-built Integration Category Contract Integrity | Pre-built dependency injection musi walidować dokładnie category-specific contract przez canonical `contract_for_category`; independent audit (`3c6f84b5ca8c5478ec3908f17fabfaf467567146`) wykrył brak DI-only `external_work` w resolverze. Parent BLOCKED do domknięcia EBH-2F-R2-R1-R1-R1. | **[ ] BLOCKED** |
| **EBH-2F-R2-R1-R1-R1** | DI-only Category Contract Resolution Integrity | Jeden canonical category-contract resolver obsługuje registry-backed i DI-only (`external_work` → `ExternalWorkIntegration`) bez local exceptions w `IntegrationProfile`; independent audit (`fd7a12f773406d1c7babb79388aba30526a17aa6`) wykrył false result typing na category-resolution boundary. Parent BLOCKED do domknięcia EBH-2F-R2-R1-R1-R1-R1. | **[ ] BLOCKED** |
| **EBH-2F-R2-R1-R1-R1-R1** | Category Integration Result Contract Typing Integrity | Publiczne category-resolution API deklaruje dokładnie legalny runtime contract (`CategoryIntegrationInstance`) dla registry-backed i DI-only bez `Any`/`object`/castów. | **[ ] CURRENT** |
| EBH-2F-R1 | Host Execution Boundary Decoupling | Finalna recertyfikacja potwierdziła migrację production Tier-3 na canonical `HostTaskExecutionPort` / `runtime.execution`, brak application/shared concrete Nexus execution coupling, exactly-one host-execution materialization owner, strong typing, brak dual production path oraz zachowanie Execution/Governance semantics. | [x] CLOSED |
| EBH-2F-R1-R1 | Harness Host Execution Wiring Contract Purity & Runtime Ownership | Concrete `NexusLoop` usunięty z `intergrax/applications/_shared/harness_host_task_execution_wiring.py`; harness seam jest governance-only, Nexus-backed helpers istnieją tylko w test fixtures, a revision admission ponownie używa canonical `EffectiveProfileRevisionAdmissionPort`. | [x] CLOSED |
| EBH-2G | RAG Contract Boundary Hardening | Certyfikacja retrieval/search/reranking/storage jako jednego pluginowalnego subsystemu z canonical contracts, jednym ownership i wymiennymi backendami. | [ ] PLANNED |
| EBH-2H | Memory Contract Boundary Hardening | Certyfikacja Memory: exactly-one ownership, canonical contracts, backend replaceability, brak bocznych persistence/context paths. | [ ] PLANNED |
| EBH-2I | Final EBH-2 Rescan | Ponowny przekrojowy audit wszystkich subsystem boundaries po lokalnych hardeningach. Sprawdzenie, czy poprawki nie stworzyły nowych cross-layer zależności, bypassów lub duplicated ownership. | [ ] PLANNED |
| HARNESS-QINF-01 | Harness Global Qualification Inventory Reconciliation | Zsynchronizować globalne `test_harness_01_gates.py` allowlist/inventory z nowszymi, osobno kwalifikowanymi EE/UCA/suspended-operation surfaces. Finding pochodzi z W3 recertification: 3 globalne gate failures nie naruszają W3, ale qualification inventory nie może pozostać czerwone ani utracić coverage przed kolejnymi Harness waves/final closure. Bez zmiany semantics produkcyjnych, chyba że reconciliation ujawni realny bypass. | [ ] PLANNED |
| HARNESS-W4 | Harness W4 — Scale / Resilience / Cancellation recertification | Reconcile and close the W4 debt wave against current repository reality and existing W4 qualification artifacts. Verify cancellation, external-operation termination, provider cancellation boundaries and no alternate execution authority. Exact subwave scope must be revalidated from canonical Harness records before implementation. | [ ] PLANNED |
| HARNESS-W5 | Harness W5 — Events / Observability delivery and export recertification | Reconcile and close W5 using existing event delivery/export/OTLP qualification records. Verify one event/evidence spine, lifecycle/composition ownership and no observability-created execution truth. | [ ] PLANNED |
| HARNESS-W6 | Harness W6 — Runtime Intelligence recertification | Reconcile W6-A…W6-E against current HEAD: typed contracts, deterministic analysis, orchestration, execution advisory boundary. Runtime intelligence remains advisory/non-authoritative and must not become a second execution/governance owner. | [ ] PLANNED |
| GOV-X1 | Governance Authority Boundary Recertification | Cross-cutting governance audit before global dependency/communication certification. Revalidate Governance ≠ Execution, proposal/permission/execution separation, authority narrowing, fresh side-effect authorization, HITL semantics, evidence-before-work and absence of governance bypasses. | [ ] PLANNED |
| EBH-3 | Dependency & Ownership Certification | Formalny audit globalnego dependency graphu wszystkich warstw i exactly-one ownership każdej odpowiedzialności. Sprawdzenie directionality, contract purity, composition owners i reverse dependencies. | [ ] PLANNED |
| EBH-4 | Communication, Composition & Bypass Certification | Audit wszystkich cross-layer communication/composition paths: event/call flows, resolvers, factories, host wiring, metadata bridges, provider seams i wszystkie sanctioned/bypass paths. | [ ] PLANNED |
| HARNESS-W7 | Harness W7 — remaining top-tier harness debt wave | Close the next Harness debt wave after EBH-3/4. Exact semantic scope and child IDs must be reconciled from `HARNESS_TOP_TIER_GAP_AUDIT.md`, `HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md` and current qualification records before implementation; do not invent a parallel authority. | [ ] PLANNED |
| HARNESS-W8 | Harness W8 / final residual harness convergence | Final residual Harness convergence wave if still open on current HEAD. Scope must be code-first and derived from canonical Harness qualification records; if already qualified, perform recertification rather than rebuild. | [ ] PLANNED |
| HARNESS-FINAL | Current-HEAD Top-Tier Harness Final Certification | **Mandatory current-HEAD recertification** (not historical closure alone). Minimum deliverables: full **A–Z Top-Tier scorecard**, **INV-1..INV-34** re-audit on exact SHA, **pluginability matrix**, **governance coverage matrix**, **durability matrix**, **recovery matrix**. Every historical/current `PARTIAL`, `GAP`, `TARGET`, or `DEFERRED` item in **frozen platform scope** must end as `CLOSED` / `ENTERPRISE QUALIFIED` or `OUTSIDE FROZEN PLATFORM SCOPE` with explicit justification and evidence—unresolved gaps may not be hidden as backlog. Historical DeepSeek/external findings must additionally be CLOSED, superseded with evidence, or explicitly non-blocking with evidence. | [ ] PLANNED / MANDATORY |
| GOV-X2 | Governance + Execution end-to-end certification | End-to-end proof that authority/approval/governance decisions propagate correctly through canonical execution and tool/effect paths without self-expansion, stale approval reuse, missing evidence or alternate execution routes. | [ ] PLANNED |
| CTRL-X | Enterprise Control-Plane Recertification | Reconcile and recertify all cross-cutting control planes on current HEAD: Security, Reliability, Cost/Budget, Evaluation, Critic/Verification, Observability/Diagnostics, Tools/Skills, Agent Distribution/Registry/Assembly, Capability Graph and Context/Prompt. Historical CLOSED is evidence, not automatic current certification. Verify mutual boundaries, exactly-one ownership and that advisory/recording planes do not become peer execution/governance authorities. | [ ] PLANNED / MANDATORY |
| STATE-X | Persistence, State & Recovery Certification | Global certification of durable and runtime state: checkpoints, continuation, evidence/trace, budgets, lineage, idempotency, policy artifacts, registries/projections and task/run state. Verify exactly-one truth owner, transactional/atomic boundaries, tenant isolation, crash/restart/resume/replay/fork consistency, stale-state rejection and no duplicate stores representing the same semantic truth. | [ ] PLANNED / MANDATORY |
| **TRACE-X** | End-to-End Traceability & Evidence Certification | Prove end-to-end **causal traceability** forward: transport identity → runtime identity → execution → task → child execution → strategy → agent → model/context decision → tool call → governance decision → side-effect authorization → provider invocation → external effect → runtime evidence/events → diagnostics → terminal outcome; and **reverse reconstruction**: effect/failure/diagnostic → execution → authority → policy/profile revision → provider → contract/version → causal parent. Mandatory coverage includes `ExecutionId`, `RunId`, `TaskId`, parent/child causality, provider/delegation/tool invocation correlation, model/context attribution, profile/policy revision attribution, side-effect authorization evidence, restart/resume continuity, terminal outcome evidence, diagnostic provenance, evidence version attribution, configured vs effective provenance. | [ ] PLANNED / MANDATORY |
| COMPAT-X | Contract, Schema & Evolution Certification | Identify frozen public/stable vs internal contracts and certify versioning/evolution rules for APIs, events, persisted schemas, plugin/provider contracts and serialization. Verify backward/forward compatibility policy, migrations, deprecation/removal rules and no compatibility shim becoming a permanent parallel authority. | [ ] PLANNED / MANDATORY |
| PROD-Q | Platform Production Qualification | Prove production readiness rather than harness/lab maturity: provider/plugin admission and qualification, startup/shutdown/resource lifecycle, strict-vs-lab mode separation, unsupported configuration handling, production bypass prevention, secrets/tenant isolation, degraded operation and fail-closed materialization. Historical `implementation complete` or harness qualification is not sufficient. | [ ] PLANNED / MANDATORY |
| **QUAL-X** | Enterprise Qualification & Regression Infrastructure Certification | Certify the **mechanical protection system** for frozen architecture—not only platform behavior. Verify: every frozen invariant has qualification evidence; corrected blockers have regression gates; architecture gates scan current closed-world surface; allowlists minimal and evidence-backed; stale inventories = 0; negative tests detect violations; critical invariants not docs-only; qualification tests deterministic; clean-checkout reproducibility; environment failure cannot auto-classify as architecture PASS; flaky tests cannot be freeze evidence; tests cannot assert names/other tests instead of invariants; qualification records match current code; mandatory architecture suite runnable as one defined freeze-oriented qualification set. | [ ] PLANNED / MANDATORY |
| EBH-5 | Replaceability & E2E Certification | Practical E2E proof that key providers, strategies and implementations can be replaced through platform contracts without modifying core mechanisms; verify real pluginability rather than test-only monkeypatching. | [ ] PLANNED |
| EBH-6 | Final Architecture Recertification | Full cross-platform recertification after all local, Harness, Governance, control-plane, state, compatibility and production-qualification work: boundaries, ownership, communication, composition, evidence, fail-closed behavior, typing and regression protection. | [ ] PLANNED |
| **EBH-7** | Comprehensive Platform Enterprise Architecture Certification | Ostateczna certyfikacja całej Integrax jako jednej platformy enterprise: hard boundaries, exactly-one ownership, canonical contracts, pluginability/replaceability, zero bypassów, zero duplicated mechanisms, correct Governance/Execution separation and validated E2E behavior. | **[ ] FINAL / MANDATORY** |
| **ARCH-FREEZE** | Architecture Freeze Certification | Formalny freeze gate po EBH-7. Governed by [Platform Enterprise Freeze Acceptance Checklist](../qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md). **Mechanical entry requirements (all must hold; `N/A` only with evidence):** mandatory roadmap `OPEN` = 0; mandatory roadmap `BLOCKED` = 0; freeze checklist `OPEN` = 0; freeze checklist `BLOCKED` = 0; unresolved architecture debt inside frozen scope = 0; unresolved Harness invariant = 0; unresolved Top-Tier frozen-scope gap = 0; failing mandatory architecture gate = 0; docs/code semantic discrepancy = 0; unversioned frozen public contract = 0; unclassified compatibility seam = 0. Additionally: canonical contract/layer/ownership/composition manifests frozen; mandatory qualification suite green; non-blocking debt register frozen; post-freeze change policy (ADR + architecture review + freeze exception + targeted recertification). | **[ ] FINAL / MANDATORY** |
| **SCENARIO-GATE** | Enterprise → Scenario transition gate | Formalny Go/No-Go do przejścia z hardeningu platformy do pełnej koncentracji na scenariuszach. **Remains BLOCKED while `ARCH-FREEZE` ≠ CLOSED.** Scenarios must not compensate for unfrozen architecture gaps. Gate CLOSED only when all mandatory rows above—including `ARCH-FREEZE`—are CLOSED with independent SHA evidence and no known enterprise blocker. | **[ ] BLOCKED** |

---

## 4. Mandatory cross-cutting audit matrix

Every parent-level closure from this point forward must explicitly assess the applicable cells below.

| Area | Required proof |
|---|---|
| Layer boundaries | No illegal imports, construction or ownership crossing layer boundaries. |
| Communication | All cross-layer calls/events/metadata bridges use sanctioned contracts and documented owners. |
| Composition | Concrete implementation selection exists only in sanctioned composition roots. |
| Ownership | Exactly one semantic owner and one canonical contract per concern. |
| Pluginability | External strategy/provider/implementation can replace defaults through platform-defined contracts where extensibility is claimed. |
| Strong typing | No weak/dynamic semantic boundary, reflection dispatch or dict/string pseudo-contract replacing an existing typed contract. |
| Bypass resistance | No direct consumer → concrete implementation path and no alternate runtime/execution/tool/governance path. |
| Governance | Authority only narrows; proposal/permission/execution remain distinct; fresh authorization exists at required side-effect seam. |
| Execution | Only canonical Execution authority admits meaningful executable work; no subsystem mints peer execution authority. |
| Evidence | Required causal/audit evidence precedes meaningful work and is attributable/reconstructable to the promised level. |
| Fail-closed | Missing required policy, authority, dependency, configuration or evidence cannot silently succeed/fallback. |
| Replaceability | Demonstrated through structural/custom implementation proof, not only monkeypatching. |
| Regression | Architecture gates + targeted functional tests protect every corrected invariant. |
| Persistence / state | Exactly one semantic truth owner per state family; recovery/replay/resume cannot create divergent truth. |
| Compatibility / evolution | Frozen contracts/schemas/events/plugins have explicit versioning, migration, deprecation and compatibility policy. |
| Production qualification | Production mode is independently qualified; lab/harness success is not accepted as production proof. |
| Freeze readiness | No unresolved architecture state remains inside the declared frozen platform scope. |

---

## 5. Closure evidence ledger

Update this section only after independent exact-SHA audit.

| Etap | Closure SHA / evidence | Notes |
|---|---|---|
| EBH-2E | `6fa75172468461dc084a2221a5fdff6f23c1fa1c` | Final LLM boundary certification accepted after independent GitHub audit. |
| ADR3-IMP-03-R1 | `255590897603f6e59ffebf4c1e4e7fde3ba747d1` | Typed declarative dispatch contract deduplication independently audited. |
| ADR3-IMP-03 / M4 | `702cb0a2024a7a98ab0bb88c1f645efeb0e7eb38` | M4 final typed declarative dispatch certification independently audited. |
| ADR3-IMP-04-R1 | `109dfd81e3a2af18affd76b0e5ac69d805783b31` | Catalog host capability contract deduplication independently audited; host Protocol now extends canonical execution-bound contract and owns only host-specific capability. |
| ADR3-IMP-04 | `b466e0a202e7984fd50f2409493159566cf03a82` | Final parent recertification independently audited on committed HEAD; no code changes required. Canonical contract, explicit identity, sanctioned composition, structural replaceability and zero legacy/reflection compatibility confirmed. |
| ADR3-IMP-05 | `b67281105f17cd2fe95d93e737470c28a043de60` | Enterprise qualification gates independently audited on exact GitHub SHA; mechanical protection confirmed for canonical ownership, no narrow/reflection regressions, metadata resolver ownership/typing, sanctioned concrete construction/gateway imports, explicit identity, UCA separation, structural replaceability, fail-closed governance and execution bypass protection. |
| HARNESS-01-R5-W3-R1-Q2 | `b67281105f17cd2fe95d93e737470c28a043de60` | Final W3/Q2 recertification independently audited on exact committed HEAD; no code changes required. Tools/WebSearch → Nexus static/dynamic/lazy resolution = 0; typed provider-neutral seams, structural pluginability, ownership, W2 regression and ADR3 documentation gates confirmed. Three global `test_harness_01_gates.py` inventory/allowlist failures were independently classified as non-W3 qualification-inventory debt and are tracked as mandatory `HARNESS-QINF-01`. |
| ENTERPRISE-FREEZE-CONTROL-01 | `324ad07d60211cb9c72bcff668e988f180e6ff5b` | Independently audited establishment of the enterprise-freeze control system: roadmap/checklist role separation, 22 `FRZ-*` families / 168 initial OPEN criteria, stage→FRZ coverage, `TRACE-X`, `QUAL-X`, strengthened `HARNESS-FINAL`, mechanical `ARCH-FREEZE`, and exact-SHA closure protocol accepted. |
| EBH-2F | `8c89bdfa046bbf124ed838b64834bdc825bb8f57` | Independent audit rejected parent closure on audited HEAD: public `IntegrationPlugin.create_integration(**kwargs: Any) -> Any`, catalog `IntegrationFactory = Callable[..., Any]`, and `getattr(plugin, "CONTRACT_SPECS", None)` on canonical registration. EBH-2F remains BLOCKED pending EBH-2F-R2. |
| EBH-2F-R2 | `7611cc8b352c29af38c964e70f675967d2e959cd` | Independent audit rejected R2 closure: `resolve` / `resolve_from_profile` → `Any`, `IntegrationContractFactory = Callable[..., Any]`, semantic `Any` on `IntegrationContractSpec`. BLOCKED pending EBH-2F-R2-R1. |
| EBH-2F-R2-R1 | `8570c8440ddf75d1256f58ecb5182d9a1a441ec5` | Independent audit rejected R2-R1 pre-built closure: `instance_for_category` validated only `PlatformIntegrationContract`, accepting wrong-category/base-only/plain-object injections. BLOCKED pending EBH-2F-R2-R1-R1. |
| EBH-2F-R2-R1-R1 | `3c6f84b5ca8c5478ec3908f17fabfaf467567146` | Independent audit rejected R2-R1-R1 closure: `contract_for_category` registry-only path broke DI-only `external_work` pre-built binding. BLOCKED pending EBH-2F-R2-R1-R1-R1. |
| EBH-2F-R2-R1-R1-R1 | `fd7a12f773406d1c7babb79388aba30526a17aa6` | Independent audit rejected R2-R1-R1-R1 closure: unified `contract_for_category` accepted, but `instance_for_category` / `resolve_from_profile` still declared `PlatformIntegrationContract` while DI-only `external_work` returns `ExternalWorkIntegration`. BLOCKED pending EBH-2F-R2-R1-R1-R1-R1. |
| EBH-2F-R2-R1-R1-R1-R1 | — | CURRENT — truthful category integration result typing (`CategoryIntegrationInstance`). |
| EBH-2F-R1 | `324ad07d60211cb9c72bcff668e988f180e6ff5b` | Final R1 recertification independently accepted on current-HEAD code. Tier-3 LKW/GCA consume `runtime.execution`; shared harness execution wiring is governance-only and contains no `NexusLoop`; `EffectiveProfileRevisionAdmissionPort` is restored; Nexus-backed materialization remains sanctioned runtime/composition ownership; structural custom `HostTaskExecutionPort` replaceability and architecture gates protect the corrected boundary. Relevant R1 surfaces were unchanged by later unrelated UCA/codecraft commits between Cursor-audited `d5f18d1...` and this exact audit SHA. The persistent GCA SQLite bootstrap failure is causally outside R1 and is tracked in the freeze checklist as `R1-SQLITE-ENV-01` for `STATE-X` / `PROD-Q` / `QUAL-X`. |
| EBH-2F-R1-R1 | `c3d5e377db989f4d7c3112f65fce552b88ee360a` | Independently audited on exact GitHub SHA. `harness_host_task_execution_wiring.py` is governance-only and contains no `NexusLoop`; canonical `EffectiveProfileRevisionAdmissionPort` is restored; Nexus-backed convenience builders moved to `tests/fixtures`; production Tier-3 continues to consume `runtime.execution`; architecture gates protect raw Nexus leakage and weak revision-admission typing. |
| HARNESS-QINF-01 | — | PLANNED — reconcile global Harness qualification inventory before HARNESS-W4/final Harness closure. |
| CTRL-X | — | PLANNED / MANDATORY — current-HEAD recertification of all cross-cutting control planes before freeze. |
| STATE-X | — | PLANNED / MANDATORY — global persistence/state/recovery certification before freeze; includes resolution/classification of `R1-SQLITE-ENV-01` where persistence semantics apply. |
| TRACE-X | — | PLANNED / MANDATORY — end-to-end traceability and evidence certification (`STATE-X` → `TRACE-X` → `COMPAT-X`). |
| COMPAT-X | — | PLANNED / MANDATORY — contract/schema/event/plugin evolution certification before freeze. |
| PROD-Q | — | PLANNED / MANDATORY — explicit production qualification before final enterprise certification; includes ensuring the SQLite bootstrap finding cannot mask production startup correctness. |
| QUAL-X | — | PLANNED / MANDATORY — qualification/regression infrastructure certification (`PROD-Q` → `QUAL-X` → `EBH-5`); `R1-SQLITE-ENV-01` is an explicit evidence item for `FRZ-REG-08` and cannot be treated as a false PASS. |
| ARCH-FREEZE | — | FINAL / MANDATORY — formal architecture freeze gate; requires checklist complete per mechanical entry requirements in §3. |
| remaining mandatory stages | — | Fill on closure. |

---

## 6. Required wording for future implementation instructions

Every task instruction in this program must include an equivalent of:

> `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` is the canonical enterprise-completion tracker. Work only on the current stage and its explicit children. Do not skip or silently reorder mandatory stages. Cursor may report READY FOR AUDIT but final CLOSED status is assigned only after independent audit of the exact GitHub commit. Any new enterprise blocker creates a child task and keeps the parent open. After independent closure, update the roadmap status and evidence ledger before starting the next mandatory stage.

Every final Cursor report with code changes must also state:

> **Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

---

## 7. Program goal

Bring the whole Integrax platform to a fully, independently recertified and formally frozen enterprise architecture state before scenario-focused development:

- hard and non-negotiable layer boundaries;
- exactly one owner per mechanism/contract/decision;
- contracts over implementations;
- full modularity, pluginability and replaceability;
- strongly typed semantic boundaries;
- one sanctioned execution/governance/tool/composition path per responsibility;
- zero hidden bypasses;
- zero duplicated mechanisms;
- complete cross-layer communication certification;
- Governance and Execution authority separation proven end-to-end;
- all cross-cutting control planes recertified on current HEAD;
- persistence/state/recovery semantics globally certified;
- end-to-end traceability and evidence certification (`TRACE-X`);
- contract/schema/plugin evolution rules frozen and explicit;
- production qualification proven independently from harness/lab maturity;
- qualification/regression infrastructure certified (`QUAL-X`);
- final enterprise closure through `EBH-7`;
- formal `ARCH-FREEZE` (checklist-complete) before `SCENARIO-GATE`;
- after freeze, any change to a frozen contract/boundary/ownership rule requires ADR, architecture review, explicit freeze exception and targeted recertification.
