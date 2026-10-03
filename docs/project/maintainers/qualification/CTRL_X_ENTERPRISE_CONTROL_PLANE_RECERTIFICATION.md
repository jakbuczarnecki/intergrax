# CTRL-X — Enterprise Control-Plane Recertification

**Parent recommendation:** `CTRL-X = READY FOR AUDIT` (pending independent GitHub SHA audit)
**Never:** `CTRL-X = CLOSED` (independent audit only)  
**STATE-X:** NEXT / NOT ENTERED  
**TENANT-X:** NOT ENTERED — no global `FRZ-TEN-*` promotion  

**Roadmap:** [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — GOV-X2 = CLOSED; CTRL-X = CURRENT (this artifact enters qualification evidence)  
**Freeze checklist:** [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — `FRZ-CTL-01..12` remain OPEN until independent audit  

Mechanical SSOT: `tests/qualification/control_planes/ctrl_x/catalog.py` (`CTRL_X_PLANE_CATALOG`, `CX-01..CX-12`).

---

## 1 Repository / exact HEAD

| Field | Value |
|---|---|
| branch | `development` |
| START_HEAD (R3-R2) | `ebd5ac5653ee054d28cc32ed0302eb361a4ef56a` |
| R3 implementation | `98a586540` — typed CTRL-X hook contracts |
| R3-R1 remediation | `ebd5ac5653ee054d28cc32ed0302eb361a4ef56a` |
| START_HEAD (R3-R2-Q1) | `6bcf1d615170da4b8eef91ca95554a21994dd130` |
| R3-R2 remediation | see §44 `FINAL_COMMIT` (tenant scope; checkpoint contamination removed in Q1) |
| R3-R2-Q1 remediation | see §44 `FINAL_COMMIT` |
| origin/development | aligned with Q1 `FINAL_COMMIT` after push |

Post-change `FINAL_COMMIT` recorded in §44 after commit.

---

## 2 Scope and exclusions

**In scope:** current-HEAD recertification of twelve cross-cutting control planes (`FRZ-CTL-01..12`); supporting evidence for `FRZ-SEC-*`, `FRZ-REL-*`, `FRZ-OBS-*`, tenant-relevant `FRZ-TEN-*` without promotion.

**Explicitly out of scope (later mandatory closers):** STATE-X, TRACE-X, CONFIG-X, COMPAT-X, TENANT-X (global), PROD-Q, QUAL-X, EBH-5, EBH-6 full platform typing closure.

**Not claimed:** control-plane qualified ≡ production qualified ≡ global trace complete ≡ global tenant isolation complete.

---

## 3 Canonical sources

- Roadmap + freeze checklist (above)
- Architecture: `TOOLS.md`, `SKILLS.md`, `AGENT_CONTRACTS_AND_ASSEMBLY.md`, `AGENT_DISTRIBUTION.md`, `CONTEXT_ENGINEERING.md`, `OBSERVABILITY.md`, `DIAGNOSTICS.md`, `RELIABILITY_FAILURE_AND_HITL.md`, `DECISION_VERIFICATION.md`, `DECISION_SYSTEM.md`, `intergrax_runtime_architecture.md`
- Security: `docs/project/technical/adr/entries/2026-06-19/ADR-SEC-001.md`
- Historical (evidence only): HARNESS-W4/W5/W6, HARNESS-FINAL, GOV-X2, EBH-3/4, INT-CONFIG-REAL-X, AW-7C, CE-01, GR-12-FINAL

---

## 4 Twelve-plane closed-world inventory

All twelve planes present. Column SSOT: catalog + architecture docs; status = current-HEAD candidate **PASS** when `CX-*` row passes mechanical matrix (§11).

| Plane | FRZ | Semantic responsibility | Canonical semantic owner | Canonical contract(s) | Runtime boundary | Composition owner | Config / profile | Plugin SPI | Consumers | Mutation path | Governance | Execution | Tenant identity | Evidence | Qualification gates | Historical | Limitations | Shadow count | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Security | CTL-01 | Cross-cutting defense, not parallel engine | Tier-3 `SecurityEnvelope` + `PluginSecurityDefenseMiddleware` | `SecurityDefensePlugin`, ADR-SEC-001 | UAEP hook timeline | Host `application_security_wiring` | security profile / plugin manifest | `security_defenses` entry points | UAEP, ToolRuntime adjacency | profile/plugin via governed host wiring | narrows/denies only | none | `tenant_id` in hook ctx | `platform.security.*` signals | PLUG-03, MEM-FINAL-6, CX-01 nodes | AW-7C substrate scoped | PROD-Q live provider trust | 0 peer SecurityEngine | PASS |
| Reliability | CTL-02 | Recovery/admission, not scheduler | `ResiliencePolicy` + `DependencyAttemptExecutionBoundary` | failure class + attempt boundary contracts | execution + tool admission | execution composition roots | resilience policy config | dependency plugins | Execution, ToolRuntime | policy/boundary config (governed) | recommend/deny admission | no second execution authority | tenant-scoped admission state | runtime events / attempt facts | HARNESS-02, W4-R1, CX-02 | HARNESS-W4 | saturation PROD-Q depth | 0 unbounded prod retry | PASS |
| Cost/Budget | CTL-03 | Global budget accounting | Execution budget ledger | `RunBudget`, `BudgetEnvelope`, `ResourceQuota` | ledger at execution bind | Nexus / execution composition | run + routing budget snapshots | n/a | Execution, tools, routing | budget reservation (local ledger) | narrows | not permission | tenant budget partition | budget exhaustion outcomes | ledger tests, CX-03 | HARNESS-FINAL | routing cost PROD-Q | 0 local budget engines | PASS |
| Evaluation | CTL-04 | Measure/score; not permission | Advisory evaluation (`TokenOptimizationAdvisory*`, quality governance) | advisory result contracts | offline/advisory modules | token-optimization / governance wiring | eval corpus config | evaluator providers | governance promotion gates (explicit) | promotion via Governance only | advisory | no execution | host-scoped queues | redaction-safe reports | CX-04 nodes | GR10 inference | global online eval TENANT-X | 0 eval→execute bypass | PASS |
| Verification | CTL-05 | Decision verification stages | Decision Verification Pipeline | `decision_verification_stage`, lifecycle contracts | post-decision stages | `decision_verification_composition` | stage wiring profile | semantic judge via ToolRuntime | Decision system | stage config (governed) | challenges decision | no execution | tenant on decision material | verification results | DS-MIG-04, CX-05 | retired Critic | external judge replaceability | 0 live CriticOrchestrator | PASS |
| Observability | CTL-06 | Record canonical evidence | Event bus + export sinks | `RuntimeEvent`, `ObservabilityExportPayload`, sink ports | bus → bounded sink → export | runtime observability wiring | export profile | `EventExportSinkPort` | diagnostics, operators | sink config | read/record | no execution truth | correlation fields | export payloads | W5-H, W5-C, CX-06 | HARNESS-W5 | TRACE-X global completeness | 0 mint ExecutionId | PASS |
| Diagnostics | CTL-07 | Interpret persisted facts | `DiagnosticOrchestrator`, `ProblemLifecycleEngine` | diagnostic ports | reconstruct/classify only | `diagnostic_runtime_wiring` | diagnostic profiles | n/a | operator surfaces | problem state (non-exec) | advisory | no effects | tenant on problems | diagnostic events | CX-07 | HARNESS-W6 scoped | TRACE-X reconstruction | 0 exec truth mint | PASS |
| Tools | CTL-08 | Intent → enforced invoke | `ToolRuntime` | tool intent / admission contracts | gateway → invoker → executor | tool composition roots | tool policy | tool plugins | agents, governance | registry/policy (governed) | permission via Governance | effects only here | tool scope per tenant | invocation records | GOV-X2 tool E2E, W4-R1, CX-08 | TOOLS.md residuals | Protocol-v2 gaps → reconciled non-blocker | 0 prod bypass in gates | PASS |
| Skills | CTL-09 | Declarative capability packs | `SkillResolver` + registry | `SkillManifest` | catalog/resolver | AgentContract assembly | skill manifests | skill plugins | AgentRegistry | catalog mutation governed | compose only | no direct backend | skill scope | composition audit | PLUG-03, CX-09 | PLUG-FINAL | transitive requires_skills gated | 0 skill→Tool bypass | PASS |
| Agent Registry | CTL-10 | Derived projection | `build_registry_projection` / read boundary | distribution revision contracts | read-only registry surface | agent distribution activation | host roster config | distribution plugins | Nexus routing | activation via distribution (governed) | not install authority | not execution | application/tenant scope | registry revision metadata | registry read boundary, CX-10 | HOST-01 | host-local register test-only | 0 install authority on registry | PASS |
| Capability Graph | CTL-11 | Dependency/impact description | `CapabilityGraph` | graph + compatibility contracts | validate/deploy gate | `capability_graph_assembly` | environment graph view | n/a | deploy/impact analysis | graph validate (governed) | no permission | no activation | tenant where modeled | deploy reports | deploy gate, CX-11 | EBH-3 | product strict gate only | 0 graph→execute | PASS |
| Context Engineering | CTL-12 | Model-call assembly owner | CE orchestrator / engine entry | CE contracts, CE-Q catalog | every canonical model call | context bootstrap composition | CE policy/budget | context plugins | Nexus LLM paths | CE config (governed) | filters context | not execution | fragment tenant guard | `ContextEngineeringReport` | CE-01, CX-12 | HARNESS-RESIDUAL | TOKEN-CE productization → future | 0 CE bypass in CE-Q15 | PASS |

---

## 5 Semantic owner matrix

One semantic owner per plane (see inventory column). **duplicate semantic owner = 0** (verified by closed-world inventory).

---

## 6 Composition owner matrix

| Plane | Composition owner |
|---|---|
| Security | Host `application_security_wiring` / harness host runtime |
| Reliability | Execution + tool composition roots (`DependencyAttemptExecutionBoundary` builder) |
| Cost/Budget | Execution budget ledger materialization at bind |
| Evaluation | Advisory module wiring (token-optimization / governance consumers) |
| Verification | `decision_verification_composition` factory |
| Observability | Runtime event delivery + export composition |
| Diagnostics | `diagnostic_runtime_wiring` |
| Tools | ToolRuntime / gateway composition |
| Skills | AgentContract + SkillResolver binding |
| Agent Registry | Distribution → `build_registry_projection` |
| Capability Graph | `capability_graph_assembly_resolver` / deploy gate |
| Context Engineering | CE bootstrap / orchestrator composition |

---

## 7 Contract/boundary matrix

See inventory «Canonical contract(s)» and «Runtime boundary». Typed contracts anchored in `intergrax/contracts/*` and plane-specific architecture docs cited in §3.

---

## 8 Authority matrix

| Plane | Observe | Recommend | Deny/narrow | Grant permission | Create execution | Mutate own state | Mutation governed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Security | Y | Y | Y | N | N | Y (profile) | Y (host/governance) |
| Reliability | Y | Y | Y | N | N | Y (policy) | Y |
| Cost/Budget | Y | Y | Y | N | N | Y (ledger) | Y (local ledger rules) |
| Evaluation | Y | Y | N | N | N | Y (results store) | Y when promotion |
| Verification | Y | Y | Y (challenge) | N | N | Y (stage state) | Y |
| Observability | Y | N | N | N | N | Y (sink config) | Y |
| Diagnostics | Y | Y | N | N | N | Y (problems) | N/A interpretive |
| Tools | Y | N | Y | via Gov | Y (effects) | Y (registry) | Y |
| Skills | Y | N | N | N | N | Y (catalog) | Y |
| Agent Registry | Y | N | N | N | N | N (derived) | via distribution |
| Capability Graph | Y | Y (impact) | N | N | N | Y (graph view) | Y |
| Context Engineering | Y | Y (rank) | Y (filter) | N | N | Y (policy) | Y |

No plane is peer Governance or peer Execution authority (GOV-X2 closure preserved).

---

## 9 Consequential mutation inventory

Consequential control-plane mutations route through `ControlPlaneMutationAuthorizationBoundary` / `ControlPlaneMutationRequest` (`intergrax/contracts/control_plane_mutation.py`). Evidence: `test_cpma_2_non_user_approver_rejected` + GR-12-FINAL control-plane qualification.

| Plane | Mutations | Classification |
|---|---|---|
| Security | security profile / plugin enable | CONSEQUENTIAL — governed |
| Reliability | resilience policy profile | CONSEQUENTIAL — governed |
| Cost/Budget | budget envelope config | LOCAL NON-CONSEQUENTIAL at ledger; host config CONSEQUENTIAL |
| Evaluation | promotion / registry of eval artifacts | CONSEQUENTIAL when tied to release |
| Verification | stage enablement | CONSEQUENTIAL — composition |
| Observability | export sink activation | CONSEQUENTIAL — host wiring |
| Diagnostics | diagnostic profile | LOCAL / host config |
| Tools | registry/policy hot reload | CONSEQUENTIAL — CPMA |
| Skills | catalog registration | CONSEQUENTIAL — plugin admission |
| Agent Registry | none at runtime (derived) | READ-ONLY projection |
| Capability Graph | strict deploy validate | CONSEQUENTIAL — deploy gate |
| Context Engineering | CE policy/plugin | CONSEQUENTIAL — host composition |

**uncatalogued consequential mutation count = 0** (GR-12-FINAL + CPMA catalog scope).

---

## 10 Cross-plane boundary matrix

| Intersection | Truth owner | Decision owner | Effect owner | Contract | Downstream widen? |
|---|---|---|---|---|---|
| Security ↔ Governance | Governance | Governance | Governance | defense hook results | No — defense cannot grant |
| Reliability ↔ Execution | Execution | Reliability admission | Execution | attempt boundary | No bypass on deny |
| Cost ↔ Routing | Budget ledger | Routing policy | Execution | RunBudget snapshots | No silent provider pick |
| Evaluation ↔ Verification | Decision | Verification stages | Decision lifecycle | stage contracts | Eval ≠ permission |
| Verification ↔ Decision | Decision | Verification | Decision | challenge result | No second decision system |
| Observability ↔ Diagnostics | Observability facts | Diagnostics | neither execution | event payloads | No truth creation |
| Tools ↔ Skills | ToolRuntime | Governance + ToolRuntime | ToolRuntime | skill tool_ids | Skill ≠ grant |
| Tools ↔ Governance | Governance | Governance | ToolRuntime | admission | No registry-alone grant |
| Agent Registry ↔ Distribution | Distribution revision | activation governance | distribution | projection build | Registry not install authority |
| Capability Graph ↔ activation | Distribution/deploy | deploy gate | distribution | graph validate | Edge ≠ permission |
| CE ↔ Memory/RAG/Prompt | CE assembly | CE | LLM adapter call | CE entry surfaces | CE not store owner |

---

## 11 CX-01..CX-12 mechanical matrix

| ID | Plane | Owner unique | Boundary unique | Typed | Composition unique | Peer Gov | Peer Exe | Bypass | Tenant | Proof nodes | Result |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CX-01 | Security | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-02 | Reliability | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-03 | Cost/Budget | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-04 | Evaluation | Y | Y | Y | Y | 0 | 0 | 0 | N/A† | 2 | PASS |
| CX-05 | Verification | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-06 | Observability | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-07 | Diagnostics | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-08 | Tools | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-09 | Skills | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-10 | Agent Registry | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-11 | Capability Graph | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 2 | PASS |
| CX-12 | Context Engineering | Y | Y | Y | Y | 0 | 0 | 0 | PASS | 3 | PASS |

† CX-04 tenant: online eval queues host-scoped — `N/A — WITH EVIDENCE`; global TENANT-X not entered.

---

## 12 Security responsibility matrix

| Data concern | Platform owner? | Layer/owner | Boundary | Evidence | CTRL-X verdict | Later PROD-Q |
|---|---|---|---|---|---|---|
| tenant-sensitive data | partial | Security + CE + Governance | hook ctx / CE fragment guard | MEM-FINAL-6, CE-Q3 | candidate evidence | PROD-Q |
| raw secrets / references | partial | host secret resolution | no cross-layer raw secret payload | HOST-Q5, SEC contracts | candidate evidence | PROD-Q |
| prompt/model input | Y (CE) | Context Engineering | CE assembly | CE-01 | candidate evidence | — |
| tool/provider payload | Y (ToolRuntime) | ToolRuntime gateway | invoke boundary | CX-08 | candidate evidence | PROD-Q |
| runtime evidence | Y (Observability) | Event bus | export payload | CX-06 | candidate evidence | TRACE-X |
| diagnostic output | Y (Diagnostics) | interpretive only | problem store | CX-07 | candidate evidence | — |
| logs/export | partial | Observability export | sink ports | W5-H | candidate evidence | PROD-Q |
| retention | N/A infra | operator/infra | documented assumption | ADR-SEC-001 | N/A — WITH EVIDENCE | PROD-Q |
| deletion/redaction | partial | CE eval reports | redaction-safe eval | CX-04 | candidate evidence | PROD-Q |
| encryption | partial | platform signals | encryption denied events | security_events | candidate evidence | PROD-Q |
| data residency | N/A | deployment | not platform-guaranteed | — | N/A — WITH EVIDENCE | PROD-Q |
| provider exposure | partial | Security + tool provider SPI | plugin admission | PLUG-03 | candidate evidence | PROD-Q |

---

## 13 Reliability / capacity matrix

| Mechanism | Owner | Bound | Backpressure | Timeout | Retry | Cancellation | Throttling | Saturation |
|---|---|---|---|---|---|---|---|---|
| Root execution deadline | Execution budget bind | global deadline | admission deny | explicit owner | RetryEngine bounded | HARNESS-02 | rate limits | explicit reject |
| Tool protected work | Tool admission | W4-R1 semaphore | reject second invoke | wait timeout | policy-bound | cancel propagate | per-tool | fail closed |
| Dependency attempts | Attempt boundary | W4 arch gate | boundary enforced | attempt timeout | classifier | cancel | CB / bulkhead | degrade/reject |
| Provider calls | External op boundary | harness proofs | backpressure | provider timeout | bounded | cancellation | throttling | typed failure |

**Finding counts:** unbounded applicable production concurrency = 0; hidden infinite retry = 0; silent fallback authority widening = 0; overload-created execution bypass = 0 (current-head gate evidence).

---

## 14 Observability / Diagnostics truth matrix

| Assertion | Count / status |
|---|---|
| Execution truth → Observability records | canonical spine (W5) |
| Observability → Execution truth creation | 0 (W5-H + CX-06) |
| Diagnostics → Execution truth creation | 0 (CX-07 fail-closed) |
| Diagnostics → Governance permission | 0 |
| Diagnostics → direct protected effect | 0 |

---

## 15 Tools / Skills separation

`SkillManifest` composes declarations; `ToolRuntime` enforces invocation (PLUG-03 positive/negative skill gateway tests). **skill requirement ≠ host capability grant.**

---

## 16 Agent Registry / Assembly

`AgentRegistry` on materialized projection: **no `register` on runtime read surface** (`test_materialized_projection_runtime_surface_has_no_register`). Activation/install authority remains distribution + governance (GOV-X2).

---

## 17 Capability Graph

Graph describes dependencies; deploy gate blocks experimental agents in strict mode. Graph catalog does not own binding identity resolution.

---

## 18 Context Engineering

`CONTEXT_ENGINEERING` exactly-one owner; CE-Q1/Q3/Q15 prove entry surfaces, tenant fragment rejection, and no direct prompt injection on canonical Nexus paths. Residual doc phrases classified in §22.

---

## 19 Tenant audit (16-question summary)

Per-plane verdicts in catalog (`tenant_verdict`). Applicable planes: Security, Reliability, Cost, Verification, Observability, Diagnostics, Tools, Skills, Agent Registry, Capability Graph, Context Engineering = **PASS**; Evaluation = **N/A — WITH EVIDENCE** (host-scoped advisory). **Global TENANT-X = NOT ENTERED.**

---

## 20 Strong typing audit

- **Core boundary pyright slice (15 modules):** 0 errors (log: `.tmp/session/ctrl-x/pyright-core-boundary-slice.log`)
- **Wide tree scan (representative directories):** 133 errors — **CTRL-X-D1 TRACKED FREEZE DEBT → EBH-6 / FRZ-TYP** (pre-existing; not control-plane authority defects). Provenance: `.tmp/session/ctrl-x/pyright-semantic-surfaces.log`
- **semantic-boundary slice after `security_events` port union:** 0 errors
- Critic legacy symbols: **0** in production (`DS-MIG-04`)

---

## 21 Bypass / duplicate scan

| Pattern | Current-HEAD finding |
|---|---|
| duplicate SecurityEngine | 0 |
| live CriticOrchestrator | 0 (deleted) |
| shadow budget engine | 0 in CTRL scope |
| observability mint ExecutionId | 0 in W5/CTRL gates |
| skill direct tool bypass | 0 in PLUG-03 gates |
| capability graph triggered execution | 0 |

TOOLS.md / CE historical residuals reconciled in §22 (not automatic blockers).

---

## 22 Historical evidence reconciliation

| Artifact | Classification |
|---|---|
| HARNESS-W4 | CURRENTLY REPRODUCED (W4-R1 nodes in CX-02/CX-08) |
| HARNESS-W5 | CURRENTLY REPRODUCED (CX-06 nodes) |
| HARNESS-W6 | CURRENT CODE CONFIRMED (diagnostics non-authoritative; scoped) |
| HARNESS-FINAL | SUPERSEDED for global claim; partial CURRENT for budget |
| GOV-X2 | CURRENT CODE CONFIRMED (Gov ≠ Exe) |
| CE-01 | CURRENTLY REPRODUCED (CX-12) |
| GR-12-FINAL | CURRENTLY REPRODUCED (cross-cutting CPMA node) |
| AW-7C | SCOPED EVIDENCE — not global SEC PASS |

**Documentation residuals:**

| Statement | Class |
|---|---|
| TOOLS partial governance / Protocol-v2 gaps | D — feature maturity; gated paths pass |
| CE non-uniform / TOKEN-CE | D / E — CE-01 supersedes for canonical paths |
| OBS universal vendor limitations | C — TRACE-X / PROD-Q |

**unclassified historical/residual statements = 0**

---

## 23 Test / gate results

| Command | passed | failed | skipped | xfailed | exit |
|---|---:|---:|---:|---:|---:|
| `uv run pytest -p no:xdist tests/qualification/control_planes/ctrl_x/ -q` | 11 | 0 | 0 | 0 | 0 |

Log: `.tmp/session/ctrl-x/pytest-ctrl-x-package.log`

---

## 24 Exact proof replay

| Metric | Value |
|---|---|
| declared | 27 |
| unique | 27 |
| requested | 27 |
| executed (passed) | 27 |
| failed | 0 |
| missing | 0 |
| unjustified skipped | 0 |

Mechanism: `tests/qualification/control_planes/ctrl_x/proof_replay.py` (excludes orchestration modules). Gate: `test_ctrl_x_exact_proof_replay_all_catalog_nodes_pass`.

---

## 25 FRZ evidence mapping (candidate only — no Cursor PASS)

| Family | CTRL-X contribution |
|---|---|
| FRZ-CTL-01..12 | CX-01..CX-12 catalog + replay |
| FRZ-SEC-01..10 | CX-01 + MEM-FINAL-6 + AW-7C scoped |
| FRZ-REL-01..11 | CX-02 + HARNESS-02/W4 |
| FRZ-OBS-01..07 | CX-06 + CX-07 |
| FRZ-TEN-* | per-plane tenant column; global OPEN |

---

## 26 Findings / debt

| ID | Class | Summary | Owner stage |
|---|---|---|---|
| — | — | **IN-SCOPE BLOCKER = 0** | — |
| CTRL-X-D1 | TRACKED FREEZE DEBT | Wide tree typing debt (committed provenance; boundary slice 0 errors) | EBH-6 / FRZ-TYP |
| CTRL-X-D2 | TRACKED FREEZE DEBT | PROD-Q provider/sandbox global SEC rows | PROD-Q |
| CTRL-X-D3 | TRACKED FREEZE DEBT | TRACE-X global observability completeness | TRACE-X |

**unclassified = 0**

---

## CTRL-X-R1 — Independent-audit remediation

| Blocker | Root cause | After R1 |
|---|---|---|
| R1-A | Production host accepted `FAIL_OPEN` defense plugins | `register_application_security_hooks` + `validate_security_wiring` reject non-`FAIL_CLOSED` before middleware attach |
| R1-B | `security_events.py` used `TYPE_CHECKING` + `object` event port | `HostOrchestrationRuntimeEventPort` only |
| R1-C | 133 wide pyright rows only under `.tmp` | Committed `typing_scope.py`, `wide_pyright_scope.py`, `wide_pyright_provenance.py`, `CTRL_X_WIDE_PYRIGHT_PROVENANCE.md` |
| R1-D | CX-04 narrowed to advisory token optimization only | Closed-world Evaluation catalog + R1-EVAL proof nodes |

**Security after:** production `FAIL_OPEN` accepted = 0; `FAIL_CLOSED` path proven; generic `object` event boundary = 0; no new Security authority.

**Typing after:** `CTRL_X_SEMANTIC_BOUNDARY_MODULES` gate = 0 pyright errors; wide diagnostics grouped with `unaccounted = 0` (see provenance module total).

**Evaluation after:** `EvaluationProfile` + `OnlineEvaluationRegistry` + shadow/advisory/offline surfaces inventoried in CX-04 catalog; evaluation ≠ Decision Verification; no direct execution/permission path in proofs.

**CTRL-X-R1 recommendation:** READY FOR AUDIT (independent GitHub audit required).

---

## CTRL-X-R2 — Contract-Pure Middleware Composition & Evidence Closure

| Blocker | Before R2 | After R2 |
|---|---|---|
| R2-A | `application_security_wiring` narrowed on `MiddlewarePipeline` + `attach_tier1_middleware_if_absent` | `HostOrchestrationMiddlewarePipelinePort.attach_runtime_middleware_if_absent` only; `HookPoint` Tier-0 owner in `intergrax/contracts/middleware_hook_point.py`; alternate port proof (R2-MW-02) |
| R2-B | Wide diagnostics auto-classified EBH-6 off manifest | Semantic plane mapping per domain; `ctrl_x_impact` on every group; `CTRL-X blocker = 0`; total diagnostics = 128 accounted |
| R2-C | Weak `hasattr` / docstring Evaluation proofs | R2-EVAL flow proofs: registry append-only, `evaluate_agent_promotion` gate, Decision Verification composition disjoint; TENANT-X debt on missing `tenant_id` on `OnlineEvaluationObservation` |
| R2-D | 15 security failures unbaselined | A/B/C identical 15 failures — `PRE-EXISTING TEST DEBT` matrix in `test_ctrl_x_r2_security_baseline.py` |

**Middleware after R2:** composition boundary = `HostOrchestrationMiddlewarePipelinePort`; concrete `MiddlewarePipeline` check in production wiring = 0; sanctioned attach = 1; FAIL_OPEN production rejection preserved.

**CTRL-X-R2 recommendation:** **BLOCKED** — independent audit requires ADR-CTRL-X-001 (middleware hook ownership & typed context). See `docs/project/technical/adr/entries/2026-10-03/ADR-CTRL-X-001-MIDDLEWARE-HOOK-CONTRACT-OWNERSHIP.md`.

---

## CTRL-X-ADR1 — Middleware hook contract (2026-10-03)

| Item | Status |
|---|---|
| ADR-CTRL-X-001 | **Proposed / Ready for Independent Architecture Audit** (not Accepted) |
| CTRL-X-R1 | **BLOCKED** pending ADR acceptance |
| CTRL-X-R2 | **BLOCKED** pending ADR acceptance |
| CTRL-X parent | **BLOCKED** pending ADR acceptance |

Trigger: unauthorized `HookPoint` ownership move; `Mapping[str, object]` hook boundary; 15 security failures without roadmap owners; wide-pyright semantic classification policy.

---

## CTRL-X-ADR1-Q1 — ADR consistency & security debt ownership (2026-10-03)

Evidence commit baseline: `ec55899956cf6c3c6290407c8c8ebecdb066911a` (`development`).

| Objective | Status | Evidence |
|---|---|---|
| **Q1-A** ADR supersession / refinement vs ADR-HARNESS-001 D1 | **resolved** | ADR-CTRL-X-001 header fields **Refines / supersedes** + **Does NOT reopen**; §ADR-HARNESS-001 relationship (Q1-A); exactly-one owner matrix for HookPoint, protocols, pipeline, port, alias |
| **Q1-B** canonical roadmap ownership for 15 security failures | **resolved** | ADR-CTRL-X-001 §22 full 15-row matrix; A/B/C provenance unchanged; HEAD re-run 15/15 fail with classified owners (QUAL-X = 15); **remaining security-test CTRL-X blockers = 0** |
| **Q1-FIX** FRZ semantic mapping in §22 security matrix | **resolved** | ADR-CTRL-X-001 §22 FRZ families aligned to `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`; **FRZ-CTL-04** in 15-test matrix = **0**; §CTRL-X-ADR1-Q1-FIX acceptance readiness |

| Item | Status |
|---|---|
| CTRL-X-ADR1-Q1-FIX | **READY FOR AUDIT** |
| CTRL-X-ADR1-Q1 | **READY FOR FINAL INDEPENDENT ACCEPTANCE** |
| CTRL-X-ADR1 | **READY FOR FINAL INDEPENDENT ACCEPTANCE** (ADR still Proposed until audit Accepts) |
| CTRL-X parent | **BLOCKED PENDING ADR ACCEPTANCE** |
| CTRL-X-R3 | **NOT ENTERED** |
| STATE-X | **NOT ENTERED** |

---

## CTRL-X-R3 — Typed middleware context (ADR-CTRL-X-001 implementation)

| Item | Value |
|---|---|
| ADR baseline (independent acceptance) | `07ef4d57a46e3d5a06bfc7a7d95cdfcd930e9b38` |
| ADR-CTRL-X-001 | **Accepted** — independent architecture audit |
| Tier-0 contracts | `intergrax/contracts/middleware_hook_semantics.py` (`MiddlewareHookInvocationContext`, payload family, `MiddlewareExecutionSubjectFacet`) |
| Cross-layer `runtime_state` on middleware ABI | **0** (internal `HookContext.runtime_state` retained for hook-registry mutation only) |

**Before → after (boundary):**

```text
HostOrchestrationMiddlewareHookContext.runtime_state: Mapping[str, object]
  → MiddlewareHookInvocationContext + typed payload + subject facet
```

**R3 invariants R3-01..R3-15:** qualified in `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r3_typed_middleware_contracts.py` (R3-T01..R3-T12 + bypass search).

**Evidence (historical R3 close):** semantic-boundary pyright 0 errors; proof replay superseded by R3-R1 evidence below.

**CTRL-X-R3 recommendation:** **READY FOR AUDIT** (superseded by R3-R1 current-head section).

---

---

## CTRL-X-R3-R2 — Tenant scope fail-closed (Security middleware)

| Item | Value |
|---|---|
| Parent | CTRL-X-R3 |
| Blocker closed | missing/blank request tenant + tenant-bound resource no longer ALLOW |
| Canonical helper | `intergrax/runtime/security/tenant_scope.py` (`normalize_tenant_scope_id`, `tenant_scope_is_valid`) |
| Plugin defense | `allow_unscoped=True` (None/None only when both absent after normalize) |
| Task intake | `TenantSecurityMiddleware` uses `allow_unscoped=False` |
| Proof replay | `requested=44` `unique=44` `executed=44` `passed=44` `failed=0` |
| CTRL-X suite | `76` tests, exit 0 |

| Item | Status |
|---|---|
| CTRL-X-R3-R2 | **READY FOR AUDIT** (superseded by Q1 current-head section below) |
| CTRL-X-R3-R1 | **READY FOR AUDIT** |
| CTRL-X-R3 | **READY FOR AUDIT** |
| CTRL-X | **READY FOR AUDIT** (not CLOSED — independent audit required) |
| STATE-X | **NEXT / NOT ENTERED** |

---

## CTRL-X-R3-R2-Q1 — Remove STATE-X scope contamination

| Item | Value |
|---|---|
| Parent | CTRL-X-R3-R2 |
| START_HEAD | `6bcf1d615170da4b8eef91ca95554a21994dd130` |
| Contamination reverted | `intergrax/runtime/long_running/checkpoint_resume_validation.py` restored to exact `ebd5ac5653ee054d28cc32ed0302eb361a4ef56a` (git blob `d7bcd0253da353d59731020ca1ae5f1ad5b13cdc`) |
| Deferred to STATE-X | checkpoint/resume authority semantics (e.g. missing `task_snapshot` → `ParentExecutionAuthority.unknown()` from `093b6570a`) — **not** in CTRL-X production delta |
| STATE-X production delta | **0** (no `long_running` / state / recovery / checkpoint / persistence production changes vs `ebd5ac…`) |

**R3-R2 production delta from `ebd5ac5653ee054d28cc32ed0302eb361a4ef56a` (exact, post-Q1):**

```text
intergrax/runtime/security/tenant_scope.py
intergrax/runtime/security/defense_plugin.py
intergrax/applications/_shared/application_security_wiring.py
```

Tenant fail-closed matrix unchanged (canonical `tenant_scope_is_valid`; plugin `allow_unscoped=True`; intake `allow_unscoped=False`; security ordering and `KIND_DEFENSE_BLOCKED` evidence preserved).

**Exact proof replay (current HEAD after Q1):** `requested=44` `unique=44` `executed=44` `passed=44` `failed=0` `errors=0` `missing=0` `skipped=0` (`proof_replay.py` on Q1 `FINAL_COMMIT`). Historical `requested=40` (pre-R3-R2) superseded.

| Item | Status |
|---|---|
| CTRL-X-R3-R2-Q1 | **READY FOR AUDIT** |
| CTRL-X-R3-R2 | **READY FOR AUDIT** |
| CTRL-X-R3 | **READY FOR AUDIT** |
| CTRL-X | **READY FOR AUDIT** (not CLOSED — independent audit required) |
| STATE-X | **NEXT / NOT ENTERED** |

---

## CTRL-X-R3-R1 — Tool payload fidelity & qualification closure

| Item | Value |
|---|---|
| Parent | CTRL-X-R3 |
| Evidence commit | current `development` HEAD after R3-R1 remediation |
| Tool arguments contract | `ToolCallHookPayload.arguments: JsonObject` (`intergrax/contracts/structured_json_value.py`) |
| Stringification at canonical payload build | **0** (`normalize_tool_arguments` / `normalize_structured_json_object`) |
| Security token scan projection | `intergrax/runtime/security/json_security_projection.py` (local to Security / built-in scanners) |
| Closed-world middleware inventory | `tests/qualification/control_planes/ctrl_x/middleware_inventory.py` |
| R3-T10 | Direct FAIL_OPEN composition rejection (no test-function-as-proof) |
| R1-TOOL-01..08 | `test_ctrl_x_r3_typed_middleware_contracts.py` |

**Exact proof replay (pre-R3-R2 baseline):** `requested=40` … `passed=40` (superseded by R3-R2 catalog nodes).

**Exact proof replay (current HEAD after R3-R2):** `requested=44` `unique=44` `executed=44` `passed=44` `failed=0` `errors=0` `missing=0` `skipped=0`.

**CTRL-X suite:** `uv run pytest -p no:xdist tests/qualification/control_planes/ctrl_x -q` → exit 0 (76 tests).

| Item | Status |
|---|---|
| CTRL-X-R3-R1 | **READY FOR AUDIT** |
| CTRL-X-R3 | **READY FOR AUDIT** |
| CTRL-X | **READY FOR AUDIT** (blocked pending independent audit only) |
| STATE-X | **NEXT / NOT ENTERED** |
| FRZ-CTL-* promotion | Independent audit only |

---

## 27 Recommendation (current HEAD)

| Item | Status |
|---|---|
| CTRL-X-R3-R2-Q1 | **READY FOR AUDIT** |
| CTRL-X-R3-R2 | **READY FOR AUDIT** |
| CTRL-X-R3-R1 | **READY FOR AUDIT** |
| CTRL-X-R3 | **READY FOR AUDIT** |
| CTRL-X parent | **READY FOR AUDIT** (not CLOSED — independent GitHub SHA audit required) |
| CX-01..CX-12 | PASS (mechanical, current-head proof replay — 44 nodes) |
| STATE-X | NOT ENTERED |
| FRZ-CTL-* promotion | Independent audit only |

---

## Owner summary (§43)

See `CTRL_X_PLANE_CATALOG` fields: semantic_owner, canonical_contract_boundary, tenant_verdict per plane. Composition owners in §6. Replaceability: plugin/SPI columns in §4.

---

## Changed files (§44)

| Commit role | SHA |
|---|---|
| R3 implementation | `98a586540` — typed CTRL-X hook contracts |
| R3-R1 remediation | `ebd5ac5653ee054d28cc32ed0302eb361a4ef56a` |
| R3-R2 remediation | `c8338ebb6` — `fix(security): fail closed on incomplete tenant scope` (tenant files; `093b6570a` also touched checkpoint — **reverted in Q1**) |
| R3-R2-Q1 remediation (`FINAL_COMMIT`) | `a4ce244a0` — `fix(ctrl-x): remove state scope contamination` |

| Kind | Paths |
|---|---|
| production (R3-R2 delta vs `ebd5ac…`, post-Q1) | `intergrax/runtime/security/tenant_scope.py` (new); `intergrax/runtime/security/defense_plugin.py`; `intergrax/applications/_shared/application_security_wiring.py` only — **not** `checkpoint_resume_validation.py` |
| Q1 restore | `intergrax/runtime/long_running/checkpoint_resume_validation.py` → exact `ebd5ac…` content (STATE-X deferred) |
| tests | `tests/unit/runtime/security/test_tenant_scope.py`; `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r3_r2_tenant_scope_fail_closed.py`; `tests/support/middleware_hook_test_context.py`; `tests/qualification/control_planes/ctrl_x/catalog.py` |
| qualification docs | `docs/project/maintainers/qualification/CTRL_X_ENTERPRISE_CONTROL_PLANE_RECERTIFICATION.md` |

**STATE-X production delta (vs `ebd5ac…`, post-Q1):** `0`

---

## Mandatory audit disclaimer

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
