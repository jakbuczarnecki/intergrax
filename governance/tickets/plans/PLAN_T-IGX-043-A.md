# PLAN T-IGX-043-A — FC-1 execution routing and Nexus construction (EQO-A remediation wave, node PT-FC1; version for the RW train)

Status: plan text for the PT-FC1 node of the RW train (RW-PLAN v13, appendix A1 block `PT-FC1 (T-IGX-043-A)`). Nothing in this plan is implemented, run or audited. No claim of closure, PASS, FRZ promotion or independent verification is made. Evidence notation: `path@H:line`, `path@B:line` and `path@T:line`, where `H` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (the head read for the first version of this plan, 2026-11-16), `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (Campaign A baseline) and `T` = `05699b633cdb1318c215f6529e449420efc1382c` (head of `arch-freeze/rw-train` read on 2026-11-17; it contains PT-WA, PR #642). Full-SHA evidence rows of the first version are in `FACT_CHECK.md`. A citation marked `@H` or `@B` was checked unchanged at `T` for every file of the write set except the seven files named in section 17 (`git diff 7960ae7f6 origin/arch-freeze/rw-train -- <file>`); section 17 lists every fact that differs at `T` and section 18 lists the differences from appendix A1.

## 1. Purpose and findings

One of four fail-closed batches of the single product-path wave (DP-11; master-plan node `PT-FC1`; authorization AR-41). It removes the places where a PRODUCT/STRICT Nexus composition can silently run with a relaxed posture, and it makes the posture of every Nexus-internal router and store explicit (OD-REM-03, C3, DP-24, DP-26, DP-27).

| Finding | Severity (dossier) | What it is at `H` | This ticket |
|---|---|---|---|
| EQOA-A4-044 | MEDIUM, `CONFIRMED_DEFECT` | `build_host_task_execution` builds a second `AgentRouter` without `production_mode` (`nexus_host_execution.py@H:57-60`), default `False` (`agent_router.py@H:49`); `HostTaskExecution.execute` routes through it whenever `task.agent_id` is empty (`host_task.py@H:223-224`). Production-only lifecycle/owner/runbook checks (`agent_routing_policy.py@H:34-57`) are skipped on the canonical host-task path. | Part A (firm) |
| EQOA-A4-009 | LOW, `ARCHITECTURE_RISK` | `AgentRouter._route_best_or_fallback` picks `list_routable_agent_ids()[0]` with `selection_reason="fallback_first_routable"`, `fallback_used=True` (`agent_router.py@H:133-143`); no architecture doc or test names it (the Campaign A dossier files do). | Part A (firm), DP-26 |
| EQOA-A4-015 (a) | MEDIUM, `UNVERIFIED_HYPOTHESIS` | Flag defaults differ: `RuntimeConfig.production_mode=True` (`host_runtime_config.py@H:406`), `NexusLoop(production_mode=False)` (`nexus_loop.py@H:255`), `HostOrchestrationLoopInitSpec.production_mode=False` (`host_orchestration_loop_init_spec.py@H:88`), `AgentRouter(production_mode=False)`. | Part A (firm), DP-27 |
| EQOA-A4-015 (b) | same | `ExternalContractorAdapterAgent.build_context` reads `harness_production_mode()` (constant `False`, `intergrax/agents/defaults.py@T:13-15`; `external_contractor_adapter_agent.py@T:113`); `OrganizationWorkerAgent.build_context` hard-codes `production_mode=False` (`organization_worker_agent.py@T:114`); a third site `intergrax/agents/authoring/acp_stub_reflex.py@T:59-65` uses the same constant. `UAEPExecutor` calls `agent.build_context(request)` (`uaep_executor.py@H:336`). Probe V2 (B-1 alpha R03 v2, run at `B`) is CONFIRMED: `ExternalContractorAdapterAgent` is mounted in the PRODUCT manifest of `governed_contractor_application`, whose sanctioned launcher is STRICT-only, and `OrganizationWorkerAgent` is registered only by `intergrax/dev_support` (lab). | Part C1 (firm; step 0 replays V2 at the train head) |
| EQOA-A4-037 | MEDIUM, `ARCHITECTURE_RISK` | `persist_human_decision` returns when `human_store is None` (`human_response.py@H:91-92`); `NexusLoop.human_decision_store` defaults to `None` (`nexus_loop.py@H:246,317`); `materialize_host_orchestration_backend` passes none (`environment_orchestration_materialization.py@H:63-105`) and `HostOrchestrationLoopInitSpec` has no such field (`@H:63-107`). Reject/escalate verdicts handled inside Nexus are therefore not written by that path. Whether another durable path records them is probe V5. | Part B (decided design D-FC1-03) |
| EQOA-A4-024 | MEDIUM, `ARCHITECTURE_RISK` | `NexusLoop.__init__` resolves task-memory and runtime-event stores from explicit arguments or ambient env vars and returns `None` (persistence off) when unset (`nexus_loop.py@H:284-291`; `task_memory/store.py@H:44-67`; `events/store.py@H:48-71`). A second path exists in Tier-3 `wire_task_memory` (`task_memory_wiring.py@H:28-53`) which turns "profile disables task memory" into `enabled=None`, i.e. env-controlled, not `False` (`@H:64-69`). | Part B (decided D-FC1-01: the Tier-1 openers stay LAB-only roots; FC-1 owns A4-024) |
| EQOA-A4-031 | LOW, `UNVERIFIED_HYPOTHESIS` (dossier) | `UAEPExecutor` assembles context only if `context_engine` and `llm_adapter` are both set (`uaep_executor.py@H:386`); the CE fail-closed guard is called only from `tool_runtime.py@H:296-299`. Probe V6 (B-1 alpha R03 v6, run at `B`) is CONFIRMED at composition level: a STRICT host composition without an LLM adapter gets a context manager without engine or adapter. | Part C2 (firm; step 0 replays V6 at the train head) |

Decisions that fix the design: DP-24 (hybrid: wiring layers fail closed on STRICT or PRODUCT, leaf selectors default-deny with an explicit typed LAB opt-in), DP-26 (fallback removed in STRICT/PRODUCT, LAB opt-in with a `fallback_used` event), DP-27 (`NexusLoop(production_mode=)` is a required keyword on product code paths; tests and labs pass it explicitly), DP-18 and DP-13/DP-14 where files overlap with seam work (section 11, coordination C-1..C-3), DP-25 (guide defaults are LAB-scoped), DP-11 (one wave).

## 2. Prerequisites (state them in the ticket; do not start before they hold)

- Milestone `M1` (PDC P9 closed) and the Operator's separate authorization for remediation product-path work (AR-39..AR-44; AR-41 for this batch). Status at `H`: `PDC_INDEX.md@H` (last changed `2bce73842`) still lists the six P9 artifacts as "Answers recorded in chat; awaiting the Operator's final confirmation" and `PDC_SHA_LADDER`/`PDC_OPERATOR_DECISIONS` as "Open"; later commits on `origin/development` are `de025fca0` ("record the Operator's confirmation of the final closure text in OD-PDC-X-CLOSE"), `feaa2a87d` (merge of PR #603), `399d194e5` and `7960ae7f6` (merge of PR #604, "fill FINAL_CLOSURE_PUBLICATION_SHA"); `WP-12/WAVE_SCHEDULE.md` records `M1` as reached on 2026-11-16. This plan records the divergence between the index text and the later commits and does not decide whether P9 is closed.
- Master-plan nodes: `SCOPE` (wave scope frozen: DP-11, DP-13..DP-33, DS-EXC, B-1/B-2 alpha, S1 diagnostics), `DS-2`, `DS-3`, `PT-B01` (probes V1, V2, V5, V6 as run R03; V4 for A4-024). Soft: `PT-G01`, `PT-D12` (seam canon). The wave head `RW` and the audit node `PT-Q02` come after this ticket.
- Operator decisions on this ticket: on 2026-11-16 the Operator accepted the recommended answer to every open question of the drafts T-IGX-043-A..D (published rendering: `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`, items 1, 2, 6 and 8 and addenda 1-4; the per-question mapping D-FC1-01..06 is in `.tmp/session/eqo-a-remediation/FC/FC-1/OPEN_QUESTIONS.md`, a session artefact). D-FC1-03 is option D (hosts with HITL inject the SQLite decision store, hosts and workers without HITL pass the typed acknowledgement; the check is presence plus not in-memory). The ceiling of up to 21 conditional product paths is stated in item 2 of that record; section 4 states how this version relates to it. The merge order (item 6) is FC-2, then FC-1, then FC-3, then the CONFIG-X scan extension; W-A and W-B merge before FC-1. This version changes the design only where section 18 says so; every other decision named above is carried unchanged.
- Approval route: the Approved section of the ticket is filled through the train wave-packet process (OD-AF-22). This plan text does not authorise implementation by itself. The remaining authorisations are listed in `AUTHORIZATION_REQUESTS.md` (session artefact).
- Train prerequisites (RW-PLAN v13 section c, lane 3): PT-WA is merged into `arch-freeze/rw-train` (it is contained in `T`), PT-WB merges before this ticket starts, PT-FC2 merges before it (merge order B, A, C). Product commits go only to a branch that merges into `arch-freeze/rw-train` with a merge commit. Step 0 re-reads every `@T` anchor against the train head at the start, because PT-WB and PT-FC2 change the train after `T`.
- Probe input: B-1 alpha run R03 at `B` (`.tmp/session/arch-freeze/b1-alpha/RESULTS_B1_ALPHA.md`, session artefact): V1 CONFIRMED at router level, V2 CONFIRMED, V4 CONFIRMED, V5 CONFIRMED, V6 CONFIRMED. The red-first rule of RW-PLAN section d applies: the new tests run once on a detached checkout of the train base and the log is kept.

## 3. Design per file

Zero new product files. New typed errors live in existing modules so that no EBH-7/B6-F10 classifier entry for a new product path is needed.

### Part A — routing posture (A4-044, A4-009, A4-015a)

| # | File | Change |
|---|---|---|
| A1 | `intergrax/runtime/nexus/agent_router.py` | `AgentRouter.__init__(registry, *, production_mode: bool, event_bus=None, lab_first_routable_fallback: bool = False)`; `production_mode` has no default (`@H:45-54` today). New `AgentRoutingError(RuntimeError)` defined in this module. In `_route_best_or_fallback` (`@H:115-143`): when `find_best_match` returns `None` and `lab_first_routable_fallback` is false, raise `AgentRoutingError` (no `AGENT_SELECTED` event is emitted because `route()` emits after selection, `@H:107-112`); the first-routable branch stays only for `lab_first_routable_fallback=True` and keeps `fallback_used=True` / `selection_reason="fallback_first_routable"` on the event (`@H:137-143`, payload `AgentSelectionPayloadV1.fallback_used` is emitted at `@H:173-180`). Construction with `production_mode=True` and `lab_first_routable_fallback=True` raises `ValueError` (contradictory posture; no silent downgrade). The explicit-`agent_id` path and the capability path are untouched; `scripts/maintenance/check_capability_routing.py@H:41-46` still requires `validate_task_for_capability_routing` and `select_best_capability_match` in this file. |
| A2 | `intergrax/runtime/nexus/nexus_loop.py` | `production_mode: bool` becomes a required keyword-only parameter of `NexusLoop.__init__` (no default; `@H:255`). New keyword `lab_first_routable_fallback: bool = False` passed to the router (`@H:361-365`). New read-only properties `agent_router` and `production_mode` next to the other properties (`@H:572-775`). No other behavior change in this part. |
| A3 | `intergrax/runtime/execution/nexus_host_execution.py` | `build_host_task_execution` uses `nexus_loop.agent_router` instead of constructing `AgentRouter(nexus_loop.registry, event_bus=...)` (`@H:57-60`); the `AgentRouter` import (`@H:29`) is removed. Single routing authority, so FRZ-OWN-04 "no shadow authority" holds and the posture of both routing sites is identical by construction. All four product callers inherit it: `environment_host_task_execution.py@H:52`, `nexus_host_execution_from_wiring_target.py@H:36`, `worker_host_task_execution_composition.py@H:93`, `debug/app.py@H:98,125`. |
| A4 | `intergrax/runtime/nexus/execution/graph_executor.py` | The default router (`router or AgentRouter(registry, event_bus=event_bus)`, `@H:258`) must now state its posture: `AgentRouter(registry, production_mode=False, event_bus=event_bus)`. This is dead on product paths (`NexusLoop` always injects its router, `nexus_loop.py@H:367-370`); the structural gate (NT-FC1-12) forbids product-code `GraphExecutor(` calls without `router=`. Decided D-FC1-04: this explicit-literal default plus the gate; the alternative (router required, 58 test edits) was not chosen. |
| A5 | `intergrax/runtime/execution/debug_lab_nexus_loop.py` | Both lab builders (`@H:26-31`, `@H:40-44`) pass `production_mode=False` and `lab_first_routable_fallback=True` explicitly. Judgment call J-2: it preserves today's lab/debug behavior as the typed LAB opt-in that DP-26 prescribes; if a lab task never reaches the fallback, the flag is inert. |
| A6 | `intergrax/runtime/execution/host_orchestration_loop_init_spec.py` | `production_mode` becomes `field(kw_only=True)` without default (`@T:86`, `@H:88`); the dataclass is `frozen=True, slots=True` (`@T:61`). Constructions of the spec: `host_orchestration_environment_spec_builder.py@T:194-217` (the Tier-1 function `build_host_orchestration_loop_init_spec`, which passes `environment.execution_mode is ExecutionMode.STRICT` at `@T:217`; PT-WA split the first version's single builder into this Tier-1 function and a Tier-3 caller, section 17) and `worker_host_task_execution_composition.py@H:80-91` (passes the caller's `production_mode`); no test constructs the spec (`git grep` over `tests`, `testing_support` and `applications/*/tests` at `H` and at `T`: 0 hits). |
| A7 | tests/labs sweep | Every `NexusLoop(` call that lacks `production_mode=` gets an explicit `production_mode=False` (behavior-preserving; the keyword exists today). Census at `T` (AST over every `*.py`, calls named `NexusLoop` or `AgentRouter`; `census.py` and its JSON output are session artefacts): 215 `NexusLoop(` sites without `production_mode=`: 197 in 92 files under `tests/`, 13 in 4 files under `applications/*/tests/` (product-path prefix), 3 in `testing_support/`, 2 in `debug_lab_nexus_loop.py` (A5); plus 7 `AgentRouter(` test calls in 5 files under `tests/` (the five of section 6 slice R1 and the sweep list) and 2 product `AgentRouter(` sites (`nexus_host_execution.py:57`, `graph_executor.py:258`; A3 and A4). The 97 `tests/` files that need an edit are the 92 `NexusLoop(` files plus the 5 `AgentRouter(` files without overlap; appendix A1 lists 100 `tests/` files because the census also finds three files whose calls already pass `production_mode=`: `test_gr10_r12_orchestration_continuation_architecture_gates.py` and `test_attempt_lifecycle_durability_gate.py` (13 `NexusLoop(production_mode=True)` sites, STRICT; see B4, C2 and R-8) and `test_ue_11f_observability_diagnostic_correlation.py` (2 sites with `production_mode=False`, no edit expected). Ten notebooks call `NexusLoop(registry)` (`agents/*/notebooks/01_*_experiment.ipynb@H:25`) and two usage docs show `NexusLoop(registry, ...)` (`applications/USAGE.md@H:199`, `intergrax/applications/USAGE.md@H:242`): same edit, listed in the write set. Every `AgentRouter(` test call gets `production_mode=False`. Contradicts WP-2's "14+" (FACT_CHECK C-02). |

### Part B — persistence posture (A4-037, A4-024; DP-24 hybrid)

Rule: in LAB (`production_mode=False`) nothing changes (env resolution and `None` stay documented lab behavior, DP-25). In `production_mode=True` the Tier-1 openers never read ambient env, and a missing store needs an explicit typed acknowledgement. This mirrors the existing precedent `disable_execution_continuation` + `validate_execution_continuation_for_composition` (`nexus_loop.py@H:280`, `durability_policy.py@H:27-56`).

| # | File | Change |
|---|---|---|
| B1 | `intergrax/runtime/task_memory/store.py` | `resolve_task_memory_persistence(..., production_mode: bool = False, disabled: bool = False)`: with `production_mode=True` the `ENV_TASK_MEMORY_DB` read (`@H:59`) is skipped; no `implementation` and no `db_path` raises new `TaskMemoryPersistenceNotConfiguredError` unless `disabled=True` (then `None`). LAB path byte-identical. |
| B2 | `intergrax/runtime/events/store.py` | Same shape for `resolve_runtime_event_persistence` (`@H:48-71`, env read `@H:69`), error `RuntimeEventPersistenceNotConfiguredError`. A bus that already carries persistence (`RuntimeEventBus.persistence`, `event_bus.py@H:187`) counts as configured (checked in B4). |
| B3 | `intergrax/runtime/human/persistence_errors.py` | Adds `HumanDecisionPersistenceNotConfiguredError` next to the existing errors (`@H:14-15,20,35`). |
| B4 | `intergrax/runtime/nexus/nexus_loop.py` | New keywords `disable_human_decision_persistence`, `disable_task_memory_persistence`, `disable_runtime_event_persistence` (all `bool = False`; precedent `disable_execution_continuation`, `@H:280`). In `__init__`: pass `production_mode`/`disabled` to B1/B2 (`@H:284-291`); after `self._human_store` is set (`@H:317`) a composition check refuses `production_mode=True` with no human store and no disable flag, and refuses an `InMemoryHumanDecisionPersistence` instance (`persistence_contract.py@H:58`) in STRICT; the event-bus check uses `self._event_bus.persistence`. `persist_human_decision` itself stays tolerant (`human_response.py@H:91-92` unchanged, J-3): the guard is at composition, where the posture is known. **Consistency rule (acknowledgement versus wired HITL tools):** the acknowledgement and wired HITL tools are mutually exclusive. If the effective tool wiring includes the HITL bundle (`HITL_BUNDLE_ID`, registered at `intergrax/tools/registry/shipped_plugins.py@H:73`; tools `hitl.*`, `intergrax/tools/providers/hitl/service.py@H:29-33`, which read `ToolWiringContext.human_decision_store`, `@H:36-42`, binding field `intergrax/tools/registry/wiring.py@H:113`), `build_harness_host_runtime` refuses `disable_human_decision_persistence=True`; and when a store is injected, the same instance is handed to the tool wiring context (carrier: `intergrax/applications/_shared/tool_wiring.py`, whose `build_application_tool_wiring` accepts a `wiring_context`, `@H:64-69`; this is the only carrier file in the write set. The three sibling modules that copy the same field, `integration_tool_wiring.py@H:54`, `notify_tool_wiring.py@H:44`, `session_tool_wiring.py@H:44`, are NOT in the write set; if implementation shows that any file besides `tool_wiring.py` must change, that is a write-set extension that needs an amended plan, a new digest and the Operator's decision, not a silent edit). The exact predicate that reads the bundle activation from the profile is located at implementation and pinned by NT-FC1-26. A host or worker that runs without wired HITL tools is the only case the acknowledgement covers; it remains a declared exception whose freeze-document statement is outside this ticket (S-04); the owner of the rule in code is the Execution Engine owner zone (WP-2 section 5, HARNESS/EBH-4 line). **Hosts that wire the HITL bundle (read-only `git grep` at `H`, FACT_CHECK F-48):** the bundle is a shipped tool plugin registered for every host that calls `build_application_tool_wiring` (`tool_wiring.py@H:91-99`, `register_shipped=True`); tools are exposed only when a host's `ToolProfile.enabled` names `hitl.*` ids. No application host names a `hitl.*` id in code: `attestation_demo`, `dispute_sim_application`, `governed_contractor_application`, `lab_application`, `poc_template_application` and the two scaffold templates build fixed lists without it. UNVERIFIED (not determinable statically): `legal_application`, `local_workspace_application` and `research_application`, whose `ToolProfile.enabled` comes from runtime settings (`settings.enabled_tool_ids`, `tool_wiring.py@H:42,90-91,24`), so a deployment could enable `hitl.*`; and `intergrax_assistant_application`, whose tool wiring was not read. The lab, attestation and assistant hosts use `DebugHitlResumeService` (`applications/*/host/factory.py`), a separate HITL consumer that is not the tool bundle. The rule is therefore a runtime refusal, not a host list. |
| B5 | `intergrax/runtime/execution/host_orchestration_loop_init_spec.py`, `environment_orchestration_materialization.py`, `host_orchestration_environment_spec_builder.py`, `intergrax/applications/_shared/host_orchestration_backend_spec_builder.py` | The spec gains `human_decision_store: HumanDecisionPersistence | None = None` and the three `disable_*` flags (kw-only, default False). `materialize_host_orchestration_backend` forwards them to `NexusLoop(...)` (`@H:63-105`; today none of the three is forwarded). The two builders carry them from the harness to the spec: the Tier-3 function `build_host_orchestration_loop_init_spec_from_environment(registry, env=...)` (`host_orchestration_backend_spec_builder.py@T:103-209`, called by `harness_host_runtime.py@T:594`) takes the four new keyword arguments and passes them to the Tier-1 function `build_host_orchestration_loop_init_spec(environment=...)` (`host_orchestration_environment_spec_builder.py@T:94-239`), which sets them on the spec (`@T:194`). The Tier-3 file was created by PT-WA and is a firm path of this ticket from this version (section 18, difference 3). |
| B6 | `intergrax/runtime/execution/host_orchestration_environment_spec_builder.py`, `worker_host_task_execution_composition.py`, conditional forwarders `intergrax/runtime/task/nexus_worker_execution.py@H:108-143`, `worker_bootstrap.py@H:92-104` | The Tier-1 builder (B5) and the worker composition (`@H:80-91`) accept and forward the new arguments. Production workers (`NexusWorkerRuntime.from_registry(production_mode=...)`) build a STRICT `NexusLoop` with no persistence arguments today, so they will be refused until they pass a store or an acknowledgement: this is the largest blast-radius item of Part B (risk R-3, decided D-FC1-03). The forwarders are edited only if the kwargs cannot be carried otherwise. |
| B7 | `intergrax/applications/_shared/harness_host_runtime.py` | `build_harness_host_runtime` gains `human_decision_store` and `disable_human_decision_persistence` (default `None`/`False`) and forwards them to the Tier-3 spec builder (`@T:594`, B5); `production_mode` there is already derived (`@H:354`). When `production_mode` and the profile disables task memory, the harness passes `disable_task_memory_persistence=True` (J-4). The existing STRICT governance-evidence requirement (`@H:532-540`) is left as is. Ten application call sites (eight host factories, `local_workspace_application/host/host_runtime_composition.py:143`, `local_workspace_application/model_runtime_proof/runtime.py:146`) and four call sites in `intergrax/` (`harness/app.py:170`, `scaffold/new_application.py:513,646`, `scaffold/new_application_product.py:571`) call this function (`git grep` at `H`, FACT_CHECK F-31); STRICT-capable call sites follow D-FC1-03: a host that has HITL passes `human_decision_store=create_sqlite_human_decision_store(...)` (`sqlite_composition.py@H:170-180`, the existing SQLite persistence composition); a host or worker without HITL passes `disable_human_decision_persistence=True`. The per-site choice is recorded in the ticket checkpoint; the edited call sites are listed as conditional product paths in section 4 (only call sites that can run STRICT are edited). |
| B8 | `intergrax/applications/_shared/task_memory_wiring.py` | `wire_task_memory_from_profile` passes `enabled=False` (not `None`) when the profile disables every memory kind and the host is STRICT/PRODUCT (`@H:64-70` today maps "not enabled" to `None`, which `wire_task_memory` resolves from env, `@H:44-47`). Is intended to close the "ambient env must not override a profile that disables task memory" leg (V4 / CK-08); that is established only by NT-FC1-19, probe V4 and the independent audit. Delta against the master-plan row (not listed there). |
| B9 | `intergrax/applications/_shared/scenario_runtime_baseline.py` (conditional) | The scenario baseline builds a spec with the Tier-3 builder (`@T:507`) and materializes it through `compose_application_host_orchestration_session` (`@T:536`); its STRICT branches (`@T:255,292,313`) show that it can run STRICT. It is edited only to pass the store or the acknowledgement of D-FC1-03 if step 0 shows a STRICT path through it. It is not in the first version of this plan and not in appendix A1 (section 18, difference 4). |

### Part C — probe-triggered parts (V2 and V6 are CONFIRMED at `B`, so both parts are firm; step 0 replays both probes at the train head)

| # | Trigger | Change |
|---|---|---|
| C1 | V2 CONFIRMED at `B`: a STRICT-reachable Tier-2 `build_context` builds its `RuntimeConfig` with `production_mode=False` | `build_context` derives `production_mode` from the request or host posture instead of a literal or the lab constant, in four files: `intergrax/agents/defaults.py@T:13-15` (the first version wrote `agents/defaults.py`, which does not exist at `H` or at `T`), `agents/organization_worker/organization_worker_agent.py@T:110-116`, `agents/external_contractor_adapter/external_contractor_adapter_agent.py@T:107-114`, `intergrax/agents/authoring/acp_stub_reflex.py@T:59-65`. The carrier of the posture is a request field or an injected host configuration that `build_context(request: RuntimeRequest)` can already read; it is fixed at step 0 and pinned by NT-FC1-24 and NT-BND-05-1. If no carrier exists without editing a contract file outside the write set, STOP: that is a write-set extension that needs an amended plan, a new digest and the Operator's decision. Then `test_harness_production_mode_is_false_for_lab` (`tests/unit/applications/test_harness_runtime_defaults.py@H:12`) is revisited. `OrganizationWorkerAgent` is lab-only (registered by `intergrax/dev_support`) and `acp_stub_reflex.py` is a reference stub; both are edited so that NT-BND-05-1 holds without an allowlist. |
| C2 | V6 CONFIRMED at `B`: a STRICT host composition can reach `UAEPExecutor` without a context engine or LLM adapter | `NexusLoop.__init__` refuses `production_mode=True` when `self._context_manager.context_engine` or `.llm_adapter` is `None` (they are passed to `UAEPExecutor` at `nexus_loop.py@H:340-352`; `uaep_executor.py@H:386` assembles only when both are set), or the existing CE guard (`tool_runtime.py@H:296-299`) is called at that composition point. The refusal is a typed error defined in an existing module of the write set (no new product file; the module is fixed at step 0). The V6 probe of B-1 alpha called the spec-builder function at its first location; at `T` that function is the Tier-3 one (section 17), so the replay must call it there. LAB (`production_mode=False`) is unchanged. |

If the replay at the train head of V2 or V6 is negative, the matching part is not implemented and the ticket records the replay as the closure evidence for that finding; that only shrinks the write set and turns NT-FC1-24 or NT-FC1-25 into a control.

### Not changed (decided, with reason)

- `human_response.py` (listed in WP-2): the leaf stays tolerant, the composition guard (B4) is the fail-closed point (J-3).
- `AgentEngine`/`UAEPExecutor` module-level defaults (`agent_engine.py@H:35,166`) are not edited (C2 refuses at the composition point).
- Other `production_mode=False` literals found while reading (`context_manager.py@H:239`, `uaep_assemble.py@H:66`, planner `list_routable_agent_ids(production_mode=False)` at `nexus_llm_plan_builder.py@H:31` and `nexus_plan_bridge.py@H:101`): not part of any named finding, left unchanged (decided D-FC1-05); they are registered as a new Campaign B observation (FACT_CHECK F-36). `context_manager.py@H:235-239` is pinned by line in `tests/qualification/trace_x/_trace_x_p4_production_composition_registry.py`.
- No document-store-backed `HumanDecisionPersistence` is added (a new product file; not chosen in D-FC1-03).

## 4. Exact write set

Counts parsed from the lists of this section: product paths firm 35 (none new; 19 source files, 12 documents and notebooks, 4 tests under `applications/`); product paths conditional 17; test and support paths firm 109 (6 new, 100 sweep files under `tests/`, 3 under `testing_support/`); test paths conditional 7; documents firm 4.

Product paths (prefix `intergrax/`, `agents/`, `applications/`; each one is ACT-N11 sensitive):

Firm, source files (19; edited, none new):

```text
intergrax/runtime/nexus/agent_router.py
intergrax/runtime/nexus/nexus_loop.py
intergrax/runtime/nexus/execution/graph_executor.py
intergrax/runtime/execution/nexus_host_execution.py
intergrax/runtime/execution/debug_lab_nexus_loop.py
intergrax/runtime/execution/host_orchestration_loop_init_spec.py
intergrax/runtime/execution/environment_orchestration_materialization.py
intergrax/runtime/execution/host_orchestration_environment_spec_builder.py
intergrax/runtime/execution/worker_host_task_execution_composition.py
intergrax/runtime/task_memory/store.py
intergrax/runtime/events/store.py
intergrax/runtime/human/persistence_errors.py
intergrax/applications/_shared/harness_host_runtime.py
intergrax/applications/_shared/task_memory_wiring.py
intergrax/applications/_shared/host_orchestration_backend_spec_builder.py
intergrax/agents/defaults.py
agents/organization_worker/organization_worker_agent.py
agents/external_contractor_adapter/external_contractor_adapter_agent.py
intergrax/agents/authoring/acp_stub_reflex.py
```

Firm, product-path documents and notebooks (sweep of the explicit-posture keyword, no logic) (12):

```text
applications/USAGE.md
intergrax/applications/USAGE.md
agents/dispute_analyst/notebooks/01_dispute_analyst_experiment.ipynb
agents/dispute_intake/notebooks/01_dispute_intake_experiment.ipynb
agents/dispute_scenario/notebooks/01_dispute_scenario_experiment.ipynb
agents/dispute_strategist/notebooks/01_dispute_strategist_experiment.ipynb
agents/intergrax_assistant/notebooks/01_intergrax_assistant_experiment.ipynb
agents/legal/notebooks/01_legal_experiment.ipynb
agents/local_indexer/notebooks/01_local_indexer_experiment.ipynb
agents/local_search/notebooks/01_local_search_experiment.ipynb
agents/local_synthesizer/notebooks/01_local_synthesizer_experiment.ipynb
agents/signoff_probe/notebooks/01_signoff_probe_experiment.ipynb
```

Firm, product-path tests (prefix `applications/`) (4; sites: 5, 1, 1, 6):

```text
applications/governed_contractor_application/tests/host/test_governed_contractor_canonical_execution.py
applications/legal_application/tests/serving/test_legal_fastapi_chat.py
applications/legal_application/tests/serving/test_legal_serving_identity.py
applications/local_workspace_application/tests/host/test_lkw_canonical_execution.py
```

Conditional product paths (17): the two B6 forwarders, `tool_wiring.py` (carrier of the injected store to the tool wiring context, consistency rule in B4), `scenario_runtime_baseline.py` (B9) and the 13 files that call `build_harness_host_runtime` (the first version counted 14 call sites; four of them are in three files, so the distinct files are 10 under `applications/` and 3 under `intergrax/`; each is edited only to pass the store or the acknowledgement of D-FC1-03, only if it can run STRICT; the per-site choice is recorded in the ticket checkpoint):

```text
intergrax/runtime/task/nexus_worker_execution.py
intergrax/runtime/task/worker_bootstrap.py
intergrax/applications/_shared/tool_wiring.py
intergrax/applications/_shared/scenario_runtime_baseline.py
applications/attestation_demo/host/factory.py
applications/dispute_sim_application/host/factory.py
applications/governed_contractor_application/host/factory.py
applications/intergrax_assistant_application/host/factory.py
applications/lab_application/host/factory.py
applications/legal_application/host/factory.py
applications/poc_template_application/host/factory.py
applications/research_application/host/factory.py
applications/local_workspace_application/host/host_runtime_composition.py
applications/local_workspace_application/model_runtime_proof/runtime.py
intergrax/harness/app.py
intergrax/scaffold/new_application.py
intergrax/scaffold/new_application_product.py
```

Tests and support outside the product prefixes (not ACT-N11 paths, but outside the EBH-7 identity allowlist, so they merge before the final pin).

New test files, firm (6; named in section 5; the sixth is the construction gate of RW-PLAN row NT-ROW-031):

```text
tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py
tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py
tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py
tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py
tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py
tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py
```

Sweep files under `tests/`, firm (100; the AST census at `T`, section 3 A7: 97 files that need the keyword edit plus the three that already pass `production_mode=` and are affected by B4 and C2, namely `test_gr10_r12_orchestration_continuation_architecture_gates.py`, `test_ue_11f_observability_diagnostic_correlation.py` and `test_attempt_lifecycle_durability_gate.py`). The seven `AgentRouter(` files are among them: `test_ebh_2d_c_r4_canonical_runtime_agent_port.py`, `test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py`, `test_agent_routing_policy.py`, `test_observability_emission_coverage.py`, `test_capability_routing_acp_con6.py`:

```text
tests/acceptance/agent_os/conftest.py
tests/acceptance/agent_os/test_acp_pattern_agents.py
tests/acceptance/agent_os/test_agent_os_scenarios.py
tests/integration/applications/architecture/test_harden_1d_durable_problem_store_failure_semantics.py
tests/integration/applications/test_unified_execution_entry_j2.py
tests/integration/applications/test_unified_execution_entry_j3.py
tests/integration/debug/test_debug_api_injected_trace_store_b09.py
tests/integration/eval/test_nexus_eval_runner.py
tests/integration/runtime/adaptive/test_nexus_adaptive_signal.py
tests/integration/runtime/long_running/test_long_running_scheduler_j4.py
tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py
tests/integration/runtime/test_graph_cancellation.py
tests/integration/runtime/test_graph_failure_recovery.py
tests/integration/runtime/test_harness_hitl_pagerduty.py
tests/integration/runtime/test_nexus_loop_echo.py
tests/integration/runtime/test_nexus_loop_governance.py
tests/integration/runtime/test_nexus_loop_guardrail_wiring.py
tests/integration/runtime/test_nexus_loop_hitl_hooks.py
tests/integration/runtime/test_nexus_loop_human_request_v2.py
tests/integration/runtime/test_nexus_loop_long_running.py
tests/integration/runtime/test_nexus_loop_research.py
tests/integration/runtime/test_nexus_loop_runtime_event_persistence.py
tests/integration/runtime/test_nexus_loop_security_wiring.py
tests/integration/runtime/test_observability_wiring.py
tests/integration/runtime/test_planning_decision_record_gate.py
tests/integration/runtime/test_platform_wiring_observability.py
tests/integration/runtime/test_sandbox_uaep.py
tests/integration/runtime/test_shadow_workspace_uaep.py
tests/integration/runtime/test_terminal_diagnostic_production_e2e.py
tests/qualification/bg_01/test_bg_01_gates.py
tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py
tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py
tests/qualification/host_01/test_host_01_gates.py
tests/qualification/mcp_server_x/reference_app/http_serve.py
tests/qualification/mcp_server_x/test_mcp_server_x_p6_reference_l3_qualification.py
tests/qualification/mcp_server_x/test_mcp_server_x_p7_failure_injection_l2.py
tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_composition_exposure.py
tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py
tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py
tests/unit/applications/test_application_security_wiring.py
tests/unit/applications/test_harness_guardrail_wiring.py
tests/unit/applications/test_http_canonical_execution.py
tests/unit/applications/test_mcp_canonical_execution.py
tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py
tests/unit/applications/test_mcp_host_mount.py
tests/unit/applications/test_mcp_server_engine.py
tests/unit/applications/test_mcp_server_p4_lifecycle.py
tests/unit/applications/test_reference_apps_mcp.py
tests/unit/architecture/test_ebh_2d_c_r4_canonical_runtime_agent_port.py
tests/unit/architecture/test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py
tests/unit/architecture/test_ebh_4_r1_r3_b6_r2_runtime_conformance.py
tests/unit/contracts/test_execution_identity.py
tests/unit/debug/test_g6_api.py
tests/unit/debug/test_h3_interaction_intake.py
tests/unit/debug/test_h5_teams_interaction_intake.py
tests/unit/runtime/architecture/test_agent_routing_policy.py
tests/unit/runtime/architecture/test_gr10_r12_orchestration_continuation_architecture_gates.py
tests/unit/runtime/architecture/test_gr10_r13_r2_production_reliability_composition_adoption.py
tests/unit/runtime/architecture/test_gr10_r9_r3_canonical_topology_slot_mse.py
tests/unit/runtime/architecture/test_gr10_r9_r4_topology_mse_composition_mandatory.py
tests/unit/runtime/architecture/test_gr10_r9_r5_topology_factory_fail_closed.py
tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py
tests/unit/runtime/architecture/test_npsc5d_r3_final_qualification.py
tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py
tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py
tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py
tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py
tests/unit/runtime/diagnostics/test_ue_11f_observability_diagnostic_correlation.py
tests/unit/runtime/events/test_observability_emission_coverage.py
tests/unit/runtime/execution/budget/test_ue_8b1r1_ledger_lifecycle.py
tests/unit/runtime/execution/budget/test_ue_8b1r2_preserve_run_budget_through_nexus_entry.py
tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py
tests/unit/runtime/execution/lineage/test_host_task_lineage_wiring.py
tests/unit/runtime/execution/lineage/test_host_task_resume_lineage_identity.py
tests/unit/runtime/execution/lineage/test_terminal_conflict_seal.py
tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py
tests/unit/runtime/execution/test_decision_optionality.py
tests/unit/runtime/execution/test_decision_orchestration_recovery.py
tests/unit/runtime/execution/test_execution_runtime.py
tests/unit/runtime/execution/test_host_task_terminal_publisher.py
tests/unit/runtime/execution/test_orchestration.py
tests/unit/runtime/execution/test_orchestration_topology_e2e_proof.py
tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py
tests/unit/runtime/execution/test_ue_11e_resume_recovery.py
tests/unit/runtime/governance/test_g3b_governance_coverage.py
tests/unit/runtime/governance/test_gov_x2_r3_post_run_tenant_contract.py
tests/unit/runtime/hooks/test_hook_parity_b06.py
tests/unit/runtime/human/test_human_decisions.py
tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py
tests/unit/runtime/long_running/test_runtime_checkpoint.py
tests/unit/runtime/nexus/test_handle_task_execution_identity.py
tests/unit/runtime/nexus/test_nexus_loop_finalization_hook_failure.py
tests/unit/runtime/nexus/test_nexus_loop_helpers.py
tests/unit/runtime/nexus/test_nexus_runtime_event_metric_scope_r3.py
tests/unit/runtime/nexus/test_nexus_runtime_event_metric_scope_r4.py
tests/unit/runtime/registry/test_agent_registry_read_boundary.py
tests/unit/runtime/registry/test_capability_routing_acp_con6.py
tests/unit/runtime/security/test_sec_ent.py
tests/unit/runtime/security/test_sec_planes.py
tests/unit/runtime/task/test_host_task_execution_run_adapter.py
```

Support modules, firm (3):

```text
testing_support/agent_distribution/dg001_canonical_multi_agent_diagnostic_harness.py
testing_support/agent_distribution/multi_agent_coordination_qualification_harness.py
testing_support/obs_universal_spine/diagnostic_execution_stack.py
```

Conditional test paths (7; each is edited only if step 0 shows that a source-token or call-shape assertion in it sees the FC-1 edit):

```text
tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py
tests/qualification/pdc_x/d7/test_d7_recurrence.py
tests/qualification/sched_01/test_sched_01_gates.py
tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py
tests/unit/applications/architecture/test_diag_platform_adoption_gate.py
tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py
tests/unit/scaffold/test_scaffold_harness_alignment.py
```

Documents, firm (4; target-behavior sentences land with the code, WP-2 section 6; each is subject to `scripts/docs/check_docs_domain_pairs.py` and the EBH-7 allowlist question; `PLATFORM_CONFIGURATION.md` defaults belong to the DP-25 doc batch, not here):

```text
docs/project/architecture/NEXUS_EXECUTION_FLOW.md
docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md
docs/project/architecture/MEMORY.md
docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md
```

Write-set growth, stated plainly: the master-plan row PT-FC1 listed 7 paths; the first version of this plan had 14 firm source files, 12 sweep documents/notebooks, 4 `applications/*/tests` files and up to 21 conditional product paths (2 forwarders, 4 agent files, up to 14 `build_harness_host_runtime` call sites, `tool_wiring.py`). This version has 19 firm source files (the Tier-3 spec builder that PT-WA created, and the four files of Part C, which are firm because V2 is CONFIRMED), the same 12 + 4, and up to 17 conditional product paths (the four agent files moved to firm; `scenario_runtime_baseline.py` is added; the 14 call sites are 13 distinct files). The conditional count (17) is within the ceiling of up to 21 conditional product paths in item 2 of the published transcription; the firm product count rises from 30 to 35, which that record does not mention (section 18, differences 3 and 5). None of the conditional paths is applied unless its trigger holds.

Forbidden for this ticket: everything not listed above, in particular `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, `governance/` except this ticket and plan, the roadmap and the FRZ checklist, `IIAC_*`, PDC-X record directories, `docs/project/maintainers/qualification/`, `intergrax/integrations/**`, the files of the other fail-closed batches (`intergrax/skills/**`, `intergrax/distributed/**`, `rag/embedding/**`, `runtime/notifications/**`, `runtime/interactions/**`, `runtime/capacity/**`, `applications/_shared/{production_capacity,queue_worker,memory,runtime_event_delivery,sandbox,interaction}_wiring.py`), and any new file under a product prefix.

## 5. Negative tests

Each runs red at the unmodified wave baseline and green after the fix; each configuration is a PRODUCT/STRICT (or LAB-default) configuration that must now be refused. Numbering is stable; the six new files are named in section 4. Parametrized variants count as one named test.

| ID | File | Configuration | Expected |
|---|---|---|---|
| NT-FC1-01 | `test_fc1_host_task_routing_posture.py::test_host_task_rejects_non_production_state_agent_in_production_loop` | `NexusLoop(registry, production_mode=True)`; `build_host_task_execution(loop, ...)`; registry holds one agent in lifecycle `DEVELOPMENT` that matches the task; task `agent_id=""` | `execute` raises (no routable agent) and the agent is not run; at baseline it routes |
| NT-FC1-02 | same file `::test_host_task_rejects_production_eligible_agent_without_owner_or_runbook` | as above, agent `PRODUCTION`, `production_eligible=True`, owner/runbook missing | refused: `AgentRoutingError` (a `RuntimeError`); the agent is not run. The policy reasons (`agent_routing_policy.py@H:51-57`) are not propagated on the `agent_id=""` route (the router raises `AgentRoutingError` / `AgentRegistry has no routable agents.`, `agent_router.py@H:134-135`); the reason strings stay asserted by the existing router-level test `test_production_mode_requires_owner_for_production_eligible_agent` |
| NT-FC1-03 | same file `::test_host_task_routes_production_agent_with_owner_and_runbook` | as above with complete metadata | routes (control, green before and after) |
| NT-FC1-04 | same file `::test_host_task_execution_uses_the_loops_router` | any `NexusLoop`; `build_host_task_execution(loop)` | the execution's router `is` `loop.agent_router`, and its posture equals `loop.production_mode` |
| NT-FC1-05 | `test_fc1_agent_router_posture_and_fallback.py::test_agent_router_requires_explicit_production_mode` | `AgentRouter(registry)` | `TypeError` |
| NT-FC1-06 | same file `::test_nexus_loop_requires_explicit_production_mode` | `NexusLoop(registry)` | `TypeError` |
| NT-FC1-07 | same file `::test_strict_router_refuses_first_routable_fallback` | `production_mode=True`, task without capability, no best match, routable agents present | `AgentRoutingError`; no `AGENT_SELECTED` event recorded |
| NT-FC1-08 | same file `::test_lab_router_without_opt_in_refuses_fallback` | `production_mode=False`, `lab_first_routable_fallback` not set, same task | `AgentRoutingError` |
| NT-FC1-09 | same file `::test_lab_opt_in_keeps_fallback_and_emits_fallback_used` | `production_mode=False`, `lab_first_routable_fallback=True`, same task | first routable agent selected, event has `fallback_used=True`, `selection_reason="fallback_first_routable"` |
| NT-FC1-10 | same file `::test_strict_router_with_lab_fallback_opt_in_is_refused_at_construction` | `AgentRouter(registry, production_mode=True, lab_first_routable_fallback=True)` | `ValueError` |
| NT-FC1-11 | same file `::test_explicit_agent_id_and_capability_paths_unchanged` | explicit `agent_id`; capability route | unchanged selection and events (control) |
| NT-FC1-12 | `test_fc1_router_and_loop_construction_gate.py::test_no_product_construction_without_explicit_posture` | AST scan of non-test code under `intergrax/`, `agents/`, `applications/` | every `AgentRouter(`/`NexusLoop(` call has `production_mode=`; every `GraphExecutor(` passes `router=` (only `nexus_loop.py` constructs it); only the allowed `AgentRouter(` construction sites remain (`nexus_loop.py`, `graph_executor.py` default); a rogue fixture module with `AgentRouter(registry)` is flagged (self-test of the gate, pattern of `testing_support/architecture/obs_diag_x3_ast_gates.py@T:110-120`; PT-WA moved that module out of the product tree) |
| NT-FC1-13 | `test_fc1_nexus_loop_persistence_posture.py::test_strict_loop_refuses_missing_human_decision_store` | `NexusLoop(registry, production_mode=True, ...)` with the other stores satisfied, no `human_decision_store`, no ack | `HumanDecisionPersistenceNotConfiguredError` |
| NT-FC1-14 | same file `::test_strict_loop_accepts_explicit_human_persistence_disable` | as NT-FC1-13 with `disable_human_decision_persistence=True` | constructs (acknowledgement path is explicit, typed) |
| NT-FC1-15 | same file `::test_strict_loop_refuses_in_memory_human_store` | `human_decision_store=InMemoryHumanDecisionPersistence()` | refused |
| NT-FC1-16 | same file `::test_strict_loop_ignores_ambient_task_memory_env` | `production_mode=True`, `INTERGRAX_TASK_MEMORY_DB=<tmp path>`, no store, no path | `TaskMemoryPersistenceNotConfiguredError`; the file at the env path is not created; with `disable_task_memory_persistence=True` the store is `None` and still no file |
| NT-FC1-17 | same file `::test_strict_loop_ignores_ambient_runtime_events_env` | same for `INTERGRAX_RUNTIME_EVENTS_DB` | `RuntimeEventPersistenceNotConfiguredError`, no file; a bus that already persists is accepted |
| NT-FC1-18 | same file `::test_lab_loop_keeps_env_resolution_and_none_default` | `production_mode=False` | behavior unchanged (control; the existing `test_resolve_*_disabled_by_default` tests stay green) |
| NT-FC1-19 | `test_fc1_harness_strict_persistence_wiring.py::test_profile_disabling_task_memory_is_not_reenabled_by_env` | STRICT environment profile with all memory flags off, env var set | `wire_task_memory_from_profile` returns `store=None`, `db_path=None`, no file (V4/CK-08) |
| NT-FC1-20 | same file `::test_strict_harness_runtime_refuses_missing_human_store_without_ack` | `build_harness_host_runtime` with a STRICT profile, no store, no ack | refused before the loop is returned |
| NT-FC1-21 | same file `::test_strict_worker_composition_refuses_missing_persistence` | `build_worker_host_task_execution_from_registry(production_mode=True, ...)` without stores/acks | refused |
| NT-FC1-22 | same file `::test_reject_and_escalate_verdicts_are_recorded_through_materialization` | scenarios of `test_nexus_loop_fails_on_human_rejection` / `test_nexus_loop_escalates_and_persists` (`test_nexus_loop_governance.py@H:126,196`) run through `materialize_host_orchestration_backend` with a recording store port (V5/NT-05) | one durable decision record per verdict |
| NT-FC1-23 | `test_fc1_nexus_loop_persistence_posture.py::test_spec_forwards_persistence_arguments` | `HostOrchestrationLoopInitSpec` with each new field; the same four arguments passed to the Tier-3 builder `build_host_orchestration_loop_init_spec_from_environment` | `NexusLoop` receives each (guards the forwarding in B5 through both builders and the materialization) |
| NT-FC1-26 | `test_fc1_harness_strict_persistence_wiring.py::test_acknowledgement_is_refused_when_hitl_tools_are_wired` | STRICT profile with the HITL tool bundle enabled and `disable_human_decision_persistence=True`; second case: store injected and the tool wiring context receives the same instance | first case refused before the loop is returned; second case: `ctx.human_decision_store is` the injected store |
| NT-FC1-24 (Part C1) | in `test_fc1_harness_strict_persistence_wiring.py` | STRICT governed-contractor composition, `agent.build_context(request)` for each of the three Tier-2 sites | `config.production_mode is True` |
| NT-FC1-25 (Part C2) | `test_fc1_nexus_loop_persistence_posture.py` | `production_mode=True`, `ContextManager` without engine/adapter, also through the harness path (Tier-3 builder) | refused; LAB unchanged |
| NT-BND-05-1 (RW-PLAN row NT-ROW-031, basis S) | `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_tier2_build_context_takes_posture_from_host_not_a_literal` | AST scan of `agents/` and `intergrax/agents/`: no `RuntimeConfig(` construction inside a `build_context` method whose `production_mode` is a literal or the `harness_production_mode()` constant; rogue-fixture self-test (pattern of NT-FC1-12); no allowlist | red at the train base on the three sites of C1, green after C1. The RW-PLAN row words the rule as "a Tier-2 agent that constructs RuntimeContext outside the sanctioned composition root fails" and does not name the root; this plan reads it as the posture rule above (section 18, difference 6); the exact scan rule is fixed at step 0 |

Totals: 26 FC-1 tests (NT-FC1-01..26) plus NT-BND-05-1 = 27 tests in six new files. NT-FC1-03, -11 and -18 are controls that are green before and after; the other 24 are expected red on the unmodified code (17 refusals: NT-FC1-01, 02, 05, 06, 07, 08, 10, 12, 13, 15, 16, 17, 19, 20, 21, 25, 26; plus 7 new-behavior or structural checks: NT-FC1-04, 09, 14, 22, 23, 24 and NT-BND-05-1). The red-first run (section 7 step 2) records which of them actually fail; a negative replay of V2 or V6 at the train head turns NT-FC1-24 or NT-FC1-25 into a control.

## 6. Regression scope (every node below was checked to exist at `H` by `git ls-tree` / `git grep`; none has been run)

Defect-pinning tests that stay valid because LAB behavior is unchanged (a change that turns any of them red must be explained, not edited away): `tests/unit/runtime/task_memory/test_task_memory_store.py::test_resolve_task_memory_persistence_disabled_by_default`, `tests/unit/runtime/events/test_runtime_event_persistence.py::test_resolve_runtime_event_persistence_disabled_by_default`, `tests/unit/runtime/nexus/test_nexus_loop_helpers.py::test_persist_human_decision_no_store`, `tests/unit/applications/test_harness_runtime_defaults.py::test_harness_production_mode_is_false_for_lab` (revisited only if C1 runs).

| Slice | Node IDs / files |
|---|---|
| R1 routing | `tests/unit/runtime/architecture/test_agent_routing_policy.py::{test_retired_agent_is_not_routable, test_production_mode_requires_owner_for_production_eligible_agent, test_agent_router_skips_deprecated_agents, test_agent_router_blocks_explicit_retired_agent}`; `tests/unit/architecture/test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py` (11 tests: `test_agent_router_has_no_can_handle_scoring_loop`, `test_router_does_not_reinvoke_can_handle_for_telemetry`, ...); `tests/unit/architecture/test_ebh_2d_c_r4_canonical_runtime_agent_port.py` (7 tests incl. `test_agent_router_has_no_concrete_agent_import`, `test_capability_routing_fail_closed_without_routable_contract`); `tests/unit/runtime/registry/test_capability_routing_acp_con6.py::{test_select_best_capability_match_prefers_higher_score, test_agent_router_rejects_class_name_in_task_metadata}`; `tests/unit/runtime/events/test_observability_emission_coverage.py::{test_agent_router_emits_agent_selected, test_agent_router_source_records_agent_selected_constant}`; `scripts/maintenance/check_capability_routing.py` |
| R2 host task | `tests/unit/runtime/execution/test_host_governance_admission_composition.py` (3), `tests/unit/runtime/execution/lineage/test_host_task_lineage_wiring.py::{test_build_host_task_execution_wires_lineage_persistence, test_host_task_root_admission_exists_before_delegate}`, `tests/unit/runtime/execution/test_host_task_terminal_publisher.py::{test_nexus_adapter_delegates_terminal_publication, test_nexus_adapter_is_typed_port_implementation}`, `tests/unit/runtime/architecture/test_host_governance_admission_wiring_architecture_gates.py::{test_nexus_host_execution_requires_explicit_admit_root_governance_identity, test_nexus_worker_from_registry_requires_explicit_governance_admission}`, `tests/unit/runtime/execution/test_agent_executor.py::test_engine_exception_propagates_unchanged`, `tests/unit/scaffold/test_npsc2_canonical_scaffold_execution.py`, `tests/fixtures/harness_host_task_execution.py` (fixture used by many) |
| R3 loop, HITL, stores | `tests/integration/runtime/test_nexus_loop_governance.py::{test_nexus_loop_pauses_for_human_request, test_nexus_loop_resumes_after_human_approval, test_nexus_loop_fails_on_human_rejection, test_nexus_loop_escalates_and_persists}`; `tests/unit/runtime/human/test_human_decisions.py::{test_human_decision_store_records_and_lists, test_persist_human_decision_uses_generic_persistence, test_nexus_loop_accepts_human_decision_persistence}`; `tests/unit/runtime/nexus/test_nexus_loop_helpers.py` (5 tests); `tests/unit/runtime/task_memory/test_task_memory_store.py` and `tests/unit/runtime/events/test_runtime_event_persistence.py` (all nodes); `tests/unit/runtime/nexus/test_nexus_loop_finalization_hook_failure.py` |
| R4 gates naming the files | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py`, `tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py`, `tests/qualification/dup_x/test_dup_x_npsc_boundary_gates.py`, `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py`, `tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py`, `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py`, `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py`, `tests/qualification/state_x/test_state_x_final_r1_closed_world.py`, `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py`, `tests/qualification/harness_01/test_harness_01_gates.py`, `tests/qualification/config_x/test_config_x_qualification_gates.py` (the STATE-X explicit classification names `events/store.py`, `_state_x_explicit_mechanism_classifications.py@H:2424`; `state_x/inventory.py@H:960,1039,1107,1167` name `nexus_loop.py` and `human_response.py`) |
| R5 sweep proof | exactly the 104 test files of the sweep proof, listed in `R5_SWEEP_FILES.txt` (session artefact, regenerated from the census at `T`): 100 under `tests/` (62 unit, 26 integration, 9 qualification, 3 acceptance; 97 edited by the sweep plus the three that already pass `production_mode=` and are affected by B4 and C2) and 4 under `applications/*/tests/`; run as exactly 5 commands (one per group, explicit file lists, no directory globs, no `-k`), one at a time under the lock, in the order unit, integration, acceptance, qualification, applications; a command that runs longer than 15 minutes is aborted and split into two explicit file lists (the split is recorded); authorised by AR-FC1-03b. Run once after the whole sweep and once at the end of the ticket (10 commands in total at most), never after every step. |
| R7 train L1 | the L1 commands of RW-PLAN v13 section e run on the PR and are recorded in the PR body (`check_intergrax_no_applications_imports.py`, `check_agent_registry_bypass.py`, `check_agents_no_tier3_imports.py`, `check_agents_no_vendor_sdk_imports.py`, `check_integration_vendor_imports.py`, `check_dependency_ownership.py`, `igx_gov ticket`, `verify_arch_freeze_plan_traceability.py`, `arch_freeze_b1_closed_world_inventory.py --strict-certify`); expected-red entries are only those registered in RW-PLAN section c |
| R6 docs | `uv run python scripts/docs/check_docs_domain_pairs.py` |

The baseline state of every R-slice at `H` is UNKNOWN until the first run; the first run is red/green evidence at the wave baseline (section 7 step 1), not a verdict on this ticket.

## 7. Test commands (one process at a time; lock `D:/Projekty/_wt/_locks/pytest.lock` by `mkdir`, release by `rmdir`, per the session rules; logs under `.tmp/session/eqo-a-remediation/IMPL/FC-1/`, red-first logs also under `.tmp/session/arch-freeze/T-IGX-043-A/red-first/`)

1. Baseline evidence before any edit (detached checkout of the train base, RW-PLAN section d red-first rule): `uv run pytest <R1 files> <R2 files> <R3 files> -p no:cacheprovider -q` (one command, multiple paths).
2. Red-first: after writing only the new test files, `uv run pytest tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py -p no:cacheprovider -q` and keep the log: the refusal tests must fail there.
3. After Part A: R1 + R2 slices, one command. After Part B: R3 slice, one command. After the sweep (step 1 of section 8) and once at the end: the R5 sweep proof (5 commands, section 6); not after every step.
4. Gates: one command over the R4 files.
5. Static: `uv run ruff check <changed python files>` and `uv run pyright <changed product files>`; `uv run python scripts/docs/check_docs_domain_pairs.py`.
6. Mechanical check without pytest: `python -I` over `scan_call_sites.py` (kept next to this plan; read-only, writes only its TSV) must report 0 `NexusLoop(`/`AgentRouter(` calls without `production_mode=` in any file.
7. Broad or diagnostic suites are not part of this ticket; S1 diagnostics and the wave-level regression are scheduled by the wave plan (see `AUTHORIZATION_REQUESTS.md`).

## 8. Order of work inside the ticket

1. Tests-first, behavior-preserving sweep (A7): add explicit `production_mode=False` to every call site. It is valid against the unmodified code (the keyword exists), so it can be reviewed and merged first and it keeps the later diff small.
2. Add the new test files; run red-first (step 2 above).
3. Part A (A1..A6), then R1/R2.
4. Probes V1/V2/V4/V5/V6 are replayed at the train head (their results at `B` come from B-1 alpha; the V6 replay calls the Tier-3 spec builder): a negative replay of V2 or V6 removes the matching Part C1/C2; a positive one confirms it.
5. Part B (B1..B8), then R3.
6. Documents; static checks; R4 gates.
7. Hand over to the wave: the ticket never self-declares `ready_for_audit` PASS; the audit is node `PT-Q02` at the exact `RW` SHA.

Coordination on the train (RW-PLAN v13, lane 3; every train PR merges into `arch-freeze/rw-train` with a merge commit): PT-WA is merged (`T`), so the restructured `host_orchestration_environment_spec_builder.py`, `host_orchestration_loop_init_spec.py`, `harness_host_runtime.py` and the new Tier-3 `host_orchestration_backend_spec_builder.py` are the base this ticket edits (section 17); PT-WB and PT-FC2 merge before this ticket starts and step 0 re-reads the three hotspot files and the Tier-3 builder against the then-current train head. Order: FC-2, then FC-1, then FC-3 (M-1: on the up to seven `applications/*/host/factory.py` call sites FC-1 merges first and FC-3 rebases onto it; FC-1's edit there is only the added store/acknowledgement argument); PT-WF after FC-1 on `nexus_loop.py`; T-AF2-RW-TENANT after FC-1 on `harness_host_runtime.py`; the gate test `test_diag_platform_adoption_gate.py` (conditional for both) is edited by PT-FC2 first. (C-3) A4-024 appears in AR-40 (W-B) and in this ticket: decided D-FC1-01, FC-1 owns it, W-B neither relocates nor duplicates the two openers; the closed-list entry naming them as LAB-only openers is a seam-document item outside this ticket.

## 9. ACT-N11 / IIAC consequence

- Product paths per `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")` (`scripts/governance/iiac_common.py@H:22`): 19 firm source files, 12 sweep documents/notebooks, 4 test files under `applications/` = 35 firm product-path files, none new; up to 17 conditional more (2 forwarders, `tool_wiring.py`, `scenario_runtime_baseline.py`, 13 files that call `build_harness_host_runtime`). Tests under `tests/` and `testing_support/` are not product paths.
- Every product path above breaks the IIAC activation until the single RB-2 re-pin at the final train product tree; this ticket does not perform, delegate or prepare the re-pin. The ACT-N11 red of train PRs is the registered expected-red entry KR-08 (RW-PLAN section c, table 1). The activation record at `T` was not re-read (not verified here).
- New product files: none (no EBH-7/B6-F10 classifier entry for a new path; the edited paths are classified by the existing rules, verification by the pin session).
- The EBH-7 identity allowlist is not read here (UNVERIFIED): docs and `tests/` edits are treated as outside it and must precede the final pin (master plan W5).

## 10. ARCH-FREEZE impact and what closes each finding (no status is promoted)

| Finding | Class (source) | Closes when |
|---|---|---|
| A4-044 | BLOCKS (WP-2 provisional; dossier CONDITIONS; probe V1 may lower it) | NT-FC1-01..04, 12 green at `RW`, V1 recorded, independent audit at the exact SHA; touches FRZ-EXE-02 (PASS, arch), FRZ-PLG-05 (OPEN), FRZ-PRD-07/08 |
| A4-009 | CONDITIONS | NT-FC1-07..11 green, doc sentence landed, audit |
| A4-015 | UNKNOWN; BLOCKS (V2 CONFIRMED at `B`) | (a) NT-FC1-05/06/12 + spec change; (b) C1 + NT-FC1-24 + NT-BND-05-1, or a negative replay of V2 at the train head; FRZ-GOV-05, FRZ-BND-05, FRZ-OWN-03 |
| A4-037 | CONDITIONS; BLOCKS if V5 shows no other durable record | V5 recorded, NT-FC1-13..15, 20..23 and 26 green, audit; FRZ-GOV-06, FRZ-AUD-04 |
| A4-024 | CONDITIONS (WP-1 DECISION); X-A4-024 worst case BLOCKS (V4) | NT-FC1-16..19 green, V4 recorded, STATE-X/CONFIG-X closed-world inventories checked (R4) because the two openers stay LAB-only roots, audit; FRZ-CFG-02, FRZ-STA-06 |
| A4-031 | UNKNOWN; V6 CONFIRMED at `B` | C2 + NT-FC1-25 green, or a negative replay of V6 at the train head; FRZ-CTL-12 |

## 11. Independent audit design (C3, DP-03, DP-12)

Separate non-implementing session, not the implementer, reviewer, IIAC signer or activation certifier; skill `/intergrax-independent-audit`; at the exact `RW` SHA after the last product change (any later product change voids it). The auditor replays: (1) every negative test red at the baseline tree and green at `RW`; (2) mutation per part (revert A3, A1 fallback branch, B4 guard one at a time; the matching tests must turn red); (3) the AST call-site script over the whole tree (0 calls without explicit posture) and a fresh fallback census over the diff (no new silent fallback); (4) LAB opt-ins are typed, explicit and rejected in production (NT-FC1-10); (5) probes V1/V2/V5/V6/V4 re-run with the same factory functions the product hosts call (composition fidelity, WP-11 section on probes); (6) the ACT-N11 path list in this plan equals the diff of product paths; (7) the sweep did not change any test expectation (diff of test files restricted to the added keyword); (8) docs claim only what the code does. Result labels: `PASS_SCOPED` only for the scope actually executed; no FRZ promotion by the auditor.

## 12. Risks

- R-1 The LAB-default-deny reading of the fallback (J-1, decided D-FC1-02) can turn lab/test flows red that reach `_route_best_or_fallback` unintentionally; `git grep` found no test that names the fallback, but flows that rely on it silently are unknown until the first run.
- R-2 The sweep touches 101 test files (plus ten notebooks and two usage docs); a wrongly placed keyword is a syntax/semantic error caught only by running them; mitigated by the AST list script and by the bounded R5 sweep proof (101 files, 5 commands).
- R-3 Part B refuses STRICT constructions that previously ran with persistence silently off. Callers at risk: the fourteen call sites of `build_harness_host_runtime` (ten in `applications/`, four in `intergrax/`), `NexusWorkerRuntime.from_registry(production_mode=...)`, `compose_application_host_orchestration_session` (`application_host_orchestration_composition.py@H:41-74`). No Tier-3 module passes a human decision store into `NexusLoop` or the spec today: `ctx.human_decision_store` is only read by the tool wiring (`tool_wiring.py@H:139`, `integration_tool_wiring.py@H:54`, `notify_tool_wiring.py@H:44`, `session_tool_wiring.py@H:44`) and the only creator in the tree is the SQLite persistence bundle (`sqlite_composition.py@H:119,170-180`) (FACT_CHECK F-30), the reason for the D-FC1-03 acknowledgement path.
- R-4 Overlaps on the train (section 8): `harness_host_runtime.py`, the two spec builders and `host_orchestration_loop_init_spec.py` were edited by PT-WA (merged); `nexus_loop.py` is shared with PT-WF (after this ticket) and `harness_host_runtime.py` with T-AF2-RW-TENANT (after this ticket); PT-WB and PT-FC2 are not merged at the time of writing, so their effect on these files is not read.
- R-5 A replay of V2 or V6 at the train head can change scope (negative: the part is dropped; a missing posture carrier for C1: an amended plan). A plan amendment changes the plan digest.
- R-6 Line-pinned registries elsewhere (TRACE-X P4 pins `context_manager.py:239`; COMPAT-X fixtures pin some `file::symbol:line`) are not expected to name the write set (grep over `tests`, `scripts`, `*.json` at `H` found only path-level references), but editing `nexus_loop.py` shifts its line numbers: STATE-X, TRACE-X P5 and `ebh_6/_l3_reachability_evidence.py` name the file; confirm by running R4.
- R-7 `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` names several of these files (24 hits); whether it needs a data refresh at the re-pin is for the IIAC session, UNVERIFIED here.
- R-8 Existing STRICT test constructions: 13 `NexusLoop(production_mode=True)` sites in two test files (`test_gr10_r12_orchestration_continuation_architecture_gates.py` 9, `test_attempt_lifecycle_durability_gate.py` 4; census at `T`) build STRICT loops with no human store, no acknowledgement and no context engine or LLM adapter. Some are negative tests of the existing continuation and attempt-lifecycle guards; B4 and C2 add checks to the same constructor, so the order of the checks inside `NexusLoop.__init__` decides whether those tests still see their own error. The two files are in the sweep proof and are edited only to pass the explicit acknowledgements (and, for C2, existing test doubles for the engine and adapter); if no such doubles exist the edit is a write-set extension.
- R-9 Harness STRICT callers in tests: 53 files under `tests/`, `testing_support/` and `applications/*/tests/` call `build_harness_host_runtime(` (`git grep` at `T`); how many of them build a STRICT environment is NOT measured. Step 0 runs a read-only census (which of the 53 pass a STRICT profile) before Part B; a test file that must change beyond the firm lists is a write-set extension (amended plan, new digest, the Operator's decision), not a silent edit.
- R-10 C2 refuses every STRICT `NexusLoop` without a context engine and LLM adapter, including a STRICT host that plans without an LLM (the case probe V6 used). That is the intended fail-closed behavior; the hosts it refuses are found by NT-FC1-20, 21 and 25 and the R3 slice.

Combined test blast radius: the 215-site sweep of this ticket and the test edits and opt-ins of FC-2 and FC-3 (and the guards of FC-4) may touch the same test files and the same PRODUCT/STRICT test environments; none of the four plans enumerates the others' files. A census of the integrated wave tree (all four tickets merged on the integration branch: which existing tests now meet more than one new guard, and which files more than one ticket edited) is required before the independent audit; it is a wave-level step, not performed by this ticket.

## 13. Rollback

Before `RW` is frozen: revert the ticket's commits on the integration branch (the sweep commit is independent and harmless alone because the keyword already exists). After the wave merges: revert the train merge; the IIAC re-activation must be redone, a revert alone does not restore it (AR-41 rollback text). The ticket introduces no data migration, no new persistent file and no setting change; LAB behavior is the same before and after except the DP-26 default (J-1), so a revert restores today's tree exactly.

## 14. Effort (planning estimate, not measured)

| Part | Size | Note |
|---|---|---|
| A1..A6 | S–M | `nexus_loop.py` is a 1307-line hotspot |
| A7 sweep | M (mechanical, scriptable) | 215 + 7 call sites, 12 docs/notebooks |
| New tests | M | 26 FC-1 tests plus NT-BND-05-1 in six files |
| B1..B8 | M–L | blast radius R-3 |
| C1 / C2 | S / S | firm (V2 and V6 CONFIRMED at `B`) |
| Docs | S | four sentences/rows |
| Independent audit | separate session | node `PT-Q02` |

## 15. Design choices applied (J-1 and J-5 are Operator-decided; J-2..J-4 are implementer choices inside the decided design)

- J-1 DP-26/DP-24 read literally: the fallback is default-deny everywhere and requires the typed LAB opt-in (decided D-FC1-02).
- J-2 Lab builders pass the opt-in so today's debug/lab behavior is preserved.
- J-3 `persist_human_decision` stays tolerant; the guard is at composition.
- J-4 The harness passes `disable_task_memory_persistence=True` only when the profile itself disables task memory in STRICT/PRODUCT.
- J-5 The host-task router is the loop's router (identity), not a second router built from a flag.

## 16. Out of scope

Any other fail-closed selector (A4-017/019/026/030/032/035/036, A4-002), seam relocations (W-A/W-B: DP-13..DP-19), the closed-list entry for the two Tier-1 openers (seam documents), planner/assembly `production_mode=False` literals (D-FC1-05), the IIAC re-pin and EBH-7 pin, FRZ promotion, any claim of closure or independent verification, GitHub settings, validator code, `scripts/`, Docker/vendor work (C10/C15 freeze).



## 17. Facts that differ at the train head `T` (the first version was read at `H`)

| Item | At `H` (first version) | At `T` | Effect on this plan |
|---|---|---|---|
| Spec builders | one file, `host_orchestration_environment_spec_builder.py`, function `build_host_orchestration_loop_init_spec_from_environment(registry, env=...)` | PT-WA split it: Tier-3 `intergrax/applications/_shared/host_orchestration_backend_spec_builder.py` (same function name, resolves the profile; `harness_host_runtime.py@T:594` calls it) and Tier-1 `host_orchestration_environment_spec_builder.py::build_host_orchestration_loop_init_spec(environment=...)` (neutral values; builds the spec at `@T:194`; `production_mode` from `environment.execution_mode` at `@T:217`) | A6, B5, B6 apply to both builders; the Tier-3 file is added to the write set; the V6 probe factory moved |
| `host_orchestration_loop_init_spec.py` | `production_mode` at `@H:88`; wiring field typed `HostOrchestrationApplicationWiringBundle` | `@T:86`; wiring field typed `HostOrchestrationApplicationWiringPort`; imports of `NexusLoop` and `AgentRegistryRead` removed | anchors only |
| `harness_host_runtime.py` | | two hunks at `@T:714-760` (`effective_environment.graph_spec` argument; `HarnessHostInternalComposition.from_capabilities`) | none: the cited anchors 354, 532-540 and 594 are unchanged |
| `docs/project/architecture/MEMORY.md` | SQLite env row at `@H:396` | `@T:410` (unrelated docs commits added a scope label, an owner table and a section) | anchor only |
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | | an owner-mapping note and table added at the top (unrelated docs commit) | the fallback-policy sentence is added below it |
| OBS-DIAG-X3 AST gates | cited as `obs_diag_x3_ast_gates.py@H:110-119` | moved by PT-WA to `testing_support/architecture/obs_diag_x3_ast_gates.py` (`@T:110-120`) | pattern reference of NT-FC1-12 |
| `test_obs_diag_x3_universal_spine_adoption.py`, `test_diag_platform_adoption_gate.py` | | both edited by PT-WA | stay conditional (source-token assertions on `NexusLoop(`); the second is shared with PT-FC2 |
| `agents/defaults.py` | named in Part C1 | does not exist at `H` or `T`; the file is `intergrax/agents/defaults.py` (`harness_production_mode` at `@T:13-15`), re-exported by `applications/_shared/runtime_defaults.py` | path corrected in Part C1 and section 4 |
| `external_contractor_adapter_agent.py` | `build_context` at `@H:107-117` | `@T:107-114` | anchor only |
| Sweep census | 215 `NexusLoop(` sites + 7 `AgentRouter(` test sites; 97 `tests/` files | identical counts at `T`; A1 lists 100 `tests/` files (three with explicit posture already) | R5 list has 104 files |
| `build_harness_host_runtime` call sites | 10 under `applications/`, 4 in `intergrax/` | the same 14 sites, in 13 distinct files; in addition `scenario_runtime_baseline.py` builds and materializes a spec | section 4, B9 |
| All other files of the write set | | `git diff 7960ae7f6 origin/arch-freeze/rw-train` is empty for every file of the write set and of the sweep list except the seven files named above (`host_orchestration_environment_spec_builder.py`, `host_orchestration_loop_init_spec.py`, `harness_host_runtime.py`, the two docs, the two gate tests) | none |

## 18. Differences from RW-PLAN v13 appendix A1 (`PT-FC1`) and how this version reconciles them

| # | Item | A1 | First version of this plan | This version |
|---|---|---|---|---|
| 1 | Sweep files under `tests/` | 100 firm (AST census rule at `T`) | 97 (92 `NexusLoop(` files plus 5 `AgentRouter(` files, no overlap counted) | 100 firm; the three extra files already pass `production_mode=`, two of them STRICT (R-8) |
| 2 | `agents/defaults.py` | `agents/defaults.py`, kind `new`, cond | `agents/defaults.py` (path does not exist) | `intergrax/agents/defaults.py`, edited, firm (difference 5) |
| 3 | Tier-3 `host_orchestration_backend_spec_builder.py` | absent | absent (the file did not exist at `H`) | firm, edited (B5) |
| 4 | `scenario_runtime_baseline.py` | absent | absent | conditional (B9) |
| 5 | Part C1 files (`organization_worker_agent.py`, `external_contractor_adapter_agent.py`, `acp_stub_reflex.py`, `intergrax/agents/defaults.py`) | cond (4) | cond | firm (V2 CONFIRMED at `B`; RW-PLAN rows NT-ROW-031 and NT-ROW-140 name basis S and B tests for them). If the Operator prefers appendix A1's class, they stay conditional and NT-FC1-24 and NT-BND-05-1 become conditional with them |
| 6 | `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` | firm, new (source `T-AF2-RW-PLAN`) | absent | firm, new; NT-BND-05-1 read as the posture rule of section 5 (RW-PLAN row NT-ROW-031 does not name the sanctioned composition root) |
| 7 | Conditional call sites of `build_harness_host_runtime` | absent from A1 (A1 cond product paths: 7) | up to 14 sites (13 distinct files), not named | named in section 4, 13 files |
| 8 | Part C2 | not a path (nexus_loop.py is firm) | cond (probe V6) | firm (RW-PLAN rows D-01 and A4-031: V6 CONFIRMED); NT-FC1-25 firm |
| 9 | File column of NT rows | NT-ROW-021 names `test_fc1_nexus_loop_persistence_posture.py` for NT-FC1-20..23 and 26; NT-ROW-140 names `test_fc1_host_task_routing_posture.py` for NT-FC1-05, 06, 12 and 24; NT-ROW-143 names `test_fc1_harness_strict_persistence_wiring.py` for NT-FC1-16..19 | NT-FC1-20..22 and 26 and NT-FC1-24 in `test_fc1_harness_strict_persistence_wiring.py`; NT-FC1-05/06 in `test_fc1_agent_router_posture_and_fallback.py`; NT-FC1-12 in the construction gate file; NT-FC1-16..18 in `test_fc1_nexus_loop_persistence_posture.py` | placement of the first version kept: the test ids and the set of six files are the same; only the file column of RW-PLAN section d differs, to be corrected there |
| 10 | Counts (firm product / firm total / conditional) | 30 / 143 / 14 | 30 firm product, up to 21 conditional product (and 7 conditional tests, 14 in A1's reading) | 35 / 148 / 24 (conditional: 17 product paths and 7 test paths); the lists in section 4 are the parsed source of these numbers |
