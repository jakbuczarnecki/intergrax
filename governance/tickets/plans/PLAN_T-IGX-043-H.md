# PLAN T-IGX-043-H — EQO-A remediation code wave, package W-B: one LLM resolver, explicit profile projections, closed provider composition roots, serving defaults, importlib prefix allowlist

Status: plan pending review (ticket lifecycle `plan_pending`). Drafted 2026-11-16 for master-plan node `PT-WB` (wave package W-B, `AR-40`). Nothing here is approved, implemented, executed or audited; no test was run while drafting. This plan certifies nothing, promotes no FRZ criterion and states no independent verification.

Notation. `H` = `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (read 2026-11-16; `git rev-parse origin/development` equals it). `B` = Campaign A baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`. In this plan `path@H:line` abbreviates the full 40-hex SHA of `H`; `FACT_CHECK.md` spells the full SHA on every row. Status words: `UNCHANGED` = `git diff B H -- <path>` empty; `CHANGED`; `GONE`. Product delta `B..H` is exactly three files, `intergrax/rag/graph/generation_visibility.py`, `intergrax/rag/graph/indexer/heuristic_graph_indexer.py`, `intergrax/rag/graph/providers/inmemory_graph_store.py` (`git diff --stat B H -- intergrax agents applications scripts/maintenance tests/unit/architecture`); none is in the W-B write set, so every file cited below is `UNCHANGED` since `B`. Other `.tmp` sources named here are session artefacts, not repository records.

## 1. Purpose and findings

W-B closes the structural half of the layer-boundary seams that the Operator ruled in decision session 2 (DP-16..DP-21). One ticket, one package of the single integrated wave (DP-11); it never merges alone. Parts:

| Part | What | Decision | Findings |
|---|---|---|---|
| A | One host LLM resolver: Tier-1 `runtime/wiring` is canonical, the Tier-3 module delegates; the routing record stops being a module global | DP-16 | EQOA-A4-016 |
| B | The mirrored profile models (12 pairs at `H`; the finding and DP-17 say 11, see `FACT_CHECK.md` F-30) become declared projections with a field map that fails on unmapped fields; projection targets may not be `extra="ignore"` (35 dropped fields: Memory 17, Cost 9, Reliability 9) | DP-17 | EQOA-A8-003 |
| C | Provider composition roots: classification of every module that imports `integrations.providers.*` / `integrations._shared.*` outside `intergrax/integrations`, a closed named list proposal, and the data that the docs and the ledger take; at `H` no module is rerouted by code (section 4.C) | DP-18 | EQOA-A4-020; A4-024 only as a classification input (owner FC-1, decided D-FC1-01) |
| D | Serving: LKW reference production goes through `run_local_workspace_hosted_application` (the only caller of `run_hosted_application`); default bind of direct entrypoints becomes `127.0.0.1` with the explicit value `0.0.0.0` as the opt-in | DP-19 | EQOA-A4-040 |
| E | Package-prefix allowlist for the importlib manifest loaders (fail closed in STRICT/PRODUCT); the three confirmed A4-007 seams are taken over from the W-A plan (section 1b) | DP-21 | EQOA-A4-007 (confirmed alpha item, three seams, owner of the loader files: this ticket); follow-on of EQOA-A7-021 (S11); EQOA-A4-004 untouched |
| G | Governed Contractor strict production serving command onto `run_hosted_application` (kept in this ticket, not split; class `cond`, not firm; serialized with `T-AF2-RW-SUPPLY`; lands only if the Operator rules B-2 A4-040 in scope, section 1b) | DP-19 | EQOA-A4-040 (S7b; B-2 alpha) |

Dependencies, not findings of this ticket: EQOA-A5-001 / EQOA-A2-009 (EBH-3 gate scan extension, DP-20) is `PT-G01` (tests/scripts, gate meaning not delegated) plus the LKW facade items of W-A; W-B only states which of its modules fall in the demo/proof exception class (section 4.C). The gates that make A, B, C, D, E standing (second-`resolve_llm_adapter` ledger gate, profile parity gate, direct-serving count gate, closed-roots ledger and the importlib prefix check) are `PT-G01`, outside this ticket.

Assignment contradiction recorded for the Operator: `WP-12/TICKETS_DRAFT.md` lists A4-024 among the findings of `PT-WB`, but decision D-FC1-01 gave A4-024 to FC-1 (`PLAN_T-IGX-043-A.md` section 8, C-3: "W-B neither relocates nor duplicates the two openers"). This plan follows D-FC1-01: W-B changes no code for A4-024 and only classifies the two openers (section 4.C).

## 1a. Confirmed Campaign B-1 alpha items routed to W-B (added 2026-11-16)

Source: `.tmp/session/eqo-a-remediation/B1_ALPHA_ROUTING.md` and `.tmp/session/arch-freeze/b1-alpha/RESULTS_B1_ALPHA.md` (section 3, rows A4-007, X-A4-016, X-A8-003, X-A4-024; relayed content, used here as data). Runs at the baseline `B`, diagnostic and non-certifying: nothing below was re-run by this plan. The probes assert the safe behaviour; "CONFIRMED" means the probe failed at `B`. Implementation of any item waits for the gate `SCOPE`; product commits go only to the branch `arch-freeze/rw-train`; the approved FC tickets are not edited for these items (their dedupe is the overlap matrix of `T-AF2-RW-PLAN`).

| Item | Alpha result | Class (source) | What W-B does | Files (section "Write set") | Closing negative tests (section "Negative tests") |
|---|---|---|---|---|---|
| A4-007 boundary seams at three points (shared with W-A) | R02 b1, b2, b4 CONFIRMED: `agent_resolution.py:42` imports any `import_path`; `migration_wiring.py:337` imports `{package}.manifest` with the package taken from a directory glob; `agents/readiness/scoreboard.py:44-57` imports `removeprefix("agents.")` of a caller string. b3 (authoring facade) PASS, A7-021 hard-boundary NOT_REPRODUCED (hard = 0) | `BLOCKS` | Part E covers all three confirmed seams with the prefix allowlist (`scoreboard.py` joins the write set as E9; it was excluded before, Q-WB-07 updated); `wiring.py` and `mcp_import_guard.py` stay as the same loader class. The authoring facade (b3) is not edited. W-A shares the finding (its S2/X-A4-004 registry-bypass work is a different gate and different files); the rule for the shared A4-007 closure is Q-WB-08 item d | `intergrax/applications/_shared/import_allowlist.py`, `agent_resolution.py`, `migration_wiring.py`, `wiring.py`, `mcp_import_guard.py`, `intergrax/agents/readiness/scoreboard.py` | NT-WB-18..23, NT-WB-30 |
| X-A4-016 tier routing records shared across tenants | R04 nt12a (Tier-1) and nt12b (Tier-3) CONFIRMED: one process-wide `_last_routing_evaluation`, the last writer wins, tenant A can consume tenant B's record; the Tier-3 consume has no caller | `BLOCKS` | the canonical Tier-1 record is tenant-scoped (section 4.A item 6): entries are keyed by adapter and by the tenant of the routing context; a consume with another tenant returns nothing | `intergrax/runtime/wiring/llm_resolver.py`, `intergrax/applications/_shared/llm_resolver.py`, `intergrax/runtime/nexus/engine/runtime_state.py` | NT-WB-02, NT-WB-03, NT-WB-27, NT-WB-28 |
| X-A8-003 silent drop in profile projection (DP-17) | R04 nt13a CONFIRMED (the probe asserts that each configured Tier-3-only value is carried or the projection refuses); nt13b PASS (the dropped values are enforced elsewhere by Tier-3 bridges) | `CONDITIONS` | Part B: declared maps, forbid targets, import-time verification; every dropped field carries a verified consumer route (the nt13b evidence made checkable). Under the recommended option A of Q-WB-03 the nt13a criterion as written would still report a drop; only option B (carry the values) or a ruling on the criterion closes it | `host_profile_slices.py`, `profile_projection.py`, `runtime_boundary_adapters.py`, `reliability_runtime_bridge.py` | NT-WB-08..13, NT-WB-29 |
| X-A4-024 task-memory disabled-profile env var | R03 v4 CONFIRMED: `task_memory_wiring.py:46-56` passes `enabled=None` (not `False`) when the profile disables every memory kind, so the env var opens the store | `CONDITIONS` | none in W-B: D-FC1-01 and FC-1 B8 already cover exactly this probe (`PLAN_T-IGX-043-A.md` B8 and `NT-FC1-19`, "STRICT environment profile with all memory flags off, env var set"); recorded here so the overlap matrix dedupes it to FC-1 (the routing table lists "FC-1 / W-B (A4-024 verdict of WP-1)") | none | closed by `NT-FC1-16..19` of `T-IGX-043-A`, not by a W-B test |

## 1b. Scope changes agreed after the first draft and Campaign B-2 alpha routing (added 2026-11-16)

Agreements between the ARCH-FREEZE orchestrator and the EQO-A planning session (questions Q-RW-02 and Q-RW-15). They are session agreements, not Operator answers; the Operator-facing questions are Q-WB-06, Q-WB-08, Q-WB-12 and Q-WB-13. Sources of the B-2 rows: `B1_ALPHA_ROUTING.md` (section "B-2 alpha") and `.tmp/session/arch-freeze/b2-alpha/RESULTS_B2_ALPHA.md` section 3; relayed content used as data, diagnostic and non-certifying, nothing re-run by this plan.

1. **A4-007 is Part E of this ticket.** The design, facts and negative tests of the three dynamic-import seams are taken from section 7a ("Part F") of `FC/WA/PLAN_T-IGX-043-G.md`, which the W-A plan drops (this plan edits no file of `FC/WA/`). The three seam files are `intergrax/applications/_shared/agent_resolution.py`, `intergrax/applications/_shared/migration_wiring.py` and `intergrax/agents/readiness/scoreboard.py`. Part E also keeps `wiring.py`, `mcp_import_guard.py` and the helper `import_allowlist.py` (DP-21 loader class, same mechanism); Q-WB-06 asks whether Part E is cut to the three seams (then 16 instead of 19 firm paths). The allowlist gate that proves "every `import_module` is dominated by a check" stays `PT-G01`; NT-WB-21 and NT-WB-30 are the ticket-local versions.
2. **Part G stays inside this ticket (no split) and is conditional (`cond`), not firm.** It is serialized with the ticket `T-AF2-RW-SUPPLY` (supply-chain remediation) on `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py`: one of the two tickets merges first and the other rebases; the order is set by the overlap matrix of `T-AF2-RW-PLAN` (section 9). B-2 A4-040 (direct `uvicorn.run` in 7 apps, one of which is that strict serving command) lands in Part G only if the Operator rules it in scope (Q-WB-12); the other six direct sites are declared non-production, counted (NT-WB-25) and not edited beyond Part D. The A4-040 ruling is open (Q-RW-16 b of the RW plan); the PROD-Q node runs of Part G need a separate authorisation (Q-RW-16 d); the approval line of this ticket does not cover Part G, and the wave packet of this ticket excludes it until the ruling.
3. **B-2 routing to W-B:**

| Item | B-2 alpha result (relayed) | Class | What W-B does | Files | Closing tests |
|---|---|---|---|---|---|
| A4-040 (B-2) | 7 apps call `uvicorn.run` directly; one is a strict production serving command (promotion trigger met, static) | conditional (DP-19 scope ruling, Operator confirmation, Q4 of the B-2 routing) | Part D: LKW reference production on the hosted runner and loopback defaults (LKW, research). Part G: the strict serving command on `run_hosted_application`, only if ruled in scope (Q-WB-12) | Part D and Part G paths of section "Write set" | NT-WB-14..17, NT-WB-25, NT-WB-26 |
| A4-020 (B-2) | provider imports and switches; classification first | conditional | Part C classifies (59 modules plus 43 area files, no product edit at `H`); W-B edits a file only if the Operator rules that file a bypass (Q-WB-13); each ruled file is routed through `IntegrationProfile` and added to the conditional product paths. Candidate files (not findings): the three IMPL modules `intergrax/runtime/execution/postgresql_decision_checkpoint_persistence.py`, `intergrax/runtime/long_running/postgresql_task_checkpoint_store.py`, `intergrax/runtime/migration/human_decision_legacy_disposition.py`, and `applications/local_workspace_application/slack_companion/companion.py` | section "Write set", subsection "conditional on a Part C ruling" | NT-WB-31 (conditional) |

The other B-2 rows (A4-025, A2-009 gate scope, A2-051, X-A6-003, A6-007) are not W-B items; A2-009 stays the `PT-G01` scan extension unless the eight imports are moved, which this plan does not do.

## 2. Prerequisites that must hold before implementation starts

| Item | Source | State read at `H` |
|---|---|---|
| Node `SCOPE` (wave scope frozen) and `DS-2` (DP-13..DP-23) | `WP-12/DAG.md` (`PT-WB` hard prerequisites `SCOPE`, `PT-G01`) | `DS-2` decided in the decision record; `SCOPE` not evidenced in the repository (orchestrator state) |
| `PT-G01` (non-product gates and tests) | `WP-12/DAG.md` row `PT-WB` | not started; W-B states its dependency (section 9) and does not carry the gates |
| Milestone `M1` (PDC P9 closed) | DP-11 | `PDC_INDEX.md@H:44-49` still lists the six P9 artifacts as "Answers recorded in chat; awaiting the Operator's final confirmation", while `OD-PDC-X-CLOSE.md` (PR #603) carries a signature line dated 2026-11-16 and the roadmap carries a PDC-X closure label. The repository texts disagree; this plan does not decide whether P9 is closed. The transcription `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (draft, session artefact) records the product-path go for AR-39..AR-44 as "granted, effective only when M1 holds", M1 itself not confirmed by that file |
| Authorisation `AR-40` (wave package W-B) and ticket approval `AR-38` | `WP-12/AUTHORIZATION_REQUESTS.md` | open; `AUTHORIZATION_REQUESTS.md` of this directory. AR-40 names DP-16..DP-19; Parts C-E also rely on DP-20/DP-21 context (see `OPEN_QUESTIONS.md` Q-WB-10) |
| `T-IGX-043-B` (FC-2) merged first | merge order B, A, C, F confirmed (transcription item 6); it owns `intergrax/applications/_shared/wiring_posture.py` (`wiring_fails_closed`), which Part E imports | not merged; Part E cannot start before it |
| W-A ticket (`PT-WA`, drafted by a parallel session) | `WP-12/TICKETS_DRAFT.md` | not available when this plan was written; overlap rules are in section 9 |
| Operator answers to the design questions of this ticket | `OPEN_QUESTIONS.md` | none given; every design below that is marked "proposal" is a recommendation, not a decision |

Hotspot rule: the three files that W-A/W-B must merge before FC-1 (`host_orchestration_environment_spec_builder.py`, `harness_host_runtime.py`, `nexus_loop.py`) are not in the W-B write set (grep at `H`: `harness_host_runtime.py` has no resolver, projection, provider-import, uvicorn or importlib reference; `nexus_loop.py` references only the two openers at `:70,199,284,288`; the builder imports the Tier-3 resolver at `:34,171` and that import is removed by W-A, section 9).

## 3. Posture rule

Parts A..D change behaviour that is not posture-conditional (one resolver; explicit projections; loopback default) except Part E, which uses the DP-24 hybrid: wiring layers fail closed when `wiring_fails_closed(env)` (FC-2's helper: `env.application_profile is PRODUCT` or `env.execution_mode is STRICT`); leaf loaders that receive no posture are default-deny only when a caller declares fail-closed, otherwise unchanged (LAB). The precise threading is a census step (section 8, step 0) and open question Q-WB-06.

## 4. Design per part (every cited line read at `H`, identical at `B`)

### 4.A One host LLM resolver (DP-16; EQOA-A4-016)

Facts at `H`. Two modules define the same resolver surface: Tier-1 `intergrax/runtime/wiring/llm_resolver.py` (190 lines) and Tier-3 `intergrax/applications/_shared/llm_resolver.py` (398 lines). Each owns a module global `_last_routing_evaluation` (`runtime/wiring/llm_resolver.py@H:20`, `_shared/llm_resolver.py@H:31`). The Tier-3 module adds (i) the live routing wrapper `wrap_routing_evaluating_adapter` when `env.llm_routing_profile` is set (`@H:249-262`), (ii) the AHI live routing hint `resolve_live_model_routing_wiring` (`@H:239-246`, defined in `llm_routing_wiring.py@H:90-134`), (iii) the capability check `required_llm_features` (`@H:295-301,315,333,346`), (iv) a `context_provider` parameter, (v) the extra API `resolve_runtime_llm_profile`, `resolve_llm_routing_decision`, `create_adapter_for_routing_evaluation`, `resolve_(optional_)environment_llm_adapter` (`@H:109-170,349-398`). The Tier-1 module has none of (i)-(v): with a routing profile it evaluates once and builds a plain adapter (`@H:113-136`). Consumers of the record: only `runtime_state.py@H:241,290` (Tier-1 global, `consume_routing_evaluation()` inside `configure_llm_tracker`); the Tier-3 `consume_routing_evaluation` (`@H:39-44`) has no caller (`git grep` over `intergrax agents applications scripts tests` at `H`: definition only), so evaluations recorded on the Tier-3 path are never emitted as a diag event by that route. The platform wrapper `RoutingEvaluatingLLMAdapter` needs only `profile_source.llm_routing_profile` (`llm_adapters/routing/evaluating_adapter.py@H:49-70,125`), so it is usable at Tier-1 without a Tier-3 type. Importers: Tier-3 module — `applications/local_workspace_application/host/agent_factories.py:59,70,94`, `model_runtime_proof/runtime.py:17`, `environment_wiring.py:25`, `llm_routing_runtime_bridge.py:119`, `routing_evaluating_adapter.py:61`, `runtime_config_bridge.py:43`, `scenario_runtime_baseline.py:57`, `host_orchestration_environment_spec_builder.py:34,171`; Tier-1 module — `workspace_routes.py:534`, `ask_service.py:100`, `hybrid_ask_service.py:325` (each `resolve_llm_adapter(None)` in the LKW), `agents/authoring/acp_run.py:398`, `acp_runtime_session_hooks.py:40`, `runtime_state.py:241`.

Design (proposal; the choices are Q-WB-01/Q-WB-02).

| # | File | Change |
|---|---|---|
| A1 | `intergrax/runtime/wiring/llm_resolver.py` (canonical) | (1) Define in this module a structural `LLMResolutionSource` (`Protocol`: read-only `llm_profile`, `llm_routing_profile`); both `RuntimeEnvironmentProfile` (`contracts/runtime_environment.py@H:32-33`) and `ApplicationEnvironmentProfile` (`environment_profile/root.py@H:249-265`) satisfy it, so the Tier-3 façade passes its profile unchanged and no projection of unrelated profile parts is needed (decouples A from Part B). (2) `resolve_llm_adapter` becomes the superset signature of both modules: adds `context_provider`, `required_llm_features` (checked with `LLMAdapterRegistry.require_capabilities` before return, a Tier-0 import), and `live_routing_hint_provider: Callable[[RoutingContext], str | None] | None` (a typed port that replaces the Tier-3 import of the AHI wiring). (3) When `env.llm_routing_profile` is set the canonical resolver wraps the adapter with the platform `RoutingEvaluatingLLMAdapter` (`profile_source=env`, default `adapter_factory` = `create_adapter_for_routing_evaluation`). (4) Moves in from Tier-3 with unchanged names: `_create_base_llm_adapter`, `create_adapter_for_routing_evaluation`, `resolve_runtime_llm_profile`, `resolve_llm_routing_decision`, `resolve_(optional_)environment_llm_adapter`. (5) Keeps `_resolve_llm_adapter_impl`, `resolve_optional_llm_adapter`, `evaluate_llm_routing`, `resolve_llm_profile`, `resolve_optional_llm_profile` and the module-level names `create_adapter`, `create_adapter_with_failover` that existing tests patch (`test_ebh_2e_r6_r1_r5…:171,175`, `…r6_r2_r3…:151,155`). (6) Removes the module global `_last_routing_evaluation`: the record is bound to the resolved adapter object and is tenant-scoped (X-A4-016): a weak-keyed registry guarded by a lock maps adapter to a per-tenant map (tenant of the `RoutingContext` at evaluation time, `None` when the context has none) to the latest evaluation; `consume_routing_evaluation(adapter, *, tenant_id)` pops only the entry of that tenant and returns `None` for any other tenant, so tenant A can never consume tenant B's record even when both use one shared adapter; the wrapper's `on_evaluated` updates the entry of the tenant it evaluated. Env-var selection (`llm_profile_from_env`) and the `LLMProviderNotConfiguredError` refusal are unchanged. |
| A2 | `intergrax/applications/_shared/llm_resolver.py` (façade) | Every public function keeps its name and signature and only delegates to A1: it passes `env` (duck-typed, no projection) and a `live_routing_hint_provider` closure built from `resolve_live_model_routing_wiring` (so the `ValueError("tenant_id is required …")` of AHI-enabled product hosts is raised at the same place). The façade keeps no adapter creation: no `create_adapter*`, no `ModelRouter`, no `wrap_routing_evaluating_adapter` call, no module global. `consume_routing_evaluation(adapter)` re-exports the A1 function. Public names stay so that the three `monkeypatch.setattr("intergrax.applications._shared.llm_resolver.resolve_llm_adapter", …)` qualification tests (`tests/qualification/pdc_x/d1_repair/…:42`, `…:47`, `p8_repair/…:39`) and `tests/unit/agents/conftest.py:55` (patches the Tier-1 name) keep working unchanged. |
| A3 | `intergrax/runtime/nexus/engine/runtime_state.py` | `configure_llm_tracker` calls `consume_routing_evaluation(core_adapter)` (`@H:241,290`) instead of the zero-argument global consume, so a Tier-3-resolved adapter's evaluation is emitted (closes the lost-record leg). |
| A4 | `intergrax/llm_adapters/USAGE.md` | The lines that tell Tier-3 hosts to prefer `resolve_llm_adapter` "from `intergrax.applications._shared.llm_resolver`" (`@H:29,344,423`) name the canonical module and describe the façade. The owner-document text (`LLM_ADAPTERS.md` names `resolve_llm_adapter()` without a module, WP-1 S3) is `PT-D12`, not this ticket. |
| A5 | `intergrax/applications/_shared/routing_evaluating_adapter.py` | No edit planned: its default factory imports the Tier-3 `create_adapter_for_routing_evaluation` (`@H:57-65`), which now delegates; census at step 0 confirms no caller depends on a different behaviour. |

Behaviour change to flag: ACP/agent callers of the Tier-1 resolver (`acp_run.py:398`, `acp_runtime_session_hooks.py:40`) gain the live-routing wrapper and capability check when their profile carries a routing profile or required features (Q-WB-02). The LKW `resolve_llm_adapter(None)` callers are unchanged (no env, no routing profile).

### 4.B Explicit profile projections (DP-17; EQOA-A8-003)

Facts at `H`. AST comparison of `intergrax/contracts/host_profile_slices.py` (Tier-0) and `intergrax/applications/contracts/environment_profile/sub_profiles.py` (Tier-3) (script run read-only on `git show` output, result in `FACT_CHECK.md` F-30): 12 classes in Tier-0, and all 12 names are also Tier-3 classes, so 12 pairs, not the 11 of the finding title and DP-17 (WP-1's own table S4a 7 + S4b 3 + S4c 2 also sums to 12). Divergent pairs: `MemoryProfile` (4 vs 21 fields, 17 only in Tier-3), `CostProfile` (2 vs 11, 9), `ReliabilityProfile` (7 vs 16, 9), `PolicyRulesProfile` (+2: `allowed_handler_ids`, `policy_enforcement_mode`), `AdaptiveProfile` (+1: `business_outcome_webhook`). Tier-0 `extra`: `ignore` on `ContextProfile`, `MemoryProfile`, `CostProfile`, `ReliabilityProfile` (`host_profile_slices.py@H:69,106,117,126`); `forbid` elsewhere. The only product code that projects Tier-3 into Tier-0 does `Target.model_validate(source.model_dump(mode="json"))`: five times in `runtime_boundary_adapters.py@H:48-62` (Memory, Context, Cost, Reliability, ExecutionBoundaryExport) and once in `reliability_runtime_bridge.py@H:34`; because the targets ignore extras, the 35 Tier-3-only fields are silently dropped. A scan of consumers (`FACT_CHECK.md` F-31) shows every one of the 35 fields is read by a Tier-3 module directly from the full profile (for example `memory_runtime_bridge.py`, `memory_wiring.py`, `cost_runtime_bridge.py`, `reliability_wiring.py`, `recovery_contract_wiring.py`) and several by Tier-1 code through `RuntimeConfig`; none is read from the Tier-0 slice (it does not have them). Whether the ACP/agent path needs retention, scope or enforcement values it never receives is the open check `X-A8-003` (Q-WB-03); this plan makes the drop declared and gated, not different.

| # | File | Change |
|---|---|---|
| B1 | `intergrax/contracts/profile_projection.py` (new, Tier-0 neutral) | `ProfileProjection` (frozen dataclass: `name`, `target`, `projected: tuple[str, ...]`, `not_projected: Mapping[str, str]` = field to non-empty reason) with `verify(source_cls)` and `project(source_obj)`. `verify` raises typed errors: `UnmappedProfileFieldError` (a source field in neither `projected` nor `not_projected`), `StaleProjectionEntryError` (a listed field missing in the source), `ProjectionTargetFieldMissingError` (a projected field not on the target), `ProjectionTargetIgnoresExtrasError` (target `model_config["extra"] != "forbid"`). `project` builds the target from `source.model_dump(mode="json", include=projected)` (the same JSON round trip as today, restricted to the declared fields). No Tier-3 import (specs carry names, classes are passed in). |
| B2 | `intergrax/contracts/host_profile_slices.py` | `extra="ignore"` becomes `"forbid"` on `ContextProfile`, `MemoryProfile`, `CostProfile`, `ReliabilityProfile` (the projection targets); the other eight already forbid. Census at step 0: constructions of these four classes with unknown keys in `tests/` and product code (`RuntimeEnvironmentProfile(` has one product construction, `runtime_boundary_adapters.py@H:51`; five test files construct it). |
| B3 | `intergrax/applications/_shared/runtime_boundary_adapters.py` | Declares `PROFILE_PROJECTIONS` for all 12 mirrored pairs (identity maps for the seven equal pairs; for the five divergent ones every Tier-3-only field is listed in `not_projected` with a reason naming the consuming route, for example `retention_days`: "consumed from the full Tier-3 profile by memory_runtime_bridge.py"), verifies each against the real Tier-3 class at import time (an unmapped new field then fails every import of the module, in CI first), and replaces the five inline `model_validate(model_dump(...))` calls with `PROFILE_PROJECTIONS[...].project(...)`. The 38 reason strings (35 + PolicyRules 2 + Adaptive 1) are written from the consumer table in `FACT_CHECK.md` F-31 and re-grepped at step 0. |
| B4 | `intergrax/applications/_shared/reliability_runtime_bridge.py` | Line 34 uses the same registry entry (`project` of the Reliability projection) instead of its own `model_validate(model_dump)`. |

Not projected in code at `H` (only declared and verified): Guardrail, ContextDecision, ApplicationSecurity, Evaluation, Orchestration, PolicyRules, Adaptive; their parity gate is `PT-G01` (S4a). Proposal Q-WB-03 recommends the explicit `not_projected` route (behaviour unchanged); the alternatives are adding Tier-0 fields (behaviour change) or running the verification first.

### 4.C Provider composition roots (DP-18; EQOA-A4-020)

Method (read-only, no test run): AST scan of every `.py` under `intergrax/runtime`, `intergrax/applications` and `applications` (excluding `tests` and `vendor_knowledge`) for imports of `intergrax.integrations.providers*` or `intergrax.integrations._shared*`, at module level or inside functions. Result at `H` (`PROVIDER_IMPORT_CLASSIFICATION.tsv`): 59 modules plus 43 `intergrax/runtime/vendor_knowledge/**` files. The 59 are 22 under `intergrax/runtime` (22 + 43 = 65, the WP-1 S5 count), 6 under `intergrax/applications/_shared` and 31 under `applications/` (17 non-script modules that import `integrations.providers.*`, the WP-1 count; 8 scripts; 6 that import only `integrations._shared`). Proposed classes (a proposal; the membership is Q-WB-04):

| Class | Count | Meaning | Treatment |
|---|---|---|---|
| ROOT | 19 | module whose purpose is to select or construct a concrete provider and hand a neutral contract onward (for example `runtime/notifications/factory.py`, `runtime/transport/bootstrap.py`, `runtime/persistence/{sqlite_composition,postgresql_opens,integration_profile_wiring}.py`, `runtime/capacity/production_adapters.py`, LKW `host/message_bus_wiring.py`) | the closed named list (`INTEGRATIONS.md` by `PT-D12`, ledger by `PT-G01`); may not grow |
| CONST | 22 | imports only constants, path resolvers, catalog manifests or one validator (for example `runtime/events/store.py`, `runtime/task_memory/store.py`, `human/store.py`, LKW `connected_source_*`) | per-file ledger entry with a count ceiling; no code change |
| IMPL | 3 | provider-bound store implementations or migration helpers inside Tier-1 (`postgresql_decision_checkpoint_persistence.py`, `postgresql_task_checkpoint_store.py`, `human_decision_legacy_disposition.py`) | named with ceiling; membership is Q-WB-04 |
| DEMO_PROOF | 9 | lab route and operator/proof scripts | exception class (DP-20 context); no code change |
| SHARED_DEMO | 6 | `integrations._shared` imports in demo/proof apps (3), `model_runtime_proof` (2) and LKW `host/factory.py` (S8b, W-A) | DP-20 exception list (`PT-G01`); `host/factory.py` is W-A's facade item |
| AREA | 43 files | `runtime/vendor_knowledge/**` (WP-1 S5a, provider-contribution area; the clause was not located) | one named area with a per-file count ceiling |

Effect on code: at `H` no ROOT/CONST/IMPL module needs to be rerouted for the list to be closed, so W-B has no product edit for DP-18; the decided rule "everything else behind `IntegrationProfile`" is satisfied by the classification plus the gate (`PT-G01`) unless the Operator rejects the membership of some module, in which case that module is added to the write set (Q-WB-04 option B). Two openers — `runtime/events/store.py` and `runtime/task_memory/store.py` (A4-024) — are CONST/LAB-only roots per D-FC1-01 and are untouched by W-B. `runtime/capacity/production_adapters.py` (FC-2), `runtime/notifications/factory.py` and `runtime/interactions/**` (FC-3) are ROOT and are edited by those tickets, not here. Deliverable of Part C: the TSV, a per-class count table, and the exact text proposal for the closed list as an input to `PT-D12` and `PT-G01`; nothing is written to `INTEGRATIONS.md`, the ledger or the roadmap by this ticket.

### 4.D Serving (DP-19; EQOA-A4-040)

Facts at `H`. Direct `uvicorn.run(` call sites (AST/grep over `intergrax agents applications scripts`): seven under `applications/` (`attestation_demo/host/main.py:19`, `governed_contractor_application/host/package_b_strict_production_serving_command.py:159`, `intergrax_assistant_application/host/main.py:19`, `lab_application/host/main.py:22`, `local_workspace_application/host/main.py:107`, `poc_template_application/host/main.py:19`, `research_application/host/main.py:101`) and three real calls under `intergrax/` (`cli/run.py:43`, `harness/app.py:215`, `scaffold/external_project/app.py:32`); `scaffold/new_application.py:689` is a call inside a generated-code string (WP-1 counted it as a fourth). The hosting lifecycle engine `run_hosted_application` (`intergrax/hosting/runner.py:297`) has one caller, `local_workspace_application/hosting/foreground.py:65`. The LKW has two serving paths with different defaults: `host/main.py:run_reference_production` binds `LOCAL_WORKSPACE_BACKEND_HOST` default `0.0.0.0` (`:105`) and calls `uvicorn.run` (`:107`), while `hosting/profile.py:75-76` defaults the same variable to `127.0.0.1`. Other `0.0.0.0` Python defaults: `research_application/host/main.py:99`, `research_application/host/settings.py:20`, `governed_contractor_application/host/package_b_strict_production_serving_command.py:155,161` (variable `GOVERNED_CONTRACTOR_BACKEND_BIND_HOST`, not `…_BACKEND_HOST`). Container images set the host variable to `0.0.0.0` explicitly (for example `local_workspace_application/docker/Dockerfile:42`, `research_application/docker/Dockerfile:42`), so for those the variable value already is the opt-in. The generic settings default is `127.0.0.1` (`intergrax/applications/contracts/settings.py:19`). The Dockerfile `CMD ["uvicorn", …, "--host", "0.0.0.0"]` lines and `docker_templates.py` are CLI direct-serve, not `uvicorn.run`; they are outside this part (Q-WB-05).

| # | File | Change |
|---|---|---|
| D1 | `applications/local_workspace_application/host/main.py` | `run_reference_production` keeps its lifecycle deploy/activate steps and then serves through `run_local_workspace_hosted_application(process_composition=composition, settings=resolved_settings)` instead of `uvicorn.run`; the `0.0.0.0` default disappears (the hosted profile already defaults to `127.0.0.1`). |
| D2 | `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py` | The two tests that capture `uvicorn.run(asgi_app)` (`:58-100`, `:114-150`) are rewritten to observe the hosted runner (the hosted runtime exposes injectable `application_factory`/`server_factory`, `hosting/runtime.py@H:107-127`) while keeping their assertions about activation before serving and store continuity. |
| D3 | `applications/research_application/host/main.py` | Host default `127.0.0.1` (`:99`); the entrypoint stays a direct `uvicorn.run` declared non-production (no hosted facade exists for research). |
| D4 | `applications/research_application/host/settings.py` | `backend_host` default `127.0.0.1` (`:20`). |
| D5 | `applications/research_application/.env.example` | `RESEARCH_BACKEND_HOST` example value (`:13`, today `0.0.0.0`) becomes the loopback value with a comment that `0.0.0.0` is the explicit opt-in. |

The declaration of "non-production" for the other direct entrypoints is documentation (`APPLICATION_HOSTING.md`, `PT-D12`) plus the count gate (`PT-G01`); this ticket adds no marker constant to code. Bind change for the governed command is part G, not D, because the qualification compose file starts that command without a bind variable (`docker-compose-local-qualification/docker-compose.yml:26` sets `INTERGRAX_GCA_STRICT_SERVE_EXEC`; no `BIND_HOST` anywhere in `deploy/`), so a loopback default would make the qualification container unreachable unless the compose file or its env example sets the opt-in.

### 4.E Importlib prefix allowlist (DP-21; follow-on of EQOA-A7-021)

Facts at `H`. Loaders that import a module named by serialized or configured data: `applications/_shared/wiring.py:57-80` (`load_callable(import_path)`, single caller `wiring.py:209` for `binding.factory_path`), `agent_resolution.py:42` (`import_path` of an `AgentBinding`) and `:99` (module derived from an already resolved agent type: `{package}.contract`), `migration_wiring.py:337` (`_load_manifest_for_package`, caller `:388`; it catches `ImportError` and returns `None`), `mcp_import_guard.py:60` (`load_mcp_server_builder`, no non-test caller found). Same class, not named by WP-1 S11 and still excluded from this ticket: `runtime/execution/context_engine_resolver.py:29` (`engine_ref` from the profile), `dev_support/agent_registry_bootstrap.py:61`, `model_inference/registry/vision_adapter_registry.py:60`, `speech_adapters/registry/speech_adapter_registry.py:43`, `integrations/_shared/p4/factories.py:28` (Q-WB-07). `intergrax/agents/readiness/scoreboard.py:44-57` is no longer excluded: the alpha run confirmed it as the third A4-007 seam (`_load_agent_instance` imports `removeprefix("agents.")` of a string from the fleet inventory, `scoreboard.py@H:41-57`; tests: `tests/unit/agents/readiness/test_production_readiness_scoreboard.py`; also read by `scripts/gates/check_agent_production_readiness.py` and `report_agent_production_readiness.py`, not edited). Tests use arbitrary top-level packages in `import_path`/`factory_path` (`echo.echo_agent.EchoAgent`, `example_agent.factory.build_agent`, `agents.search.factory:build`), so a global default-deny would break them.

Seam facts taken over from the W-A plan (section 7a, read there at `H`; re-read here for the three files): the three confirmed seams check the imported object only after the import has run (module-level side effects included). `agent_resolution.py:21-47` `resolve_agent_type` does `importlib.import_module(module_path)` of any serialized `import_path` and checks the `Tier2Agent` subclass afterwards; its second `import_module` (`:99`, `{package}.contract` of an already resolved class) is not caller-supplied and is not changed. `migration_wiring.py:332-337` `_load_manifest_for_package` imports `f"{package}.manifest"` where `package` comes from a directory glob of `*_application` folders (`:53`) and swallows `ImportError` into `None`, which `check_application_migrations` reports as "migrations present but manifest could not be loaded" (`:388-390`). `scoreboard.py:44-57` `_load_agent_instance` imports `removeprefix("agents.")` of a module string. Tier-2 agent import paths are bare top-level names (`echo.echo_agent.EchoAgent` in `tests/unit/applications/contracts/test_agent_resolution.py@H:17-31`; the scoreboard strips `agents.`), so a literal prefix such as `agents.` would describe no real path and would not refuse the probe's synthetic `zz_synth_t2.agent`: for these seams the prefix must be the registered package name. Rule of this design: the permitted set is derived from one existing registered source, never a second hand-typed list; which source supplies each set (the agent registry bootstrap `intergrax/dev_support/agent_registry_bootstrap.py:61`, the product manifest registry, the fleet inventory read by `load_fleet_inventory`, or the repository `agents/` top-level packages when the inventory path is caller-supplied) is UNVERIFIED and is step 0 of Part E (Q-WB-06). Refusal is before any import call: `agent_resolution` raises the existing `AgentImportError`; `migration_wiring` returns `None` (the existing caller already reports it; the package must also be a bare identifier); `scoreboard` raises `RuntimeError`; `wiring.py` and `mcp_import_guard.py` raise the new typed errors of E1. Not in Part E (census items, handed to `PT-G01`): `context_engine_resolver.py:29`, `intergrax/cli/doctor_*`, `dev_support/agent_registry_bootstrap.py:61` (Q-WB-07). Existing tests that drive these seams with real packages (`test_agent_resolution.py`, `test_migration_wiring.py`, `test_production_readiness_scoreboard.py`) must stay green; one using a package outside the registered set is edited (conditional paths). Classification consequence: `intergrax/agents/readiness/scoreboard.py` has no EBH-7/B6-F10 C4 rule (section 10).

| # | File | Change |
|---|---|---|
| E1 | `intergrax/applications/_shared/import_allowlist.py` (new) | `ImportPrefixAllowlist` (frozen; whole-package prefixes without trailing dot; `permits(path)` = equal or `startswith(prefix + ".")`; rejects empty, relative and non-identifier paths), no hand-typed platform constant (the first draft proposed `("intergrax", "agents")`; replaced by the registered-source rule above), `require_import_allowed(path, allowlist, fail_closed, purpose)`, and two typed errors that are not `ImportError` subclasses (`ImportPathNotAllowedError`, `ImportAllowlistRequiredError`) so `_load_manifest_for_package` cannot turn a refusal into `None`. Placed under `intergrax/applications/_shared/` because that prefix is classified by the EBH-7/B6-F10 C4 rules (`COMPOSITION_CONFIGURATION`, `_ebh7_f10_canonical_inputs.py@H:272-276`) while `intergrax/utils/` has no rule (`UNKNOWN`, `classify_change` `_ebh7_f10_ecap_normative_matrix.py@H:654-660`). |
| E2..E5 | `wiring.py`, `agent_resolution.py`, `migration_wiring.py` (package from a directory glob: the allowlist covers the application packages), `mcp_import_guard.py` | Each loader gains keyword-only `allowed_prefixes: ImportPrefixAllowlist | None = None` and `fail_closed: bool = False` and calls `require_import_allowed` before `importlib.import_module`; with `fail_closed=True` and no allowlist it raises `ImportAllowlistRequiredError`; with an allowlist it enforces in every posture. Host composition supplies the set derived from the registered source plus the application's own package (`wiring_fails_closed(env)` decides `fail_closed`). `agent_resolution.py` and `migration_wiring.py` use the refusal forms of the seam facts above (`AgentImportError`, `None`) instead of the E1 typed errors. |
| E9 | `intergrax/agents/readiness/scoreboard.py` (confirmed A4-007 seam) | `_load_agent_instance` requires the module string to start with `agents.` and its top-level package to belong to the inventoried agent set before `importlib.import_module`, otherwise `RuntimeError`; the check is local to the module (no import of the E1 helper from a Tier-2-facing path); this loader has no posture input (a readiness tool), so the check is unconditional (no LAB opt-out), proposed in Q-WB-06. The two gate scripts that call the scoreboard are not edited. |
| E6..E8 (conditional) | `environment_conformance.py:65`, `environment_diff_wiring.py:108`, `roster_agent_contract_authority.py:117,131,137` | Threading edits only if the step-0 census shows the call is reachable from a STRICT/PRODUCT host build (nine product callers of `resolve_agent_type_from_binding`/`resolve_agent_contract_from_binding` at `H`). |

### 4.G Governed Contractor strict serving on `run_hosted_application` (inside this ticket, not split; conditional on the open A4-040 ruling)

`package_b_strict_production_serving_command.py` (199 lines) opens a strict serving session (`open_gca_strict_production_serving_session_from_single_node_production_profile_v1`), serves with its own `uvicorn.run(factory, host=…, factory=True)` (`:159`) and owns close handling (`:166-185`); the LKW precedent is a four-module hosting adapter (`hosting/{profile,runtime,boundary,foreground}.py`, `hosting/profile.py` builds a `HostedApplicationProfile`). Part G adds a Governed Contractor hosting package of the same shape, makes the command call `run_hosted_application`, moves the session close into the hosted lifecycle and sets the bind default to `127.0.0.1` with the explicit opt-in documented in the compose file, the single-node env example and the runbook. Write set: section "Write set". It rewrites three PROD-Q/R7-Q qualification files that pin this command (`tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py` patches `sys.modules["uvicorn"]` and asserts the command shape; `…_p1_strict_production_reachability.py`; `_prod_q_r1_r7_q_p1_strict_reachability.py`): that is a gate-meaning edit and the C10/C15 freeze applies (no production configuration change without separate authorisation). It stays inside this ticket (no split, section 1b) and keeps its own approval questions (Q-WB-08, Q-WB-12). Order against `T-AF2-RW-SUPPLY`: both tickets touch `package_b_strict_production_serving_command.py`; they are serialized, whichever merges second rebases onto the first, and neither rewrites the other's hunks (section 9). Part G lands only if the Operator rules B-2 A4-040 in scope (Q-WB-12); without that ruling the Part G paths and tests of the write set are not implemented. They carry class `cond` in the write set and are outside the firm approval scope.

## Write set

Branch: product commits of this ticket go only to the branch `arch-freeze/rw-train` (the RW train integration branch); no other branch receives them. Using that branch needs the Operator's authorisation (`AUTHORIZATION_REQUESTS.md`, AUTH-WB-06). One path per line below; product paths are ACT-N11 sensitive (every path matching `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`, `scripts/governance/iiac_common.py@H:22`).

### Product paths, firm (17; the two new files are `intergrax/contracts/profile_projection.py` and `intergrax/applications/_shared/import_allowlist.py`)

```
intergrax/runtime/wiring/llm_resolver.py
intergrax/applications/_shared/llm_resolver.py
intergrax/runtime/nexus/engine/runtime_state.py
intergrax/contracts/profile_projection.py
intergrax/contracts/host_profile_slices.py
intergrax/applications/_shared/runtime_boundary_adapters.py
intergrax/applications/_shared/reliability_runtime_bridge.py
intergrax/applications/_shared/import_allowlist.py
intergrax/applications/_shared/wiring.py
intergrax/applications/_shared/agent_resolution.py
intergrax/applications/_shared/migration_wiring.py
intergrax/applications/_shared/mcp_import_guard.py
intergrax/agents/readiness/scoreboard.py
applications/local_workspace_application/host/main.py
applications/research_application/host/main.py
applications/research_application/host/settings.py
applications/research_application/.env.example
```

### Product paths, conditional (only if the step-0 census requires; part E threading)

```
intergrax/applications/_shared/environment_conformance.py
intergrax/applications/_shared/environment_diff_wiring.py
intergrax/applications/_shared/roster_agent_contract_authority.py
```

### Product paths, conditional on a Part C ruling (B-2 A4-020; none at `H`)

Only the files that the Operator rules a bypass (Q-WB-13); candidates named in section 1b, not findings. Nothing is edited without the ruling; the list below is the full candidate set, one path per line.

```
intergrax/runtime/execution/postgresql_decision_checkpoint_persistence.py
intergrax/runtime/long_running/postgresql_task_checkpoint_store.py
intergrax/runtime/migration/human_decision_legacy_disposition.py
applications/local_workspace_application/slack_companion/companion.py
```

### Product paths, part G (class `cond`: conditional on the open A4-040 ruling, Q-WB-12; inside this ticket, Q-WB-08; serialized with `T-AF2-RW-SUPPLY` on the first path; 4 new under `hosting/`, 4 modified)

```
applications/governed_contractor_application/host/package_b_strict_production_serving_command.py
applications/governed_contractor_application/hosting/__init__.py
applications/governed_contractor_application/hosting/profile.py
applications/governed_contractor_application/hosting/runtime.py
applications/governed_contractor_application/hosting/boundary.py
applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml
applications/governed_contractor_application/deploy/docker-compose-local-qualification/RUNBOOK.md
applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.env.example
```

### Documentation under a product prefix

```
intergrax/llm_adapters/USAGE.md
```

(For part G the runbook above is also documentation.)

### Tests under a product prefix (counted as product paths by the prefix rule)

```
applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py
```

### Tests, new (not product paths; outside the EBH-7 identity allowlist, so merged before the final pin)

```
tests/unit/applications/test_wb_llm_resolver_canonical.py
tests/unit/applications/test_wb_llm_resolver_delegation.py
tests/unit/contracts/test_wb_profile_projection.py
tests/unit/applications/test_wb_profile_projection_registry.py
tests/unit/applications/test_wb_serving_bind_defaults.py
tests/unit/applications/test_wb_import_allowlist.py
```

Conditional on a Part C ruling (Q-WB-13), one more new file:

```
tests/unit/applications/test_wb_provider_roots.py
```

### Tests, existing, retargeted only (patch strings or source pins; the two `tests/unit/architecture/` files are gate tests, Q-WB-09)

```
tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py
tests/unit/llm_adapters/routing/test_evaluating_adapter.py
tests/unit/llm_adapters/routing/test_routing_metering.py
tests/unit/llm_adapters/routing/test_secondary_evaluating_wrap.py
tests/unit/architecture/test_ebh_2e_r6_r2_r3_routing_evaluator_end_to_end_injection.py
tests/unit/architecture/test_ebh_2e_r6_r1_r5_tier1_failover_chain_ordering_convergence.py
```

### Tests assigned by the RW plan (firm; paths from appendix A1 of `T-AF2-RW-PLAN`, new; the content is owned by that plan and is not designed here, Q-WB-14)

```
tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py
tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py
```

### Tests, part G (class `cond`, only with Part G)

```
tests/unit/applications/test_wb_gca_hosted_strict_serving.py
tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py
tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py
tests/qualification/prod_q/_prod_q_r1_r7_q_p1_strict_reachability.py
```

### Documents owned elsewhere, not edited here

```
docs/project/architecture/LLM_ADAPTERS.md
docs/project/architecture/INTEGRATIONS.md
docs/project/architecture/APPLICATION_HOSTING.md
docs/project/architecture/PLATFORM_FOUNDATION.md
```

(`PT-D12`; the roadmap, the FRZ checklist and EDER rows belong to `PT-D09`/`PT-D10`/EDER.)

### Part view and rules


The same product paths grouped by part:

| Part | Firm product files | New |
|---|---|---|
| A | `intergrax/runtime/wiring/llm_resolver.py`, `intergrax/applications/_shared/llm_resolver.py`, `intergrax/runtime/nexus/engine/runtime_state.py`, `intergrax/llm_adapters/USAGE.md` (doc) | 0 |
| B | `intergrax/contracts/host_profile_slices.py`, `intergrax/applications/_shared/runtime_boundary_adapters.py`, `intergrax/applications/_shared/reliability_runtime_bridge.py` | 1: `intergrax/contracts/profile_projection.py` |
| C | none (data deliverable) | 0 |
| D | `applications/local_workspace_application/host/main.py`, `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`, `applications/research_application/host/main.py`, `applications/research_application/host/settings.py`, `applications/research_application/.env.example` | 0 |
| E | `intergrax/applications/_shared/wiring.py`, `agent_resolution.py`, `migration_wiring.py`, `mcp_import_guard.py`, `intergrax/agents/readiness/scoreboard.py` | 1: `intergrax/applications/_shared/import_allowlist.py` |
| E conditional | `intergrax/applications/_shared/environment_conformance.py`, `environment_diff_wiring.py`, `roster_agent_contract_authority.py` (only if the census requires) | 0 |
| C conditional (B-2 A4-020) | the files ruled a bypass among the four candidates of section "Product paths, conditional on a Part C ruling" | 0 |
| G (`cond`, inside this ticket) | `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py`, `…/deploy/docker-compose-local-qualification/docker-compose.yml`, `…/deploy/docker-compose-local-qualification/RUNBOOK.md`, `…/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.env.example` | 4: `applications/governed_contractor_application/hosting/{__init__,profile,runtime,boundary}.py` |

Totals by class: firm = 19 product paths (17 modified including the documentation file and the test under `applications/`, 2 new) and 14 tests under `tests/` (6 new, 6 retargeted, 2 new assigned by the RW plan), 33 paths in all; conditional (`cond`) = 15 product paths (3 Part E threading callers, 4 Part C candidates, 8 Part G) and 5 tests (the Part C test and 4 Part G tests), 20 paths in all; at most 34 product paths if every condition is ruled in. Part G is conditional on the open A4-040 ruling and is excluded from the firm approval scope.

Tests and census-found edits (under `tests/`, not product paths, but outside the EBH-7 identity allowlist and therefore merged before the final pin): new files `tests/unit/applications/test_wb_llm_resolver_canonical.py`, `test_wb_llm_resolver_delegation.py`, `test_wb_profile_projection_registry.py`, `test_wb_serving_bind_defaults.py`, `test_wb_import_allowlist.py`, `tests/unit/contracts/test_wb_profile_projection.py` (6; plus `tests/unit/applications/test_wb_gca_hosted_strict_serving.py` for G and the conditional `tests/unit/applications/test_wb_provider_roots.py` for a Part C ruling). Existing test files edited only to retarget patch paths or source pins: `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` (`:54,58`), `tests/unit/llm_adapters/routing/{test_evaluating_adapter.py:52, test_routing_metering.py:93, test_secondary_evaluating_wrap.py:66}`, `tests/unit/architecture/test_ebh_2e_r6_r2_r3_routing_evaluator_end_to_end_injection.py` (`:187,191` patch strings and the three tests `:92-124` that read or introspect both resolver files) and `…r6_r1_r5_tier1_failover_chain_ordering_convergence.py` (`:12` imports `_create_base_llm_adapter` from the Tier-3 module). The two `tests/unit/architecture/` files are gate tests (`pytest.mark.gate`): whether re-expressing their source pins is a gate-meaning change is not decided here (Q-WB-09). For G only: `tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py`, `…_p1_strict_production_reachability.py`, `_prod_q_r1_r7_q_p1_strict_reachability.py`.

Documentation: only `intergrax/llm_adapters/USAGE.md` (A4) and, for G, the runbook above. Documents owned elsewhere and not edited here: `LLM_ADAPTERS.md`, `INTEGRATIONS.md`, `APPLICATION_HOSTING.md`, `PLATFORM_FOUNDATION.md` (`PT-D12`, Operator-owned canon text), the roadmap and FRZ checklist (`PT-D09`/`PT-D10`), EDER rows.

Forbidden paths: `.github/`, `pyproject.toml`, `uv.lock`, `scripts/` (gates are `PT-G01`, `T-IGX-043-E/F`), the roadmap, the FRZ checklist, `IIAC_*`, PDC-X record directories, `governance/` except this ticket and its plan, Docker files other than the four Part G paths, `intergrax/runtime/execution/host_orchestration_*`, `harness_host_runtime.py`, `nexus_loop.py` (W-A/FC-1), `intergrax/runtime/notifications/**`, `intergrax/runtime/interactions/**` (FC-3), `intergrax/runtime/capacity/**`, `memory_wiring.py`, `wiring_posture.py` (FC-2), every `intergrax/integrations/**` file.

## Negative tests

New test files (names are proposals; the files do not exist yet). Every test fails at `H`/`B` unless marked control (06, 07, 13, 17, 24) or record (25); 26 needs Part G and 31 a Part C ruling. Node ids are `file::function`.

| id | configuration | expected refusal | node id |
|---|---|---|---|
| NT-WB-01 | AST of `applications/_shared/llm_resolver.py` | no call or import of `create_adapter`, `create_adapter_with_failover`, `ModelRouter`, `wrap_routing_evaluating_adapter`; no module-level assignment named `_last_routing_evaluation`; every public function body calls the Tier-1 module (red at `H`) | `tests/unit/applications/test_wb_llm_resolver_delegation.py::test_nt_wb_01_facade_creates_no_adapters` |
| NT-WB-02 | two adapters resolved with two routing profiles in two threads | `consume_routing_evaluation(adapter_a)` returns A's evaluation and never B's; Tier-1 module has no attribute `_last_routing_evaluation` (red at `H`) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_02_routing_record_is_per_adapter` |
| NT-WB-03 | adapter resolved through the Tier-3 façade with `llm_routing_profile` set; `configure_llm_tracker` with that adapter | the routing-rule diag event is emitted (red at `H`: the Tier-3 record is never consumed) | `tests/unit/applications/test_wb_llm_resolver_delegation.py::test_nt_wb_03_tier3_routing_record_is_consumed` |
| NT-WB-04 | Tier-1 `resolve_llm_adapter(env, required_llm_features=("X",))` with an adapter that lacks `X` | `LLMCapabilityMismatchError` (red at `H`: Tier-1 has no check) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_04_required_features_mismatch_refused` |
| NT-WB-05 | Tier-1 env with `llm_routing_profile` | returned adapter is a routing-evaluating wrapper and re-evaluates before a call (red at `H`) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_05_routing_profile_wraps_adapter` |
| NT-WB-06 | no `llm_profile`, no `INTERGRAX_LLM_*` | `LLMProviderNotConfiguredError` from both entry points (control, green at `H`) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_06_unconfigured_provider_refused` |
| NT-WB-07 | evaluator raises `AllowlistViolationError` | propagates unchanged (control) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_07_allowlist_violation_propagates` |
| NT-WB-08 | synthetic source class with one extra field and an identity map | `UnmappedProfileFieldError` naming the field | `tests/unit/contracts/test_wb_profile_projection.py::test_nt_wb_08_unmapped_source_field_refused` |
| NT-WB-09 | synthetic target with `extra="ignore"` | `ProjectionTargetIgnoresExtrasError` | `tests/unit/contracts/test_wb_profile_projection.py::test_nt_wb_09_ignore_target_refused` |
| NT-WB-10 | map entry for a field that the source does not have | `StaleProjectionEntryError` | `tests/unit/contracts/test_wb_profile_projection.py::test_nt_wb_10_stale_map_entry_refused` |
| NT-WB-11 | the 12 real pairs | `verify` passes; `not_projected` sizes are exactly 17 / 9 / 9 / 2 / 1 for Memory / Cost / Reliability / PolicyRules / Adaptive and every reason is non-empty (red at `H`: no registry) | `tests/unit/applications/test_wb_profile_projection_registry.py::test_nt_wb_11_twelve_pairs_verified_with_exact_unprojected_counts` |
| NT-WB-12 | `MemoryProfile.model_validate({"retention_days": 7})` and the same for Cost, Reliability, Context | `ValidationError` (red at `H`: `extra="ignore"`) | `tests/unit/applications/test_wb_profile_projection_registry.py::test_nt_wb_12_slices_forbid_unknown_keys` |
| NT-WB-13 | `application_profile_to_runtime_profile(profile with retention_days=7)` | result equals the previous output for all Tier-0 fields (control: no behaviour change) | `tests/unit/applications/test_wb_profile_projection_registry.py::test_nt_wb_13_projection_output_unchanged` |
| NT-WB-14 | LKW `run_reference_production` with `LOCAL_WORKSPACE_BACKEND_HOST` unset | hosted runner is called; no `uvicorn.run`; bind host `127.0.0.1` (red at `H`: `0.0.0.0`) | `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_14_lkw_reference_production_hosted_loopback` |
| NT-WB-15 | AST of `run_reference_production` | no call named `uvicorn.run` | `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_15_lkw_reference_production_has_no_uvicorn_run` |
| NT-WB-16 | research `host/main.py` and `ResearchBackendSettings()` with the variable unset | bind `127.0.0.1` (red at `H`) | `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_16_research_default_loopback` |
| NT-WB-17 | `LOCAL_WORKSPACE_BACKEND_HOST=0.0.0.0` | honoured (control: the explicit value is the opt-in) | `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_17_explicit_all_interfaces_honoured` |
| NT-WB-18 | `fail_closed=True`, allowlist `("intergrax","agents")`, `load_callable("os.system")` (the test supplies the set; in production the set comes from the registered source of section 4.E) | `ImportPathNotAllowedError`; `importlib.import_module` is not called (red at `H`) | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_18_load_callable_outside_allowlist_refused` |
| NT-WB-19 | `fail_closed=True`, no allowlist | `ImportAllowlistRequiredError` | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_19_fail_closed_without_allowlist_refused` |
| NT-WB-20 | `resolve_agent_type(agent_type=None, import_path=...)` with the registered Tier-2 package set: the probe's synthetic `zz_synth_t2.agent.Agent` and `subprocess.Popen` (A4-007 b1); the registered `echo.echo_agent.EchoAgent` as control inside the same test | `AgentImportError` before `importlib.import_module` is called for the two refused names; `echo.echo_agent.EchoAgent` resolves | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_20_resolve_agent_type_outside_allowlist_refused` |
| NT-WB-21 | `_load_manifest_for_package` with the registered application package set: `"os"`, a dotted name `"a.b"`, the probe's synthetic `"zz_synth_app"` (A4-007 b2; the package arrives from a directory glob of `*_application` folders); `check_application_migrations` over a folder whose name is outside the set | `None` without any `importlib.import_module` call; `check_application_migrations` reports "migrations present but manifest could not be loaded" (no exception escapes) | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_21_migration_manifest_outside_allowlist_refused` |
| NT-WB-22 | `load_mcp_server_builder("os", symbol="system")` with allowlist | refused | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_22_mcp_builder_outside_allowlist_refused` |
| NT-WB-23 | prefix boundary: `intergrax_evil.x`, `..x`, empty, `intergrax` alone vs `intergrax.x` | `intergrax_evil.x`, relative and empty refused; `intergrax.x` permitted | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_23_prefix_boundary` |
| NT-WB-24 | LAB (`fail_closed=False`, no allowlist) `echo.echo_agent.EchoAgent` | loads (control) | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_24_lab_unrestricted_control` |
| NT-WB-25 | source scan of `applications/**` product `*.py` for `uvicorn.run(` | count equals the declared set (9 after D1; recorded, not a standing gate: the standing count gate is `PT-G01`) | `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_25_uvicorn_run_site_count_record` |
| NT-WB-26 | package-B command with ready session | serves through `run_hosted_application`; default bind `127.0.0.1`; session closed once | `tests/unit/applications/test_wb_gca_hosted_strict_serving.py::test_nt_wb_26_package_b_serves_through_hosted_runner` |
| NT-WB-27 | one adapter shared by two tenants; evaluate A, evaluate B, consume A, consume B (the nt12 interleaving, X-A4-016) | each tenant consumes only its own evaluation; `consume_routing_evaluation(adapter, tenant_id="A")` never returns B's | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_27_shared_adapter_record_is_tenant_scoped` |
| NT-WB-28 | record stored for tenant A; consume with tenant B and with `tenant_id=None` | `None` for both (a record with a tenant is not returned to another or an unscoped consumer) | `tests/unit/applications/test_wb_llm_resolver_canonical.py::test_nt_wb_28_foreign_tenant_consume_returns_nothing` |
| NT-WB-29 | the 38 `not_projected` entries of the registry (X-A8-003, nt13b made checkable) | every reason names an existing module whose source contains the field name; a field with no consumer found fails the test (refusal at verification time) | `tests/unit/applications/test_wb_profile_projection_registry.py::test_nt_wb_29_every_unprojected_field_has_a_verified_consumer` |
| NT-WB-30 | `agents.readiness.scoreboard._load_agent_instance("os", path)`, `"subprocess"`, `"agents_evil.x"`, `"agents.zz_synth_t2.agent"` (a top-level package outside the inventoried set; A4-007 b4) | `RuntimeError` before `importlib.import_module` is called; an inventoried `agents.<name>` is permitted | `tests/unit/applications/test_wb_import_allowlist.py::test_nt_wb_30_scoreboard_loader_outside_agents_refused` |
| NT-WB-31 | conditional on a ruling (Q-WB-13): AST scan of the files ruled a bypass in Part C (candidates: the three IMPL modules and `slack_companion/companion.py`) | none imports `intergrax.integrations.providers*` or `intergrax.integrations._shared*`; the provider choice is reached through `IntegrationProfile` only (red at `H` for a ruled file; exists only if a file is ruled) | `tests/unit/applications/test_wb_provider_roots.py::test_nt_wb_31_ruled_bypass_files_use_integration_profile` |

Counts: 31 tests = 23 negative tests that fail at `H` or refuse a configuration (01-05, 08-12, 14-16, 18-23, 27-30) + 5 controls that are green at `H` (06, 07, 13, 17, 24) + 1 scan record (25) + 1 test for part G (26) + 1 conditional negative test for a Part C ruling (31, exists only if a file is ruled a bypass). Without part G and without a Part C ruling: 29 tests, 23 negative. The alpha probes nt12a/nt12b/nt13a, R02 b1/b2/b4 and R03 v4 are the confirming probes at `B`; NT-WB-02/03/27/28 close X-A4-016, NT-WB-18..23/30 close A4-007 (b1 by 20, b2 by 21, b4 by 30), NT-WB-08..13/29 close X-A8-003 (X-A4-024 is closed by `NT-FC1-16..19`); NT-WB-14..17/25/26 carry B-2 A4-040 and NT-WB-31 carries B-2 A4-020 (conditional).

## 7. Regression scope (209 node IDs, each defined at `H`, written out in `REGRESSION_NODES_VERIFIED.txt`)

| Part | Existing files (node count) |
|---|---|
| A | `tests/unit/applications/test_llm_routing_resolver.py` (1), `test_environment_llm_wiring.py` (1), `tests/unit/llm_adapters/test_llm_provider_selection.py` (8), `routing/test_evaluating_adapter.py` (3), `routing/test_routing_metering.py` (9), `routing/test_secondary_evaluating_wrap.py` (2), `tests/unit/architecture/test_ebh_2e_r1…` (10), `…r4…` (6), `…r6_r1_r5…` (9), `…r6_r2_r3…` (5), `…r6_r2_routing_evaluator_usage_tracker_composition_ports.py` (7), `tests/unit/agents/authoring/test_acp_routing_trace_integration.py` (1), `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` (1) = 63 |
| B | `tests/unit/applications/test_environment_profile.py` (10), `test_acp_session_host_wiring.py` (4), `test_harness_reliability_wiring.py` (10), `tests/unit/runtime/kernel/test_session_reliability_acp_prod4.py` (2), `tests/unit/architecture/test_ebh_2d_b_environment_profile_contract_ownership.py` (34), `tests/unit/integrations/providers/llm_guardrail/test_pba_fix_c_guardrail_provider_config.py` (11), `applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py` (21) = 92 |
| D | `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py` (5, edited), `tests/hosting/test_hosted_foreground.py` (5), `tests/hosting/test_hosted_profile.py` (6) = 16 |
| E | `tests/unit/applications/test_migration_wiring.py` (5), `contracts/test_agent_resolution.py` (6), `test_production_factory_invocation_ac5.py` (14), `test_mcp_surface_opt_in.py` (6), `test_mcp_public_api_surface.py` (2), `tests/unit/agents/readiness/test_production_readiness_scoreboard.py` (5) = 38 |

Total 209 (63 + 92 + 16 + 38). Not run by this ticket, listed for the audit: the three qualification tests that monkeypatch the Tier-3 `resolve_llm_adapter` (`tests/qualification/pdc_x/d1_repair/test_d1_repair_durable_projection_inputs.py:42`, `…/test_d1_repair_pipeline_hook.py:47`, `…/p8_repair/test_r6_durable_projection_slug.py:39`) and `tests/unit/agents/conftest.py:55` (Tier-1 patch): they are inputs of the step-0 patch-target census, not part of the bounded commands, because their run time is not bounded here (Q-WB-09 lists the census).

## 8. Test commands (one process at a time under the shared test lock; none run when drafted) and order of work

Order of work inside the ticket (red-first: step 1 tests are written and shown red against the unmodified tree at the parent commit, then the fix):

0. Preflight: `git fetch`, record `origin/development`; re-run the three scans (`git grep` for `llm_resolver` importers, patch targets and `consume_routing_evaluation`; AST scan of provider imports; `git grep` for `uvicorn.run`, `importlib.import_module`); freeze the census into the ticket checkpoint; confirm `T-IGX-043-B` is merged and `wiring_posture.py` exists; decide E6-E8; stop if any census number differs from this plan by more than the named tolerance (resolver importers ±2, provider-import classes ±3, `uvicorn.run` sites ±1).
1. Part B1+B3 tests, then `profile_projection.py`, then the registry and the two bridges (B2 last: it flips `extra`).
2. Part A tests, then A1, A2, A3, then the retargeting edits R1-R6; V2.
3. Part E tests, `import_allowlist.py`, the four loaders (+ conditional callers).
4. Part D tests, D1-D5; V4.
5. Part C deliverables (TSV, list text) re-generated against the step-0 head.
6. A4 and the USAGE.md text; static checks (`git diff --stat`, forbidden-path scan).
7. Part G (class `cond`; inside this ticket; only if the Operator rules B-2 A4-040 in scope, Q-WB-12; order against `T-AF2-RW-SUPPLY` on the serving command file as set by the overlap matrix). A Part C bypass ruling (Q-WB-13), if any, adds its files after step 5.
8. Hand over; the ticket never self-declares `ready_for_audit` PASS.

Commands (V1..V7; the pytest paths are the node-ID files of section 7):

- V1: `uv run pytest tests/unit/applications/test_wb_llm_resolver_canonical.py tests/unit/applications/test_wb_llm_resolver_delegation.py tests/unit/contracts/test_wb_profile_projection.py tests/unit/applications/test_wb_profile_projection_registry.py tests/unit/applications/test_wb_serving_bind_defaults.py tests/unit/applications/test_wb_import_allowlist.py -p no:cacheprovider`
- V2 (A): the 13 Part A files of section 7 in one command.
- V3 (B): the 7 Part B files in one command.
- V4 (D): `uv run pytest applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py applications/local_workspace_application/tests/hosting/test_hosted_foreground.py applications/local_workspace_application/tests/hosting/test_hosted_profile.py -p no:cacheprovider`
- V5 (E): the 6 Part E files in one command.
- V6: `git grep` census commands of step 0 (no process fan-out).
- V7: `uv run python -m scripts.governance.igx_gov ticket` before push.
- G (if ruled in scope): `uv run pytest tests/unit/applications/test_wb_gca_hosted_strict_serving.py tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py -p no:cacheprovider`; PROD-Q node runs need the Operator's separate authorisation (freeze C10/C15).

No diagnostic, broad or S1 suite is part of this ticket.

## 9. Overlaps and merge order

| With | Shared files / dependency | Rule |
|---|---|---|
| W-A (`PT-WA`) | `host_orchestration_environment_spec_builder.py:34,171` imports the Tier-3 resolver (one of the S1a imports); `runtime_boundary_adapters.py` and `host_profile_slices.py` are the neutral-contract hosts W-A will extend | W-B does not edit the builder; the Tier-3 façade keeps `resolve_environment_llm_adapter`/`resolve_llm_adapter` so W-A can pre-resolve the adapter on the Tier-3 side (the builder already accepts `llm_adapter`, `@H:136,169-171`). Proposal (Q-WB-08, item c): Part B merges before W-A on the two profile files, so W-A's new projections use `ProfileProjection` |
| FC-2 (`T-IGX-043-B`) | `wiring_posture.py`; ROOT modules `production_adapters.py`, `memory_wiring.py` | FC-2 merges first (order B, A, C, F); W-B imports the helper and edits none of its files |
| FC-1 (`T-IGX-043-A`) | `nexus_loop.py`, `harness_host_runtime.py`, `host_orchestration_environment_spec_builder.py`, the two openers | none edited by W-B; FC-1 rebases onto W-A/W-B on the three hotspot files (transcription item 6); W-B only lists the openers in Part C |
| FC-3 (`T-IGX-043-C`) | `runtime/notifications/factory.py`, `runtime/interactions/**`, `guardrail_runtime_bridge.py`, seven `applications/*/host/factory.py` | none edited by W-B; classified ROOT |
| `T-IGX-043-E/F` | gate scripts, CONFIG-X scan | none |
| `PT-G01` | gates: single resolver (ledger), profile parity (S4a), direct-serving count (S7a), closed roots (S5), importlib prefix check (S11), EBH-3 scan extension (DP-20: a gate-meaning change, not delegated) | W-B writes only its own tests; standing gates and any expected-red policy are `PT-G01`'s (Q-WB-10); W-B does not carry or order them |
| `T-AF2-RW-SUPPLY` | `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py` (Part G) and `applications/_shared/*_wiring.py` (`wiring.py`, `migration_wiring.py` in Part E; the master plan lists `PT-FC2`, `PT-WB`, `T-AF2-RW-SUPPLY` as overlapping there, `MEP-AF2_v2.1_final.md:388`, session artefact) | serialized, never concurrent; which merges first is fixed by the overlap matrix of `T-AF2-RW-PLAN`; the second rebases and keeps the first's hunks; this plan assumes no order |
| W-A (`T-IGX-043-G`), A4-007 | W-A drops section 7a (Part F) of its plan and W-B carries the three seams (Q-RW-02) | W-B edits no file of `FC/WA/`; the overlap matrix records A4-007 under W-B for `agent_resolution.py`, `migration_wiring.py`, `scoreboard.py`; W-A keeps the registry-bypass gate work |
| `PT-D12` | `LLM_ADAPTERS.md`, `INTEGRATIONS.md`, `APPLICATION_HOSTING.md` text | W-B delivers the class table and the count table as input; edits nothing |

## 10. ACT-N11 / IIAC consequence and ARCH-FREEZE impact

All firm product files match `PRODUCT_PATH_PREFIXES` (`iiac_common.py@H:22`); any merge breaks the IIAC activation until the single RB-2 re-pin (DP-58, ACT-N11), and a revert does not restore it. W-B merges only inside the integrated wave train `RW`, never alone. New tests and docs are outside the EBH-7 identity allowlist and must be in place before the final pin. Classification of the new and edited paths for EBH-7/B6-F10 (rules read at `H` in `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:273-599` and the longest-prefix lookup `_ebh7_f10_ecap_normative_matrix.py:654-660`): `intergrax/agents/readiness/scoreboard.py` matches no rule (the list has `intergrax/applications/`, `contracts/`, `runtime/`, `llm_adapters/` and others, but no `intergrax/agents/` prefix and no exact-path rule), so `classify_change` returns `UNKNOWN`, which is a BLOCKER at the final pin; this plan edits no F10 input (the rendered matrix is regenerated by `PT-D11`) and hands the path to `PT-Q04` as the W-A plan did for its own `agents/` paths (Q-WB-06 sub-question). The two new paths: `intergrax/contracts/profile_projection.py` falls under rule `intergrax/contracts/` (`CONTRACT`, `_ebh7_f10_canonical_inputs.py@H:277-281`); `intergrax/applications/_shared/import_allowlist.py` under `intergrax/applications/` (`COMPOSITION_CONFIGURATION`, `@H:272-276`); the Part G package under the enumerated `applications/governed_contractor_application/` rule. This plan proposes no re-pin of its own.

| Finding | Class (WP-1 / master plan) | What W-B does | What closes it (not claimed here) |
|---|---|---|---|
| A4-016 | `BLOCKS` (effective; linked FRZ-OWN-03/05, FRZ-PLG-04) | one canonical resolver, delegating façade, context-bound record | NT-WB-01..07 green + `LLM_ADAPTERS.md` names the canonical module (`PT-D12`) + the ledger gate against a second `resolve_llm_adapter` (`PT-G01`) + independent audit at the exact wave SHA |
| A8-003 | `CONDITIONS` (escalates if `X-A8-003` shows a needed value dropped) | declared field maps, forbid targets, import-time verification | NT-WB-08..13 + parity gate (`PT-G01`) + the Q-WB-03 ruling + audit |
| A4-020 | `CONDITIONS` (effective `BLOCKS` via FRZ-BND-05, FRZ-CFG-06, FRZ-CTR-02, FRZ-OWN-05, FRZ-PLG-04) | classification and the closed-list proposal; no reroute at `H` | `INTEGRATIONS.md` list (`PT-D12`) + ledger and count ceilings (`PT-G01`) + audit; FC-2/FC-3 for the fail-closed edits of their ROOT modules |
| A4-040 (with B-2 alpha: 7 direct apps, one strict serving command) | `CONDITIONS` (worst case `BLOCKS` if STRICT serving is found on a direct path, Campaign B CK-S1) | LKW reference production on the hosted runner; loopback defaults; Part G (inside this ticket) moves the strict serving command only if the Operator rules it in scope | declaration in `APPLICATION_HOSTING.md` (`PT-D12`) + count gate (`PT-G01`) + G decided + audit; FRZ-PRD-04/05/09 live parts stay PROD-Q |
| A4-020 (with B-2 alpha, conditional) | see above | files ruled a bypass go through `IntegrationProfile` (conditional paths, Q-WB-13); NT-WB-31 | the Operator's ruling per file + NT-WB-31 + audit |
| A4-024 | `CONDITIONS` | none (FC-1) | FC-1's closure |
| A4-007 (alpha CONFIRMED, shared with W-A) | `BLOCKS` | the allowlist at the three confirmed seams (E3, E4, E9; E2 and E5 same class); the three seams are taken over from the W-A plan | NT-WB-18..23, NT-WB-30 green + AST boundary check at `B`/`H` (not this ticket) + the W-A part of the finding (Q-WB-08 d) + audit |
| X-A4-016 (alpha CONFIRMED) | `BLOCKS` | tenant-scoped routing record | NT-WB-02/03/27/28 green + audit; part of A4-016 closure |
| X-A8-003 (alpha CONFIRMED) | `CONDITIONS` | explicit projection with verified consumer routes | NT-WB-08..13/29 green + Q-WB-03 ruling (nt13a criterion) + audit |
| X-A4-024 (alpha CONFIRMED) | `CONDITIONS` | none (FC-1 `NT-FC1-16..19`) | FC-1's closure |
| A7-021 (follow-on) | `UNKNOWN`/effective `BLOCKS` | the allowlist code | AST boundary check at `B` and `H` documented (not this ticket) + audit |

## 11. Independent audit design (separate non-implementing session at the exact wave head SHA)

The auditor is neither the implementer nor a reviewer of this ticket (C3, DP-03, `PT-Q02`), works in a scratch worktree at the audited SHA and replays: (1) mutation check: for each negative test, remove the corresponding hunk in the scratch tree and show the test fails, then restore; (2) census diff: the step-0 censuses re-run at the wave SHA (zero `_last_routing_evaluation`; `resolve_llm_adapter` definitions only in the canonical module and the delegating façade; zero `model_validate(...model_dump` of profile slices outside the registry; `extra="ignore"` absent on the four slices; `uvicorn.run` sites equal the declared set; loader sites equal E2-E5 plus the declared exclusions; Part C counts 19/22/3/9/6/43 unchanged or each change explained); (3) ACT-N11 reconciliation: `git diff --name-only <RW base> <RW SHA>` intersected with the three prefixes equals the declared firm set plus the other tickets' files; (4) regression V1..V5 at that SHA; (5) behaviour check of A: the ACP callers with a routing profile receive the wrapper and the live evaluation is emitted once (record consumed once); (6) text check: no sentence in the ticket, plan or docs claims closure, PASS or independence. The verdict vocabulary and publication are the auditor's and the Operator's; this plan asserts none.

## 12. Risks, rollback, effort, size

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | A4: ACP callers of the Tier-1 resolver start live re-evaluation and capability checks (behaviour change) | Q-WB-02; NT-WB-04/05; ACP routing tests in V2 |
| R-2 | Existing tests patch Tier-3 private names (`_create_base_llm_adapter`, `create_adapter_for_routing_evaluation`) and two gate tests read both resolver sources | names preserved at Tier-1; retargeting only; gate edits flagged (Q-WB-09) |
| R-3 | Weak-keyed record: an adapter wrapped after resolution loses its record (silent no-diag, as today when no record exists); an adapter class that is not weak-referenceable or not hashable cannot be registered | NT-WB-03; documented; the record is keyed on the object returned by the resolver; the step-0 census of adapter classes decides between the registry and a typed attribute on the wrapper (Q-WB-01) |
| R-4 | B2 flips four slices to `forbid`: any caller passing unknown keys now fails | step-0 census; B2 is the last edit of Part B |
| R-5 | Import-time `verify` makes a future Tier-3 field without a map entry fail every import of the adapter module | intended fail-closed behaviour; the parity gate gives the same signal earlier |
| R-6 | D1 changes LKW reference production semantics (restart policy `on_failure(3)`, signal handling, app factory `create_local_workspace_backend_app` with host readiness) | the hosted facade is the existing production path; D2 keeps the two behavioural assertions; Q-WB-05 |
| R-7 | G: the PROD-Q package-B serving gate and the qualification compose file pin the current command and default bind; the same command file is edited by `T-AF2-RW-SUPPLY` | Part G is a separable unit of this ticket (no separate ticket), needs the Operator ruling on scope and gates, freeze C10/C15; serialization with RW-SUPPLY in section 9 |
| R-8 | E: a refusal raised as `ImportError` would be swallowed by `_load_manifest_for_package` | errors are not `ImportError` subclasses; NT-WB-21 |
| R-9 | Combined blast radius with FC-2/FC-3 STRICT guards on shared test environments | the integrated-tree census of FC-2 section 8a is repeated once on the wave tree; this ticket adds no STRICT-only guard except E |

Rollback: before merge, revert the ticket's commits on the wave branch; each part is a separable unit (A together with R1-R6; B as a unit; D; E; G). After merge a revert is possible per part but does not restore the IIAC activation; the RB-2 rebind must be redone. The Tier-3 façade keeps public names, so a revert of A restores the previous two-resolver state without touching callers.

Effort (S/M/L): A `M`, B `M`, C `S` (data), D `S`, E `M`, G `L`; the ticket is `L` overall (19 firm product paths and 14 firm tests; 15 conditional product paths and 5 conditional tests, Part G among them; at most 34 product paths). No split: the ticket stays one unit by agreement of the ARCH-FREEZE orchestrator and the EQO-A planning session (Q-RW-02, Q-RW-15). The earlier split proposal (H-1 = A + B, H-2 = C + E, H-3 = D + G) is withdrawn from the plan and kept only as an alternative in Q-WB-08.
