# T-IGX-043-H — EQO-A remediation code wave, package W-B: one LLM resolver, explicit profile projections, closed provider composition roots, serving defaults, importlib prefix allowlist

## Scope

Product-path ticket for master-plan node `PT-WB` of the integrated remediation wave (DP-11), covering the Operator decisions DP-16 (the Tier-1 `runtime/wiring` host LLM resolver is canonical, the Tier-3 resolver delegates, the routing record stops being a module global), DP-17 (the mirrored profile models become declared projections with a field map that fails on unmapped fields; no `extra="ignore"` on a projection target), DP-18 (a closed named list of provider composition roots, everything else behind `IntegrationProfile`, the list may not grow), DP-19 (direct `uvicorn.run` entrypoints are non-production and counted; production serving goes through `run_hosted_application`; default bind `127.0.0.1` with the explicit value `0.0.0.0` as the opt-in) and DP-21 (a package-prefix allowlist for the importlib manifest loaders). Findings: EQOA-A4-016 (with alpha item X-A4-016), EQOA-A8-003 (with X-A8-003), EQOA-A4-020, EQOA-A4-040, EQOA-A4-007 (alpha, three loader seams); EQOA-A4-024 only as a classification input because decision D-FC1-01 gave it to FC-1 (`T-IGX-043-A`), which contradicts the master-plan row of `PT-WB` (open question Q-WB-11); EQOA-A5-001, EQOA-A2-009 and EQOA-A7-021 only as dependencies (the gates and the scan extension are `PT-G01`, a gate-meaning change owned elsewhere; DP-20 is context, not scope). Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589); the code was re-read at `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42` and every file in the write set is unchanged since the baseline (the only product delta is three `intergrax/rag/graph/` files).

Parts (plan sections 4.A..4.G): A resolver (4 files incl. `intergrax/llm_adapters/USAGE.md`); B profiles (3 modified, 1 new: `intergrax/contracts/profile_projection.py`); C provider roots (a data deliverable, no product edit at `H`: 59 modules plus 43 `vendor_knowledge` files classified ROOT 19, CONST 22, IMPL 3, DEMO_PROOF 9, SHARED_DEMO 6, AREA 43); D serving (LKW reference production onto the hosted facade, loopback defaults for LKW and research); E importlib allowlist (5 modified incl. `intergrax/agents/readiness/scoreboard.py`, 1 new: `intergrax/applications/_shared/import_allowlist.py`, up to 3 conditional callers); G Governed Contractor strict serving on `run_hosted_application`, kept inside this ticket (no split), class `cond` (not firm, not covered by this ticket's approval line), serialized with `T-AF2-RW-SUPPLY`, landing only if the Operator rules B-2 A4-040 in scope (RW plan Q-RW-16 b; its PROD-Q node runs need a separate authorisation, Q-RW-16 d) (4 new, 4 modified, three PROD-Q gate files rewritten). Write set by class: firm = 19 product paths (17 modified, 2 new) and 14 tests under `tests/` (6 new, 6 retargeted, two of them gate tests and a ruling question, 2 new assigned by the RW plan: Q-WB-14), 33 paths in all; conditional (`cond`) = 15 product paths (3 Part E threading callers, 4 Part C bypass candidates, 8 Part G) and 5 tests (the Part C test and 4 Part G tests), 20 paths in all; 209 regression node IDs written out and verified by definition search; 23 negative tests among 31 (NT-WB-26 needs Part G and NT-WB-31 a Part C ruling). Campaign B-1 alpha confirmed items routed to W-B (plan section 1a): A4-007 (three loader seams, `BLOCKS`, shared with W-A; `intergrax/agents/readiness/scoreboard.py` joins Part E), X-A4-016 (the routing record must be tenant-scoped, `BLOCKS`), X-A8-003 (`CONDITIONS`) and X-A4-024 (`CONDITIONS`, closed by FC-1's `NT-FC1-16..19`, no W-B file). Campaign B-2 alpha routing (plan section 1b): A4-007 is taken over from the W-A plan as Part E (facts and design of its section 7a); A4-040 is conditional on the DP-19 scope ruling (Part G lands only if the Operator rules it in scope; the strict serving command file is serialized with `T-AF2-RW-SUPPLY`); A4-020 is conditional (W-B edits a file only if the Operator rules it a bypass, up to four candidates, plan section "Write set"). Product commits of this ticket go only to the branch `arch-freeze/rw-train`. The exact paths (sections "Write set" and "Negative tests" of the plan), the forbidden paths, the per-part design with `path@sha:line`, the named negative tests, the audit design, risks, rollback and effort are in the plan. The size of the ticket is an open question (Q-WB-08); the ticket is kept whole by agreement of the ARCH-FREEZE orchestrator and the EQO-A planning session (Q-RW-02, Q-RW-15), which is not an Operator answer; the closed-list membership (Q-WB-04), the gate-test edits (Q-WB-09) and the other questions are in `OPEN_QUESTIONS.md`; none has an Operator answer.

Overlaps: W-A (`PT-WA`) removes the Tier-3 resolver import from `host_orchestration_environment_spec_builder.py` (W-B does not edit it and keeps the Tier-3 function names so W-A can pre-resolve the adapter; Part B proposed to merge before W-A on `runtime_boundary_adapters.py` and `host_profile_slices.py`); FC-2 (`T-IGX-043-B`) merges first and owns `wiring_posture.py`, which Part E imports; FC-1, FC-2 and FC-3 edit their own ROOT modules (W-B edits none and none of the three hotspot files); `PT-G01` owns the standing gates; `PT-D12` owns the canon text; `T-AF2-RW-SUPPLY` is serialized with Part G on the serving command file and shares the `applications/_shared/*_wiring.py` area with Part E (order set by the overlap matrix of `T-AF2-RW-PLAN`).

Prerequisites before implementation starts: master-plan nodes `SCOPE` and `DS-2` (DP-13..DP-23, decided in the decision record), `PT-G01` stated as a dependency, milestone `M1` as confirmed by the Operator (at the head read for this draft, `PDC_INDEX.md` lines 44-49 still say the P9 artifacts await the Operator's final confirmation while `OD-PDC-X-CLOSE.md` merged by PR #603 carries a signature line dated 2026-11-16; this ticket does not decide whether P9 is closed), authorisation `AR-40` and the approval line of this ticket, `T-IGX-043-B` merged. This ticket merges only inside the integrated wave train; every merge of its product files breaks the IIAC activation until the single RB-2 re-pin (ACT-N11), and a revert does not restore it. After the wave: independent audit at the exact wave SHA by a session that did not implement or review it. Nothing in this ticket claims closure, PASS, FRZ promotion or independent verification.

## Plan digest

Plan digest: 33b428e5c010a9d33061ba2f6b3a55e06cd48c68ab2f46ac1ed68a785e39f7e4

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-H.md`](plans/PLAN_T-IGX-043-H.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T09:37:59+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD.md@e2c519b6d8ea4fbfb9b66cf57df766625f08ea36
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

One process at a time under the shared test lock; none run when this ticket was drafted. The commands V1..V7 of plan section 8 are the contract:

- V1 new tests: `uv run pytest tests/unit/applications/test_wb_llm_resolver_canonical.py tests/unit/applications/test_wb_llm_resolver_delegation.py tests/unit/contracts/test_wb_profile_projection.py tests/unit/applications/test_wb_profile_projection_registry.py tests/unit/applications/test_wb_serving_bind_defaults.py tests/unit/applications/test_wb_import_allowlist.py -p no:cacheprovider`
- V2 resolver regression (Part A): the 13 files of plan section 7, group A, in one command (63 node IDs).
- V3 profile regression (Part B): the 7 files of group B in one command (92 node IDs).
- V4 serving regression (Part D): `uv run pytest applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py applications/local_workspace_application/tests/hosting/test_hosted_foreground.py applications/local_workspace_application/tests/hosting/test_hosted_profile.py -p no:cacheprovider`
- V5 importlib regression (Part E): the 6 files of group E in one command (38 node IDs).
- V6 step-0 censuses by `git grep` and AST scan (no pytest); repeated by the auditor.
- V7 `uv run python -m scripts.governance.igx_gov ticket` before push.
- Part G (class `cond`, outside the firm contract; only if the Operator rules B-2 A4-040 in scope): `uv run pytest tests/unit/applications/test_wb_gca_hosted_strict_serving.py tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py -p no:cacheprovider`; PROD-Q node runs need a separate authorisation (freeze C10/C15).
- Reviewer check: `git diff --stat` lists exactly the plan's write set; the product-path files are the declared ones.
- Independent wave audit at the exact RW SHA (plan section 11) is a separate session and is not part of this contract.

## Out of scope

`intergrax/runtime/execution/host_orchestration_*`, `harness_host_runtime.py`, `nexus_loop.py` and the two persistence openers (W-A, FC-1); `intergrax/runtime/notifications/**`, `runtime/interactions/**`, `runtime/capacity/**`, `memory_wiring.py`, `wiring_posture.py` and every provider-selector change (FC-2, FC-3); `intergrax/integrations/**`; the standing gates, the seam ledger, the EBH-3 scan extension and the count gates (`PT-G01`, gate-meaning changes not delegated); the owner-document text of `LLM_ADAPTERS.md`, `INTEGRATIONS.md`, `APPLICATION_HOSTING.md`, `PLATFORM_FOUNDATION.md` (`PT-D12`), the roadmap, the FRZ checklist, EDER rows; other importlib loaders (`context_engine_resolver.py` and five more, Q-WB-07); Dockerfile and docker-templates CLI direct-serve lines; the six direct `uvicorn.run` sites other than the LKW, research and Part G command (declared non-production and counted only); IIAC activation and the RB-2 re-pin; GitHub settings; `.github/`, `pyproject.toml`, `uv.lock`; production configuration, Docker migration, new vendor integrations, S1 or diagnostic suites; live capacity or PROD-Q qualification; FRZ promotion.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_f10529af-ffa4-4b47-9469-cfaffee42f5e

(the PT-WB implementer session; recorded here, outside the plan text and outside the three approval lines).

## Implementer checkpoint (outside the plan text and the approval block)

Recorded after the independent review of PR #654. The plan, its digest and the approval lines are unchanged.

- A4-007 seams b1 and b2: residual. Part E lands the loader capability only (`allowed_prefixes`, `fail_closed`, typed refusals; `scoreboard.py` seam b4 is closed unconditionally). No product caller passes an allowlist or `fail_closed`, so b1 (`resolve_agent_type`, `load_callable`) and b2 (`_load_manifest_for_package`) still import without a prefix check on production paths. PT-WB does not claim A4-007 closed, and NT-WB-20 and NT-WB-21 prove the loader refusal, not the closure of b1/b2. Ruling of the ARCH-FREEZE orchestrator: keep the lane order (PT-WB before PT-FC2); do not widen the write set. Owner: PT-FC2 (creates `wiring_posture.py`). Closure condition: threading callers edited after `wiring_posture.py` is on the train, plus a product derivation of the permitted sets from the registered source.
- Step-0 census of the threading callers E6-E8 (read at the PR head): product callers of `resolve_agent_type_from_binding` / `resolve_agent_contract_from_binding` are `wiring.py:109,127,219`, `environment_conformance.py:65`, `environment_diff_wiring.py:108`, `roster_agent_contract_authority.py:117,131,137`; `load_callable` has one caller, `wiring.py:214`; `check_application_migrations` is called by `scripts/maintenance/check_application_migrations.py:24` and `scripts/gates/check_application_production_gates.py:229`; `load_mcp_server_builder` has no product caller. E6-E8 are not edited.
- Deviation from plan A2: the Tier-3 module still builds the Tier-3 wrapper type (`RoutingEvaluatingLLMAdapter` subclass), supplied to the canonical resolver through the `RoutingAdapterWrapper` port, so existing `isinstance` checks against the Tier-3 type keep working; the canonical default wraps with the platform class. The Tier-3 module creates no base adapter. A later ticket may retarget those `isinstance` checks to the platform class.
- Suppression: `# noqa: UP047` on `build_registry_from_manifest` in `wiring.py` because `test_production_factory_invocation_ac5.py` matches the exact header `def build_registry_from_manifest(`.
- Residuals handed over: NT-OWN-05-1 guards 1 of the 5 named responsibilities (host LLM resolver); the other four (plugin discovery, context-window tables, tool-input guard, evaluator contract) have no step-0 scan. The closed provider-root list keeps `applications/local_workspace_application/host/factory.py`, which no longer imports a provider package since PT-WA (removing it would tighten the guard; for PT-G01). The routing record of a run without a tenant is keyed as unscoped and consumed only by an unscoped run.
- Out-of-write-set red tests caused in part by this change, for the known-red register: `tests/acceptance/llm_routing/test_mid_run_budget_routing_e2e.py::test_materialize_runtime_config_wraps_evaluating_adapter` and `tests/acceptance/llm_routing/test_production_metering_routing_e2e.py::test_production_metering_mid_run_budget_swap_with_tracker` (red at the train head already; they also patch Tier-3 private names that moved to `intergrax.runtime.wiring.llm_resolver`).
- Contracts ratchet by the prescribed scripts (`typ_measure.py`, `measure_typ05.py`) over `intergrax/contracts`, base vs head: Any 138 to 138, object_annot 357 to 357, getattr 11 to 11, type_ignore 0 to 0, TYP-05 lookup tables 5 to 5.
- The rewritten LKW bootstrap tests are red at the train head in this environment (memory provider admission), so their new assertions that the hosted runner receives the activated composition are not shown green here.

## RW traceability (GA.2R)

Traceability only (RW-PLAN-AUDIT finding M7, ticket T-IGX-078-B). This section grants no write authority and changes no scope, plan, plan digest, approval or lifecycle state.

| FRZ row of the RW CODE_FIX inventory | Negative test ids in the RW plan |
|---|---|
| FRZ-BND-05 | NT-BND-05-1, NT-BND-05-2 |
| FRZ-OWN-03 | NT-OWN-03-1 |
| FRZ-OWN-05 | NT-OWN-05-1 |
| FRZ-PRD-05 | NT-PRD-05-1, NT-PRD-05-2 |

Binding: `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json`. Lane, hard predecessors, merge position and mandatory tests of this ticket are in the live plan `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md`.

## AF-W3-PT-WB-01 executor settlement (2026-11-27)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-WB | T-IGX-043-H`. Authorization: AR-40 / OD-AF-PKT1. Base and execution SHA: `245d47177c7814234a8d98fc062ed1afce3b735d` (`origin/development`); PT-WA and SCOPE closure ancestors confirmed. No firm write-set product edits in this session (Parts A–E already on `development`); this pass is verification, OP-B evidence accounting, and ticket settlement.

**Implementation verdict:** `IMPLEMENTED_FIRM_SCOPE_WITH_EXACT_RESIDUALS` (not full ticket DONE: FRZ-BND-05 OP-B remains `EVIDENCE_MISSING`).

| Area | Status |
|---|---|
| Part A canonical LLM resolver + tenant-scoped routing | `ALREADY_IMPLEMENTED`; V1 green (37); NT-OWN-03 green at execution SHA |
| Part B explicit profile projections | `ALREADY_IMPLEMENTED`; V1 registry/projection tests green |
| Part C closed provider roots | `ALREADY_IMPLEMENTED` (data/gate); **NT-BND-05-2 FAIL** — `intergrax/applications/_shared/queue_worker_wiring.py` import outside `CLOSED_PROVIDER_ROOTS` (cross-ticket residual; file not in T-IGX-043-H write set; Q-WB-13 does not authorize A4-020 bypass edits) |
| Part D serving lifecycle | `ALREADY_IMPLEMENTED`; NT-PRD-05 / NT-WB-15 nodes green in targeted run; V4 local failures (LKW bootstrap / hosted foreground) match prior checkpoint (memory provider admission in this environment) |
| Part E importlib allowlist | `ALREADY_IMPLEMENTED`; NT-WB-20/21 green in V1; **b1/b2 caller threading** residual → PT-FC2 per prior checkpoint |
| Part G | `CONDITIONAL_NOT_AUTHORIZED` (Q-WB-12); not executed |
| FRZ-BND-05 OP-B (NT-BND-05-1) | `EVIDENCE_MISSING`; `BLOCKER=CANONICAL_NEGATIVE_TEST_WRITE_SET_MISSING` — `test_rw_bnd05_runtime_context_construction.py` not in firm write set; test still asserts `RuntimeConfig.production_mode` posture, not RuntimeContext construction scan |
| FRZ-OWN-05 | Historical `RED_AT_PREFIX` preserved; **NT-OWN-05-1 FAIL** at execution SHA: AST census includes `applications/*/docker/runtime-context/**/llm_resolver.py` mirrors (not firm implementation); host-resolver duplication guard otherwise satisfied at pin for canonical + façade only |

**Qualification (this session, one pytest process):** V1 PASS (37/37). V2 90 passed, 1 failed (`test_environment_llm_wiring_without_llm_selection_does_not_materialize_adapter`, `ConfigurationError` — out of firm WB write set). V3 89 passed, 3 failed (ACP session host / LKW profile closure — harness/enterprise coupling). V4 15 passed, 4 failed (LKW production bootstrap / hosted foreground — same class as prior LKW admission note). V5 68 passed, 1 xfail (known TRANSFERRED), 1 failed (`test_mcp_public_api_surface` import surface — outside Part E claim). Ruff (core write-set sample): PASS. Pyright (llm_resolver + profile_projection): 0 errors. V7 `igx_gov ticket`: NOT PASS locally (`FileNotFoundError`: `uv` not on PATH for regression-gate subprocess).

**OP-B / progressive evidence:** `OP_B_PENDING=12`; no `af2.red_first_ga2r.v1` publication (TASK-08 skipped). `FRZ_PROMOTION: NONE`.

**PT-WA handoff preserved:** NT-WA-23 roster isolation; harness.policy_smoke ×2 → T-IGX-043-A; OTLP ×2 → T-IGX-043-B; Pyright PT-WA not claimed full PASS; dynamic-import census not published as evidence.

**Lifecycle:** remains `plan_approved`; semantic close `IMPLEMENTED_FIRM_SCOPE_WITH_EXACT_RESIDUALS`; G2 OPEN; RB2 NOT_PERFORMED; independent PT-Q02 audit still required.

## AF-W3-PT-WB-02 executor settlement (2026-11-27)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-WB | AF-W3-PT-WB-02`. Authorization: AR-40 / OD-AF-PKT1 / OD-AF-W3-BND05-01 variant A.
Baseline: `c74d42f11c1806de418363ad31572ee9b8005364`; SCOPE closure ancestor `0a4755281f46653a8b741a26c8e26b0af4555775` confirmed at preflight.

**Implementation verdict:** `BND05_DETECTOR_AND_OWN05_MIRROR_DELIVERED`; FRZ-BND-05 OP-B remains `EVIDENCE_MISSING` (no mechanism-aligned `test_scan_flags_a_rogue_agent` prefix FAIL → pin PASS at identical `test_blob_sha`).

| Deliverable | Result |
|---|---|
| OD-AF-W3-BND05-01 | `DECIDED` variant A in operator register |
| Write-set amendment | `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` on T-IGX-043-H; preserved on T-IGX-043-A |
| FRZ-BND-05 detector | `scan_runtime_context_construction` + seeded `test_scan_flags_a_rogue_agent`; posture scan preserved |
| FRZ-BND-05 red-first | `EVIDENCE_MISSING` — self-test PASS at prefix overlay and at HEAD |
| FRZ-OWN-05 mirror fix | `applications/*/docker/runtime-context/**` excluded from resolver census; NT-OWN-05-1 green |
| V6 census | @ execution SHA: 2 canonical resolver definitions; 4 Tier-2 `RuntimeContext.build` unauthorized residuals (T-IGX-043-A) |
| V7 `igx_gov ticket` | PASS with `uv` on PATH |
| Cross-ticket | `queue_worker_wiring.py` closed-root violation preserved for PT-FC2 |
| OP-B settlement | `OP_B_PENDING=12`; `RED_AT_PREFIX=21`; `EVIDENCE_MISSING=12` |

**Lifecycle:** `plan_approved`; not full DONE; G2 OPEN; FRZ promotion none; PT-Q02 PENDING.

**Evidence SHA reconciliation (WB-01 vs WB-02):** WB-01 cited `245d47177c7814234a8d98fc062ed1afce3b735d` (EC-W3-07 merge) while published package commit was `c74d42f11` (parent `1d0181051fa034b5f45101755002e48f5dc35715`).

## AF-W3-PT-WB-03 executor settlement (2026-11-27)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-WB | AF-W3-PT-WB-03`. Authorization: AR-40; OD-AF-PKT1 (historical); OD-AF-W3-BND05-01 variant A; OD-AF-CLOSE-B2; OD-AF-CLOSE-B6.
Verified baseline ancestor: `3eb76b5ebafd69f4696db5be57f7085b3802e22f` on `origin/development`.

**Implementation verdict:** `BND05_PRODUCT_GATE_AND_PRE_FIX_RED_BASELINE`; FRZ-BND-05 OP-B remains `EVIDENCE_MISSING` (pre-fix RED captured; not `RED_AT_PREFIX` promotion).

| Deliverable | Result |
|---|---|
| Detector | Canonical symbol identity for `RuntimeContext` construction (`import`, alias, module alias, qualified reference); local same-name class excluded |
| Product gate | `test_scan_flags_a_rogue_agent` enforces Tier-2 scan; **RED** on four real `RuntimeContext.build` sites |
| Synthetic controls | `test_detector_synthetic_runtime_context_construction_controls` — PASS independent of product |
| Documented census (non-asserting) | `_DOCUMENTED_UNAUTHORIZED_CENSUS_SITES` — four paths; `boundary_demo_agent.py` **not** in T-IGX-043-A write set |
| OWN-05 | Mirror exclusion via `_resolver_definitions`; synthetic tests use the real census helper |
| Plan digest | Recomputed after section 13 amendment (OD-AF-W3-BND05-01 variant A) |

**Authority census (Tier-2 `RuntimeContext.build`, unauthorized):**

| Path | Line | Owner |
|---|---|---|
| `agents/boundary_demo/boundary_demo_agent.py` | 96 | unresolved product authority (outside T-043-A write set) |
| `agents/external_contractor_adapter/external_contractor_adapter_agent.py` | 112 | T-IGX-043-A / PT-FC1 |
| `agents/organization_worker/organization_worker_agent.py` | 118 | T-IGX-043-A / PT-FC1 |
| `intergrax/agents/authoring/acp_stub_reflex.py` | 69 | T-IGX-043-A / PT-FC1 |

**AF-W3-PT-WB-01 V2–V5 residual accounting (nine failures, execution SHA `245d47177c7814234a8d98fc062ed1afce3b735d`):**

| Node bucket | Count | Representative failure | Owner / remedy |
|---|---|---|---|
| V2 | 1 | `test_environment_llm_wiring_without_llm_selection_does_not_materialize_adapter` | out of firm WB write set |
| V3 | 3 | ACP session host / LKW profile closure harness | enterprise coupling; not WB firm claim |
| V4 | 4 | LKW production bootstrap / hosted foreground (memory provider admission) | environment admission |
| V5 | 2 | `test_mcp_public_api_surface` FAIL; one xfail TRANSFERRED | outside Part E surface claim |

Preserved handoffs: `queue_worker_wiring.py` → PT-FC2; importlib b1/b2 → PT-FC2; Part G `CONDITIONAL_NOT_AUTHORIZED`.

**Pre-fix RED baseline (canonical node):** see `.tmp/proof/af-w3-pt-wb-03/` after commit (log SHA256, test blob SHA, execution SHA). `RED_FIRST_EVIDENCE=INCOMPLETE`; `OP_B_PENDING=12`; `FRZ-BND-05` not promoted.

**Lifecycle:** `plan_approved`; not full DONE; G2 OPEN; PT-FC1 product fixes required for GREEN on canonical node; PT-Q02 PENDING.
