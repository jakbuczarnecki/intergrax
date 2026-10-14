# PROD-Q-R1-R7-Q-P0 — Qualification inventory & gate plan

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** — inventory and gate plan only; **does not** enter **PROD-Q-R1-R7-Q** execution, **does not** claim **R7-Q PASS**, **does not** promote any **FRZ-*** row |
| **Bounded child** | **PROD-Q-R1-R7-Q-P0** |
| **Parent** | **PROD-Q-R1-R7** — **BLOCKED** / **NOT CLOSED** |
| **Mandatory successor stage** | **PROD-Q-R1-R7-Q** — **NOT ENTERED** |
| **Branch** | `development` |
| **START_HEAD / inventory baseline** | `64f0f4474267672d4c18be683d9c33eb8a5afdfa` |
| **Origin alignment** | `origin/development` @ `64f0f4474267672d4c18be683d9c33eb8a5afdfa` (verified before authoring) |
| **Method** | Static source inspection + existing test discovery + recorded qualification SHAs; **no pytest / pyright executed in P0** |

## Canonical sources (read scope)

| Source | Role |
|--------|------|
| [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) | Architecture authority — §14.4, §15, §16; DEC-01–DEC-09 / DEC-09-EXT |
| [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) | **FRZ-*** applicability (contribution only) |
| [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) | Program stage disposition |
| [`PROD_Q_R1_R7_PREREQUISITE_CANONICAL_ACCEPTANCE_ALIGNMENT.md`](PROD_Q_R1_R7_PREREQUISITE_CANONICAL_ACCEPTANCE_ALIGNMENT.md) | D1–D4 lineage vs **R7-Q** entry |

**Disposition rule:** Historical **D1–D4** acceptance and slice qualification **≠** **PROD-Q-R1-R7-Q PASS**. P0 records what is reusable vs what still requires an **R7-Q** gate at **current HEAD**.

---

## A. Source and commit baseline

| Item | Value |
|------|-------|
| Repository | `jakbuczarnecki/intergrax` |
| Branch | `development` |
| **EXPECTED_START_HEAD** | `64f0f4474267672d4c18be683d9c33eb8a5afdfa` |
| **Verified** | `git merge-base --is-ancestor 64f0f4474267672d4c18be683d9c33eb8a5afdfa HEAD` → **true** @ P0 start |
| ADR ratification lineage | DEC-01–DEC-09 / DEC-09-EXT @ ADR R6 (decision provenance; not production qualification) |
| Implementation evidence pins (slice — not re-run in P0) | D1+D1b `17e928e4…`; D1-R6 `c25e7efe…`; D3 `f2dd8ecb…` (incl. D3-R1); D4 impl `9b6efa8d…`, evidence `0dd51fe1…` |

---

## B. Closed-world contract / owner / caller inventory (@ HEAD `64f0f447…`)

Legend: **SO** = semantic owner · **CO** = composition owner · **PC** = production caller (strict path).

### B.1 Deployment & physical unit (A1)

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Deployment port | `RuntimeDeploymentAdapter` · `intergrax/agent_distribution/deployment.py` | **Agent Distribution** | `VenvBundleColocatedRuntimeDeploymentAdapter` · `intergrax/agent_distribution/venv_bundle_colocated_deployment_adapter.py` | `wire_production_lifecycle_services` | `ActivationService.prepare_candidate` via `ProductionLifecycleLauncher.deploy_and_activate` |
| Prepared unit / drain | `PreparedDeployment`, `DrainStatus`, instance records | **AD** | `deployment.py` + SQLite / in-memory stores | D2 lifecycle wiring | `ActivationService`, coordinator paths |
| Production artifact binding | `resolve_production_artifact_root`, `assemble_production_registry_projection_input_bundle` | **Applications shared** | `production_registry_projection_input_bundle.py` | Launcher + D3 descriptor store | `ProductionLifecycleLauncher.deploy_and_activate` |
| Reference-only fake | `FakeInMemoryRuntimeDeploymentAdapter` | **AD (test/reference)** | `fake_in_memory_runtime_deployment_adapter.py` | `open_reference_production_process_lifecycle` | **Not** `open_production_process_lifecycle` |

### B.2 Activation & admission

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Activation orchestration | `ActivationService` | **AD** | `activation.py` | `ProductionLifecycleLauncher` | `deploy_and_activate` after governance |
| Physical coordination | `ActivationDomainCoordinator` | **AD** | `activation_domain_coordinator.py` | `build_activation_domain_coordinator_for_stores` | Colocated reserve + coordinator-verified open |
| Serving execution admission | `ServingExecutionEligibilityEvidence`, store-backed admission | **AD** (eligibility) · **Apps** (bridge) | `StoreBackedColocatedServingUnitExecutionAdmission` | `bind_colocated_serving_runtime_after_activation` | GCA task execution path |
| Topology guard | `validate_colocated_serving_admission_topology` | **Apps** | `colocated_serving_runtime_composition.py` | `bind_colocated_serving_runtime_after_activation` | Before handoff |

### B.3 Governance (canonical, not execution)

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Control-plane mutation auth | `ControlPlaneMutationAuthorizationBoundary` | **Governance** | `ProductionControlPlaneMutationAuthorizationBoundary` alias · `production_control_plane_governance_wiring.py` | D2/D4 injection | `authorize_scoped_control_plane_mutation` in launcher / strict serving |
| Tenant binding | `ApplicationEnvironmentTenantResolver` | **AD + Governance** | Injected on `ProductionControlPlaneGovernance` | D2 `open_production_process_lifecycle` | Activation authorization + D2 negative tests |

### B.4 Production lifecycle composition (DEC-03)

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Sanctioned opener | `open_production_process_lifecycle` | **Apps** | `production_process_lifecycle.py` | Self (single slot) | `open_strict_production_serving_session` · tests |
| Service wiring | `wire_production_lifecycle_services` | **Apps** | `production_lifecycle.py` | Called only from production opener | Requires non-fake `RuntimeDeploymentAdapter` + governance |
| Deploy / activate glue | `ProductionLifecycleLauncher` | **Apps (orchestration)** | `production_lifecycle.py` | Composition root | Strict serving session optional activation |
| Durable AD stores | `SqliteAgentDistributionStoreBundle` | **AD** | `sqlite_stores.py` | `build_durable_production_agent_platform_runtime` | Production opener |

### B.5 D3 recovery & projection

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Durable projection inputs | `RegistryProjectionInputStore` | **AP-10 seam** | `DescriptorBackedRegistryProjectionInputStore` · `durable_registry_projection_input_store.py` | D2/D3 wiring | `register` / `get` on restart |
| Post-restart reconcile | `reconcile_production_serving_environment_on_process_open` | **AD** (attest) · **Apps** (compose) | `production_process_serving_reconciliation.py` | `open_production_process_lifecycle` | Mandatory targets from durable serving store |
| Traffic-serving attest | `ActivationService.reconcile_traffic_serving_physical_unit_after_process_restart` | **AD** | `activation.py` | Reconciliation pipeline | Process open |

### B.6 D1b handoff & GCA strict serving (D4)

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Runtime handoff | `ServingRuntimeHandoff`, `establish_serving_runtime_handoff` | **Apps** | `serving_runtime_handoff.py`, `colocated_serving_runtime_composition.py` | After activation bind | Strict GCA factory |
| Colocated accessor | `VenvBundleColocatedServingUnitRuntimeAccessor` | **Apps** | `venv_bundle_colocated_serving_unit_runtime_accessor.py` | Handoff establishment | Resolver + physical gate |
| Strict serving session | `open_strict_production_serving_session` | **Apps** | `strict_production_serving_launcher.py` | D2 opener → D3 reconcile → optional activate → bind → GCA | Operator external ASGI |
| Strict ASGI factory | `create_governed_contractor_strict_production_asgi_app` | **GCA** | `strict_production_serving.py` | Post-handoff only | uvicorn factory entry |
| Legacy entry (fail-closed) | `main.run()` | **GCA** | `main.py` | — | **Rejected** in production (D4-03) |

### B.7 Applicable existing qualification modules (closed-world)

| Stage | Primary test modules |
|-------|---------------------|
| D0 / D0-R2 | `tests/unit/agent_distribution/test_prod_q_r1_r7_d0_*.py` |
| D1 | `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py` |
| D1b + admission | `tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py`, `test_prod_q_r1_r7_d1_r2_*` … `d1_r4_*` |
| D1-R6 chain | `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6*.py` |
| D2 | `tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py` |
| D3 / D3-R1 | `tests/unit/applications/test_prod_q_r1_r7_d3_*.py` |
| D4 | `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` |
| PROD-Q-R1 (adjacent) | `tests/unit/runtime/enterprise_reliability/test_prod_q_r1_pis_01_*.py`, `tests/unit/applications/governed_contractor/test_prod_q_r1_r*.py` |
| **R7-Q (missing)** | No `tests/qualification/**/test_prod_q_r1_r7_q*`; ADR §15 `tests/qualification/prod_q/` fake-reachability gate **not** specialized for R7 strict graph @ HEAD |

---

## C. Qualification matrix (21 mandatory dimensions)

**Coverage status legend**

| Status | Meaning |
|--------|---------|
| **SLICE-EVIDENCED** | Behavioral tests exist in D0–D4 modules; recorded SHA in slice qualification docs |
| **HEAD-STATIC** | Code/test presence verified @ `64f0f447…` without execution in P0 |
| **R7-Q-GAP** | Mandatory §15 / operator R7-Q scope not yet proven as integrated qualification |

**Classification legend:** see §D.

| # | Invariant (summary) | SO | Platform contract | CO | Production caller | FRZ (primary) | Code evidence (@ HEAD) | Existing test path | Evidence SHA (slice) | Coverage | Missing R7-Q gate | Runtime / resources | Classification | Next action |
|---|---------------------|----|--------------------|----|-------------------|---------------|------------------------|-------------------|----------------------|----------|-------------------|---------------------|----------------|-------------|
| 1 | Physical prepare + meaningful readiness on **real A1** adapter | AD | `RuntimeDeploymentAdapter.prepare` / `check_readiness` | `wire_production_lifecycle_services` | `ActivationService.prepare_candidate` | FRZ-PRD-04, FRZ-PRD-09 | `venv_bundle_colocated_deployment_adapter.py` | `test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py` (`test_prepare_*`, `test_runtime_initialization_*`) | `17e928e4…` (D1 bundle) | SLICE-EVIDENCED | Integrated strict path: deploy → READY on production opener w/ real bundle | Local `tmp_path` venv artifact | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice) | Reuse D1 module in R7-Q wave 1 replay @ HEAD |
| 2 | Immutable artifact identity + digest validation | AD + Apps | Revision + materialization record; `file://` production locator | Bundle assembly | `deploy_and_activate` | FRZ-PRD-03, FRZ-PRD-08 | `production_registry_projection_input_bundle.py`, adapter digest checks | D1: `test_wrong_digest_*`, `test_missing_artifact_*`, `test_wrong_locator_scheme_*` | `17e928e4…` | SLICE-EVIDENCED | End-to-end: governance → persist revision → prepare with **operator-trusted** artifact root policy | Trusted artifact roots config | **MISSING TARGETED GATE** | R7-Q wave 2 E2E strict launcher |
| 3 | Serving-unit identity + liveness attestation | AD | `ServingUnitAttestation`, `attest_serving_unit` | Activation gates | `ActivationService` commit/ready paths | FRZ-PRD-09, FRZ-SEC-04 | D0 orchestration + D1 live unit | D0 `test_prod_q_r1_r7_d0_*`; D1 attestation rows | `2b44cee…` D0; `17e928e4…` D1 | SLICE-EVIDENCED | Attestation on **traffic-serving** unit after full D4 session | Same as row 1 | **MISSING TARGETED GATE** | Combine D4 `test_d4_stale_physical_*` + live HTTP optional |
| 4 | Provider-neutral activation semantics | AD | `ActivationService` + port injection | Launcher | `deploy_and_activate` | FRZ-PLG-*, FRZ-RPL-* | No vendor branches in `activation.py` (D0 regression) | D0 `test_alternate_adapter*`; D1 `test_alternate_adapter_satisfies_protocol` | `2b44cee…`, `17e928e4…` | SLICE-EVIDENCED | Prove neutrality with **second contract-conforming** adapter through **production** opener (not only D1 class) | In-test Protocol impl | **MISSING TARGETED GATE** | Extend D2 `test_d2_typed_replaceable_adapter_wiring` pattern to full activate |
| 5 | Structural replaceability (second conforming implementation) | AD | `RuntimeDeploymentAdapter` Protocol | D2 injection | `open_production_process_lifecycle(..., deployment_adapter=)` | FRZ-RPL-*, FRZ-CTR-01 | D2 stand-in adapter | `test_d2_typed_replaceable_adapter_wiring` | D2 doc session (HEAD-STATIC) | SLICE-EVIDENCED (structural) | ADR §15: second impl + same `ActivationService` graph on **strict GCA** path | Local only | **MISSING TARGETED GATE** | R7-Q wave 2 |
| 6 | Authorized activation w/ canonical Governance | Governance + AD | `ControlPlaneMutationAuthorizationBoundary` | D2/D4 governance wiring | `deploy_and_activate(..., principal=)` | FRZ-SEC-04, FRZ-GOV-* | `production_control_plane_governance_wiring.py` | D2 `test_d2_governance_deny_*`; D4 `test_d4_governance_deny_*` | HEAD-STATIC | SLICE-EVIDENCED | Operator **production mutation policy profile** (not reference harness) | Policy fixture | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Operator-approved production boundary profile for R7-Q |
| 7 | DENY, stale revision, concurrency / CAS failures | AD | `atomic_commit_*`, coordinator linearization | D1-R6 coordinator | Activation + serving stores | FRZ-PRD-08, FRZ-STA-* | `activation_domain_coordinator.py`, stores | D1-R6 `r4`–`r8`, `r5_serving_admission_linearization`; D0 stale revision | `c25e7efe…` | SLICE-EVIDENCED | Cross-layer: DENY before any physical prepare on strict opener | SQLite local | **MISSING TARGETED GATE** | R7-Q adversarial matrix row |
| 8 | Cross-tenant negative scenarios | Governance + AD | Tenant resolver + authz evidence | D2 | `open_production_process_lifecycle` | **FRZ-TEN-*** | Resolver injection only | D2 `test_d2_wrong_tenant_resolver_blocks` only | HEAD-STATIC | **R7-Q-GAP** | Multi-tenant E2E: tenant A env cannot activate/serve tenant B | Two tenant fixtures | **MISSING TARGETED GATE** | **R7-Q-P1-EXEC** tenant pack; align `tests/qualification/tenant_x/` scope |
| 9 | Failure before physical readiness → zero serving | AD | `prepare_candidate` fail-closed | Launcher | `deploy_and_activate` | FRZ-PRD-03 | Activation + adapter negatives | D1 readiness/attest failures; activation tests | `17e928e4…` | SLICE-EVIDENCED | Strict session: failed prepare never opens handoff / ASGI | Local | **MISSING TARGETED GATE** | D4 negative extension |
| 10 | Activated revision → runtime resolver identity continuity | Apps | `ServingRuntimeHandoff`, `VenvBundleRuntimeAgentFactoryResolver` | `establish_serving_runtime_handoff` | GCA strict factory | FRZ-PRD-04, FRZ-OWN-03 | `serving_runtime_handoff.py`, `venv_bundle_runtime_agent_factory_resolver.py` | `test_prod_q_r1_r7_d1b_serving_runtime_handoff.py` | `17e928e4…` | SLICE-EVIDENCED | Prove **same** `serving_unit_ref` + revision id at ASGI request boundary | Local | **MISSING TARGETED GATE** | R7-Q wave 2 HTTP smoke |
| 11 | No manual registry seeding / reference shortcuts | Apps | Forbids `reference://` on production bundle | Launcher | `assemble_production_registry_projection_input_bundle` | FRZ-PRD-08 | `production_registry_projection_input_bundle.py` | D1b negatives; D3 descriptor authority | `17e928e4…`, `f2dd8ecb…` | SLICE-EVIDENCED | Static reachability: strict graph imports no harness registry seed helpers | CI static gate | **MISSING TARGETED GATE** | Add `tests/qualification/prod_q/` R7-Q reachability module |
| 12 | Startup failure + resource cleanup | Apps | Slot + store cleanup | `open_production_process_lifecycle` | Strict launcher | FRZ-PRD-04, FRZ-PRD-05, FRZ-PRD-10 | `production_process_lifecycle.py`, `strict_production_serving_launcher.py` | D2 invocation open failure rows; D4-R2-a/b | `0dd51fe1…` D4 | SLICE-EVIDENCED | Full stack cleanup: AD DB + continuation + PIS after failed strict session | Local SQLite | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice) | Replay D2+D4 regressions @ HEAD in execution child |
| 13 | Recovery after process restart | AD + Apps | D3 reconcile + attest | `open_production_process_lifecycle` | Process open | FRZ-PRD-12, FRZ-REC-* | `production_process_serving_reconciliation.py` | `test_prod_q_r1_r7_d3_*`, `d3_r1_*` | `f2dd8ecb…` | SLICE-EVIDENCED | Restart **strict serving** process: reconcile → bind → ASGI still consistent | Subprocess or in-test re-open | **MISSING TARGETED GATE** | R7-Q wave 3 |
| 14 | Stale READY after physical-unit loss | AD | Attest before rely; reconcile fail-closed | D3 pipeline | Open + reconcile | FRZ-PRD-12, FRZ-PRD-09 | `reconcile_traffic_serving_physical_unit_after_process_restart` | D3 `test_d3_stale_*`, `test_d3_open_fails_reconciliation_on_empty_adapter_*`; D4 `test_d4_stale_physical_*` | `f2dd8ecb…`, `0dd51fe1…` | SLICE-EVIDENCED | Operator-visible outcome after stale READY w/ **real** unit table loss | Local | **MISSING TARGETED GATE** | R7-Q integrated scenario |
| 15 | Strict execution admission + request-level rejection | Apps + AD | `StoreBackedColocatedServingUnitExecutionAdmission` | Handoff bind | GCA execution | FRZ-SEC-04, FRZ-PRD-08 | `colocated_serving_unit_task_execution.py` | D1-R2–R4, D1-R6 admission suites | `c25e7efe…` | SLICE-EVIDENCED | Request-level deny when gate closed / forged evidence (HTTP) | Local ASGI client | **MISSING TARGETED GATE** | R7-Q wave 2 |
| 16 | Drain, stop, rollback | AD | Adapter drain ports + `ActivationService` rollback | Launcher / admin | Ops paths | FRZ-PRD-05 | Adapter `begin_drain` / `stop` / `resume_serving` | D1 drain/stop tests; D1-R6 rollback evidence | `17e928e4…`, `c25e7efe…` | SLICE-EVIDENCED | Drain/stop during **active strict serving** session | Local | **MISSING TARGETED GATE** | R7-Q operational wave |
| 17 | Unsupported topology + missing credentials/config | AD + Apps | Topology enum + production wiring | Composition guards | Open / activate | FRZ-PRD-03 | `validate_colocated_serving_admission_topology`; D1 OCI negative | D1 `test_unsupported_topology_*`; D1b; D4 missing governance | `17e928e4…` | SLICE-EVIDENCED | Missing production credentials profile fail-closed (not lab harness) | Config fixtures | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Document operator config negatives in R7-Q |
| 18 | Fake deployment adapter reachability **= 0** on strict production graph | Apps | DEC-03 strict graph | D2 module boundary | Strict GCA chain | FRZ-PRD-07, FRZ-PRD-08 | No fake in `production_process_lifecycle.py` | D2 `test_d2_production_process_lifecycle_module_has_no_fake_*`; D4 `test_d4_fake_adapter_rejected` | HEAD-STATIC | **R7-Q-GAP** | ADR §15: AST/import closure `create_governed_contractor_strict_production_*` → `FakeInMemory*` count 0 | CI static | **MISSING TARGETED GATE** | **R7-Q-P1-EXEC** qualification module |
| 19 | No silent fallback / alternate production paths | Apps | Single opener; fail-closed | D2/D4 | Strict launcher | FRZ-PRD-07, FRZ-PRD-08 | Reference path isolated | D2 rejects fake; D4-03 legacy `run()` | `0dd51fe1…` | SLICE-EVIDENCED | Prove no runtime fallback to `open_reference_*` from strict factory | Static + behavioral | **MISSING TARGETED GATE** | R7-Q reachability gate |
| 20 | Provider invocation ambiguity / duplicate external effects | Execution / ER | `ProviderInvocationStore`, dispatch gate | GCA composition | Strict factory | FRZ-PRD-08, FRZ-REL-* | `SQLiteProviderInvocationStore` wired in D2 opener | `test_prod_q_r1_pis_01_*` (PROD-Q-R1 scope) | PROD-Q-R1 lineage | **R7-Q-GAP** (adjacent) | Deploy/activate must not double materialize / ambiguous PIS keys on strict path | SQLite | **IN-SCOPE BLOCKER** (parent **PROD-Q-R1**) | R7-Q records dependency; gate in R7-Q-P2 with PIS negatives |
| 21 | Evidence continuity, diagnostics, exact-SHA reproducibility | Program | Qualification docs + CI logs | — | — | FRZ-REG-02/03/09 | This P0 doc + slice qualifications | — | `64f0f447…` (P0 baseline) | HEAD-STATIC | Single R7-Q qualification record @ execution SHA | CI artifact store | **MISSING TARGETED GATE** | **PROD-Q-R1-R7-Q-EXEC** deliverable |

---

## D. Gap classification summary

| Class | Count (of 21) | Dimensions |
|-------|----------------|------------|
| **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice; not R7-Q PASS) | 2 | 1, 12 |
| **MISSING TARGETED GATE** (primary R7-Q work) | 14 | 2, 3, 4, 5, 7–11, 13–16, 18, 19, 21 |
| **REQUIRES REAL INFRASTRUCTURE** | 0 mandatory for A1 local qualification; optional | External CI-built immutable `file://` bundles @ scale; multi-host SQLite (**TRACKED FREEZE DEBT** per D1-D1b ARC-03) |
| **IN-SCOPE BLOCKER** | 1 | 20 (parent **PROD-Q-R1** PIS / dispatch — must not block inventory but gates full R7-Q closure) |
| **TRACKED FREEZE DEBT** | 0 exclusive rows | Cross-process SQLite trust; `UnsupportedVenvBundleMaterializationAdapter` (materialization wave, not R7-Q redesign) |
| **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | 2 | 6, 17 (operator production governance + credentials profile) |

**Mandatory gates already proven (slice level, reusable):** D0 attestation; D1 A1 adapter lifecycle; D1b handoff; D1-R6 coordination/admission; D2 production opener negatives; D3 restart reconcile; D4 strict launcher + ASGI admission.

**Mandatory gates missing for R7-Q:** integrated strict-path E2E; cross-tenant adversarial pack; global fake-reachability static gate per §15; second-adapter strict-path replaceability; operator production policy profile evidence; exact-SHA execution record @ HEAD.

---

## E. Proposed smallest bounded qualification waves

Preserves all §15 rows without redesigning D1–D4.

| Wave | Child ID (proposed) | Scope | Reuses |
|------|---------------------|-------|--------|
| **1** | **PROD-Q-R1-R7-Q-P1-EXEC** | @ HEAD: replay consolidated pytest bundle from D0–D4 modules (single command); add **static** strict-graph fake-reachability gate (`tests/qualification/prod_q/`) | All modules in §B.7 |
| **2** | **PROD-Q-R1-R7-Q-P2-EXEC** | Integrated local E2E: `open_strict_production_serving_session` → optional activate → ASGI factory → bounded HTTP admission smoke; identity continuity row 10 | D4 tests as template |
| **3** | **PROD-Q-R1-R7-Q-P3-EXEC** | Adversarial matrix: cross-tenant (8), DENY/CAS (7), stale READY (14), drain/stop (16), pre-readiness fail (9) on **strict** graph | D2/D3/D4 negatives |
| **4** | **PROD-Q-R1-R7-Q-P4-EXEC** (optional) | Operator CI artifact + trusted roots (2, 17); subprocess restart (13) if not covered in P3 | Real infra only if local `tmp_path` insufficient |

**STOP:** No wave may mark **PROD-Q-R1-R7-Q PASS** without independent audit @ exact SHA.

---

## F. Estimated test categories & resource risks (not executed in P0)

| Category | Est. cases | Risk |
|----------|------------|------|
| Replay D0–D4 unit gates | ~120–180 | Windows: single `uv run pytest` batch (test-process budget) |
| New qualification static gate | ~5–15 | Low CPU |
| Strict ASGI HTTP smoke | ~10–25 | Port binding; keep localhost |
| Cross-tenant adversarial | ~15–30 | Fixture complexity |
| Subprocess restart | ~5–10 | Timing flakes — use deterministic reconcile hooks |
| Parallel uv / xdist | Avoid | Per `intergrax-test-process-budget` |

---

## G. FRZ mapping (R7-Q contribution intent — all remain **OPEN**)

| FRZ ID | R7-Q dimensions | Notes |
|--------|-----------------|-------|
| FRZ-PRD-01 | 18, 19 | Lab vs production separation |
| FRZ-PRD-03 | 2, 9, 17 | Fail-closed config/topology |
| FRZ-PRD-04 | 1, 10, 12, 13 | Startup lifecycle |
| FRZ-PRD-05 | 12, 16 | Shutdown/drain |
| FRZ-PRD-07 | 18, 19 | Strict vs reference |
| FRZ-PRD-08 | 2, 7, 11, 15, 18, 19, 20 | Bypass / admission |
| FRZ-PRD-09 | 1, 3, 14 | Readiness semantics |
| FRZ-PRD-10 | 12 | Operator-visible failures |
| FRZ-PRD-12 | 13, 14 | Recovery |
| FRZ-SEC-04 | 3, 6, 15 | Unactivated serve blocked |
| FRZ-TEN-* | 8 | **No global PASS** from D4 bounded review |
| FRZ-REG-02/03/09 | 21 | Regression gates |
| FRZ-CTR-*, FRZ-PLG-*, FRZ-RPL-*, FRZ-TYP-* | 4, 5, 11 | Contracts / replaceability |

---

## H. Tenant Isolation Audit scope (R7-Q)

| In scope | Out of scope (separate program) |
|----------|----------------------------------|
| `ApplicationEnvironmentTenantResolver` on production opener + launcher authz | Global **TENANT-X** / **FRZ-TEN-*** closure |
| Cross-tenant DENY on activation and serving intent (dim 8) | Unrelated tier tenant data planes |
| D2 `test_d2_wrong_tenant_resolver_blocks` as regression anchor | Full enterprise tenant inventory scan |
| D4 bounded tenant review ([`PROD_Q_R1_R7_D4_STRICT_GCA_PRODUCTION_LAUNCHER.md`](PROD_Q_R1_R7_D4_STRICT_GCA_PRODUCTION_LAUNCHER.md) §Tenant) | Multi-app production deployment matrix |

**Tenant verdict (P0):** **NOT CERTIFIED** — local contract evidence only; **R7-Q** must add adversarial cross-tenant strict-path gates before parent **PROD-Q-R1-R7** can approach closure.

---

## I. Entry criteria — next execution child (**PROD-Q-R1-R7-Q-P1-EXEC** or operator-renamed **R7-Q** runner)

1. **Baseline:** `development` @ SHA ≥ `64f0f4474267672d4c18be683d9c33eb8a5afdfa`; no concurrent overlapping edits to strict production graph during run.
2. **Parent stage:** **PROD-Q-R1-R7-D4** operator-closed @ evidence `0dd51fe1…` (unchanged).
3. **D2 / D3:** **READY FOR AUDIT** — execution child may proceed without waiting for independent D2/D3 closure, but R7-Q PASS still requires no regressions vs their test modules.
4. **P0 artifact:** This inventory accepted for audit (this document @ commit recorded in execution report).
5. **Explicit non-goals:** No FRZ PASS promotion; no **PROD-Q-R1-R7** / **PROD-Q** parent closure; no production code changes unless a failing gate requires a corrective child.
6. **Deliverables of execution child:** Exact-SHA pytest log under `.tmp/proof/<run-id>/`; qualification record `PROD_Q_R1_R7_Q_EXECUTION_EVIDENCE.md` (future); static fake-reachability gate if missing.

---

## Structural & architecture audit (@ HEAD — no new architecture required)

| Check | Finding |
|-------|---------|
| Contracts over implementations | `RuntimeDeploymentAdapter` Protocol retained; A1 impl present |
| Layer boundaries | `test_deployment_port_has_no_applications_import`; GCA strict path consumes handoff (D1b/D4) |
| Exactly-one SO / CO | AD activation vs Apps composition vs Governance authz — matches ADR §6.1 |
| Strong typing | Pydantic + Protocol on closed-world surfaces (no R7-Q redesign) |
| Replaceability | Protocol + D2 injection; full R7-Q proof pending |
| No duplicate authority / shadow registry | D1-R4 ARC resolutions; coordinator path |
| Fail-closed | D2/D3/D4 negatives |
| Governance ≠ Execution | Preserved |
| Proposal ≠ permission ≠ execution | Preserved |
| Observability non-authoritative | Unchanged |
| **ARCHITECTURE DECISION REQUIRED** | **None** for R7-Q if execution stays within ADR §14.4 / §15 |

**Note:** ADR §3 closed-world table describes **reference-era** fake adapter defaults; @ `64f0f447…` production symbols in §B supersede for qualification planning. P0 does **not** amend ADR status.

---

## Stage status (unchanged)

| Stage | Status after P0 |
|-------|-----------------|
| **PROD-Q-R1-R7-Q** | **NOT ENTERED** |
| **PROD-Q-R1-R7-Q-P0** | **READY FOR AUDIT** (this document) |
| **PROD-Q-R1-R7** | **BLOCKED** / **NOT CLOSED** |
| **FRZ-PRD-*** | **OPEN** |

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.*
