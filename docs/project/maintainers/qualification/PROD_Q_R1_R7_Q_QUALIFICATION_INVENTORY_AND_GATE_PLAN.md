# PROD-Q-R1-R7-Q-P0 — Qualification inventory & gate plan

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** — inventory and gate plan only; **does not** enter **PROD-Q-R1-R7-Q** execution, **does not** claim **R7-Q PASS**, **does not** promote any **FRZ-*** row |
| **Bounded child** | **PROD-Q-R1-R7-Q-P0** (+ **PROD-Q-R1-R7-Q-P0-R1** defect corrections @ HEAD below) |
| **Parent** | **PROD-Q-R1-R7** — **BLOCKED** / **NOT CLOSED** / **NOT QUALIFIED** |
| **Mandatory successor stage** | **PROD-Q-R1-R7-Q** — **NOT ENTERED** |
| **Branch** | `development` |
| **START_HEAD / inventory baseline** | `7d217a04573fbed0479f8fb596673e1537801446` (PRE-AUDIT HEAD = verified @ P0-R1) |
| **Origin alignment** | `origin/development` @ `7d217a04573fbed0479f8fb596673e1537801446` |
| **Method** | Static source inspection + targeted closed-world verification; **no pytest / pyright executed in P0-R1** |

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
| **EXPECTED_START_HEAD** | `7d217a04573fbed0479f8fb596673e1537801446` |
| **Verified** | `git merge-base --is-ancestor 7d217a04573fbed0479f8fb596673e1537801446 HEAD` → **true** @ P0-R1; working tree clean; `development` tracks `origin/development` |
| **Closed-world @ HEAD** | `RuntimeDeploymentAdapter` · `VenvBundleColocatedRuntimeDeploymentAdapter` · `open_production_process_lifecycle` · `open_strict_production_serving_session` · `VenvBundleColocatedServingUnitRuntimeAccessor` — present; D1/D2/D4 replaceability & strict-topology tests listed in §B.7 |
| ADR ratification lineage | DEC-01–DEC-09 / DEC-09-EXT @ ADR R6 (decision provenance; not production qualification) |
| Implementation evidence pins (slice — not re-run in P0) | D1+D1b `17e928e4…`; D1-R6 `c25e7efe…`; D3 `f2dd8ecb…` (incl. D3-R1); D4 impl `9b6efa8d…`, evidence `0dd51fe1…` |

---

## B. Closed-world contract / owner / caller inventory (@ HEAD `7d217a04…`)

Legend: **SO** = semantic owner · **CO** = composition owner · **PC** = production caller (strict path).

### B.1 Deployment & physical unit (A1)

| Concern | Contract | SO | Implementation | CO | PC |
|---------|----------|----|----------------|----|-----|
| Deployment port | `RuntimeDeploymentAdapter` · `intergrax/agent_distribution/deployment.py` | **Agent Distribution** | `VenvBundleColocatedRuntimeDeploymentAdapter` · `intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py` | `wire_production_lifecycle_services` | `ActivationService.prepare_candidate` via `ProductionLifecycleLauncher.deploy_and_activate` |
| Prepared unit / drain | `PreparedDeployment`, `DrainStatus`, instance records | **AD** | `deployment.py` + SQLite / in-memory stores | D2 lifecycle wiring | `ActivationService`, coordinator paths |
| Production artifact binding | `resolve_production_artifact_root`, `assemble_production_registry_projection_input_bundle` | **Applications shared** | `production_registry_projection_input_bundle.py` | Launcher + D3 descriptor store | `ProductionLifecycleLauncher.deploy_and_activate` |
| Reference-only fake | `FakeInMemoryRuntimeDeploymentAdapter` | **AD (test/reference)** | `deployment.py` (reference class) | `open_reference_production_process_lifecycle` | **Not** `open_production_process_lifecycle` |

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

### B.8 Replaceability acceptance model (P0-R1 — reconciled with DEC-01 / strict D4)

R7-Q must **not** require an arbitrary second physical adapter through `open_strict_production_serving_session`. Ratified **A1** + D4 launcher pin the sanctioned physical implementation. Replaceability is three **separate** evidence-backed claims:

| Claim | Contract / mechanism | SO | CO | Evidence @ HEAD (reuse; not R7-Q PASS by itself) | R7-Q disposition |
|-------|----------------------|----|----|--------------------------------------------------|------------------|
| **A. Platform contract replaceability** | `RuntimeDeploymentAdapter` injected into `wire_production_lifecycle_services` → `ActivationService` | **AD** | **Apps** (`production_lifecycle.py`) | D0 `test_d0_11_alternate_adapter_satisfies_protocol`; D1 `test_alternate_adapter_satisfies_protocol`; D2 `test_d2_typed_replaceable_adapter_wiring` | **SLICE-EVIDENCED** — activation-domain logic unchanged across conforming adapters on **production opener wiring** |
| **B. Strict production A1 topology** | `open_strict_production_serving_session` + `_require_venv_production_adapter` | **Apps** | D4 strict launcher | `strict_production_serving_launcher.py` (`isinstance` + `reject_reference_runtime_deployment_adapter`); D4 `test_d4_fake_adapter_rejected`; D2 fake module negatives | **HEAD-STATIC** — reference/fake adapters rejected; **only** `VenvBundleColocatedRuntimeDeploymentAdapter` admitted on strict GCA path (**DEC-01**) |
| **C. Structural extensibility (sanctioned port)** | Same `RuntimeDeploymentAdapter` Protocol at `open_production_process_lifecycle(..., deployment_adapter=)`; future non-A1 physical providers require a **new conforming adapter** + explicit launcher/composition policy (out of scope for current strict A1 qualification) | **AD** (port) · **Apps** (strict pin) | D2 opener vs D4 launcher | ADR §15 row “Provider-neutral injection” + “Replaceability”; DEC-01 A1 first | **No `STOP — ARCHITECTURE DECISION REQUIRED`** for R7-Q within ADR §14.4 / §15 — existing port supports alternate implementations on **D2 production wiring**; strict serving **intentionally** narrows to VENV_BUNDLE A1 |

**Architecture note:** ADR §15 “Second adapter implementation” applies to **platform** qualification (Claim A + D2 injection), not to mandating a non-VENV adapter on the **strict GCA** launcher (Claim B).

---

## C. Qualification matrix (21 mandatory dimensions)

**Coverage status legend**

| Status | Meaning |
|--------|---------|
| **SLICE-EVIDENCED** | Behavioral tests exist in D0–D4 modules; recorded SHA in slice qualification docs |
| **HEAD-STATIC** | Code/test presence verified @ `7d217a04…` without execution in P0-R1 |
| **R7-Q-GAP** | Mandatory §15 / operator R7-Q scope not yet proven as integrated qualification |

**Classification legend:** see §D.

| # | Invariant (summary) | SO | Platform contract | CO | Production caller | FRZ (primary) | Code evidence (@ HEAD) | Existing test path | Evidence SHA (slice) | Coverage | Missing R7-Q gate | Runtime / resources | Classification | Next action |
|---|---------------------|----|--------------------|----|-------------------|---------------|------------------------|-------------------|----------------------|----------|-------------------|---------------------|----------------|-------------|
| 1 | Physical prepare + meaningful readiness on **real A1** adapter | AD | `RuntimeDeploymentAdapter.prepare` / `check_readiness` | `wire_production_lifecycle_services` | `ActivationService.prepare_candidate` | FRZ-PRD-04, FRZ-PRD-09 | `venv_bundle_colocated_runtime_deployment_adapter.py` | `test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py` (`test_prepare_*`, `test_runtime_initialization_*`) | `17e928e4…` (D1 bundle) | SLICE-EVIDENCED | Integrated strict path: deploy → READY on production opener w/ real bundle | ≤5 min narrow replay when scheduled (D1 module only) | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice; background — not auto R7-Q PASS) | **PROD-Q-R1-R7-Q-P2** integrated strict activate |
| 2 | Immutable artifact identity + digest validation | AD + Apps | Revision + materialization record; `file://` production locator | Bundle assembly | `deploy_and_activate` | FRZ-PRD-03, FRZ-PRD-08 | `production_registry_projection_input_bundle.py`, adapter digest checks | D1: `test_wrong_digest_*`, `test_missing_artifact_*`, `test_wrong_locator_scheme_*` | `17e928e4…` | SLICE-EVIDENCED | End-to-end: governance → persist revision → prepare with **operator-trusted** artifact root policy | Trusted artifact roots config | **MISSING TARGETED GATE** | R7-Q wave 2 E2E strict launcher |
| 3 | Serving-unit identity + liveness attestation | AD | `ServingUnitAttestation`, `attest_serving_unit` | Activation gates | `ActivationService` commit/ready paths | FRZ-PRD-09, FRZ-SEC-04 | D0 orchestration + D1 live unit | D0 `test_prod_q_r1_r7_d0_*`; D1 attestation rows | `2b44cee…` D0; `17e928e4…` D1 | SLICE-EVIDENCED | Attestation on **traffic-serving** unit after full D4 session | Same as row 1 | **MISSING TARGETED GATE** | Combine D4 `test_d4_stale_physical_*` + live HTTP optional |
| 4 | Provider-neutral activation semantics (**Claim A** — platform port) | AD | `ActivationService` + `RuntimeDeploymentAdapter` injection | `wire_production_lifecycle_services` / launcher | `deploy_and_activate` | FRZ-PLG-*, FRZ-RPL-* | `activation.py` (no vendor branches) | D0 `test_d0_11_alternate_adapter_satisfies_protocol`; D1 `test_alternate_adapter_satisfies_protocol`; D2 `test_d2_typed_replaceable_adapter_wiring` | `2b44cee…`, `17e928e4…` | SLICE-EVIDENCED | Scheduled narrow replay @ HEAD of D0/D1/D2 replaceability rows (≤5 min/task); **not** arbitrary adapter on strict GCA | Local Protocol stub | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice; scheduled replay for R7-Q record) | **PROD-Q-R1-R7-Q-P2** full activate-on-opener if gap found |
| 5 | Structural replaceability — sanctioned port + strict A1 pin (**Claims B + C**) | AD (port) · Apps (strict pin) | `RuntimeDeploymentAdapter` @ D2 opener; `VenvBundleColocatedRuntimeDeploymentAdapter` @ D4 launcher | D2 `open_production_process_lifecycle`; D4 `open_strict_production_serving_session` | FRZ-RPL-*, FRZ-CTR-01 | `strict_production_serving_launcher.py`; D2 `_AltRuntimeDeploymentAdapter` wiring | D2 `test_d2_typed_replaceable_adapter_wiring`; D4 `test_d4_fake_adapter_rejected` + `reject_reference_*` | HEAD-STATIC + D2 slice | SLICE-EVIDENCED (D2 wiring) + HEAD-STATIC (D4 pin) | P1: static strict-graph + D4 fake rejection reuse; **no** second physical adapter on strict launcher | ≤5 min (D4 negative subset) | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (structural); integrated strict activate in P2 | **PROD-Q-R1-R7-Q-P1** reachability; **PROD-Q-R1-R7-Q-P2** E2E |
| 6 | Authorized activation w/ canonical Governance | Governance + AD | `ControlPlaneMutationAuthorizationBoundary` | D2/D4 governance wiring | `deploy_and_activate(..., principal=)` | FRZ-SEC-04, FRZ-GOV-* | `production_control_plane_governance_wiring.py` | D2 `test_d2_governance_deny_*`; D4 `test_d4_governance_deny_*` | HEAD-STATIC | SLICE-EVIDENCED | Operator **production mutation policy profile** (not reference harness) | Policy fixture | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Operator-approved production boundary profile for R7-Q |
| 7 | DENY, stale revision, concurrency / CAS failures | AD | `atomic_commit_*`, coordinator linearization | D1-R6 coordinator | Activation + serving stores | FRZ-PRD-08, FRZ-STA-* | `activation_domain_coordinator.py`, stores | D1-R6 `r4`–`r8`, `r5_serving_admission_linearization`; D0 stale revision | `c25e7efe…` | SLICE-EVIDENCED | Cross-layer: DENY before any physical prepare on strict opener | SQLite local | **MISSING TARGETED GATE** | R7-Q adversarial matrix row |
| 8 | Cross-tenant negative scenarios | Governance + AD | Tenant resolver + authz evidence | D2 | `open_production_process_lifecycle` | **FRZ-TEN-*** | Resolver injection only | D2 `test_d2_wrong_tenant_resolver_blocks` only | HEAD-STATIC | **R7-Q-GAP** | Multi-tenant E2E: tenant A env cannot activate/serve tenant B | Two tenant fixtures; ≤5 min adversarial pack when scheduled | **MISSING TARGETED GATE** | **PROD-Q-R1-R7-Q-P3** tenant pack; align `tests/qualification/tenant_x/` scope |
| 9 | Failure before physical readiness → zero serving | AD | `prepare_candidate` fail-closed | Launcher | `deploy_and_activate` | FRZ-PRD-03 | Activation + adapter negatives | D1 readiness/attest failures; activation tests | `17e928e4…` | SLICE-EVIDENCED | Strict session: failed prepare never opens handoff / ASGI | Local | **MISSING TARGETED GATE** | D4 negative extension |
| 10 | Activated revision → runtime resolver identity continuity | Apps | `ServingRuntimeHandoff`, `VenvBundleRuntimeAgentFactoryResolver` | `establish_serving_runtime_handoff` | GCA strict factory | FRZ-PRD-04, FRZ-OWN-03 | `serving_runtime_handoff.py`, `venv_bundle_runtime_agent_factory_resolver.py` | `test_prod_q_r1_r7_d1b_serving_runtime_handoff.py` | `17e928e4…` | SLICE-EVIDENCED | Prove **same** `serving_unit_ref` + revision id at ASGI request boundary | Local | **MISSING TARGETED GATE** | R7-Q wave 2 HTTP smoke |
| 11 | No manual registry seeding / reference shortcuts | Apps | Forbids `reference://` on production bundle | Launcher | `assemble_production_registry_projection_input_bundle` | FRZ-PRD-08 | `production_registry_projection_input_bundle.py` | D1b negatives; D3 descriptor authority | `17e928e4…`, `f2dd8ecb…` | SLICE-EVIDENCED | Static reachability: strict graph imports no harness registry seed helpers | CI static gate | **MISSING TARGETED GATE** | Add `tests/qualification/prod_q/` R7-Q reachability module |
| 12 | Startup failure + resource cleanup | Apps | Slot + store cleanup | `open_production_process_lifecycle` | Strict launcher | FRZ-PRD-04, FRZ-PRD-05, FRZ-PRD-10 | `production_process_lifecycle.py`, `strict_production_serving_launcher.py` | D2 invocation open failure rows; D4-R2-a/b | `0dd51fe1…` D4 | SLICE-EVIDENCED | Narrow D2+D4 failure rows when scheduled (≤5 min); not consolidated D0–D4 batch | Local SQLite | **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice) | **PROD-Q-R1-R7-Q-P2** if regressions appear |
| 13 | Recovery after process restart | AD + Apps | D3 reconcile + attest | `open_production_process_lifecycle` | Process open | FRZ-PRD-12, FRZ-REC-* | `production_process_serving_reconciliation.py` | `test_prod_q_r1_r7_d3_*`, `d3_r1_*` | `f2dd8ecb…` | SLICE-EVIDENCED | Restart **strict serving** process: reconcile → bind → ASGI still consistent | Subprocess or in-test re-open | **MISSING TARGETED GATE** | R7-Q wave 3 |
| 14 | Stale READY after physical-unit loss | AD | Attest before rely; reconcile fail-closed | D3 pipeline | Open + reconcile | FRZ-PRD-12, FRZ-PRD-09 | `reconcile_traffic_serving_physical_unit_after_process_restart` | D3 `test_d3_stale_*`, `test_d3_open_fails_reconciliation_on_empty_adapter_*`; D4 `test_d4_stale_physical_*` | `f2dd8ecb…`, `0dd51fe1…` | SLICE-EVIDENCED | Operator-visible outcome after stale READY w/ **real** unit table loss | Local | **MISSING TARGETED GATE** | R7-Q integrated scenario |
| 15 | Strict execution admission + request-level rejection | Apps + AD | `StoreBackedColocatedServingUnitExecutionAdmission` | Handoff bind | GCA execution | FRZ-SEC-04, FRZ-PRD-08 | `colocated_serving_unit_task_execution.py` | D1-R2–R4, D1-R6 admission suites | `c25e7efe…` | SLICE-EVIDENCED | Request-level deny when gate closed / forged evidence (HTTP) | Local ASGI client | **MISSING TARGETED GATE** | R7-Q wave 2 |
| 16 | Drain, stop, rollback | AD | Adapter drain ports + `ActivationService` rollback | Launcher / admin | Ops paths | FRZ-PRD-05 | Adapter `begin_drain` / `stop` / `resume_serving` | D1 drain/stop tests; D1-R6 rollback evidence | `17e928e4…`, `c25e7efe…` | SLICE-EVIDENCED | Drain/stop during **active strict serving** session | Local | **MISSING TARGETED GATE** | R7-Q operational wave |
| 17 | Unsupported topology + missing credentials/config | AD + Apps | Topology enum + production wiring | Composition guards | Open / activate | FRZ-PRD-03 | `validate_colocated_serving_admission_topology`; D1 OCI negative | D1 `test_unsupported_topology_*`; D1b; D4 missing governance | `17e928e4…` | SLICE-EVIDENCED | Missing production credentials profile fail-closed (not lab harness) | Config fixtures | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Document operator config negatives in R7-Q |
| 18 | Fake deployment adapter reachability **= 0** on strict production graph | Apps | DEC-03 strict graph | D2 module boundary | Strict GCA chain | FRZ-PRD-07, FRZ-PRD-08 | No fake in `production_process_lifecycle.py` | D2 `test_d2_production_process_lifecycle_module_has_no_fake_*`; D4 `test_d4_fake_adapter_rejected` | HEAD-STATIC | **R7-Q-GAP** | ADR §15: AST/import closure `create_governed_contractor_strict_production_*` → `FakeInMemory*` count 0 | ≤5 min static + D4 negative reuse | **MISSING TARGETED GATE** | **PROD-Q-R1-R7-Q-P1** |
| 19 | No silent fallback / alternate production paths | Apps | Single opener; fail-closed | D2/D4 | Strict launcher | FRZ-PRD-07, FRZ-PRD-08 | Reference path isolated | D2 rejects fake; D4-03 legacy `run()` | `0dd51fe1…` | SLICE-EVIDENCED | Prove no runtime fallback to `open_reference_*` from strict factory | Static + behavioral | **MISSING TARGETED GATE** | R7-Q reachability gate |
| 20 | Provider invocation ambiguity / duplicate external effects (**parent dependency**) | **Execution / ER** (PIS contract) · **GCA** (`GovernedProviderInvocationDispatchGate` — dispatch SO) | `ProviderInvocationStore` + GR-7 recovery; dispatch gate in `provider_invocation_lifecycle.py` | GCA `production_external_work_composition.py` | Strict factory external-work path | FRZ-PRD-08, FRZ-REL-* | `SQLiteProviderInvocationStore` wired in D2 opener | `test_prod_q_r1_pis_01_*` (**PROD-Q-R1** store contract only) | PROD-Q-R1 / PIS-ADR lineage | **R7-Q-GAP** (parent) | Qualify: no ambiguous PIS keys / unsafe repeat dispatch on **strict** external-work path; **do not** assert duplicate external effects @ HEAD without a failing strict-path contract test | SQLite; parent gate TBD | **PARENT DEPENDENCY — UNQUALIFIED RISK** (not proven defect @ HEAD) | **PROD-Q-R1** closure + **PROD-Q-R1-R7-Q-P3** strict-path PIS negatives when parent unblocks |
| 21 | Evidence continuity, diagnostics, exact-SHA reproducibility | Program | Qualification docs + CI logs | — | — | FRZ-REG-02/03/09 | This P0 doc + slice qualifications | — | `7d217a04…` (P0-R1 baseline) | HEAD-STATIC | Single R7-Q qualification record @ execution SHA; **full D0–D4 replay** only via operator-approved budget gate (§E.1) | CI artifact store | **MISSING TARGETED GATE** | **PROD-Q-R1-R7-Q-EXEC** deliverable |

---

## D. Gap classification summary

| Class | Count (of 21) | Dimensions |
|-------|----------------|------------|
| **ALREADY EVIDENCED — CURRENT HEAD VERIFIED** (slice; not R7-Q PASS) | 4 | 1, 4, 5, 12 |
| **MISSING TARGETED GATE** (primary R7-Q work) | 12 | 2, 3, 7–11, 13–16, 18, 19, 21 |
| **REQUIRES REAL INFRASTRUCTURE** | 0 mandatory for A1 local qualification; optional | External CI-built immutable `file://` bundles @ scale; multi-host SQLite (**TRACKED FREEZE DEBT** per D1-D1b ARC-03) |
| **PARENT DEPENDENCY — UNQUALIFIED RISK** | 1 | 20 (**PROD-Q-R1** PIS/dispatch — gates full **R7-Q** closure; **not** downgraded to freeze debt) |
| **TRACKED FREEZE DEBT** | 0 exclusive rows | Cross-process SQLite trust; `UnsupportedVenvBundleMaterializationAdapter` (materialization wave, not R7-Q redesign) |
| **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | 2 | 6, 17 (operator production governance + credentials profile) |

**Mandatory gates already proven (slice level, reusable background):** D0 attestation; D1 A1 adapter lifecycle; D1b handoff; D1-R6 coordination/admission; D2 production opener negatives + replaceability wiring; D3 restart reconcile; D4 strict launcher + ASGI admission + fake rejection.

**Mandatory gates missing for R7-Q:** integrated strict-path E2E; cross-tenant adversarial pack; global fake-reachability static gate per §15; operator production policy profile evidence; exact-SHA execution record @ HEAD; parent **PROD-Q-R1** PIS/dispatch qualification for dim 20.

### D.1 Dimension 20 — PIS / dispatch (parent blocker classification)

| Field | Record |
|-------|--------|
| **Canonical PIS owner** | **Execution / enterprise reliability** — `ProviderInvocationStore` contract (`intergrax/runtime/…`); durable SQLite adapter **PROD-Q-R1-PIS-01** scope |
| **Canonical dispatch owner** | **GCA host** — `GovernedProviderInvocationDispatchGate` (`applications/governed_contractor_application/host/provider_invocation_lifecycle.py`); wired from `production_external_work_composition.py` |
| **Exact parent dependency** | **PROD-Q-R1** (not closed) — dispatch-safety / repeat-dispatch policy per **PROD-Q-R1-ADR-03** §2.1–§2.3 |
| **Behavior requiring qualification** | Strict-path external provider work: intent durability, recovery ambiguity, and **no** unsafe second `execute()` for the same `invocation_id` under crash/re-entry |
| **Evidence present @ HEAD** | `test_prod_q_r1_pis_01_*` — **store** contract only; ADR documents architectural dispatch-gap analysis — **not** a current-HEAD failing strict GCA integration test |
| **Evidence missing** | Strict production graph test proving duplicate external effects **or** proving dispatch safety end-to-end on sanctioned composition |
| **Finding type** | **Unqualified risk** (parent ADR analysis) — **not** recorded as **proven defect** @ HEAD without failing contract/test on strict path |
| **R7-Q rule** | Dim 20 remains **unresolved parent-level dependency**; full **R7-Q** closure blocked until **PROD-Q-R1** qualifies dispatch + strict-path negatives scheduled in **P3** |

---

## E. Bounded qualification waves

Preserves all ADR §15 rows without redesigning D1–D4. **Execution policy (all children):** narrow changed-invariant test selection; **sequential** `uv run pytest`; `-p no:xdist`; **≤5 minutes cumulative per Cursor task**; no parallel `uv`/Python jobs; no automatic retries; **no** full D0–D4 suite in one batch without explicit operator approval.

Historical D0–D4 evidence may be cited as **background** only — it does **not** auto-count as **R7-Q @ current HEAD** certification.

### E.1 Mandatory full-replay qualification gate (operator-scheduled)

| Gate | Intent | Status |
|------|--------|--------|
| **R7-Q FULL-D0-D4-REPLAY** | Operator-approved resource budget; replays all §B.7 modules @ exact SHA for scoped regression record | **CLOSED — SCOPED D0–D4 REGRESSION / OPERATOR ACCEPTED** (2026-10-15) — **21** modules, **209** passed tests, **five** distinct execution SHAs (batches A, B, C replay, D, E); rollup audit baseline `8ee834e4c570d3372f47a1381416a38cae0bd30e`; decision `PROD-Q-R1-R7-Q-EXEC-E1-OPERATOR-CLOSURE`; evidence [`PROD_Q_R1_R7_Q_EXEC_E1_FINAL_ROLLUP_AUDIT.md`](PROD_Q_R1_R7_Q_EXEC_E1_FINAL_ROLLUP_AUDIT.md) §16. **Scope restriction:** no single atomic 21-module pytest or CI monolithic artifact asserted. Original Batch C remains **PROCESS EXIT 0 — EVIDENCE INCOMPLETE**; Batch C replay **PASS — INDEPENDENTLY ACCEPTED**. **Does not** satisfy **R7-Q PASS** (remaining §C integrated/operator-profile gates). |

#### E.1.1 Execution readiness package (preparation only — **NOT EXECUTED** @ task `PROD-Q-R1-R7-Q-P3-FINAL-CLOSURE-RECORD`)

*Historical at preparation time (docs baseline `28c9bc40…`, P3 closure record). Execution and operator scoped closure completed subsequently — see §E.1 table and rollup audit §16.*

| Field | Value |
|-------|-------|
| **Parent / stage** | **PROD-Q-R1-R7-Q** → gate **E.1** / child identity **PROD-Q-R1-R7-Q-EXEC** (full D0–D4 replay) |
| **Documentation baseline @ preparation** | `28c9bc40d1cdcbf62700e961677b4048c0753b85` (`development`) |
| **Prerequisite gates** | **PROD-Q-R1-R7-Q-P3** = **CLOSED — OPERATOR ACCEPTED** (2026-10-15); **P1**/**P2** evidence preserved; **D4** independently accepted @ `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e`; **E.1** does **not** inherit PASS from P3 integration tests |
| **Execution order (mandatory)** | **D0** attestation → **D1** physical deployment adapter → **D1b** runtime handoff (+ **D1-R2…R4** admission closure modules) → **D1-R6** coordinator/admission chain → **D2** production process opener → **D3** (+ **D3-R1**) restart/reconcile → **D4** strict GCA launcher; then adjacent **PROD-Q-R1** PIS modules in §B.7 only when operator schedule includes dim **20** replay |
| **Canonical test modules (@ §B.7 — exact paths)** | **D0/D0-R2:** `tests/unit/agent_distribution/test_prod_q_r1_r7_d0_serving_unit_attestation.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d0_r2_idempotent_attestation.py` · **D1:** `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py` · **D1b + D1-R2…R4:** `tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py`, `tests/unit/applications/test_prod_q_r1_r7_d1_r2_serving_admission_authority.py`, `tests/unit/applications/test_prod_q_r1_r7_d1_r3_serving_execution_admission_closure.py`, `tests/unit/applications/test_prod_q_r1_r7_d1_r4_serving_admission_authority_closure.py` · **D1-R6:** `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_activation_domain_coordinator.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r1_activation_domain_coordinator_corrective.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r3_coordinator_scoped_gate_closure.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r4_physical_gate_atomicity.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r5_serving_admission_linearization.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r6_coordination_lease_ownership.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r7_lease_cleanup_rollback_admission_evidence.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r8_transaction_safety_closure.py`, `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r9_physical_admission_contract_equivalence.py` · **D2:** `tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py` · **D3/D3-R1:** `tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py`, `tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py` · **D4:** `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` · **PROD-Q-R1 adjacent:** `tests/unit/runtime/enterprise_reliability/test_prod_q_r1_pis_01_*.py`, `tests/unit/applications/governed_contractor/test_prod_q_r1_r*.py` (glob — resolve exact files @ execution SHA) |
| **Historical evidence baselines (reuse reference)** | D0/D1/D1b/D1-R6/D2/D3/D4 SHAs cited in P3 decision package §A; P3 slice SHAs **not** substitutes for replay @ execution SHA |
| **Resource / timeout policy** | §F: **one** sequential `uv run pytest <paths> -p no:xdist` per operator-approved batch; **≤5 minutes cumulative per Cursor task** unless operator expands budget; **no** parallel `uv`/Python jobs; **no** automatic retries; STOP on timeout/failure — do not force-push or silently reconcile |
| **Operator approval** | Explicit **execution authorization** required — preparation task does **not** grant E.1 run consent |
| **Evidence capture** | Logs under `.tmp/proof/<run-id>/`; record exact execution SHA; qualification deliverable **PROD-Q-R1-R7-Q-EXEC** |
| **Remaining R7-Q dimensions** | §C matrix (21 dimensions) — P3 closes wave **3** only; dims **1–21** parent **NOT QUALIFIED** until independent audit + **E.1** when mandated |
| **Verdict** | **READY FOR EXECUTION AUTHORIZATION** (prerequisites for documented replay inventory met; **E.1 NOT EXECUTED**) |

### E.2 Wave children

| Wave | Child ID | Scope | Reuses |
|------|----------|-------|--------|
| **1** | **PROD-Q-R1-R7-Q-P1 — Strict Production Reachability Gate** | (1) AST/static strict graph inventory; (2) fail-closed legacy/reference entry checks; (3) fake/reference adapter exclusion at sanctioned production boundary; (4) **reuse** existing narrow D2/D4 test evidence (no HTTP E2E); (5) new regression tests **only** for uncovered static invariants (`tests/qualification/prod_q/`) | D2 module fake negatives; D4 `test_d4_fake_adapter_rejected`, D4-03; §B.8 Claim B |
| **2** | **PROD-Q-R1-R7-Q-P2** | Integrated local E2E: `open_strict_production_serving_session` → optional activate → ASGI factory → bounded HTTP admission smoke; rows 2, 3, 10 | D4 tests as template |
| **3** | **PROD-Q-R1-R7-Q-P3** | Adversarial matrix: cross-tenant (8), DENY/CAS (7), stale READY (14), drain/stop (16), pre-readiness fail (9); strict-path PIS negatives when parent unblocks (20) | D2/D3/D4 negatives |
| **4** | **PROD-Q-R1-R7-Q-P4** (optional) | Operator CI artifact + trusted roots (2, 17); subprocess restart (13) if not covered in P3 | Real infra only if local `tmp_path` insufficient |

**STOP:** No wave may mark **PROD-Q-R1-R7-Q PASS** without independent audit @ exact SHA; **E.1** scoped D0–D4 regression = **CLOSED — OPERATOR ACCEPTED** (2026-10-15) — **not** a substitute for remaining §C qualification.

**Next mandatory stage:** **R7-Q §C remaining mandatory integrated / operator-profile qualification gates** (dims **2, 3, 6, 7–11, 13–16, 17–19, 21** per §C summary; dim **20** parent **PROD-Q-R1**). **P1** = **READY FOR AUDIT** @ `8ec49660…`; **P2** = **CLOSED — operator exact-SHA acceptance** @ `8e43b160447ca43de6631fe2a1b7f99948b64f2f`; **P3** = **CLOSED — OPERATOR ACCEPTED** @ docs baseline `28c9bc40d1cdcbf62700e961677b4048c0753b85`; **E.1** = **CLOSED — SCOPED D0–D4 REGRESSION / OPERATOR ACCEPTED** @ rollup `8ee834e4…` (qualification [`PROD_Q_R1_R7_Q_EXEC_E1_FINAL_ROLLUP_AUDIT.md`](PROD_Q_R1_R7_Q_EXEC_E1_FINAL_ROLLUP_AUDIT.md) §16).

---

## F. Test categories & resource policy (not executed in P0-R1)

| Category | Selection | Policy |
|----------|-----------|--------|
| Per-task pytest | Named modules/tests only (invariant-scoped) | One sequential `uv run pytest … -p no:xdist`; ≤5 min/task |
| P1 static gate | New `tests/qualification/prod_q/` rows | Low CPU; no HTTP |
| P2 HTTP smoke | Localhost ASGI client | Port binding; bounded cases |
| P3 adversarial | Tenant / CAS / stale / PIS | Fixture complexity; split across tasks |
| Full D0–D4 replay | §B.7 all modules | **Operator approval only** — separate scheduled gate **E.1** |
| Parallel uv / xdist | **Forbidden** | `intergrax-test-process-budget` |

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

## I. Entry criteria — next execution child (**PROD-Q-R1-R7-Q-P1 — Strict Production Reachability Gate**)

1. **Baseline:** `development` @ SHA ≥ `7d217a04573fbed0479f8fb596673e1537801446`; no concurrent overlapping edits to strict production graph during run.
2. **Parent stage:** **PROD-Q-R1-R7-D4** operator-closed @ evidence `0dd51fe1…` (unchanged); **PROD-Q-R1-R7** / **PROD-Q-R1-R7-Q** remain **NOT ENTERED** / **NOT QUALIFIED**.
3. **D2 / D3:** **READY FOR AUDIT** — P1 may proceed without waiting for independent D2/D3 audit closure; R7-Q PASS still requires no regressions vs their modules when replayed under **E.1**.
4. **P0 artifact:** This inventory @ P0-R1 commit accepted for audit.
5. **Explicit non-goals:** No FRZ PASS promotion; no **PROD-Q-R1-R7** / **PROD-Q** parent closure; no production code changes unless a failing gate requires a corrective child; **no** consolidated 120–180 test batch in P1.
6. **P1 deliverables:** Static reachability artifacts; narrow D2/D4 log under `.tmp/proof/<run-id>/` if tests re-run; new qualification static module only for uncovered invariants; **no** HTTP E2E / cross-tenant / infra-heavy tests in P1.

---

## Structural & architecture audit (@ HEAD — no new architecture required)

| Check | Finding |
|-------|---------|
| Contracts over implementations | `RuntimeDeploymentAdapter` Protocol retained; A1 impl present |
| Layer boundaries | `test_deployment_port_has_no_applications_import`; GCA strict path consumes handoff (D1b/D4) |
| Exactly-one SO / CO | AD activation vs Apps composition vs Governance authz — matches ADR §6.1 |
| Strong typing | Pydantic + Protocol on closed-world surfaces (no R7-Q redesign) |
| Replaceability | §B.8 Claims A–C reconciled — platform port evidenced (D0/D1/D2); strict A1 pin evidenced (D4); **no** arbitrary second adapter on strict launcher |
| No duplicate authority / shadow registry | D1-R4 ARC resolutions; coordinator path |
| Fail-closed | D2/D3/D4 negatives |
| Governance ≠ Execution | Preserved |
| Proposal ≠ permission ≠ execution | Preserved |
| Observability non-authoritative | Unchanged |
| **ARCHITECTURE DECISION REQUIRED** | **None** for replaceability within ADR §14.4 / §15 (§B.8). Parent **PROD-Q-R1** PIS dispatch policy remains open per **PROD-Q-R1-ADR-03** — tracked under dim 20, not R7-Q launcher redesign |

**Note:** ADR §3 closed-world table describes **reference-era** fake adapter defaults; @ `7d217a04…` production symbols in §B supersede for qualification planning. P0-R1 does **not** amend ADR ratification status.

---

## Stage status (unchanged)

| Stage | Status after P0 |
|-------|-----------------|
| **PROD-Q-R1-R7-Q** | **NOT ENTERED** |
| **PROD-Q-R1-R7-Q-P0** | **READY FOR AUDIT** (this document @ P0-R1 corrections) |
| **PROD-Q-R1-R7** | **BLOCKED** / **NOT CLOSED** |
| **FRZ-PRD-*** | **OPEN** |

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.*
