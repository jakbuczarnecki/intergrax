# PLAN T-IGX-051-N — T-AF2-RW-SUPPLY-AB: attestation on install and upgrade, mandatory trust composition, revocation source, lifecycle evidence consumer, no-target serving stop (Parts A and B of T-IGX-051-B)

Draft, 2026-11-18, **revision 2** after the independent review `REVIEW_DRAFT_051N.md` (label of another session: REVIEW_FINDINGS, 2 HIGH, 6 MEDIUM, 6 LOW, 2 INFO; every item is answered in section 17). Revision 1 is kept unchanged as `PLAN_T-IGX-051-N.md` (digest `209c5ad7…35e2`). Nothing was run against the product and nothing was changed outside `.tmp`. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan; labels of other sessions ("DRY_RUN_FINDINGS", "REVIEW_FINDINGS", "RW_PLAN_AUDIT_PASS", "HELD", "STOP S-1") are quoted with their source. Defects are described by ID and title only.

Evidence notation `path@SHA:line`. `T` = `4c49cf877e65ad12d2fd4215186691515412a368` (`origin/arch-freeze/rw-train`, merge of PR #675). `P` = `700754f5e88ed36bdab32194667510b9ce2ba581` (head of PR #676, branch `af2/rw-supply`, draft, open, base `arch-freeze/rw-train`, merge base `T`; fetched 2026-11-18 for revision 2). Between revision 1's `8078e7557` and `P` only `intergrax/core/plugins/installed_digest.py` and its test changed, so every citation of revision 1 holds at `P`. `D` = `d0466feba1583cd5fccaf4d7121a68f68c13fbf2` (`origin/development`). Every write-set path is byte-identical at `D`, `T` and `P`, except `intergrax/agent_distribution/trust.py` (lint-only commit of PR #676) and `intergrax/applications/_shared/deploy_activate_pipeline.py` (PR #676 Part C and its review fix). `FACT_CHECK_rev2.md` holds the claim-by-claim table.

## 0. Origin and binding inputs

- **Signed parent plan.** `T-IGX-051-B` (T-AF2-RW-SUPPLY), plan digest `969407ba56159dc4be23820146609c3d284a49ff6ca1332c8c2229677e2cc562` (draft copy `w3-drafts/rw-supply/PLAN_T-IGX-051-B.md`, Parts A and B at `:53-76`, section 5b at `:256-275`; the digest reproduces). Its implementer stopped at write-set stop **S-1** ("Parts A and B need paths outside the write set", `w3-drafts/rw-supply/impl/PR_BODY.md`): making `ProductionAgentPlatformAdminConfig.package_trust` required breaks the three constructors P1-P3 of plan 5b, which Q-SUP-11 kept out of scope. PR #676 carries Parts C to I only.
- **Orchestrator decision** (session record `.tmp/session/arch-freeze/FOLLOW_UPS.md:165`, quoted as a record, not as approval): merge C-I as a partial and draft `T-IGX-051-N` (Parts A and B plus the 5b caller files) for a later Operator packet; "the 3 caller files are not 'one path' so S2 does not apply".
- **DS-EXC dispositions** (plan of record `agent-env/plans/pasted-content-id-e2b1-integrax-melodic-aho.md:290`, `:297`): tenant, security, fail-closed and execution-protection defects on a reachable path are `REMEDIATION_REQUIRED_BEFORE_FREEZE` with `DEFER_ALLOWED = no`, and the Operator decides DS-EXC. EX-D1-01 (HIGH), EX-D1-02, EX-D1-04 and PDC-D1-R017 therefore cannot stay residuals by an orchestrator decision: **every open part this plan leaves is an Operator question** (Q-AB-02, Q-AB-06, Q-AB-07, section 14). A-01 is the subject of PR #676 Part C; this ticket builds on that code.
- **RW-PLAN v21** (`rw-plan/RW_PLAN.md`): A1 block `T-AF2-RW-SUPPLY` (`:1929-1963`), negative-test rows NT-ROW-075..078 (`:599-602`), Q-SUP-11 (`:1200`), Q-SUP-02 (`:1129`). RW-PLAN v21 has **no A1 block for this ticket**; the write set below is a proposed block (Q-AB-01).
- **PR #676 at `P`.** Part C resolves the trust admission from the composed capability runtime before `persist_candidate_revision` (`deploy_activate_pipeline.py@P:115-127`, `:190`, `:211-212`). Its review fix renamed the state to `COMPOSED_RUNTIME_RECORDS_ONLY` and documents that, until Parts A and B land, a revoked digest is admitted on the production composition (`:100-112`; review item M-2, `FOLLOW_UPS.md:167`).
- **Approval route.** The ticket is new and carries a gate-meaning part (GM-AB-1), so S2 never applies to it: this digest and every later digest (including a step-0 re-read amendment after PR #676 merges) need the Operator's own approval in a wave packet under S1 (standing delegation record at `T`, Exclusions).

## 1. Purpose and findings

| Finding | Source label | What this ticket does | Open part and its route |
|---|---|---|---|
| EX-D1-01 | DS-EXC, HIGH | Part A: when attestation enforcement is composed (always, by the production factory), install and upgrade verify the attestation material carried by the trust record; a caller-asserted record without verified material is refused under a policy that forbids unsigned packages | publisher keys are PROD-Q items (RES-AB-4); installations admitted before enforcement keep serving unverified (RES-AB-5, Q-AB-10) |
| EX-D1-02 | DS-EXC | Part B: `ProductionAgentPlatformAdminConfig` gets a required `package_trust` composition; the factory passes coordinator, verifier, revocation, policy and time sources into the admin service; both launchers reach them through PR #676's composed-runtime resolution | **stays open on the shipped GCA strict path** after this ticket: it composes no capability runtime, so the pipeline takes the `NOT_COMPOSED` pass-through while the strict launcher still runs `deploy_and_activate` for an activation intent (`single_node_profile_v1_strict_startup.py@P:44`, `deploy_activate_pipeline.py@P:126`, `strict_production_serving_launcher.py@P:436-440`): **Operator question Q-AB-07** |
| EX-D1-04 | DS-EXC (Operator decision in the plan of record: a revoked revision without a trusted target stops serving) | Part B: `respond_to_emergency_revocation` turns the no-safe-target outcome into a governed serving stop through the existing post-cutover failure primitive, with an explicit audit event | the reason set is **Operator question Q-AB-04**; the automatic trigger (EX-D1-03) is NOT_COMPOSED and owned by RW-GATES (RES-AB-3) |
| PDC-D1-R017 | PARTIAL row | Part B: a production consumer of `lifecycle_event_sink` writes the lifecycle events (and the stop event) to the security audit trail of the owning tenant | needs the COMPAT-X classification GM-AB-1 (**Operator question Q-AB-02**); a refusal is a DS-EXC re-disposition of PDC-D1-R017 by the Operator, pre-specified as variant R in section 3 |

## 2. Design rules

1. **Fail closed at the composition, explicit 'not composed' below it** (signed rule 1). `ProductionAgentPlatformAdminConfig.package_trust` is required with no default; a `None` member is a `ValueError` naming the input; `is_lab` is keyword-only and required (no default), and a non-lab composition whose policy permits unsigned packages is refused at construction (Part B). A lab value is built by `lab_package_trust_composition()`; the strict startup refuses it. A direct construction of `AgentPlatformAdminService` keeps today's attestation behaviour when the new keyword is absent (about 30 existing callers, signed dry-run finding B-01).
2. **No second trust authority.** Verification uses `AgentPackageAttestationVerifier` / `Ed25519PackageAttestationVerifier` and the existing reason codes (`UNSIGNED_FORBIDDEN`, `MALFORMED_EVIDENCE`, `trust.py@P:207-208`). No new cryptography.
3. **No new `Any`, `object` or reflection dispatch** on any edited surface (OD-AF-11). No path under `intergrax/contracts/` is edited.
4. **No secrets** in code, logs, argv or test output; test keys are generated at run time; nothing echoes a signature or key; error messages name inputs, never values.
5. **C10/C15 freeze.** No Docker, vendor integration or production configuration change; no new operator environment variable; the strict serving command is not edited.
6. **Pinned text and structure kept** (FACT_CHECK_rev2 F-35, F-36, F-38, F-39): no `getattr(`, `setattr(`, `hasattr(`, `__dict__`, `eval(` or `exec(` in `admin_service.py` or `deploy_activate_pipeline.py`; `.mark_post_cutover_failure(` is called only from `admin_service.py`; `_authorize_admission(` stays before `persist_candidate_revision(` and `_authorize_activation(` before `.commit_activation(` in the pipeline; `production_agent_capability_runtime.py` imports nothing from `tests.`, `testing_support.`, `applications.` or `agents.`, contains neither `singleton` nor `get_global`, keeps the `delegated_subtask_child_execution: ProductionDelegatedSubtaskChildExecutionPort` field and contains neither `ChildExecutionRunner` nor `as_child_execution_port`; the strict startup keeps exactly one `open_gca_strict_production_serving_session(` call and the text `bind_process_scoped_release_on_close(writer_hold.release)`; `canonical_agent_lifecycle_composition.py` contains none of the Stage-15 forbidden fragments (`._private`, `AgentRegistry(`, `registry.register(`, `NexusLoop(`); the factory passes no `llm_profile_source` (RW-GATES NC table, EX-D3-02; `test_p6m_revision_llm_pin.py` passes it itself); the `trust_admission_not_composed` value stays (pinned, `test_rw_supply_trust_hook_before_persist.py@P:164`).
7. **No new tenant carrier.** No new class field whose name contains `tenant`; the adapter of Part B keeps its resolver as an instance attribute set in `__init__` (TENANT-X carrier pin counts class-body annotations only, F-44). Its `tenant_id=` keyword call is a new `function.tenant_kwarg.site` TENANT-X candidate, recorded in S5 (not pinned).
8. **One change, one signature.** Every caller of a changed signature is in the write set; the census is in section 4.

## 3. Parts

Two commit groups after the lint commits: Part A, then Part B (B composes A's enforcement). Part B has two pre-specified variants so that one digest covers both answers to Q-AB-02: **variant E** (evidence consumer, GM-AB-1 accepted; the default) and **variant R** (GM-AB-1 refused; section 3, end of Part B).

### Part A — attestation verified on install and upgrade (EX-D1-01)

State at `P`. `install_agent` and `upgrade_agent` admit a package with `assert_install_admission` (`admin_service.py@P:654-660`, `:834-840`), which checks revocation, source and freshness and has no signature step (`package_trust.py@P:544-609`). The service builds a coordinator without a verifier (`admin_service.py@P:327`). `AgentInstallationTrustRecord` carries evidence references and no signature material (`trust.py@P:131-145`). `forbid_unsigned_or_unqualified` defaults to true (`trust.py@P:231`) and is read only in `evaluate` within `intergrax/agent_distribution` (`package_trust.py@P:237`). The idempotent replay of an existing installation returns before any trust step (`admin_service.py@P:599-615`, `:791-805`).

Design.
- `trust.py`: new frozen model `AgentTrustAttestationMaterial` with `attestation_id`, `key_id`, `algorithm`, `publisher_id` and `signature_b64`, all `str`, stripped, non-empty, `extra="forbid"`. `algorithm` is a `str` because the enum `AgentPackageAttestationAlgorithm` lives in `package_attestation.py`, which imports `trust.py` (`package_attestation.py@P:17`); the admin step converts it and treats an unknown value as `MALFORMED_EVIDENCE`. `AgentInstallationTrustRecord` gains `attestation_material: tuple[AgentTrustAttestationMaterial, ...] = ()`; stored records without the field still parse (the stores persist `model_dump_json`, `sqlite_stores.py@P:100-101`).
- `admin_service.py`: new frozen dataclass `PackageAttestationEnforcement` (one member `verifier: AgentPackageAttestationVerifier`), placed in `admin_service.py` and not in `trust.py` as the signed text said, because the reverse import would be a cycle (Q-AB-05). New keyword `package_attestation_enforcement: PackageAttestationEnforcement | None = None`; `None` is the explicit 'not composed' state (today's behaviour, named in the class docstring). When composed, a private `_assert_trust_record_attested(trust_record, identity, policy)` runs directly after `assert_install_admission` in `install_agent` and `upgrade_agent`, before the candidate installation is created: each material item is verified for the exact digest-pinned identity (`AgentPackageAttestationVerificationRequest`, `package_attestation.py@P:106-115`); the result must be verified, its publisher must equal `trust_record.publisher_identity_ref`, and a `trust_evidence_refs` entry must name the item's attestation id. A failed or mismatching item raises `AgentPackageTrustError(MALFORMED_EVIDENCE)`; no verified item under `forbid_unsigned_or_unqualified=True` raises `AgentPackageTrustError(UNSIGNED_FORBIDDEN)`. The step runs before any metadata, authority or installation write of the same method (`admin_service.py@P:662-690`), so a refused install or upgrade persists nothing.
- Not done: attestation re-verification inside `assert_revision_trust_admissible` (`admin_service.py@P:1754`) and on the idempotent replay (Q-AB-10; residual RES-AB-5); `package_trust.py`, `package_attestation.py` and `emergency_revocation_response.py` are not edited.

### Part B — trust composition, revocation source, evidence consumer, strict-startup refusal, no-target stop (EX-D1-02, EX-D1-04, PDC-D1-R017)

State at `P`. `ProductionAgentPlatformAdminConfig` has no trust field (`production_agent_capability_runtime.py@P:103-113`) and the factory passes no coordinator, revocation, policy, time source or event sink (`:181-228`). The config is constructed in exactly three places in the repository (`git grep` over every tracked path, F-18): `testing_support/canonical_agent_lifecycle_composition.py@P:471` (also reused by `testing_support/enterprise_agent_lifecycle_composition.py@P:39,115`), `platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py@P:335` and `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py@P:360`; all three pass keywords only. No product code constructs it. The GCA strict startup takes an optional composition (`single_node_profile_v1_strict_startup.py@P:44`) and calls the profile loader first (`:56-59`); the public `open_gca_strict_production_serving_session` (`strict_gca_production_process_composition.py@P:80-104`) and `open_strict_production_serving_session` (`strict_production_serving_launcher.py@P:378`, `:425`) accept a composition too. Lifecycle events carry no tenant or environment attribute (`events.py@P:25-32`, `installation_service.py@P:90-94`); the tenant resolver needs application and environment ids (`control_plane_governance.py@P:51-60`). `_emit_events` runs after the store writes of a successful install (`admin_service.py@P:690-693`) and before `raise` on upgrade failure paths (`:868-876`, and the later failure branches). On a revoked serving revision with no trusted target the emergency service returns `BLOCKED_NO_SAFE_TARGET` and nothing stops serving (`emergency_revocation_response.py@P:292-332`, `:336-380`, `:439-474`); `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE` is returned before any target is evaluated, for a missing installation or trust record, not only for a revocation (`:292-314`, `:589-612`). The admin service returns the response unchanged (`admin_service.py@P:1580-1590`). A stop primitive exists: `ActivationService.mark_post_cutover_failure` marks the deployment instance `FAILED` (`activation.py@P:583-621`) and returns no events (`:621`, `events.py@P:36-40`); serving execution admission refuses a missing or non-`SERVING` instance (`activation_domain_coordinator.py@P:170-175`); the admin service authorizes and calls it in `handle_post_cutover_failure` (`admin_service.py@P:1684-1699`, `:2395-2424`); the authorization needs a mutation boundary and a tenant resolver (`:2406-2407`).

Design (variant E, default).
- `production_agent_capability_runtime.py`:
  - frozen dataclass `ProductionPackageTrustComposition` with `attestation_verifier: AgentPackageAttestationVerifier`, `revocation_state_source: Callable[[], AgentPackageTrustRevocationState]`, `policy_source: Callable[[], AgentPackageTrustPolicy]`, `evaluation_time_source: Callable[[], datetime]`, `lifecycle_evidence_trail: SecurityAuditSink` (`security_audit_trail.py@P:264-277`), and keyword-only `is_lab: bool` **without a default**. `__post_init__` raises `ValueError` naming any `None` member, and raises `ValueError` when `is_lab` is false and `policy_source()` (evaluated once) returns a policy with `forbid_unsigned_or_unqualified=False`. Limit, stated in the class docstring: a keyless or otherwise weak verifier cannot be detected from the outside; it fails every material item (`ed25519_package_attestation_verifier.py@P:86-99`, unknown key), so it refuses rather than admits under the non-lab policy.
  - `ProductionAgentPlatformAdminConfig.package_trust: ProductionPackageTrustComposition`, required, placed before the defaulted fields.
  - `lab_package_trust_composition() -> ProductionPackageTrustComposition`: `Ed25519PackageAttestationVerifier()` without keys, empty revocation state, `AgentPackageTrustPolicy(forbid_unsigned_or_unqualified=False)` (read only by the Part A step), the system clock, the in-memory `SecurityAuditTrail`, `is_lab=True`.
  - The factory builds `AgentPackageTrustCoordinator(attestation_verifier=...)`, `PackageAttestationEnforcement(verifier=...)` and a private class `_LifecycleEvidenceSink(trail, environment_tenant_resolver)` whose single method `record_events(events)` maps each event to one `self._trail.append(tenant_id=..., action=event.event_type, actor_id="agent_distribution.lifecycle", resource="installation:" + aggregate id, metadata=event attributes)`; the tenant comes from `admin_config.environment_tenant_resolver` with the application and environment ids that the admin service adds (next bullet). The factory passes the coordinator, the three sources, the enforcement and `lifecycle_event_sink=sink.record_events` into `AgentPlatformAdminService`. The `self._trail.append(..., tenant_id=...)` call in `record_events` is a new production call site for the COMPAT-X R3 inventory (`_compat_x_event_evolution_inventory.py@P:618-649`; unknown sites are `UNCLASSIFIED`, `:605-608`; the gates fail, `:1295-1297`, `_compat_x_event_evolution_r3_r1.py@P:1084-1088`), so it needs GM-AB-1 (conditional paths C-1 and C-1b).
- `admin_service.py` (in addition to Part A):
  - `_emit_events` passes to the sink copies of the events with two added attributes, `application_id` and `application_environment_id`, from the calling method; `audit_event_types` and the `TransitionResult` values are unchanged; the `lifecycle_event_sink` keyword keeps its `None` default (pinned, `test_d1_repair_scope_and_evidence.py@P:152-155`) (Q-AB-03).
  - **Sink failure contract** (applies to every sink): on a success path a raising sink is re-raised as a new typed `LifecycleEvidenceSinkError` (cause chained); the store writes that preceded it are committed and stay committed (documented in the method docstrings and the error docstring; the caller sees the error and the installation exists). On a failure path (the `except` branches that emit and then `raise`), the original error is raised with the sink error as its cause (`raise original from sink_exc`); the original error is never replaced. No sink error is swallowed (Q-AB-13).
  - `respond_to_emergency_revocation`: after the service responds, when the action is `BLOCKED_NO_SAFE_TARGET`, the response carries impacts, and the reason is one of `NO_PRIOR_REVISION`, `ROLLBACK_TARGET_UNTRUSTED`, `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE` or `ROLLBACK_FAILED`, the method reads the serving revision's deployment instance, authorizes a failure mark with `_authorize_mark_post_cutover_failure` (mutation id `emergency-revocation-stop:<revision>:<pointer revision>`) and calls `self._activation_service.mark_post_cutover_failure` with failure evidence `emergency-revocation:<reason>`. It then emits one explicit event `agent_trust.emergency_serving_stopped` (attributes: `runtime_revision_id`, `reason_code`, `failure_evidence_ref`, plus the two context attributes) through `_emit_events`, so it reaches the PDC-D1-R017 sink, and returns `response.model_copy(update={"audit_event_types": response.audit_event_types + (stop event type,)})`; action and reason are unchanged. `SERVING_POINTER_MISMATCH` and `ROLLBACK_CONFLICT` never stop (a newer state exists). Concurrency: a `RuntimeActivationError` from the mark (the serving revision changed after the response, `activation.py@P:597-598`) means the revoked revision no longer serves; the method emits `agent_trust.emergency_serving_stop_superseded` and returns the report without a stop. A `RuntimeActivationConflict` (the instance record changed) propagates (fail closed; the caller retries). A missing instance (`value None`, `:605-606`) emits the stop event with `instance_state=absent`; serving admission already refuses a missing instance. A governance denial raises as on the rollback path; a direct construction without a mutation boundary or tenant resolver now raises on a stop reason instead of returning the report (fail closed, part of Q-AB-04). The stop applies to every construction (Q-AB-04). Serving pointer and revision states are not changed, so the existing assertions on `traffic_serving_revision_id` stay valid (`test_ac6_trust_lifecycle_e2e.py@P:482-498`, `test_d1_adversarial_lifecycle.py@P:651-668`).
- `deploy_activate_pipeline.py` (signed A1 path of Part B): `COMPOSED_RUNTIME_RECORDS_ONLY` (`@P:100-112`) becomes `COMPOSED_RUNTIME` with value `trust_admission_composed_runtime` (public enum change; `RevisionTrustAdmissionState` is in `__all__`). The docstring states what the composed admission checks: the installation trust records against the composed revocation, policy and time sources; attestation material is verified at install and upgrade (Part A), not here; **a lab composition (`is_lab=True`) has an empty revocation state and admits unsigned packages**. No control-flow change; no test pins the old value (F-58).
- `single_node_profile_v1_strict_startup.py`: before the profile loader, a given `application_composition` whose `admin_config.package_trust.is_lab` is true raises `ValueError` (names `package_trust`, no value); `None` stays accepted (the shipped path; Q-AB-07); no new environment variable. The single `open_gca_strict_production_serving_session(` call and `bind_process_scoped_release_on_close(writer_hold.release)` stay (F-36). The other strict entry points that accept a composition (`open_gca_strict_production_serving_session`, `open_strict_production_serving_session`, `production_process_lifecycle.py@P:388`) are outside the write set and do not refuse a lab composition: **Operator question Q-AB-06**.
- The three constructors pass `package_trust=lab_package_trust_composition()` (P1-P3). They are test and proof support, not strict roots.
- Not edited: `package_b_strict_production_serving_command.py` (forbidden, PT-WB Part G and A4-040), both lifecycle launchers, `strict_production_serving_launcher.py`, `production_process_lifecycle.py`, `strict_gca_production_process_composition.py`.

**Variant R (GM-AB-1 refused by the Operator, Q-AB-02).** Pre-specified so that the same digest covers it: `ProductionPackageTrustComposition` has no `lifecycle_evidence_trail` member, the factory builds no `_LifecycleEvidenceSink` and passes no `lifecycle_event_sink`, C-1 and C-1b are not edited, NT-RW-SUP-04a and 04b are held out of the merge set (written, red-first recorded, not merged), and the stop event still goes through `_emit_events` (reaching any sink a caller composes). PDC-D1-R017 is then open by an **explicit Operator DS-EXC re-disposition** recorded with the Q-AB-02 answer, never by a default. Everything else is identical. The implementer chooses the variant only from the recorded Q-AB-02 answer; with no answer recorded, the work stops before Part B.

## 4. Census of callers and pinned tests (read at `P`)

| Group | Files | Disposition |
|---|---|---|
| constructors of `ProductionAgentPlatformAdminConfig` | 3 (P1-P3, F-18); 0 in `scripts/`, `intergrax/`, `agents/`, `applications/`; no `admin_config` use elsewhere in `tests/`, `testing_support/`, `platform_proofs/`, `scripts/` (F-19) | edit: one keyword each |
| users of P1 (canonical and enterprise compositions) | 17 files (F-20) | no edit: they receive the lab composition through P1 |
| users of P2 | `tests/integration/platform_proofs/test_ai_incident_investigation_agent_platform_e2e.py` | no edit |
| direct constructions of `AgentPlatformAdminService`, callers of `install_agent`/`upgrade_agent` | about 30 files (signed dry run) | no edit: the enforcement keyword defaults to `None` |
| constructor wrappers | `test_me13_marketplace_agent_distribution_execution_e2e.py`, `pdc_x/p6_model/test_p6m_revision_llm_pin.py` (F-39) | no edit: the factory adds keywords but not `llm_profile_source` |
| callers of `respond_to_emergency_revocation` (4 files at `P`) | `tests/integration/agent_distribution/test_ac6_trust_lifecycle_e2e.py` (blocked `ROLLBACK_TARGET_UNTRUSTED` at `:482-498`: now reaches the new mark authorization), `tests/qualification/pdc_x/d1/test_d1_adversarial_lifecycle.py` (same at `:651-664`), `tests/qualification/pdc_x/d1_repair/test_d1_repair_emergency_authorization.py` (deny boundary at `:52-61`; which call is denied matters), `tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py:245` (rollback path, unaffected); plus the service-level `test_emergency_revocation_response.py` (no admin service, unaffected) | no edit expected; S0-1 checks per file: boundary and resolver present, boundary admits the mark-post-cutover-failure mutation, no authorization or evaluator call count, no `SERVING` instance assertion after a blocked response; **a hit stops the work** |
| sink pins | `pdc_x/d1_repair/test_d1_repair_scope_and_evidence.py` | no edit: keyword default kept, extra attributes only; its failure-path test (`:133`) keeps the original error |
| admission-state pin | `tests/unit/applications/test_rw_supply_trust_hook_before_persist.py@P:164` (`trust_admission_not_composed`) | no edit: that value stays |
| text pins | F-35, F-36, F-38 | kept (rule 6) |
| PROD-Q strict-startup pins | `tests/qualification/prod_q/test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate.py:253-262`, `:680-689`; `_prod_q_r1_r7_q_p1_strict_reachability.py:71-108` | not run (PROD-Q authorization); the text pins are checked by `git grep` at the head |
| COMPAT-X R3 inventory and its pinned evidence set | `tests/qualification/compat_x/_compat_x_event_evolution_inventory.py`; `tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py@P:296-313` (hard-codes the three allowed `NON_EVENT_APPEND` evidence members) | cond C-1 and C-1b (GM-AB-1, variant E only) |
| COMPAT-X P7 counters | `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`, `test_compat_x_pdc_x_p7_repin_delta.py` | no edit: red at `T` already (6 nodes, PR #676 record M-1); counter rule in section 7 |

## 5. Write set

Branch rule: product commits only to `arch-freeze/rw-train`, after PR #676 has merged. Counts: **5 firm product paths (ACT-N11)**, **4 firm non-product paths** (3 edited support/test files, 1 new test file), **2 conditional non-product paths**; firm total 9, conditional 2. Every path exists at `T` or is new with an existing parent (`git ls-tree`).

### Product paths (ACT-N11), firm

```
intergrax/agent_distribution/trust.py
intergrax/agent_distribution/admin_service.py
intergrax/applications/_shared/production_agent_capability_runtime.py
intergrax/applications/_shared/deploy_activate_pipeline.py
applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py
```

All five are firm A1 paths of the signed 051-B block (`RW_PLAN.md:1933-1938`). `deploy_activate_pipeline.py` gets a state rename and docstring only.

### Non-product paths, firm (not ACT-N11: outside `intergrax/`, `agents/`, `applications/`)

```
testing_support/canonical_agent_lifecycle_composition.py
platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py
tests/unit/applications/test_ac4_phase9_production_composition_e2e.py
tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py
```

The first three are P1-P3 of the signed plan 5b; the fourth is new (parent exists) and is the signed A1 path for NT-RW-SUP-01..04.

### Non-product paths, conditional (variant E only; condition: the Operator accepts GM-AB-1, Q-AB-02)

```
tests/qualification/compat_x/_compat_x_event_evolution_inventory.py
tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py
```

- C-1 (inventory): one evidence member `AGENT_DISTRIBUTION_LIFECYCLE_EVIDENCE_SINK` in `RuntimeEventCallSiteClassificationEvidence` (`:104-121`), its entry in `_NON_EVENT_APPEND_EVIDENCE` (`:124-130`), and one classification row bound to **all three** of: path `intergrax/applications/_shared/production_agent_capability_runtime.py`, the qualname that `_enclosing_qualname` (`:481-495`) computes for `_LifecycleEvidenceSink.record_events` (expected `record_events`; the implementer confirms it and binds the exact value), and callee `self._trail.append`; disposition `NON_EVENT_APPEND` (the call delegates to a `SecurityAuditSink` whose evidence-plane `append` is itself classified, by analogy with the wiring row at `:595-599`).
- C-1b (pinned evidence set): the set literal in `test_cx_r3_p0_mechanical_persistence_discovery_classified` (`test_compat_x_event_evolution_r3_p0_gates.py@P:305-313`) gains the same fourth member; nothing else in that test changes.

Neither path is in the signed 5b list; revision 1 found C-1 and the review found C-1b. Signed 5b paths not needed by Parts A and B: C1-C3b (tenant scope and plan-context default, now `T-IGX-051-K`); C4 (the F35 test pins only sites inside `security_audit_trail.py`, `:33-58`); C5 (`repin_pdc_x_p7_delta.json`: its test is red at `T` without this ticket); C6 and C7 (no SQLite schema or STATE-X classification is touched).

### Edited existing tests

`tests/unit/applications/test_ac4_phase9_production_composition_e2e.py` (P3, one keyword); C-1b under GM-AB-1. No other.

### Docs

None.

**Forbidden (a need for any of these stops the work):** `intergrax/applications/_shared/production_lifecycle.py` (PT-WF), `reference_production_lifecycle.py`, `strict_production_serving_launcher.py`, `production_process_lifecycle.py`, `security_audit_trail_wiring.py`; `package_b_strict_production_serving_command.py`, `strict_gca_production_process_composition.py` and every other `applications/governed_contractor_application/host/` file; `intergrax/agent_distribution/package_trust.py`, `package_attestation.py`, `emergency_revocation_response.py`, `activation.py`, `serving_execution_admission.py`; `intergrax/runtime/security/security_audit_trail.py` (PR #676 Part E file); `intergrax/contracts/**`; the COMPAT-X P7 fixture and every other COMPAT-X file than C-1 and C-1b; the EBH-6 typing registry, the EBH-7 F10 inputs and the TENANT-X carrier pin (`tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py`, a firm path of RW-TENANT); `.github/`, `pyproject.toml`, `uv.lock`; the roadmap, the FRZ checklist, PDC-X and IIAC records; gate scripts; Docker or vendor work; real keys, real revocation feeds or live backends.

## 6. Negative tests

Types as in the signed plan: R = refused or detected configuration (red at the pre-change commit), S = structural (red because the structure is absent), C = control. Node ids are `file::test` in `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py`. RW-PLAN rows NT-ROW-075..078 (077 gains nodes 03c-03e, part of the Q-AB-01 amendment). **4 NT ids, 16 nodes: R 10, S 2, C 4.** NT-RW-SUP-04a and 04b exist only in variant E; in variant R they are held out of the merge set. No expected-red test merges inside this ticket.

| ID | NT row | Configuration | Expected result | Node | Type |
|---|---|---|---|---|---|
| NT-RW-SUP-01 | 075 (EX-D1-01) | install through the production factory (non-lab composition, throwaway Ed25519 key), trust record without material | `AgentPackageTrustError` `UNSIGNED_FORBIDDEN`; nothing persisted | `test_sup_01a_unsigned_install_refused` | R |
| NT-RW-SUP-01 | 075 | upgrade with material whose signature does not match the digest | `MALFORMED_EVIDENCE`; installation unchanged | `test_sup_01b_tampered_upgrade_refused` | R |
| NT-RW-SUP-01 | 075 | evidence ref naming a signature without material, or material whose publisher differs from the record | refused | `test_sup_01c_forged_evidence_ref_refused` | R |
| NT-RW-SUP-01 | 075 | valid signature with the throwaway key | admitted | `test_sup_01d_valid_attestation_admitted` | C |
| NT-RW-SUP-02 | 076 (EX-D1-02) | `ProductionAgentPlatformAdminConfig` without `package_trust`; a composition with a `None` member; a composition without `is_lab`; `is_lab=False` with a policy that permits unsigned packages | `TypeError`; `ValueError` naming the member; `TypeError`; `ValueError` | `test_sup_02a_factory_requires_trust_composition` | S |
| NT-RW-SUP-02 | 076 | revoked digest through `ProductionLifecycleLauncher.deploy_and_activate` with a composed capability runtime | refused before any store write | `test_sup_02b_revoked_digest_refused_production_launcher` | R |
| NT-RW-SUP-02 | 076 | the same through `ReferenceProductionLifecycleLauncher` | refused | `test_sup_02c_revoked_digest_refused_reference_launcher` | R |
| NT-RW-SUP-02 | 076 | GCA strict startup given a composition built with `lab_package_trust_composition()`; the profile loader replaced by a sentinel | `ValueError` before the loader is reached | `test_sup_02d_strict_startup_refuses_lab_trust_composition` | R |
| NT-RW-SUP-03 | 077 (EX-D1-04) | revoked serving revision with no prior revision; and with a revoked prior revision (`ROLLBACK_TARGET_UNTRUSTED`) | report action `BLOCKED_NO_SAFE_TARGET`; instance `FAILED` with `failure_evidence_ref == "emergency-revocation:<reason>"`; serving execution admission refuses; report `audit_event_types` contain `agent_trust.emergency_serving_stopped`; pointer unchanged | `test_sup_03a_revoked_without_target_stops_serving` | R |
| NT-RW-SUP-03 | 077 | stale request (`SERVING_POINTER_MISMATCH`) | no stop; instance `SERVING` | `test_sup_03b_stale_request_does_not_stop_serving` | C |
| NT-RW-SUP-03 | 077 | `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE` (an enabled entry without a trust record) and `ROLLBACK_FAILED` (rollback raises) | stop as in 03a, with each reason in the evidence ref | `test_sup_03c_evidence_unavailable_and_rollback_failed_stop` | R |
| NT-RW-SUP-03 | 077 | `ROLLBACK_CONFLICT`; and the serving revision changes between the response and the mark | no stop; the second case emits `agent_trust.emergency_serving_stop_superseded` and returns the report | `test_sup_03d_conflict_and_concurrent_change_do_not_stop` | C |
| NT-RW-SUP-03 | 077 | after a stop, a new trusted revision is deployed and activated | activation succeeds; serving admission admits it | `test_sup_03e_trusted_activation_after_stop_recovers` | C |
| NT-RW-SUP-04 | 078 (PDC-D1-R017) | install through the production composition with an evidence-plane trail over an in-process port | the install events (and, in 03a, the stop event) appear as audit entries of the owning tenant; the chain verifies | `test_sup_04a_install_event_reaches_evidence_plane` | R (variant E) |
| NT-RW-SUP-04 | 078 | composition whose `lifecycle_evidence_trail` is `None` | `ValueError` at composition | `test_sup_04b_missing_trail_refused_at_composition` | S (variant E) |
| NT-RW-SUP-04 | 078 | a raising sink: on a successful install, and on an upgrade failure path | success path: `LifecycleEvidenceSinkError` with the cause chained, the installation persisted; failure path: the original error, with the sink error as `__cause__` | `test_sup_04c_sink_failure_contract` | R |

**Red-first (S0) and revert-red (S0b).**
- Module imports: the merged file imports at module level only names that exist at the pre-change commit; names that this ticket introduces are imported inside the helpers that need them. So 03a, 03c and 04c (service API and direct construction only) run at the pre-change commit inside the merged file and fail with an assertion there (no stop, no stop event; the sink error replaces the original error on the failure path); 03b, 03d and 03e pass there.
- Probe for the other R/S nodes (01a-c, 02a-d, 04a-b): the new surface does not exist at the pre-change commit, so, as PR #676 did, a probe copy kept in the draft folder (`w3-drafts/supply-ab/impl/red_first/test_rw_supply_ab_probe.py`, never merged) keeps the **same node ids and assertion lines** and drives them through the old surface (the factory without `package_trust`): the unsigned or tampered install is admitted; `pytest.raises(TypeError)` does not raise; the revoked digest is admitted because the factory composes an empty revocation state; the sentinel loader is reached; no audit entry appears. Each must fail with an assertion, not an import or type error. The PR body records the probe-to-merged diff.
- Revert-red at the head (S0b, on the S0 scratch worktree): revert one product hunk at a time and run the mapped nodes; each must fail with an assertion: Part A step → 01a-c; trust-composition wiring in the factory → 01a, 02b, 02c, 04a; `__post_init__` checks → 02a; strict refusal → 02d; stop → 03a, 03c; stop event → 03a, 04a; concurrency handling → 03d; sink contract → 04c. The log is kept for the auditor.

## 7. Regression scope and commands

One process at a time under `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed; `owner.txt`; `rmdir` right after, also on failure; only the creator removes it); logs under `.tmp/session/arch-freeze/w3-drafts/supply-ab/impl/`; one `uv run pytest` per step with several paths; `--basetemp` under `D:/Projekty/_wt/_scratch/<slug>`; no broad or diagnostic suite; no pytest option beyond the repository configuration except `--basetemp`.

Regression set (S2, one command): `tests/unit/agent_distribution`; `tests/integration/agent_distribution`; `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`, `test_ac3_phase5_production_lifecycle_e2e.py`, `test_rw_supply_trust_hook_before_persist.py`, `test_production_composition_architecture_gate.py`, `test_me16_c1_application_skill_host_wiring.py`, `test_registry_projection_descriptor_corruption.py`; `tests/integration/platform_proofs/test_ai_incident_investigation_agent_platform_e2e.py`; `tests/integration/marketplace/test_me13_marketplace_agent_distribution_execution_e2e.py`; `tests/qualification/pdc_x/d1`, `d1_repair`, `p6_model/test_p6m_revision_llm_pin.py`, `p8_repair/test_r6_durable_projection_slug.py`; `tests/unit/runtime/architecture/test_platform_execution_unification_p0_bypass_inventory.py`, `test_platform_execution_unification_u4_child_execution_closure.py`; `tests/qualification/trace_x/test_trace_x_p5_r1_qualification_gates.py`; `tests/qualification/compat_x`; `tests/qualification/tenant_x`; `tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py`.

Known reds (not caused here; compared at the pre-change commit and the head by failed node-id set and message): the canonical and launcher e2e reds of the PR #676 log (strict-mode memory, skill pinning, contract authority), `test_ac3_phase5_production_lifecycle_e2e.py` (5 of 8, development commit `9e92e9376`, `FOLLOW_UPS.md:160`), `test_compat_x_pdc_x_p7_repin_delta.py` (6 nodes, PR #676 record M-1), the TENANT-X carrier pin (832 against pin 831, red since RW-RECOVERY; planned expected-red, owner RW-TENANT). **Counter rule:** a known-red node whose message differs only in COMPAT-X counters is not a regression when S5 lists the raw-candidate id delta and attributes every new id to a file of this ticket; any other difference in the failed set or a message is a regression.

Not run: `tests/qualification/prod_q` (PROD-Q needs its own authorization); the strict-startup text pins are checked by `git grep` instead.

| Step | Command | Purpose |
|---|---|---|
| S0 | the merged file (03a/03c/04c) and the probe (other R/S nodes) at the pre-change commit, detached worktree `D:/Projekty/_wt/intergrax/rw-supply-ab-s0` created from the main checkout's git metadata and removed afterwards | red-first per section 6 |
| S0b | revert-red per product hunk at the head, same worktree | non-vacuity of the merged nodes |
| S1 | the new test file plus `tests/unit/agent_distribution` and `pdc_x/d1_repair` after each part | per part |
| S2 | the regression set above, one command, at the pre-change commit and at the head | no new failed node (counter rule) |
| S3 | `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` on the changed Python files | style and typing |
| S4 | wave-packet contract check (`.tmp/session/arch-freeze/closed-world/typ_measure.py`, `rw-plan/work/typ05/measure_typ05.py`) over `intergrax/contracts/**` | nothing new (no contract file is edited; the run records it) |
| S5 | COMPAT-X P7 counters and raw candidate identities (with the id delta and its attribution), the TENANT-X carrier count **and the total TENANT-X candidate count** (the sink adds a `function.tenant_kwarg.site` candidate, `tests/qualification/tenant_x/_discovery.py@P:158-170`), all at the pre-change commit and the head; `git grep` of the PROD-Q text pins | records for the PR body |

Merge-time PR checks (workflow files at `P`): `ci-smoke`, `governance-fast` (KR-08 only), `evol-x-b4-preflight`, `governance-policy-pin`, `ticket-governance`, `ent-audit-fast`, `ent-audit-pr-gate`; `rag-gate` and `conformance` are path-filtered and not triggered by this write set.

## 8. Order of work

0. Start check: PR #676 merged into the train; the Q-AB-02 answer recorded (else stop before Part B); re-read every `@P` citation of sections 3-4 at the merged head; stop if a write-set path changed in meaning (a meaning change is an amendment, re-signed under S1). S0-1 census of the 4 callers of `respond_to_emergency_revocation` (section 4 row: boundary and resolver present, the boundary admits the mark mutation, no authorization or evaluator call count, no `SERVING` assertion after a blocked response; a hit stops the work); S0-2 the trail metadata key policy accepts the event attribute keys (`installation_slot_id`, `reason_code`, `prior_active_installation_id`, `application_id`, `application_environment_id`, `runtime_revision_id`, `failure_evidence_ref`); S0-3 a minimal composition can drive both launchers (else 02b/c drive each launcher's pipeline context, which both pass to `run_deploy_and_activate`); S0-4 COMPAT-X and TENANT-X baselines (S5).
0b. One mechanical `ruff check --fix` and `ruff format` commit per touched file that fails at the head, before its first behavioural hunk (at `P`: I001 in `production_agent_capability_runtime.py`, `single_node_profile_v1_strict_startup.py` and P1-P3; format hunks in `production_agent_capability_runtime.py` (2) and P1-P3 (11, 8, 11); `trust.py`, `admin_service.py`, `deploy_activate_pipeline.py`, C-1 and C-1b clean or to be re-checked at the merged head).
1. Write the test file and the probe; run S0.
2. Part A (`trust.py`, `admin_service.py`).
3. Part B in the recorded variant (`production_agent_capability_runtime.py`, `admin_service.py`, `deploy_activate_pipeline.py`, strict startup, P1-P3 in the same commit group as the required field; in variant E, C-1 and C-1b in the same commit group as the sink).
4. S0b, S2, S3, S4, S5.

Commits carry `Ref: governance/tickets/T-IGX-051-N.md`; no AI-tool attribution.

## 9. Lane, overlaps and merge order

Proposed lane **5** (RW-PLAN section c amendment, Q-AB-01): after `T-AF2-RW-SUPPLY` (PR #676) merges; parallel with lane-5 tickets that share no path (RW-TRUST, PT-FC3-F). Overlaps computed against the RW-PLAN v21 A1 scopes (`rw_plan.v1.json`) and every draft plan under `w3-drafts/` and `w3-packet-*/` (path grep):

| Shared path | Other ticket | Rule |
|---|---|---|
| all 5 product paths and the new test file | T-AF2-RW-SUPPLY (T-IGX-051-B), A1 firm | this ticket starts after PR #676 merges; it implements the A1 rows 075-078 that 051-B leaves open |
| `intergrax/agent_distribution/admin_service.py` (firm) and `test_ac4_phase9_production_composition_e2e.py` (cond) | T-AF2-RW-TENANT (T-IGX-051-A), lane 7, already signed (`w3-drafts/rw-tenant/T-IGX-051-A.md:41`) | default: this ticket first and TENANT rebases onto the install/upgrade hunks; **reverse rule**: if 051-A merges first, this ticket rebases onto its install CAS hunks and re-runs S1/S2 (a meaning change is an amendment) |
| `platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py` (inside the firm glob `platform_proofs/scenarios/ai_incident_investigation/*`) | T-AF2-RW-GATES (T-IGX-051-F), lane 4, already signed (`w3-drafts/rw-gates/T-IGX-051-F.md:32`) | either order, the second rebases; 051-F's expected edit in the glob is the agent module (`PLAN_T-IGX-051-F.md:166`); if 051-F has already formatted the file, this ticket drops its lint commit for it |
| semantic, no shared path | T-AF2-RW-GATES NC gate (EX-D1-03 `respond_to_emergency_revocation`, EX-D3-02 `llm_profile_source`, `PLAN_T-IGX-051-F.md:96,100`) | this ticket adds no STRICT/PRODUCT root call of either; the factory passes no `llm_profile_source` |
| none | PT-FC1 (T-IGX-043-A), PT-FC2-E (T-IGX-043-E part 1: `scripts/maintenance/check_production_capacity_adapters.py` only), PT-FC3, PT-FC3-F, PT-WF, PT-T01, RW-TRUST, RW-TYP, T-IGX-051-J, -K, -L, -M, T-IGX-052-E | no shared path. PT-WF edits `production_lifecycle.py`, which NT-RW-SUP-02b drives (re-run 02b if PT-WF lands first). PT-FC1's conditional `test_canonical_agent_lifecycle_architecture_gate.py` scans P1 text (rule 6 keeps it green). No other ticket lists a COMPAT-X R3 file |

Merge order: T-IGX-051-B (#676) → T-IGX-051-N → T-IGX-051-A (reverse rule above if 051-A is first); T-IGX-051-F in either order on the shared proof file. RW-TYP's rule (after the last contract editor) does not apply: no contract surface is edited. The follow-up node of Q-AB-06/Q-AB-07 (if the Operator takes the default) comes after this ticket.

## 10. ACT-N11, EBH-6, TENANT-X, EBH-7

- **ACT-N11.** The 5 firm product paths are product paths by `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")` (`scripts/governance/iiac_common.py@D:22`); they break the IIAC activation until the single RB-2 re-pin (DP-58, PT-Q03), which this ticket does not perform; expected red only KR-08 (`governance-fast`). The 6 non-product paths (`testing_support/`, `platform_proofs/`, `tests/`, firm and conditional) are not ACT-N11 paths; their commits still carry the `Ref:` trailer.
- **EBH-6 typing registry** (`tests/qualification/ebh_6/_contracts_typing_qualification.py`, 11 entries, exact match on path, line, symbol and rule over top-level public functions of `intergrax.applications._shared` modules imported by a `*_wiring.py` seed; sweep record `w3-packet-4/EBH6_TYPING_PIN_SWEEP.md` sections 2-3). No write-set path is a registry file, and neither `production_agent_capability_runtime.py` nor `deploy_activate_pipeline.py` is imported by a seed (`git grep`, 0 hits), so the gate does not scan them. Not edited. Constraint: no new `_shared` import in a seed file; every new public function fully annotated.
- **TENANT-X carrier pin** (`tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py@P:13-33`: replay head total 831, carriers of kind `class.field.tenant`). Already red (832, RW-RECOVERY); planned expected-red with owner RW-TENANT, re-pinned once at the TENANT head. Not edited. This ticket adds no carrier (rule 7); S5 records the carrier count and the total candidate count.
- **EBH-7 F10 inputs.** No write-set path is named there (`git grep` over `tests/qualification/ebh_7`, 0 hits).

## 11. Gate-meaning parts

| ID | Part | What changes | Ruling |
|---|---|---|---|
| GM-AB-1 | B, variant E (cond C-1 and C-1b) | the COMPAT-X R3 inventory classifies a new production `.append(..., tenant_id=...)` site (the lifecycle evidence consumer) as `NON_EVENT_APPEND` with a new evidence member bound to path, qualname and callee, **and** the pinned allowed-evidence set of `test_cx_r3_p0_mechanical_persistence_discovery_classified` gains that member | Operator (it widens a positive-evidence allowlist; never under S2); draft default: accept. Refusal = variant R plus an Operator DS-EXC re-disposition of PDC-D1-R017 |

No other part changes what a gate asserts. The strict-startup refusal and the serving stop are product behaviour (the stop is the Operator's EX-D1-04 disposition; its reason set is Q-AB-04).

## 12. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | The canonical and launcher e2e tests are already red at the base for unrelated reasons, so P1's effect is proven only by equal failed-node sets | compare node ids and messages at both commits; 02b/c build their own minimal composition (S0-3) |
| R-2 | A caller of the emergency response depends on the stop not authorizing, on a call count or on a `SERVING` instance | S0-1 census of the 4 files; a hit outside the write set stops the work |
| R-3 | The trail's metadata key policy rejects an event attribute key | S0-2; a rejection surfaces as `LifecycleEvidenceSinkError` under the sink contract, never silently |
| R-4 | GM-AB-1 is refused | variant R, pre-specified (same digest); 04a/04b held out; PDC-D1-R017 open only by the Operator's explicit DS-EXC re-disposition recorded with Q-AB-02 |
| R-5 | Other strict entry points accept a lab composition; the shipped GCA strict path runs with the trust check not composed | Operator questions Q-AB-06 and Q-AB-07; never reported as closed |
| R-6 | Merge conflict on `admin_service.py` with RW-TENANT, on `production_validation.py` with RW-GATES | section 9 rules, including the reverse rule |
| R-7 | Stored trust records gain a field | default `()`; old rows parse (`sqlite_stores.py@P:100-101`); none of the `content_digest_for_model` users in `intergrax/agent_distribution` takes the installation trust record (F-17); rollback limit in section 13 |
| R-8 | PR #676 changes again before it merges | step 0 re-reads every citation at the merged head; a meaning change is an amendment re-signed by the Operator (S1) |
| R-9 | A sink error after the store writes leaves an installed package with an error to the caller | stated contract (committed state, typed error); NT-RW-SUP-04c |

## 13. Rollback

Revert Part B, then Part A; P1-P3 revert with Part B (the required field); in variant E, C-1 and C-1b revert with the sink. The stop marks an instance `FAILED` through the existing primitive; recovery is the existing rollback or a new trusted activation (NT-RW-SUP-03e). **Data limit:** `AgentInstallationTrustRecord` is `extra="forbid"` (`trust.py@P:134`), so rows written with `attestation_material` cannot be parsed by reverted code: a revert of Part A after any durable install with material needs a data step (strip the field from stored rows) and is otherwise allowed only before such writes. The IIAC rebind must be redone after any revert. No setting or production configuration change.

## 14. Open questions with draft defaults

| ID | Who | Question | Draft default |
|---|---|---|---|
| Q-AB-01 | orchestrator, then **Operator** (digest in a wave packet, S1 only) | RW-PLAN amendment: new node `T-AF2-RW-SUPPLY-AB` with the A1 block of section 5 (9 firm, 2 cond), rows NT-ROW-075..078 re-owned (077 with nodes 03c-03e), lane 5, the overlap pairs of section 9 (051-F pair, 051-A reverse rule) | amend as drafted |
| Q-AB-02 | **Operator** (gate-meaning GM-AB-1) | classify the lifecycle evidence consumer in the COMPAT-X R3 inventory and extend the pinned evidence set (C-1, C-1b) | accept (variant E). Refusal = variant R **and** an explicit DS-EXC re-disposition of PDC-D1-R017 by the Operator |
| Q-AB-04 | **Operator** (EX-D1-04 behaviour) | which no-safe-target reasons stop serving, and for every construction or only when composed | stop on `NO_PRIOR_REVISION`, `ROLLBACK_TARGET_UNTRUSTED`, `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE`, `ROLLBACK_FAILED`; never on pointer mismatch or conflict; every construction. Note for the ruling: `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE` is broader than "revoked without a trusted target": it is returned before any target is evaluated, for a missing installation or trust record (`emergency_revocation_response.py@P:292-314`, `:589-612`), so serving also stops on missing trust evidence (consistent with `assert_revision_trust_admissible`, `admin_service.py@P:1773-1777`); and a direct construction without a mutation boundary now raises on a stop reason instead of returning the report. One test per reason (03a, 03c) and per non-stop case (03b, 03d) |
| Q-AB-06 | **Operator** (DS-EXC: EX-D1-02, `DEFER_ALLOWED = no`) | other strict entry points accept a lab composition: `open_gca_strict_production_serving_session` (`strict_gca_production_process_composition.py@P:80-104`, bypasses the new single-node refusal), `open_strict_production_serving_session` (`strict_production_serving_launcher.py@P:378`, `:425`) and `production_process_lifecycle.py@P:388` | a named before-freeze follow-up node `T-AF2-RW-SUPPLY-STRICT` (`DEFER_ALLOWED = no`, lane after this ticket) that moves the lab refusal into the shared strict launcher (one place covering all strict entry points); alternative: amend this ticket with `strict_production_serving_launcher.py` as a firm path |
| Q-AB-07 | **Operator** (DS-EXC: EX-D1-02, `DEFER_ALLOWED = no`) | the shipped GCA strict path (`application_composition=None`) runs `deploy_and_activate` for an activation intent with the trust admission `NOT_COMPOSED`, so a revoked digest is admitted there (`strict_production_serving_launcher.py@P:436-440`, `deploy_activate_pipeline.py@P:126`) | the same follow-up node: a strict launch with an activation intent and no composed trust admission is refused (fail closed), unless a standalone revision-trust admission (Q-SUP-02) is composed; before freeze. EX-D1-02 stays open on that path after this ticket |
| Q-AB-03 | orchestrator | event context for the sink: the admin service adds `application_id` and `application_environment_id` to the events it passes to the sink | yes; no sink signature change |
| Q-AB-05 | orchestrator | `PackageAttestationEnforcement` in `admin_service.py` instead of `trust.py` (import cycle) | yes |
| Q-AB-08 | orchestrator | red-first: merged nodes on the old API where possible, a non-merged probe with identical node ids and assertions otherwise, plus revert-red at the head | yes |
| Q-AB-09 | orchestrator | the lab composition uses `forbid_unsigned_or_unqualified=False`, read only by the Part A step; a non-lab composition with that policy is refused | yes |
| Q-AB-10 | orchestrator | re-verify attestation material in `assert_revision_trust_admissible` and on the idempotent replay | no (install and upgrade only, as signed); recorded as RES-AB-5 |
| Q-AB-11 | orchestrator | 051-F and this ticket on `production_validation.py` | either order; the second rebases |
| Q-AB-12 | orchestrator | rename `COMPOSED_RUNTIME_RECORDS_ONLY` to `COMPOSED_RUNTIME`, docstring naming the lab limits | yes (value not pinned by any test) |
| Q-AB-13 | orchestrator | sink failure contract (typed error on success paths with committed state; original error kept, sink error chained, on failure paths) and the concurrency rule of the stop (`RuntimeActivationError` → superseded event, no stop; `RuntimeActivationConflict` → propagates) | yes |

Residuals if the defaults hold (each with its route): RES-AB-1 EX-D1-02 on the shipped GCA strict path (**Operator, Q-AB-07**); RES-AB-2 strict entry points without a lab refusal (**Operator, Q-AB-06**); RES-AB-3 automatic revocation response (EX-D1-03, NOT_COMPOSED, RW-GATES gate); RES-AB-4 production keys, revocation feed and evidence backend (PROD-Q); RES-AB-5 installations admitted before enforcement keep serving without attestation re-verification (idempotent replay `admin_service.py@P:599-615`, `:791-805`; revision admission, Q-AB-10; owner orchestrator, recorded for the freeze record).

## 15. Authorization requests

| ID | Request | Needed for |
|---|---|---|
| AB-A1 | Operator approval of this plan digest in a wave packet (S1 only), with an independent review of revision 2 | any edit |
| AB-A2 | Operator rulings Q-AB-02 (GM-AB-1), Q-AB-04, Q-AB-06, Q-AB-07 | Part B variant, stop reasons, residual routes |
| AB-A3 | one detached worktree under `D:/Projekty/_wt/intergrax/` for S0 and S0b, removed afterwards | S0, S0b |
| AB-A4 | pytest under the shared lock, one process at a time, the files of section 7 | S0-S2 |

## 16. Effort

| Part | Size | Note |
|---|---|---|
| A attestation | M | one model field, one enforcement step |
| B composition, sink, stop with event, strict refusal, state rename, P1-P3, C-1/C-1b | M | five product hunks, three one-keyword support edits, one classification row, one set literal |
| Whole ticket | M | 5 product paths, 2 commit groups after the lint commits |

## 17. Review dispositions (revision 2)

Source: `REVIEW_DRAFT_051N.md` (another session's record; quoted as a label, not as a verdict of this plan).

| Item | Severity | Disposition |
|---|---|---|
| H-1 pinned evidence set in `test_compat_x_event_evolution_r3_p0_gates.py:296-313` | HIGH | added as conditional C-1b; GM-AB-1 covers both edits; counts 2 conditional (sections 4, 5, 11) |
| H-2 residuals of DEFER_ALLOWED = no items routed to the orchestrator | HIGH | Q-AB-06 and Q-AB-07 are Operator questions with defaults (follow-up node `T-AF2-RW-SUPPLY-STRICT`, before freeze); GM-AB-1 refusal is variant R plus an explicit Operator DS-EXC re-disposition; variant R pre-specified so one digest covers both branches; EX-D1-02 stated open on the shipped GCA strict path (sections 1, 3, 14; ticket) |
| M-1 stop leaves no audit evidence | MEDIUM | explicit `agent_trust.emergency_serving_stopped` event through `_emit_events`, added to the report; 03a asserts it and the failure evidence ref |
| M-2 sink failure semantics | MEDIUM | contract defined (Part B, Q-AB-13); node 04c |
| M-3 self-declared `is_lab` | MEDIUM | `is_lab` keyword-only and required; non-lab with a permissive policy refused; verifier-strength limit stated; 02a extended |
| M-4 probe adequacy | MEDIUM | merged nodes on the old API where possible (03a, 03c, 04c); probe with identical node ids and assertions; revert-red step S0b |
| M-5 S0-1 census misses the mark authorization | MEDIUM | census of the 4 callers with boundary, mark admission and call counts; fail-closed raise without a boundary stated (section 4, step 0, Q-AB-04) |
| M-6 NT coverage of the reason set | MEDIUM | nodes 03c (R), 03d (C, conflict and concurrent change), 03e (C, recovery); concurrency rule defined; 16 nodes |
| L-1 counter shift vs known-red comparison | LOW | counter rule in section 7 |
| L-2 rollback of persisted rows | LOW | data limit in section 13 |
| L-3 callee and qualname inconsistency | LOW | one method `record_events`, callee `self._trail.append`, row bound to path, qualname and callee |
| L-4 renamed docstring | LOW | docstring names the lab limits |
| L-5 051-A and 051-F already signed | LOW | reverse-order rule for 051-A; signed status noted |
| L-6 TENANT-X candidate count | LOW | S5 records the total candidate count; rule 7 |
| I-1 DUP-X/EBH-5 consumption claim | INFO | removed (FACT_CHECK_rev2 F-40; dry-run rev2) |
| I-2 pre-enforcement installations | INFO | residual RES-AB-5 |
| the cache-provider-disabling pytest option (fails under `--strict-config`) | brief | removed; section 7 allows no pytest option beyond the repository configuration except `--basetemp` |
