# T-IGX-041-A — PDC-X batch B-P7-D1: repair of the IA-D1 audit findings for domain D1 (AGENT-LIFECYCLE), loop 1

## Scope

Repair ticket for the independent audit `PDC_P7_D1_AUDIT_REPORT.md` (IA-D1, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL), under Operator option B of `DELEGATION-PDC-P7-REPAIR.md`. In scope: F-D1-02 (activation, rollback and the deploy/activate pipeline re-check revocation of the target packages), F-D1-03 authorization part (principal, tenant scope and policy on `respond_to_emergency_revocation`, no caller-supplied revocation snapshot), F-D1-05 (application-ownership guard on install into an occupied slot and on upgrade), F-D1-06 (no leftover `VERIFIED` candidate after a failed upgrade; optional inert event sink so failure-path transitions have an event), F-D1-11 (all 18 red D1 suites root-caused and repaired as test hygiene, classification or one conditional product fix), and by-product test pins for F-D1-12 (mutants M2, M7, M10) and F-D1-14. Excluded with reason and reopen condition (plan section "Excluded findings"): F-D1-01 (attestation never verified on install/upgrade; the trust record has no signature material and no production composition supplies a verifier), F-D1-03 trigger, F-D1-04 (Operator decision), F-D1-07, F-D1-08 (no production caller, F-04 family), the production wiring of the trust sources and of the pipeline hook (the production composition injects no revocation source), and the LOW/INFO findings. Product paths: yes (`intergrax/agent_distribution/admin_service.py`; hotspot files named exactly: `intergrax/applications/_shared/deploy_activate_pipeline.py` and, conditionally, `intergrax/applications/_shared/durable_registry_projection_input_store.py`). This ticket is not a certification verdict and not a closure of D1; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 identity gate red until the final re-pin. Plan: `PLAN_T-IGX-041-A.md`.

## Plan digest

Plan digest: 5bd3933a0f6f7052955fd5108a46b49335ba08a1ddd8526686e74f36100e93d2

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-A.md`](plans/PLAN_T-IGX-041-A.md).

## Approved

Approved: operator-delegate @ 2026-11-15T09:44:27+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md`; all passed before this line was written; read at the batch branch `claude/pdc-x-041a-f76aae`, plan commit `b435d5bce`, which amended the first plan version `fdebf2482` after the delegate's first check refused to sign it):

- Why a first signature was refused: the first plan version said the pipeline `run_deploy_and_activate` "built and validated" its revision moments earlier; the code (`deploy_activate_pipeline.py:98`) takes a caller-supplied `projection_input.runtime_revision`, the pipeline has production callers and no trust check. The amended plan states this truthfully, brings the pipeline into scope (Deliverable 2b) and names the file as a second exact hotspot. Everything below is re-checked from scratch on the amended version.
- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`5bd3933a0f6f7052955fd5108a46b49335ba08a1ddd8526686e74f36100e93d2`).
- The 17 non-document paths of the plan write set equal the write set of the batch row `B-P7-D1` in `PDC_IMPLEMENTATION_WBS.md` and in the generator (`scripts/qualification/pdc_x_wbs.py`); no wildcard; `uv run python scripts/qualification/pdc_x_wbs.py --check` is OK. The generator lives in `scripts/qualification/`, not under `scripts/governance/`. Nothing under `scripts/governance/` or `.github/`.
- Hotspots under `intergrax/applications/_shared/` are exactly two, named in full: `deploy_activate_pipeline.py` (Deliverable 2b: one optional context field and one call) and `durable_registry_projection_input_store.py` (conditional on Deliverable 7e, condition stated; unchanged if the root cause is a stale fixture). No path under `intergrax/contracts/`.
- In-scope findings and every excluded finding are listed with reason and reopen condition. Exclusions are of the kinds option B allows: F-D1-01 (missing trust input and composition), F-D1-03 trigger, F-D1-07, F-D1-08 (no production caller, F-04 family), the production wiring of the trust sources and of the pipeline hook (missing production composition), LOW/INFO debt. F-D1-04 is excluded as an Operator decision; the plan leaves `emergency_revocation_response.py` unchanged (read: no prior revision still returns `BLOCKED_NO_SAFE_TARGET`).
- Factual sentences re-read against the code and found true: `retire_agent` calls `_assert_installation_application_scope` (`admin_service.py:962`) while `install_agent`/`upgrade_agent` do not; `mark_verified` / `promote_verified_to_active` / `mark_failed_candidate` positions in `upgrade_agent`; `activate_revision` (1392) and `_execute_governed_rollback` (2323) have no revocation re-check; the pipeline order (`_authorize_admission` 150 before `persist_candidate_revision` 167, `_authorize_activation` 188 before `commit_activation` 217); GR-10 `PreModelPolicyConfigurationError` in `pre_model_principal.py:67-89`; the 18 ids of `failed_ids.txt` match the plan table and are covered by the write set; `AgentPackageTrustCoordinator` verifier is optional (`package_trust.py:86-88`) and `build_production_agent_platform_admin_service` (`production_agent_capability_runtime.py:181-225`) passes no coordinator, revocation, policy or time source, so the service uses an empty revocation state (`admin_service.py:321-326`) and it is the only production constructor; `AgentRegistry.mark_retired` and `SkillHostLifecycleService` have no production caller; `respond_to_emergency_revocation` has callers only in the three listed test files and two frozen audit probes; nothing in production calls `respond_to_current_revocation` apart from that method.
- No gate is weakened or deleted. The three classification edits add assertions (exact allowlist with a no-silent-staleness check; static authorization-before-commit order check for the pipeline); the fingerprint comparison of the durable CAS test is not loosened; the ACP fixture is non-autouse and a negative test keeps the fail-closed gate. Existing tests in the write set were read (`test_pattern_probe_runs`, `test_ac6_architecture_gates`, the two bypass inventories, the canonical e2e): the planned edits are stale-fixture hygiene and classification only.
- No registry, scheduler, engine, trust, policy or audit authority is added: the revocation check reuses the injected coordinator and sources; the pipeline hook and the `lifecycle_event_sink` are optional, default-off injection points that decide and store nothing. The plan states it is not a certification verdict and issues no `Approved:` line.
- Implementer (planning session `plan-041a`) and this delegate session are different sessions; the delegate edited only this approval block and the lifecycle state.
- Concerns recorded for the Operator, not blockers: (1) the revocation checks of Deliverables 2, 3 and 8 take effect only when a revocation-state source is injected; the production composition injects none (declared exception in the plan), so production enforcement needs the wiring ticket; (2) the activation/rollback check treats an enabled entry without installation or trust record as a denial, which is stricter than the build path and may surface in existing tests (the plan makes that a stop condition); (3) `respond_to_emergency_revocation` becomes subject to policy (DENY / REQUIRE_HUMAN) when a prior revision exists, which can delay an emergency rollback; (4) the WBS row still names OD-PDC-SEQ-01 as merge gate while the delegation names the standing authorizations; the delegation text governs.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d1_repair tests/qualification/pdc_x/d1 tests/unit/agents/authoring tests/unit/agent_distribution tests/integration/agent_distribution tests/unit/runtime/registry tests/unit/skills tests/qualification/pdc_x/xd/test_xd_04_agent_host.py -q
uv run ruff check intergrax/agent_distribution/admin_service.py intergrax/applications/_shared/durable_registry_projection_input_store.py tests/qualification/pdc_x/d1_repair tests/unit/agents/authoring/conftest.py
uv run pyright intergrax/agent_distribution/admin_service.py tests/qualification/pdc_x/d1_repair
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- All excluded findings listed in the plan; the fail-closed rule for a revoked revision without a trusted rollback target (F-D1-04, Operator decision); verifier wiring, production composition of the lifecycle verbs, the revocation-feed trigger.
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, every `intergrax/applications/_shared/` file except the two named in the plan, `intergrax/agent_distribution/activation.py`, `emergency_revocation_response.py`, `package_trust.py`, `testing_support/`, the TENANT-X carrier re-pin (T-IGX-041-I), the EBH-7 re-pin, the IIAC activation record and rebind, FRZ promotion, OD signatures, certification.

## Lifecycle state

plan_approved
