# PDC-P7 IA-D1 — independent audit of domain D1 (AGENT-LIFECYCLE)

Stage `PDC-X/P7-D1` · policy `IIAC-2026-11-05.1` (OD-PDC-IIAC-SCOPE, Option A) · date 2026-11-15

| Item | Value |
|------|-------|
| Role | `IA-D1`, independent auditor (clean session). Not the implementer of B-D1 (T-IGX-036-J) or of any later D1 change, not a signer, not a certifier. No product, test, script or governance file was edited. |
| Audited SHA | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (`origin/development`; audit worktree HEAD is this commit) |
| Attribution SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (ACT-N11 activation pin), used only to separate pre-existing reds from PDC-X changes |
| Scope | 22 rows `PDC-D1-R001..R022` of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`; owner paths of those rows; B-D1 write set; ADR-PDC-01; ownership rows for D1 |
| **Overall verdict** | **PARTIAL** (manifest value `CONDITIONAL`; the manifest schema has no `PARTIAL`) |
| Rows | 10 PASS · 1 PASS_SCOPED · 11 PARTIAL · 0 FAIL |
| Findings | 3 HIGH · 6 MEDIUM · 7 LOW · 4 INFO (20, `F-D1-01..20`) |
| Repair required | yes, in a D1-owned ticket by a different session, then a fresh IA-D1 re-audit (loop 1 of 2) |

This is an audit, not a certification and not a closure. It issues no `Approved:` line, changes no row status in the matrix and makes no FRZ or production-readiness claim.

## 1. Verdict in one paragraph

The upgrade, retire, rollback and skill-lifecycle components that B-D1 added are real, well covered at component level (65 pdc_x tests pass; a 10-mutant sample kills 7), and tenant scoping holds for every verb. The domain is not complete because three claims do not hold on the composed path: attestation is never verified before an agent install or upgrade (F-D1-01), a revoked package can be restored or activated by a normal operator action (F-D1-02), and the emergency revocation response is an unauthenticated mutating entry point that nothing triggers (F-D1-03). Lifecycle "evidence" is a returned tuple of strings that nothing consumes (F-D1-06), retire never reaches the runtime registry (F-D1-07), and several D1-owned suites that the matrix cites as evidence are red, identically at the pin and at the audited SHA (F-D1-11).

## 2. Method

1. Read `OD-PDC-IIAC-SCOPE`, the D1 rows (`DOMAIN_ID = PDC-D1`), `PDC_OWNERSHIP_MATRIX.md`, `ADR_PDC_01_AGENT_UPGRADE_RETIRE.md`, the B-D1 section of `PDC_IMPLEMENTATION_WBS.md` and ticket `T-IGX-036-J`. The dossier and addenda 1..4 were used as leads only.
2. Read the owner code of every row (admin service, installation and upgrade-admission services, activation, emergency revocation, package trust, registry projection, skills host lifecycle, plugin discovery and admission, contracts) the D1 test harness, and the R003/R004/R007/R013/R017/R020/R018 sections of the 4 D1 test files (the remaining test bodies were run, not read line by line).
3. Ran the cited tests (section 3), then 14 adversarial probes (`adversarial_scripts/`), then a 10-mutant sample on a throwaway checkout (section 5). One pytest process at a time; scratch outside the repository.
4. Re-verified the routed leads (section 6) and attributed every red test to the pin or to PDC-X with a second checkout.
5. Layer check: no import of top-level `agents/` or `applications/` from the audited `intergrax/` paths (`intergrax.applications._shared` is a sub-package of `intergrax/`; the one use in `agent_distribution/sqlite_stores.py` is lazy).

Environment: the shared `.venv` of the main checkout (resolves `intergrax` from the audit worktree, tzdata present), Windows 11, pytest 8.4.2.

## 3. Test evidence

| Run | Result |
|-----|--------|
| `tests/qualification/pdc_x/d1` | 65 passed, 94 s, process exited |
| `tests/unit/agent_distribution`, `tests/integration/agent_distribution`, `tests/unit/runtime/registry`, `tests/unit/agents/authoring`, `tests/unit/skills`, `xd_04` (one process) | 18 failed, 1581 passed; summary printed after 43 s, process did not exit (F-D1-20) |
| the 18 failed ids at the pin `2d3eceb6c` | 18 failed: all pre-existing (F-D1-11) |
| `tests/qualification/plug_03`, `plug_04`, `plug_final` | 27 passed, process exited |
| `xd_04` alone | 3 passed, process exited |
| probes A/B/C | 31 PROBE lines, 0 probe errors |
| mutation sample | 7 killed, 3 survived |

## 4. Per-row verdicts

Vocabulary: PASS, PASS_SCOPED (Operator-authorized exclusion only), PARTIAL, FAIL. "Evidence" names the artifact under `iiac_evidence/PDC-X/D1/`.

| Row | Claim | Verdict | Evidence and reason | Findings |
|-----|-------|---------|---------------------|----------|
| R001 | One AgentContract identity and version per agent | PARTIAL | Holds by code read (`AgentRegistry.register` rejects a duplicate id; id pattern enforced). The named gate "identity uniqueness gate" does not exist and no test asserts the duplicate rejection. | F-D1-14 |
| R002 | Typed agent authoring contract | PARTIAL | Cited evidence is "existing ACP unit suites": 12 of them fail at the audited SHA and at the pin (`PreModelPolicyConfigurationError`); cause not isolated by this audit. | F-D1-11 |
| R003 | Package identity, provenance and attestation verified before install | PARTIAL | Digest pinning, revocation, status and evidence-presence checks hold and are tested. Attestation is not verified on install or upgrade; the trust record is caller-asserted (probe P8). | F-D1-01 |
| R004 | Single agent registry authority | PASS | `AgentRegistry` defined once; distribution plane never imports it; registry has no install API (code read, 3 gate tests green). Gate is name-based. | F-D1-12 |
| R005 | Capability declarations and dependency resolution | PASS | `validate_contract_metadata` plus resolution in `AgentRegistry.register`; `tests/unit/runtime/registry` all green. | — |
| R006 | Agent assembly validation (contract + lifecycle metadata) | PASS | `validate_lifecycle_metadata` and cognitive-pattern checks; registry unit suites green. | — |
| R007 | Tenant-scoped install, bind and activate | PARTIAL | Tenant scope verified for install, bind, upgrade, retire, activate, rollback (tests green; mutants M4/M5 killed). No application-ownership guard on install and upgrade; policy authorization on upgrade/retire is not pinned (M2 survives). | F-D1-05, F-D1-12 |
| R008 | Installed != configured != effective != authorized | PASS | `xd_04` (configured roster differs from effective roster, both durable) green; installation states and bindings separate in code. A related e2e is red for a stale-fixture reason. | F-D1-11 (limitation) |
| R009 | Dynamic acquisition AC-4; doc status | PARTIAL | Architecture doc reconciled (line 2002). The plan doc row for Phase 10 still reads READY_FOR_INDEPENDENT_AUDIT. | F-D1-15 |
| R010 | Version-pinned runtime binding | PASS | `RuntimeRevision` pins package digests and, after B-P6-M, the LLM profile digest; unit suites green. The LLM pin is opt-in. | F-D1-16 |
| R011 | UPGRADE is an explicit governed operation with a compatibility check | PASS | `upgrade_agent` is install-candidate, verify, admission check, promote; reject leaves the previous revision active; 20+ tests green; mutants M1, M8, M9 killed. Edge cases are LOW. Trust gap of the candidate is F-D1-01 (row R003). | F-D1-13 |
| R012 | Rollback safety | PARTIAL | Failed activation and failed promotion roll back correctly (tests green). A normal rollback or activation to a revoked package is allowed (probes P9, P13). | F-D1-02 |
| R013 | Revocation and emergency revocation reach active executions | PARTIAL | Rollback on revocation with a trusted prior works. No-target case leaves the revoked revision serving and admitting; the entry point has no authorization and no trigger. | F-D1-02, F-D1-03, F-D1-04 |
| R014 | RETIRE is an explicit operation | PARTIAL | `retire_agent` works at the distribution plane (roster exclusion, refusal of new installs, evidence kept; mutants M3, M4, M6 killed). `AgentRegistry.mark_retired` has no production caller. | F-D1-07 |
| R015 | Cross-host agent reuse | PARTIAL | Facade is a one-to-one delegate; no test exercises reuse across hosts or environments. | F-D1-17 |
| R016 | Activation continuity across restart | PASS | `xd_04` restart tests green; durable restart e2e tests in the wide run pass. | — |
| R017 | Every lifecycle transition emits canonical evidence | PARTIAL | One event per transition exists and is returned; nothing consumes or persists it; failure paths persist transitions with no event. | F-D1-06 |
| R018 | Skill disable/remove/upgrade kept separate from agent lifecycle | PARTIAL | Component implemented and tested, separate from the agent plane. No production caller; a BaseException uninstalls the skill; pre-release candidates replace releases. | F-D1-08, F-D1-09, F-D1-10 |
| R019 | Runtime factory beyond VENV_BUNDLE | PASS_SCOPED | Authorized exclusion: OD-PDC-SCOPE-03 Option A (2026-11-12). | F-D1-18 |
| R020 | One owner for plugin discovery | PASS | Only `core/plugins/discovery.py` scans entry points (static scan over `intergrax/`, `agents/`, `applications/`); the RAG loader and Vendor Knowledge catalog delegate. Gate is syntax-based. | F-D1-12, F-D1-16 |
| R021 | plug_03/04/final have an owning stage | PASS | Ownership matrix assigns them to D1; the three suites pass (27). | F-D1-18 |
| R022 | plug_03 external chunker entry-point flow green | PASS | `tests/qualification/plug_03` green at the audited SHA. | F-D1-18 |

## 5. Mutation sample (throwaway checkout of the audited SHA)

| Mutant | Result |
|--------|--------|
| M1 admission check bypassed in `upgrade_agent` | killed |
| M2 policy authorization removed from `upgrade_agent` and `retire_agent` | **survived** |
| M3 application-scope check removed from `retire_agent` | killed |
| M4 tenant scope removed from `retire_agent` | killed |
| M5 tenant scope removed from `upgrade_agent` | killed |
| M6 retired slot accepts candidate installations | killed |
| M7 candidate with the same version accepted as an upgrade (`<=` to `<`) | **survived** |
| M8 bound-tool removal check ignored | killed |
| M9 capability removal check ignored | killed |
| M10 rejection `reason_code` dropped from the `failed_candidate` event | **survived** |

The sample is not a measurement. M2 means no D1 test uses a DENY boundary on upgrade or retire.

## 6. Re-verification of routed leads

| Lead | Result at the audited SHA |
|------|---------------------------|
| F-04 (no production caller) | Confirmed for D1: upgrade/retire facade, `mark_retired`, skill lifecycle, emergency response, attestation verifier (F-D1-01, 03, 07, 08). Whether host wiring is a PDC-X claim is an orchestrator decision. |
| F-11 | Confirmed (probe P12): F-D1-09. |
| F-12 | Confirmed (probe P12): F-D1-10. |
| F-21 | Confirmed by code read: provenance lookup keyed by the mapping key, opt-in: F-D1-16. |
| F-26 | Partly reproduced: 0.x acceptance (`0.1.0 -> 0.9.0` admitted), replay before authorization and a leftover VERIFIED candidate are real (F-D1-13). The pre-release ordering claim is not reproducible: agent package versions are PEP 440 normalised and fail SemVer parse, so they fail closed. |
| F-37 | Confirmed by code read: the LLM profile pin is skipped for unpinned revisions, no production source: F-D1-16. |
| F-39 / F-53 | Re-verified for D1 suites: 18 reds, all also red at the pin: F-D1-11. |
| F-09 | A pytest process that included `xd_04` did not exit once; not bisected to a test: F-D1-20. |
| F-33 | Not re-run; the D1 paths audited here add no import of top-level `agents/` or `applications/`. |
| F-05 | Outside D1 (D2/D3); FINDING-9 and FINDING-10 were not re-run. |

## 7. Findings

Full evidence, reproduction and suggested repair per finding: `iiac_evidence/PDC-X/D1/iiac_audit_d1_findings.json`. Severity key as in the P7-00 dossier: HIGH = a requirement claim is false on the composed path in a security-relevant way; MEDIUM = design or evidence weakness; LOW = edge or test-strength; INFO = record.

| ID | Sev | Rows | Finding | Owner path |
|----|-----|------|---------|------------|
| F-D1-01 | HIGH | R003 | No production path verifies package attestation before agent install or upgrade; trust record is caller-asserted | `agent_distribution/admin_service.py`, `package_trust.py`, `dynamic_acquisition.py` |
| F-D1-02 | HIGH | R012, R013 | `rollback_revision` and `activate_revision` do not re-check revocation of the target packages | `admin_service.py`, `activation.py` |
| F-D1-03 | HIGH | R013 | `respond_to_emergency_revocation` has no principal, no tenant scope, no policy check, accepts a caller-supplied snapshot, and has no trigger | `admin_service.py`, `emergency_revocation_response.py` |
| F-D1-04 | MEDIUM | R013 | With no trusted rollback target the response is advisory; the revoked revision keeps admitting | `emergency_revocation_response.py`, `serving_execution_admission.py` |
| F-D1-05 | MEDIUM | R007 | `install_agent`/`upgrade_agent` have no application-ownership guard (only `retire_agent` has) | `admin_service.py`, `installation.py` |
| F-D1-06 | MEDIUM | R017 | Lifecycle evidence is a returned tuple; nothing consumes it; exception paths persist transitions with no event | `admin_service.py`, `events.py` |
| F-D1-07 | MEDIUM | R014, R004 | `retire_agent` does not reach `AgentRegistry`; `mark_retired` and the facade have no production caller | `runtime/registry/agent_registry.py`, `agent_manager_command_facade.py` |
| F-D1-08 | MEDIUM | R018 | Skill disable/remove/upgrade have no production caller | `skills/host_lifecycle.py` |
| F-D1-11 | MEDIUM | R002, R008, R013, R015 | 18 D1-owned suites red at the audited SHA and at the pin | `tests/integration/agent_distribution`, `tests/unit/agent_distribution`, `tests/unit/agents/authoring` |
| F-D1-09 | LOW | R018 | `upgrade_skill` restores only for `Exception` (F-11) | `skills/host_lifecycle.py` |
| F-D1-10 | LOW | R018 | Skill upgrade ignores pre-release tags (F-12) | `skills/host_lifecycle.py`, `contracts/semver.py` |
| F-D1-12 | LOW | R004, R007, R020 | Gates miss simple bypasses; 3 of 10 mutants survive (M2, M7, M10) | `tests/qualification/pdc_x/d1/*` |
| F-D1-13 | LOW | R011 | Upgrade admission edge cases (0.x rule, replay before authorization, leftover VERIFIED candidate) | `semver_compat.py`, `upgrade_admission.py`, `admin_service.py` |
| F-D1-14 | LOW | R001 | The named identity-uniqueness gate does not exist | `agent_registry.py`, `agent_contract_meta.py` |
| F-D1-15 | LOW | R009 | Plan document still says READY_FOR_INDEPENDENT_AUDIT | `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md` |
| F-D1-16 | LOW | R010, R020 | Opt-in hardening: LLM profile pin and plugin provenance not enforced by default | `admin_service.py`, `core/plugins/admission.py`, `core/catalog_bootstrap.py` |
| F-D1-17 | INFO | R015 | No test for cross-host reuse | `agent_manager_command_facade.py` |
| F-D1-18 | INFO | R019, R021, R022 | Positive record: exclusion valid, ownership recorded, plug suites green | — |
| F-D1-19 | INFO | R011, R012, R014, R016 | Positive record: 65 pdc_x tests, restart continuity, no layer violation in audited paths | — |
| F-D1-20 | INFO | R013, R016 | Process-exit debt R-05: one non-exiting pytest process in the wide run | `tests/qualification/pdc_x/xd` |

### Suggested repair grouping (D1 ticket by a different session)

1. Trust path: F-D1-01, 02, 03, 04 (one change set around `admin_service.py`, `package_trust.py`, `emergency_revocation_response.py`; needs an Operator decision for the no-target rule in F-D1-04).
2. Evidence and wiring: F-D1-06, 07, 08, 16 (needs the orchestrator decision whether host composition wiring is a PDC-X claim, F-04).
3. Scope and test strength: F-D1-05, 12, 13, 14, 09, 10.
4. Red suites and docs: F-D1-11, 15, 17.

## 8. Known reds not re-reported

Not re-reported as new: EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x` and Docker scope (C10/C15 freeze). They were not run by this audit. The `pdc_x` process-exit debt (R-05, fixed in T-IGX-036-Z) was observed once in this session: see F-D1-20.

## 9. Limitations — what this audit did NOT cover

- Rows were judged from the code paths and tests named above. `intergrax/agents/authoring` (R002) was not reviewed beyond the cited suites; the 12 ACP failures were not root-caused.
- Durable stores (SQLite, physical admission, lease and coordinator code), the VENV_BUNDLE adapter, PostgreSQL providers, real object storage, live providers and any network path were not exercised.
- Concurrency, performance and crash behaviour beyond what the 65 pdc_x tests and the probes exercise; Windows only.
- Cross-domain rows (`xd_01..xd_12` except `xd_04`), other domains' tests, `igx_gov ticket`, the full `tests/unit` tree, CI contexts, G-DOC checks and FRZ counters.
- The 10-mutant sample is not a measurement; mutants ran against LF-normalised copies in a throwaway checkout.
- `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` was read as UTF-8 with replacement (the file is not valid UTF-8); nothing was edited.
- Severity assignments are the auditor's; F-D1-04 could be graded HIGH by an Operator who treats "fail closed" as unconditional.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders for a later certifier.
- Any change to the audited D1 paths after `c08efe46a` voids this audit for the changed paths.

## 10. Independence and provenance

IA-D1 did not implement, sign or certify any D1 batch. Commit authorship is shared across sessions (one git author), so independence rests on session separation as stated by the orchestrator. The audited worktree was used read-only for product code; probes live under `.tmp/` and are archived as evidence; mutants were applied only to a throwaway detached checkout (removed after the audit).

## 11. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/D1/`: `iiac_audit_session_manifest.PDC-X.d1.json`, `audit_scope.json`, `test_matrix.txt`, `iiac_audit_d1_findings.json`, `logs/` (test runs, pin attribution, probe outputs, mutation results), `adversarial_scripts/` (`probe_d1_a.py`, `probe_d1_b.py`, `probe_d1_c.py`, `run_mutants.py`).
