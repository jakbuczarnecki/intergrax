# T-IGX-051-B — T-AF2-RW-SUPPLY: supply chain and security (attestation, revocation, trust hook order, digests, fail-open set, strict validator, secrets, evidence sink, retention)

## Scope

Product-path ticket of the RW train (master-plan node `T-AF2-RW-SUPPLY`, lane 4; branch `arch-freeze/rw-train` only; DP-11 single integrated wave). Binding scope: `RW_PLAN.md` v13 Appendix A1 block `T-AF2-RW-SUPPLY (T-IGX-051-B)`, its negative-test rows NT-ROW-075..096 and NT-ROW-159, its regression row and its lane and overlaps. The ticket closes nothing by itself; closure conditions are in plan section 11. Draft of 2026-11-17, **revision 2** after the step-0 dry run (`dry-run/DRY_RUN_T-IGX-051-B.md`, 10 findings, each answered in plan section 17); written against `origin/arch-freeze/rw-train` `05699b633cdb1318c215f6529e449420efc1382c` (the merge of PT-WA); every product path of the write set is identical there and at `origin/development` `2727a2c2364a97492202b931375772a400279a9b`, and unchanged at the train head `881428fb231f34215a67364aa5788a4f2f986e83` (dry run).

Findings addressed, by ID and title only (labels quoted with their source: DS-EXC, the PDC exception and debt registers, the RW-PLAN):

- EX-D1-01 attestation not verified on the composed install/upgrade path (plan part A);
- EX-D1-02 revocation source, trust coordinator and pipeline hook not composed in the production path, both launchers; EX-D1-04 revoked revision without a trusted rollback target keeps serving; PDC-D1-R017 lifecycle evidence sink has no production consumer (part B);
- A-01 trust hook runs after the revision is persisted (part C);
- EX-D6-03 no installed-file digest check for plugins; FRZ-SEC-06 security-defense loader without provenance on a strict or product root, basis C (part D);
- EX-D6-01 and F-D6-02 audit trail without tamper evidence; F-D6-03 listing truncation and stored secret-shaped values (part E);
- fail-open set A-12: F-D3-12 / PDC-D3-R018 quota disables itself on a malformed value, F-D6-09 poisoning default differs by path, F-D6-11 tenant-less resource open to a scoped tenant and identity not normalised, F-D5-12 registry lock lease (class C), F-D8-14 legal hold overwritable and lossy (part F);
- F-D6-10 / PDC-D6-R024 strict-config validator accepts weak or disabled settings; EX-D6-02 and F-D6-04 production host composes no D6 controls (requalify first, class C) (part G);
- F-D6-14 / PDC-D6-R007 static secret-gate evasions; PDC-D6-R010 sandbox gate checks presence, not strength (part H);
- EX-D8-01 no retention composition and no AUDIT_EVIDENCE producer (part I).

Related items named in the RW-PLAN row but owned elsewhere, not edited here: FRZ-PLG-05 and FRZ-EXE-02 (PT-FC1), A4-040 (PT-WB; the scope ruling is open, Q-RW-16 b). This ticket takes no position on A4-040 and does not edit the bind-host default, the `uvicorn.run` block, the compose file, the RUNBOOK or the env example.

Write set (exact lists in the plan, section 5; equal to Appendix A1 and to `rw_plan.v1.json`, checked path by path): **20 firm product paths** (19 files and the glob `intergrax/core/plugins/*`, 9 tracked files at the head), **2 conditional product paths** (`intergrax/core/security_bootstrap.py`, `intergrax/applications/_shared/security_runtime_bridge.py`, conditions in the plan), **11 firm test and script paths** (9 new test files, 2 new scripts), firm total 31, conditional total 2; 19 negative-test ids written in 50 test nodes; **the merge set is 18 ids in 48 nodes: NT-RW-SUP-11 and 12a are held out of the merge set** (kept in the worktree or the draft folder) until Q-SUP-11 or the split ticket gives them a fixer, and **no expected-red test merges inside this ticket**. No docs, no edited existing tests in the base write set. The design keeps direct constructions of the admin service, the pipeline context and `PluginProvenanceAdmission` unchanged ('not composed' states) and enforces only through the production composition, so the existing callers (about 26 and 23 files in the dry run) need no edit. **Proposed additions to A1 (plan section 5b; not in scope until RW-PLAN is amended and the digest is recomputed): 3 firm** (P1-P3: the three constructors of `ProductionAgentPlatformAdminConfig`, `testing_support/canonical_agent_lifecycle_composition.py`, `platform_proofs/.../production_validation.py`, `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`) **and 8 conditional** (C1-C3b: the poisoning-default and tenant-scope paths; C4-C7: pins of the trail append site and the SQLite schema constant). **Proposed split: T-IGX-051-I** (F-D6-09 and the semantic half of F-D6-11, an Operator decision, paths in plan section 5b). `package_b_strict_production_serving_command.py` is firm in A1 and is not edited by this plan. Open items with owner: NT-RW-SUP-11 and 12a held out (Operator for Q-SUP-05, orchestrator for Q-SUP-11; F-D6-09 and the semantic half of F-D6-11 stay open), EX-D1-02 for the GCA strict path if it composes no admin service (orchestrator), EX-D6-03 composition (orchestrator), secret-gate true positives (Operator); registered classes KR-08 and KR-09 (plan section 7); planned expected-red of existing `tests/qualification/tenant_x` pin nodes if this ticket adds tenant carriers (owner RW-TENANT, registered by the orchestrator). The shared strict serving command is serialized after PT-WB (Part G of PT-WB is conditional and out of scope here). ACT-N11: every product path above breaks the IIAC activation until the single RB-2 re-pin; this ticket does not perform it and must not merge outside the wave.

Contract surfaces: the ticket edits `intergrax/contracts/agent_run.py`; the wave-packet contract check (`.tmp/session/arch-freeze/closed-world/typ_measure.py` and `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py`) must report no new `Any`, `object` or reflection metric and no new string-keyed lookup table (OD-AF-11).

Scope hold: no edit of `intergrax/`, `agents/` or `applications/` before the wave of this ticket and the delegate approval below. Commits carry the trailer `Ref: governance/tickets/T-IGX-051-B.md`.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan.

## Plan digest

Plan digest: 969407ba56159dc4be23820146609c3d284a49ff6ca1332c8c2229677e2cc562

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-B.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/rw-supply/PLAN_T-IGX-051-B.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with a copy of `scripts/governance/plan_digest.py` from `origin/development` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-18T08:24:46+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with several paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it); exact files in plan sections 6 and 7:

- Red-first (S0): the nine new test files (NT-RW-SUP-11 and 12a are written but held out of the merge set until their paths are added or the split is made) and the two new scripts' tests on a detached worktree at the pre-change commit; every R, S and G test fails for a behavioural reason (not an import error alone); the control tests (C) and the three conditional rows (NT-RW-SUP-08, NT-RW-SUP-13, NT-SEC-06-1) are recorded and, if green at the base, re-dispositioned by the RW-PLAN rule; the log is kept for the auditor.
- After each part: the new test file of the part plus its regression directories; at the end the RW-PLAN regression row (`tests/unit/agent_distribution`, `tests/unit/core/plugins`, `tests/unit/runtime/security`, `tests/qualification/pdc_x/xd`, `scripts/maintenance/check_immutable_security_audit_trail.py`, `scripts/maintenance/check_sandbox_policy_wiring.py`) and the additional files named in plan section 7.
- Gates: the two new scripts and the two existing audit-trail and sandbox-wiring scripts; the contract check above; `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on the changed Python files; a mechanical lint-and-format commit per touched file before its first behavioural hunk (the dry run found findings in six touched files).
- No broad or diagnostic suite, no Docker, no vendor integration, no real keys, revocation feeds or live backends (C10/C15 freeze); in-process fixtures are labelled as stubs in the PR body. Secrets and key material in tests are generated at run time; nothing echoes a candidate secret.
- The independent audit is a separate session at the exact wave head (node `PT-Q02`).

## Out of scope

- A4-040 and everything in PT-WB Part G (bind-host default, `uvicorn.run` block, compose file, RUNBOOK, env example); `intergrax/applications/_shared/production_lifecycle.py` (PT-WF); the paths and tests listed in plan sections 4 and 5b until an amendment adds them (among them the two lifecycle launcher files, the serving command, `package_trust.py`, `environment_wiring.py`, `plan_context_invocation.py`, `application_security_wiring.py`, `raw_secret_gate.py`, `multi_region_audit_trail.py`, `worker_bootstrap.py`, `agent_runtime_context.py`, the governed artifact callers, and the existing tests whose pins the change inverts).
- Real publisher keys, a real revocation feed, a live evidence backend or a real sandbox provider (PROD-Q); the EX-D8-01 responsibility-statement alternative (a documentation artefact); holder-scoped release of legal holds; the revocation-feed subscriber of EX-D1-03.
- The IIAC re-pin (RB-2), the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_eaf9ef72-d1c1-44d0-acbc-fac05f02edf7
