# T-IGX-051-N — T-AF2-RW-SUPPLY-AB: attestation on install and upgrade, mandatory trust composition, revocation source, lifecycle evidence consumer, no-target serving stop (Parts A and B of T-IGX-051-B)

## Scope

Product-path ticket of the RW train (proposed master-plan node `T-AF2-RW-SUPPLY-AB`, proposed lane 5; branch `arch-freeze/rw-train` only; DP-11 single integrated wave). It is the follow-up for Parts A and B of `T-IGX-051-B` (T-AF2-RW-SUPPLY, signed plan digest `969407ba56159dc4be23820146609c3d284a49ff6ca1332c8c2229677e2cc562`), whose implementer stopped at write-set stop S-1, so PR #676 carries Parts C to I only. The orchestrator recorded the decision to draft this ticket for a later Operator packet (session record `FOLLOW_UPS.md:165`). The DS-EXC dispositions of the findings below are `REMEDIATION_REQUIRED_BEFORE_FREEZE` with `DEFER_ALLOWED = no`, so every open part is an Operator question. RW-PLAN v21 has **no A1 block** for this ticket: the write set in plan section 5 is a proposed block and needs the RW-PLAN amendment of Q-AB-01. Draft of 2026-11-18, **revision 2** after the independent review `REVIEW_DRAFT_051N.md` (label: REVIEW_FINDINGS; every item answered in plan section 17), written against `origin/arch-freeze/rw-train` `4c49cf877e65ad12d2fd4215186691515412a368` and the PR #676 head `700754f5e88ed36bdab32194667510b9ce2ba581`; every write-set path is identical at the train head and at `origin/development` `d0466feba1583cd5fccaf4d7121a68f68c13fbf2`.

Findings addressed, by ID and title only (labels quoted with their source: DS-EXC, the PDC registers, RW-PLAN):

- EX-D1-01 attestation not verified on the composed install/upgrade path (plan part A);
- EX-D1-02 revocation source and trust coordinator not composed in the production path (both launchers reach the composed sources through PR #676 Part C); EX-D1-04 a revoked revision without a trusted rollback target keeps serving; PDC-D1-R017 the lifecycle evidence sink has no production consumer (plan part B).

**EX-D1-02 stays open on the shipped GCA strict path after this ticket**: that path composes no capability runtime, so the trust admission is `NOT_COMPOSED` while the strict launcher still deploys and activates for an activation intent. Its route is Operator question Q-AB-07; the other strict entry points that accept a lab composition are Operator question Q-AB-06 (draft default for both: a before-freeze follow-up node `T-AF2-RW-SUPPLY-STRICT`).

Write set (exact lists in plan section 5): **5 firm product paths**, all firm rows of the signed T-AF2-RW-SUPPLY A1 block (`intergrax/agent_distribution/trust.py`, `intergrax/agent_distribution/admin_service.py`, `intergrax/applications/_shared/production_agent_capability_runtime.py`, `intergrax/applications/_shared/deploy_activate_pipeline.py` (state rename and docstring only), `applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py`); **4 firm non-product paths**: the three constructors of `ProductionAgentPlatformAdminConfig` found in the whole repository (signed 5b P1-P3: `testing_support/canonical_agent_lifecycle_composition.py`, `platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py`, `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`) and the new `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py`; firm total 9; **2 conditional non-product paths**, `tests/qualification/compat_x/_compat_x_event_evolution_inventory.py` and `tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py`, only under the gate-meaning ruling GM-AB-1 (variant E). None of the other signed 5b conditional paths is needed (plan section 5). ACT-N11: the 5 product paths count (prefixes `intergrax/`, `agents/`, `applications/`); the 6 non-product paths do not. Negative tests: **4 NT ids (NT-RW-SUP-01..04, RW-PLAN rows NT-ROW-075..078) in 16 nodes (R 10, S 2, C 4)**, with red-first at the pre-change commit and a revert-red step at the head; NT-RW-SUP-04a/04b exist only in variant E and are held out of the merge set in variant R; no expected-red test merges inside this ticket.

The required `package_trust` field leaves no caller unbuilt: the only three constructors are in the write set; the 17 support and test users of the canonical composition receive the lab composition through it; no script, product module or other test constructs the config (plan section 4).

Lane: after T-AF2-RW-SUPPLY (PR #676) merges; before T-AF2-RW-TENANT by default, with a reverse rule if the already-signed 051-A merges first (shared `admin_service.py`; TENANT also lists the phase-9 test as conditional); serialized with the already-signed T-AF2-RW-GATES on the proof file inside its firm glob (either order, the second rebases); no shared path with T-IGX-043-A, T-IGX-043-E part 1 or the other in-flight drafts. Merge order: 051-B → 051-N → 051-A (plan section 9).

Gate-meaning part: **GM-AB-1** (Operator, Q-AB-02): the COMPAT-X R3 inventory classifies the new production lifecycle-evidence consumer site and the pinned allowed-evidence set of the R3-P0 test gains its member. A refusal selects the pre-specified variant R and needs an explicit Operator DS-EXC re-disposition of PDC-D1-R017. Other Operator decisions: **Q-AB-04** (the reasons on which serving stops; `ACTIVE_TRUST_EVIDENCE_UNAVAILABLE` is broader than "revoked"), **Q-AB-06**, **Q-AB-07**, and the digest itself (S1 only; S2 never applies to a gate-meaning ticket). Not edited here: the EBH-6 typing registry (no write-set file is pinned or scanned), the TENANT-X carrier pin (already red at 832 against 831, owner RW-TENANT; this ticket adds no carrier and records the carrier and candidate counts), the EBH-7 F10 inputs, the COMPAT-X P7 fixture (its test is red at the train head without this ticket). Registered class: KR-08 (ACT-N11, `governance-fast`).

Contract surfaces: none edited; the wave-packet contract check (`.tmp/session/arch-freeze/closed-world/typ_measure.py`, `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py`) is run and must report nothing new (OD-AF-11).

Scope hold: no edit of `intergrax/`, `agents/` or `applications/` before the wave of this ticket and the delegate approval below. Commits carry the trailer `Ref: governance/tickets/T-IGX-051-N.md`.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan.

## Plan digest

Plan digest: c2c6c1382453e8efeb3d8e93d4a659ee015951bc172bd32cbc26a9774bde6597

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-N.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/supply-ab/PLAN_T-IGX-051-N_rev2.md`; the plan is copied unchanged into `governance/tickets/plans/PLAN_T-IGX-051-N.md` when the ticket is committed). The digest was computed with `python -I scripts/governance/plan_digest.py` (identical to `origin/development`) on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-19T17:37:29+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run when this ticket was drafted, except read-only checks of the dry run (`dry-run/DRY_RUN_T-IGX-051-N_rev2.md`). One process at a time, one `uv run pytest` per verification step with several paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it); exact files in plan sections 6 and 7:

- Start check (step 0): PR #676 merged; the Q-AB-02 answer recorded (else stop before Part B); every cited line re-read at the merged head; census S0-1..S0-4 (the 4 callers of the emergency response, trail metadata keys, launcher composition, COMPAT-X and TENANT-X baselines).
- Lint (step 0b): one mechanical ruff fix-and-format commit per touched file that fails at the head, before its first behavioural hunk.
- Red-first (S0): at the pre-change commit, the merged nodes that use only the existing API (03a, 03c, 04c) fail with an assertion inside the merged file; the other R and S nodes fail with an assertion in a non-merged probe that keeps the same node ids and assertion lines; the C nodes pass. Revert-red (S0b): at the head, reverting each product hunk turns its mapped nodes red with an assertion. Logs are kept for the auditor.
- After each part: the new test file plus `tests/unit/agent_distribution` and `tests/qualification/pdc_x/d1_repair`.
- Regression (S2): the set of plan section 7, in one command, at the pre-change commit and at the head; known reds compared by failed node-id set and message, with the COMPAT-X counter rule.
- Static (S3): `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on the changed Python files. Contract check (S4) and records (S5: COMPAT-X counters and id delta, TENANT-X carrier and candidate counts, PROD-Q text pins) as in plan section 7.
- Not run: `tests/qualification/prod_q` (PROD-Q needs its own authorization); its strict-startup text pins are checked by `git grep`.
- No broad or diagnostic suite, no Docker, no vendor integration, no real keys, revocation feeds or live backends (C10/C15 freeze); in-process fixtures are labelled as stubs in the PR body; key material is generated at run time and never echoed.
- The independent audit is a separate session at the exact wave head (node `PT-Q02`).

## Out of scope

- Parts C to I of T-IGX-051-B (PR #676), the held NT-RW-SUP-11 and 12a (T-IGX-051-K), the revocation-feed subscriber (EX-D1-03, NOT_COMPOSED, RW-GATES gate), attestation re-verification at revision admission and on the idempotent replay (Q-AB-10, residual RES-AB-5).
- The lab refusal at the other strict entry points and a trust admission for the shipped GCA strict path (Operator questions Q-AB-06, Q-AB-07; draft default: follow-up node `T-AF2-RW-SUPPLY-STRICT` before freeze); the lifecycle launchers, `strict_production_serving_launcher.py`, `production_process_lifecycle.py`, `strict_gca_production_process_composition.py`, `package_b_strict_production_serving_command.py` (PT-WB Part G, A4-040), `package_trust.py`, `package_attestation.py`, `emergency_revocation_response.py`, `activation.py`, `security_audit_trail.py`.
- Real publisher keys, revocation feeds or evidence backends (PROD-Q); the COMPAT-X P7 re-pin and every COMPAT-X file other than C-1 and C-1b; the EBH-6 typing registry, the EBH-7 pin and F10 inputs, the TENANT-X carrier pin; the IIAC re-pin (RB-2), FRZ status promotion, stage closure, `.github/`, `pyproject.toml`, `uv.lock`, gate scripts, the roadmap, the FRZ checklist, PDC-X and IIAC records, Docker or vendor work.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_eaf9ef72-d1c1-44d0-acbc-fac05f02edf7
