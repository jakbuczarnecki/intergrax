# T-IGX-051-A — T-AF2-RW-TENANT: tenant binding (sandbox sessions, object keys, tool context, artifact read, observation tenant, install CAS, MCP tenant, unscoped memory)

## Scope

Product-path ticket of the RW train (master-plan node `T-AF2-RW-TENANT`, lane 7; branch `arch-freeze/rw-train` only; DP-11 single integrated wave). Binding scope: `RW_PLAN.md` **v18** Appendix A1 block `T-AF2-RW-TENANT (T-IGX-051-A)`, its negative-test rows NT-ROW-065..074, its regression row and its lane and overlaps. The ticket closes nothing by itself; closure conditions are in plan section 11. Draft revision 2 of 2026-11-17 (revision 1: plan digest `77e5d3db0922e3bed45a1821a0a7a6de2161c60119f1e68992af130e6ffc32d2`, written against v13, 14 / 22 / 0; revised after the orchestrator's step-0 dry run at the train head `881428fb231f34215a67364aa5788a4f2f986e83`, record `dry-run/DRY_RUN_T-IGX-051-A.md`, and after the RW-PLAN v14 ruling Q-TEN-01 that amended A1). Evidence is read at `origin/arch-freeze/rw-train` `05699b633cdb1318c215f6529e449420efc1382c` and at the dry-run head; none of the 74 v18 paths differs between them, and `harness_host_runtime.py` still differs from `origin/development` (PT-WA).

Serialization: the ticket starts only after PT-WA and PT-WB and after PT-WF, PT-FC1, T-AF2-RW-SUPPLY, PT-FC3, PT-FC2 and T-AF2-RW-TRUST have merged into `arch-freeze/rw-train` (RW-PLAN section c: shared paths `harness_host_runtime.py`, `policy_wiring.py`, `admin_service.py`, `environment_wiring.py`, `memory_wiring.py`, `memory_provider_admission.py` and the `tests/unit/rag/*` glob). It runs in parallel only with T-AF2-RW-TYP; PT-T01 starts after it. Every shared file is read at the train head at that time, not at the draft base.

Findings addressed, by ID and title only (labels quoted with their source: DS-EXC, the PDC exception and debt registers, the RW-PLAN):

- EX-D4-04 hosted sandbox session opened under the fixed labels `harness`/`bootstrap` (plan part A);
- EX-D8-03 `ToolWiringContext` has no tenant and the raw `storage.*` tools are not tenant-bound; EX-L-02 object keys and audit of raw `storage.*` and `workspace.*` tools (part B);
- EX-D8-04 the governed artifact read capability is tenant-wide, with no run or principal binding (part C);
- EX-D5-05 (PDC-D5-R015) `tenant_id` absent from the base observation model, strict host without a tenant (part D);
- EX-L-09, the G-12 part: `install_agent` into an occupied slot (part E);
- R-P8-01 an MCP host without effective harness auth trusts a client-asserted tenant or user (part F);
- F-D2-11, tenant part: reindex queue hooks are global and tenant-blind (part G);
- EX-D2-06 (PDC-D2-R002): STRICT forbids unscoped memory, an Operator decision, class C (part H);
- the TENANT-X carrier re-pin that the new tenant fields require (part I, request GAP-10, found by the dry run as DR-TEN-02).

Related items named in the RW-PLAN row but owned elsewhere, not edited here: FRZ-TEN-10 (moved to PT-WF Part JA by dedupe D-04), FRZ-TEN-01 and the G-09 part of EX-L-09 (PT-WF, DP-23), FRZ-CFG-02, FRZ-CFG-06 and FRZ-SEC-04 (owner PT-FC3). This ticket promotes no FRZ row.

Write set (exact lists in the plan, section 5; equal to Appendix A1 v18, checked path by path by script): **19 firm product paths**, **13 firm test paths** (8 new, 5 edited), **9 conditional product paths** and **33 conditional test paths**; firm total 32, conditional total 42; 9 negative-test ids (NT-RW-TEN-01..09) in 45 test nodes. The conditional paths are edited only under the rulings and census results named in the plan (Q-TEN-02 option 1, Q-TEN-03 option B, a cross-process compare, the backend contract, and tests that the step-0 census shows turning red). No docs, no scripts. Requests that A1 v18 does not satisfy and that are not part of the write set until RW-PLAN is amended and the digest recomputed: GAP-10 (`tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py`, the carrier pin; proposed as a firm path of this ticket, Operator-visible), GAP-11 (the application MCP facades that would pass `fails_closed`) and GAP-12 (PT-WD impact-map paths; not this ticket's scope). `policy_wiring.py` and `harness_host_runtime.py` are firm paths whose expected edit is none after PT-WF and PT-FC1. The first commit of the ticket is a behaviour-free lint commit of the staged files (ruff 0.15.20, plan section 8, group L0). ACT-N11: every product path above breaks the IIAC activation until the single RB-2 re-pin; this ticket does not perform it and must not merge outside the wave.

Contract surfaces: the write set has no path under `intergrax/contracts/`; the wave-packet contract check (`.tmp/session/arch-freeze/closed-world/typ_measure.py` and `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py`) is expected to report nothing new and must report no new `Any`, `object` or reflection metric and no new string-keyed lookup table if an amendment ever adds a `contracts/` path (OD-AF-11).

Implementer session field: left empty by the drafter; the implementer session id is written before the approval commit, because the strict validator rejects an empty value once a delegate approval block exists.

Scope hold: no edit of `intergrax/`, `agents/` or `applications/` before the wave of this ticket and the delegate approval below. Commits carry the trailer `Ref: governance/tickets/T-IGX-051-A.md`.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan.

## Plan digest

Plan digest: 295addbec8e2055707aba46eab83e385375b3091c6a235ec583a082d5f55940e

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-A.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/rw-tenant/PLAN_T-IGX-051-A.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with a copy of `scripts/governance/plan_digest.py` from `origin/development` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval block is written later by the delegate after plan review and the wave packet; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with several paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it); exact files in plan sections 6 and 7:

- Red-first (S0): the eight new test files (marker `unit`, no `gate` on the threaded and host-composing files) on a detached worktree at the pre-change commit; every R and S node fails by an assertion (not an import or collection error), a test of a new API first asserts the old surface; the control nodes (C) and the two conditional nodes (`test_ten_06f`, `test_ten_08a`) are recorded and, if green at the base, re-dispositioned by the RW-PLAN rule; `test_ten_04d` is shared with PT-WF and may already be green at the base; the log is kept for the auditor.
- After each part: the new test file of the part plus its regression directories; at the end the RW-PLAN regression row (`tests/qualification/tenant_x`, `tests/unit/runtime/sandbox`, `tests/unit/tools/providers`, `tests/unit/agent_distribution`, `tests/qualification/mcp_server_x`, `tests/unit/applications/test_mcp_identity_mode_fail_closed.py`, `tests/unit/rag/governance`) and the additional files named in plan section 7.
- Gates: the contract check above (run once at the ticket head); `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on the changed Python files; the five ci-smoke df5 nodes of `test_diag_foundation_5_destructive_proof.py` (regression); the three TENANT-X pin nodes (`test_tx_p0_r3_head_q02_class_field_carrier_count_stable`, `test_real_data_has_no_violation`, `test_gate_helper_raises_on_a_wrong_digest`), green only after the Part I re-pin.
- No broad or diagnostic suite, no Docker, no vendor integration, no real credentials, object stores or sandbox providers (C10/C15 freeze); in-process fakes are labelled as stubs in the PR body. Audit events and test output carry no key bodies, object bodies or secrets.
- The independent audit is a separate session at the exact wave head (node `PT-Q02`).

## Out of scope

- The requests GAP-10 to GAP-12 of plan section 4 until an amendment adds them; the conditional paths of A1 v18 unless their condition holds (`admin_models.py`, the store files, `online_evaluation.py`, `nexus_loop.py`, the eval tool, `online_evaluation_registry.py`, the `SandboxHostBackend` contract, tests the census does not turn red); the application MCP facades `applications/*/mcp/server.py`; the PT-WD impact-map files.
- FRZ-TEN-10 and the G-09 untenanted evaluation registry (PT-WF Part JA), `production_lifecycle.py`, `evaluation_wiring.py` (PT-WF), `retention.py` and the registry store files (T-AF2-RW-SUPPLY), anything under `intergrax/contracts/`.
- Tenant isolation inside a real sandbox, object store or revocation backend (PROD-Q); migration of objects written before the key change; the B-1 check X-A8-022 and the independent audit that R-P8-01 waits for.
- The IIAC re-pin (RB-2), the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work.

## Lifecycle state

plan_pending

## Implementer session

Implementer session:
