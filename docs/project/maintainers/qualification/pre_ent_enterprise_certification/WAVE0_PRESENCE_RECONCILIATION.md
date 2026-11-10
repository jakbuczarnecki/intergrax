# PRE-ENT Wave 0 — presence reconciliation

**Evidence HEAD:** `4a932d0cf5487b72e433805e372e3855fcd3d987`  
**Branch:** `development`  
**Date:** 2026-11-10  
**Executor:** T-IGX-034-A (orchestrator/implementer Wave 0 only — no PASS, no closure)

**P8 integration ancestry:** `427646f41de0df992cd54c0e157de8f36b560f39` is an ancestor of HEAD (`merge-base --is-ancestor` exit 0). Historical product batch C1–C4 commits are in the current tree; ticket lifecycle files for 030-B/C, 032-A, 033-A were not advanced past `plan_approved` after merge.

## Presence matrix

| Item | On HEAD? | Ticket lifecycle (governance) | Wave 1 action |
|------|----------|-------------------------------|---------------|
| **T-IGX-030-B** — `RuntimeErrorCode` contract home | **Yes** — `intergrax/contracts/runtime_error_code.py` @ `31a6c3fe1`; tests `test_runtime_error_code_contract_home.py`; spot pytest EBH-4 R1 gate **8 passed, exit 0** @ HEAD | `plan_approved` (lags git) | **qual-only** → Wave 2 full contract + hist ticket lifecycle hygiene; **no implement** |
| **T-IGX-030-C** — `pool.submit` terminalization (no dep boundary) | **Yes** — `invoker.py:1915-1919` matches plan; test `test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` @ `30f67c94a` | `plan_approved` (lags git) | **qual-only**; **no implement** |
| **T-IGX-032-A** — KV precedence in pinning wiring | **Yes** — `skill_execution_pinning_wiring.py:36-39` @ `b6a056e7d`; unit test `test_skill_execution_pinning_wiring_store_precedence.py` | `plan_approved` (lags git) | **qual-only** (incl. P2-C qual module in Wave 2); **no implement** |
| **T-IGX-033-A** — CLI Tier-0 upward edges (BND-03 static) | **Yes** — `external_work.py` is receipt-only; gate `test_bnd03_tier0_upward_edges_gate.py` @ `8bfee643f` | `plan_approved` (lags git) | **qual-only**; **no implement** for static CLI scope |
| **HIST-F-BND-03B** — `product_manifest_registry.py` dynamic Tier-0→Tier-3 | **Yes (defect present)** — `importlib.import_module(module_name)` on Tier-3 manifest modules (`lines 15-31`) | Outside 033-A scope (explicit out of scope) | **implement** — **T-IGX-034-B** worktree `bnd-03b` |
| **HIST-F-DUR-01** — pinning durability semantics | **Partial** — validator is correct (`execution_pinning_durability.py`); `KvSkillExecutionPinningStore` / `DocumentStoreSkillExecutionPinningStore` return literal `is_durable=True` (`execution_pinning_persistence.py:291-292, 387-389`); in-memory fallback reports `False` | Plan residual R1 in T-IGX-032-A | **implement** (if qual confirms gap) — **T-IGX-034-C** worktree `skill-dur`, serial after pinning qual |
| **HIST03-F-015** — P2-C qualification path | **Yes** (test path) — `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py`; product fix via 032-A | Owned by 032-A | **qual-only** in Wave 2; tied to 032-A, not new code in Wave 1 |
| **T-IGX-031-A** — F-006 / F-016 tests | **Yes** (closed batch) | `closed` | **none** |

## Read-only pytest (030-B spot check)

`tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py` @ HEAD `4a932d0cf` — **8 passed, exit 0**. Confirms T-IGX-030-B presence on HEAD; not a substitute for Wave 2 serial qualification or independent audit.

## Lifecycle vs git (summary)

| Ticket | Lifecycle in file | Git reality on HEAD |
|--------|-------------------|---------------------|
| T-IGX-030-A | `closed` | Aligned (tests-only) |
| T-IGX-031-A | `closed` | Aligned |
| T-IGX-030-B/C, 032-A, 033-A | `plan_approved` | Product commits merged (see short SHAs above); **reconcile** to `quality_gate` / `ready_for_audit` after Wave 2 qual, not before |
| T-IGX-034-A | `quality_gate` | Wave 0 deliverables complete @ HEAD `4a932d0cf` |

## Operator blockers (unchanged by Wave 0)

| Blocker | Affects |
|---------|---------|
| **G-SKILL** | SKILL-X full §3.0.4 scope (upgrade/disable/remove) — operator must choose G-SKILL-A or G-SKILL-B before autonomous impl |
| **OD-HIST04-A** | EVOL-X formal ratification |
| **OD-HIST04-C** | TOOL-LIFE-X label vs open operator decision |
| **OD-HIST04-B** | IIAC material paths — no edits in campaign |
| **ACT-N11 / IIAC** | Product paths on HEAD may require current IIAC activation evidence for promotion (not re-litigated in Wave 0) |

## Wave 1 parallel batch routing (from plan WBS)

| Batch | Owner ticket | Condition after Wave 0 |
|-------|--------------|----------------------|
| BND-03B manifest edge | T-IGX-034-B | **required** — code still missing |
| DUR-01 semantics | T-IGX-034-C | **conditional** — confirm via qual |
| Invoker 030-C / H01-F015 | T-IGX-034-D | **skip implement** — already on HEAD; qual-only |
| REPLAY gates | T-IGX-034-E | Wave 1 per plan (not presence-checked in depth here) |
| EBH-7 F10 clean SHA | T-IGX-034-F | Wave 1 |
| QUAL-X B–D | T-IGX-034-G | Wave 1 |
| DIST-SCALE Q-03 | T-IGX-034-H | Wave 1 + operator WL-7 note |
| Matrix publication | T-IGX-034-I | Wave 4+ |

## 030-C / BND-03B — code still needed?

| Scope | Still needs product code on HEAD? |
|-------|-----------------------------------|
| **T-IGX-030-C** | **No** — terminalization present |
| **HIST-F-BND-03B** | **Yes** — dynamic manifest import remains in Tier-0 |
