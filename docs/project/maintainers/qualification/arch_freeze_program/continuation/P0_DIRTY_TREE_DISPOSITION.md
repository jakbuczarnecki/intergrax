# P0 — main checkout dirty tree disposition

**Packet:** ARCH-FREEZE closure P0 (fc-train bootstrap). **Not** train application; **not** FCS.

| Field | Value |
| --- | --- |
| Read pin (published tree) | `origin/development` @ `11787a7e196cbfea32a982bfa260f02a34fe05fc` (**INTEGRATION_HEAD**; not G2/FCS/M3) |
| Train reference pin | `origin/arch-freeze/rw-train` @ `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a` (ancestor of INTEGRATION_HEAD) |
| Main checkout action | **read-only** — no commit, stash, reset, or clean |
| Artifact export (outside repo) | `D:/Projekty/_wt/intergrax/_scratch/af-fc-p0/main_status_porcelain.txt`, `main_diff.patch`, `main_diff_stat.txt` |
| Worktree train branch | `arch-freeze/fc-train` @ `D:/Projekty/_wt/intergrax/af-fc-p0` |
| Date | 2026-11-25 |

## Summary counts

| Class | Count | Meaning |
| --- | ---: | --- |
| **Total dirty paths** | 165 | `git status --porcelain` on main checkout @ export time |
| **ARCH-FREEZE unpublished (local)** | 65 | In-scope W3/RW/FC/gate product or ticket/docs work; stays local until reviewed and cherry-picked onto `arch-freeze/fc-train` |
| **Foreign local edit** | 100 | Other campaigns, housekeeping, or unrelated product; **not** applied on fc-train in P0 |

Path lists: `_scratch/af-fc-p0/arch_freeze_paths.txt`, `_scratch/af-fc-p0/foreign_paths.txt` (UTF-8).

## Classification rules (P0)

1. **ARCH-FREEZE unpublished** — paths in the RW/W3 write envelope (`intergrax/applications/_shared/**`, selected `intergrax/runtime/**`, RW maintenance/governance gates, ARCH-FREEZE qualification tests, tickets `T-IGX-043*` / `046*` / `051*` / `065-A`, related plans, seam/policy/EBH-7 matrix touches tied to freeze gates).
2. **Foreign local edit** — EQO / ENT-AUDIT-X / EVOL-X / SFX, unrelated ticket IDs, `agent-env/plans` paste cleanup, token benchmarks, `scripts/demo*` removal, general CI/docs outside freeze train scope, application hosts not in the W3 batch contract.

## ARCH-FREEZE unpublished — disposition

**Action:** retain on main checkout only. Orchestrator P2+ may review and port slices onto `arch-freeze/fc-train`. P0 does **not** merge dirty tree into fc-train.

Dominant buckets among the 65 paths:

| Bucket | Examples (count indicative) |
| --- | --- |
| Harness / wiring product | `intergrax/applications/_shared/*_wiring.py`, `harness_host_runtime.py` (~11) |
| Qualification / gate tests | `tests/unit/applications/test_fc2_*`, `test_rw_*`, `test_*_fail_closed`, architecture typ/seam/closed-world (~22) |
| Maintenance gates | `scripts/maintenance/check_*_wiring.py`, `check_frozen_contract_typ_ratchet.py` (~5) |
| Governance hooks | `scripts/hooks/igx_stop_guard.py`, `igx_subagent_guard.py` |
| Freeze traceability | `scripts/governance/verify_arch_freeze_plan_traceability.py` |
| Tickets / plans (RW family) | `T-IGX-043-*`, `046-*`, `051-*`, `065-A`, `PLAN_T-IGX-043-*` |
| Registry / matrix | `governance/architecture/seam_ledger.yaml`, `policy-registry.yaml`, EBH-5 matrix, `closed_world.py` |

## Foreign local edit — disposition

**Action:** no train carry in P0; operator/local sessions own merge or discard.

| Bucket | Representative paths |
| --- | --- |
| EQO closure sessions | `enterprise_quality_oversight/eqo_closure/**` (deletions + jsonl edits) |
| ENT / EVOL / SFX docs | `ent_audit_x/**`, `evol_x/**`, `scenario_factory_x/**` |
| Unrelated tickets | `T-IGX-019`..`036`, `055`, `070`, `072`, `074`, deleted `071-A`, `900-A`, etc. |
| Agent-env / demo cleanup | `agent-env/plans/*`, `scripts/demo*`, `tests/unit/demo*` |
| Token benchmarks | `benchmarks/token_optimization/fixtures/v1/**` (deletions) |
| Application / CI outside W3 batch | `applications/research_application/**`, `.github/workflows/unit-tests.yml`, `intergrax/platform/enterprise_audit/**` |

## Blockers

None for P0 documentation commit. Dirty tree itself is **not** a publication blocker; it is inventoried and quarantined by classification.

## Evidence index

| Artifact | Location |
| --- | --- |
| Porcelain | `D:/Projekty/_wt/intergrax/_scratch/af-fc-p0/main_status_porcelain.txt` |
| Unified diff | `D:/Projekty/_wt/intergrax/_scratch/af-fc-p0/main_diff.patch` |
| Diff stat | `D:/Projekty/_wt/intergrax/_scratch/af-fc-p0/main_diff_stat.txt` |