# EQO parallel orchestration — worktrees, groups, merge ladder

**Generated:** 2026-11-21 (operator session)  
**Baseline branch:** `eqo/closure-v2-phase-0-9` @ `4af9a00a0abe164c9ff07b38b1251704dfbba311`  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Analysis source:** EQO-CLOSURE v2 plan §3–4, `EQO_MASTER_COMPLETION_WBS.json`, `EQO_MASTER_DEPENDENCY_DAG.json`, `EQO_EXECUTION_AGENT_INSTRUCTION.md`, parallelization review (post-WB02).

Pins at orchestration start (`EQO_PIN_BIND.json`, verified):

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

Write set for parallel band: **only** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` (plus `enterprise_debt_register/reconciliation.json` when E-01 steward runs). **No** product code, FRZ, or roadmap edits. Pasms D (#680) **out of scope** for these worktrees.

---

## 1. Worktree layout

Parent: `D:/Projekty/_wt/intergrax/` (per `AGENTS.md`). Each slug ≤ 24 chars.

| Slug | Path | Branch | Role |
|------|------|--------|------|
| `eqo-par-s1` | `D:/Projekty/_wt/intergrax/eqo-par-s1` | `eqo/par/s1-wb030408` | Group S1 — W3 DES (doc) WB03/04/06/08 |
| `eqo-par-s2` | `D:/Projekty/_wt/intergrax/eqo-par-s2` | `eqo/par/s2-wb0712` | Group S2 — W3 DES (doc) WB07/09/10/11/12 |
| `eqo-par-s3` | `D:/Projekty/_wt/intergrax/eqo-par-s3` | `eqo/par/s3-w4x` | Group S3 — X-04, X-05, W4-02 tranche |
| `eqo-par-integ` | `D:/Projekty/_wt/intergrax/eqo-par-integ` | `eqo/par/integ` | Optional integrator — merge par branches, matrix/README |
| `eqo-par-plan` | `D:/Projekty/_wt/intergrax/eqo-par-plan` | `eqo/par/plan-doc` | Runbook and orchestration docs only |

**Band-C pytest worktree (EXE, not parallel doc agents):** `D:/Projekty/_wt/intergrax/eqo-w3` @ TRAIN_PIN — shared by all `*-EXE` and `EQO-C-04` after DES complete.

**Main checkout** `D:/Projekty/intergrax`: read-only for fetch; agents do not commit there during parallel phase.

---

## 2. Parallel groups (phase: post-WB02 DES wave)

Hard rule inside each WB: `DES → EXE → EVD → DIS → AUD → CLS` (DAG). Parallelism is **across** workstreams for DES and **across** agents for disjoint directories; EXE is **serialized** on pytest lock.

### Group S1 — `eqo/par/s1-wb030408`

| Task IDs (this phase) | Domain / workstream | Write set |
|----------------------|---------------------|-----------|
| `EQO-W3-WB03-DES` | Fail-closed, HITL — WB03 | `eqo_closure/runs/wb03/DES_PREREG.json`, scope notes |
| `EQO-W3-WB04-DES` | Security — WB04 | `eqo_closure/runs/wb04/` |
| `EQO-W3-WB06-DES` | Configuration — WB06 | `eqo_closure/runs/wb06/` |
| `EQO-W3-WB08-DES` | MCP — WB08 | `eqo_closure/runs/wb08/` |

**Prohibited in S1 this wave:** `EQO_MASTER_CLOSURE_MATRIX.md`, `README.md`, `EQO_A_REM_MASTER_FINDING_MATRIX.jsonl`, any `*-EXE` (no pytest in S1 worktree).

### Group S2 — `eqo/par/s2-wb0712`

| Task IDs | Workstream | Write set |
|----------|------------|-----------|
| `EQO-W3-WB07-DES` | WB07 | `eqo_closure/runs/wb07/` |
| `EQO-W3-WB09-DES` | WB09 | `eqo_closure/runs/wb09/` |
| `EQO-W3-WB10-DES` | WB10 | `eqo_closure/runs/wb10/` |
| `EQO-W3-WB11-DES` | WB11 | `eqo_closure/runs/wb11/` |
| `EQO-W3-WB12-DES` | WB12 | `eqo_closure/runs/wb12/` |

WB08 DES is owned by **S1** only (no duplicate).

### Group S3 — `eqo/par/s3-w4x`

| Task IDs | Eligibility | Write set |
|----------|-------------|-----------|
| `EQO-X-04` | `EXECUTABLE_NOW` — ROADMAP-REPLAY counter at both pins (read-only) | memo under `eqo_closure/` |
| `EQO-X-05` | `EXECUTABLE_NOW` — TENANT-X read | memo under `eqo_closure/` |
| `EQO-W4-02` (tranche 01) | `EXECUTABLE_IN_PARALLEL` with W3 | `eqo_closure/w4/W4-02_TRANCHE_01.md` only — **no** finding matrix edit |

### Integrator — `eqo/par/integ` (optional)

After S1–S3 push:

- Merge `eqo/par/s1-wb030408`, `s2-wb0712`, `s3-w4x` into integration branch.
- Single steward updates `EQO_MASTER_CLOSURE_MATRIX.md`, `README.md` phase index, shared rows — **one commit** per integration phase.

### Out of parallel groups (single-writer / serial)

| Item | Rule |
|------|------|
| All `EQO-W3-WBnn-EXE` | One holder of pytest lock; queue WB03 → WB04 → … after DES |
| `EQO-C-04` | Band C + pytest — same lock as W3 EXE |
| `EQO-E-02` apply | One steward on `register.jsonl` |
| `EQO-W3-WBnn-AUD` | Session E only — not same session as EXE executor |
| PR #680 / pasms D | Separate train tickets — EQO does not implement |

---

## 3. Pytest lock (EXE holder)

Only **one** live `uv run pytest` (or full EXE contract) across all EQO band-C worktrees.

- **Lock path:** `D:/Projekty/_wt/_locks/pytest.lock` (directory lock: create exclusive, remove on release).
- **DES / DIS / EVD documentation** does **not** require the lock.
- **Parallel doc agents (S1–S3)** must **not** run pytest EXE in this wave.
- Evidence pattern: see `eqo_closure/runs/wb02/TENANT_ISOLATION_EVIDENCE.md`.

Test contract for qualification commits (when EXE runs): one process at a time; logs under `.tmp/session/` in the active worktree; `Ref: governance/tickets/T-IGX-046-F.md`.

---

## 4. Merge ladder

```text
eqo/par/s1-wb030408  ──┐
eqo/par/s2-wb0712    ──┼──► eqo/closure-v2-phase-0-9  ──► development  (PR #686)
eqo/par/s3-w4x       ──┘         ▲
eqo/par/plan-doc     ─────────────┘ (runbook; merge before or with integrator)
         optional: eqo/par/integ pre-merges par branches locally, then fast-forward integrator → closure branch
```

**Recommended order**

1. Merge `eqo/par/plan-doc` (this runbook) into `eqo/closure-v2-phase-0-9`.
2. When S1–S3 DES commits are pushed, merge each par branch into `eqo/closure-v2-phase-0-9` (or use `eqo/par/integ` to resolve `runs/wb**/` conflicts — disjoint paths should merge cleanly).
3. Integrator commit on `eqo/closure-v2-phase-0-9`: matrix + README only.
4. Open or update **PR #686** — `eqo/closure-v2-phase-0-9` → `development`.
5. Post-merge on `development`: queued EXE on `eqo-w3` @ TRAIN_PIN per WBS order.

**Conflict policy:** If two branches touch the same file, **stop** and route through `eqo-par-integ`; never force parallel edits to matrix or finding register.

---

## 5. Prohibited parallelisms

1. Two EXE pytest processes without the shared lock.
2. Two agents committing the same file (matrix, README, `register.jsonl`, same `runs/wbNN/*`).
3. AUD and EXE in the same agent session (pasmo E).
4. WB AUD vs WB EXE by the same human on the same workstream.
5. Product / FRZ / roadmap edits under `T-IGX-046-F` scope.

---

## 6. Operator next steps

1. Confirm all five worktrees exist and sit at closure branch tip (`git worktree list` from main checkout).
2. Let S1–S3 subagents finish DES commits and **push** their `eqo/par/*` branches.
3. Parent or integrator: merge par branches → `eqo/closure-v2-phase-0-9`; update matrix once.
4. Schedule EXE sequence on `eqo-w3` with lock holder (tester role).
5. Keep `EQO_PIN_BIND.json` current; if HEAD moves, run `EQO-RV-01` / `EQO-W0-01` before next EXE wave.

---

## 7. References

- `EQO_PARALLEL_ORCHESTRATION.md` (this file) — `eqo/par/plan-doc`
- `EQO_EXECUTION_AGENT_INSTRUCTION.md`
- `EQO_MASTER_COMPLETION_WBS.json`, `EQO_MASTER_DEPENDENCY_DAG.json`
- `docs/project/technical/devos/PARALLEL-EXECUTION.md`
- Master plan: `eqo_closure_master_7499b85e.plan.md` (Cursor plans store)