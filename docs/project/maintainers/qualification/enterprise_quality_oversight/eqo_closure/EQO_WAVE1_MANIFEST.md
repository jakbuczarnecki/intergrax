# EQO wave 1 — parallel execution manifest

**Generated:** 2026-11-21 (operator session)  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set (this wave):** only `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` — no product paths, FRZ, roadmap, `scripts/`, or `tests/`.

Pins at wave start (`EQO_PIN_BIND.json`, verified):

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Merge base (every `eqo/wave-1/*` branch)

| Ref | SHA | Role |
|-----|-----|------|
| `origin/eqo/closure-v2-phase-0-9` | `0310ce581139f73854477b03b5a1ba5a0921b3b0` | DES S1–S3, runbook, `runs/wb03`–`wb12`, W4 tranche 01, X reads |
| `origin/development` | merged on branch tip | TRAIN alignment + ticket context on `development` |
| Manifest branch tip (after merge) | `b9a8312903351e0e61fa38bd1b087503734bab10` | `eqo/wave-1/manifest` |

Parent worktree: `D:/Projekty/_wt/intergrax/`. Slugs `eqo-wv-*` (≤ 24 chars). Main checkout `D:/Projekty/intergrax` stays read-only during the wave.

---

## Five parallel groups

Hard rule: disjoint paths per group; no two agents edit the same file. DES doc work does **not** take the pytest lock. **AUD** and **EXE** stay in separate sessions (pasmo E).

| Group | Slug | Branch | WBS / objective | Write set (disjoint) |
|-------|------|--------|-----------------|----------------------|
| **G-W4** | `eqo-wv-w4` | `eqo/wave-1/w4` | `EQO-W4-02` tranches 02–04 | `eqo_closure/w4/W4-02_TRANCHE_02.md` … `_04.md` only — no finding matrix |
| **G-X** | `eqo-wv-x` | `eqo/wave-1/x` | `EQO-X-07` (N-0001/0002), refresh `EQO-X-04` / `EQO-X-05` reads on pins | `eqo_closure/x/**`, X memos — `EQO-X-04` read-only |
| **G-E3** | `eqo-wv-e3` | `eqo/wave-1/e3` | `EQO-E-03` operator packet draft | `eqo_closure/EQO_E03_OPERATOR_PACKET_DRAFT.md` only — no `register.jsonl` |
| **G-MANIFEST** | `eqo-wv-manifest` | `eqo/wave-1/manifest` | Wave SSOT + worktree bootstrap | this file; bootstrap merges only |
| **G-INT** | `eqo-wv-integ` | `eqo/wave-1/integ` | Post-push integrator (serial after G-W4/G-X/G-E3) | `EQO_MASTER_CLOSURE_MATRIX.md`, `README.md` — **one** steward commit after branches push |

**Prohibited in parallel groups this wave:** `EQO_A_REM_MASTER_FINDING_MATRIX.jsonl`, `enterprise_debt_register/register.jsonl`, product paths, FRZ/roadmap, concurrent edits under the same `runs/wbNN/`.

`EQO-W3-WB08-*` (MCP) is **not** in wave-1 G-EXE queue; schedule in a follow-on wave after WB03–12 DIS.

---

## G-EXE (serialized — not a parallel group)

| Item | Value |
|------|--------|
| Slug / path | `eqo-wv-exe` → `D:/Projekty/_wt/intergrax/eqo-wv-exe` |
| Branch | `eqo/wave-1/exe` |
| Pytest worktree | `eqo-w3` @ **TRAIN_PIN** `28c27cb896d02e0185b9fa02047411fa152f6469` |
| Lock | `D:/Projekty/_wt/_locks/pytest.lock` — **one** live `uv run pytest` process |
| Tasks | `EQO-W3-WBnn-EXE` → `EVD` → `DIS` for **WB03, WB04, WB06, WB07, WB09, WB10, WB11, WB12** |
| Command source | `eqo_closure/runs/wbNN/DES_PREREG.json` — verify SHA before run |
| Commits | one commit per WB after DIS; **no** matrix/README until last WB in queue |
| Also serialized here (later) | `EQO-C-04` shares the same lock when scheduled |

---

## Worktree registry (bootstrap)

| Slug | Path | Branch | Setup |
|------|------|--------|-------|
| `eqo-wv-manifest` | `D:/Projekty/_wt/intergrax/eqo-wv-manifest` | `eqo/wave-1/manifest` | `origin/eqo/closure-v2-phase-0-9` + merge `origin/development` |
| `eqo-wv-exe` | `D:/Projekty/_wt/intergrax/eqo-wv-exe` | `eqo/wave-1/exe` | same merge base |
| `eqo-wv-w4` | `D:/Projekty/_wt/intergrax/eqo-wv-w4` | `eqo/wave-1/w4` | same |
| `eqo-wv-x` | `D:/Projekty/_wt/intergrax/eqo-wv-x` | `eqo/wave-1/x` | same |
| `eqo-wv-e3` | `D:/Projekty/_wt/intergrax/eqo-wv-e3` | `eqo/wave-1/e3` | same |
| `eqo-wv-integ` | `D:/Projekty/_wt/intergrax/eqo-wv-integ` | `eqo/wave-1/integ` | same (integrator idle until parallel pushes land) |
| `eqo-w3` | `D:/Projekty/_wt/intergrax/eqo-w3` | detached @ TRAIN_PIN | pytest-only; shared by G-EXE |

Verify: `git -C D:/Projekty/intergrax worktree list | findstr eqo-wv`

---

## Merge ladder (after wave completes)

```text
eqo/wave-1/w4  ──┐
eqo/wave-1/x   ──┤
eqo/wave-1/e3  ──┼──► eqo/wave-1/integ ──► development (PR)
eqo/wave-1/exe ──┘         ▲
eqo/wave-1/manifest ───────┘ (manifest + bootstrap merge commits)
```

1. Each parallel group **push**es its branch when done.
2. **G-INT** merges `eqo/wave-1/{w4,x,e3,exe}`; resolves only integrator-owned files if needed.
3. Open PR: integrator → `development` (or operator merge policy).
4. Post-merge: queued **AUD** per WBS in **separate** sessions; update matrix once on integrator branch.

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.

---

## References

- `EQO_PARALLEL_ORCHESTRATION.md` (DES S1–S3 ladder)
- `EQO_EXECUTION_AGENT_INSTRUCTION.md`
- `EQO_MASTER_COMPLETION_WBS.json`, `EQO_MASTER_DEPENDENCY_DAG.json`