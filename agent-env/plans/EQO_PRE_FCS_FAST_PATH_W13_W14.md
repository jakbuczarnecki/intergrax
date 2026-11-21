# EQO pre-FCS fast path — integration waves 13–14

**Ticket:** `governance/tickets/T-IGX-046-F.md`  
**Baseline:** `origin/development` post waves 1–12 (~`94f4c6cf`); **pins:** DOC_PIN `8cee3c5a`, TRAIN_PIN `28c27cb`  
**Constraints:** docs write set in `eqo_closure/` only unless authorized train EXE; no `EQO_CLOSED`; no Session E self-PASS; merge on `ticket-governance` + `governance-policy-pin` only; parallel worktrees + single pytest lock.

## 1. Minimum remaining integration waves

| Wave | Theme | Rationale |
|------|-------|-----------|
| **13** | Remaining W3 EXE @ TRAIN_PIN + doc-only disposition lanes | Closes the only executable pytest gap before honest CLS refresh (WB02/03/05/08/12 not in wave-6/10 bundles). |
| **14** | W3 terminal CLS band + matrix/W9-01 prep + ticket checkpoint | Depends on wave-13 logs; mirrors wave-5 (CLS) + wave-12 (matrix) without repeating COR/DRV (wave 9 done). |

**Do not** fold W7/W8 into pre-FCS waves (`EQO_W7_W8_DEFERRED_BLOCKERS.md`). **Do not** add wave 15 for W9 terminal — that waits post-FCS.

### Wave 13 parallel lanes

| Lane | Branch pattern | Worktree | Deliverable |
|------|----------------|----------|-------------|
| **G-EXE-A** | `eqo/wave-13/exe-wb02`, `exe-wb03`, `exe-wb05` | `eqo-w13-exe-02`… | `pytest_wbNN_exe.log`, `RUN_MANIFEST.json` |
| **G-EXE-B** | `eqo/wave-13/exe-wb08`, `exe-wb12` | `eqo-w13-exe-08`… | same (WB08 first-time EXE) |
| **G-E** | `eqo/wave-13/e` | `eqo-w13-e` | `x/N-0001|N-0002/EDER_E03_ROUTE.md` refresh; **no** `reconciliation.json` / register apply (E-02 report-only) |
| **G-C04** | `eqo/wave-13/c04` | `eqo-w13-c04` | `runs/c/C04_STATUS.md` doc cross-ref to `C04_BLOCKED_RECORD.md`; no new PASS claim |
| **G-W4** | `eqo/wave-13/w4` | `eqo-w13-w4` | `w4/W4-03_OWNER_BATCH_ENGLISH_MEMO.md` + matrix note `WAIT_FOR_OPERATOR` (B-2 registry already 186/186 @ wave-11) |
| **G-INT** | `eqo/wave-13/integ` | `eqo-w13-integ` | `EQO_WAVE13_MANIFEST.md`, README row 39, matrix EXE column refresh |

**Pytest:** `eqo-w3` @ TRAIN_PIN, `uv sync --all-extras`, lock `D:/Projekty/_wt/_locks/pytest.lock` (same as wave 10).

### Wave 14 parallel lanes

| Lane | Branch pattern | Worktree | Deliverable |
|------|----------------|----------|-------------|
| **G-CLS-A** | `eqo/wave-14/cls-wb02`…`wb05`, `wb12` | `eqo-w14-cls-NN` | `runs/wbNN/W3_CLS.md` **terminal disposition** (non-PASS language) |
| **G-CLS-B** | `eqo/wave-14/cls-wb08` + remaining refresh 01,04,06,07,09–11 | `eqo-w14-cls-NN` | include first WB08 `W3_CLS.md` |
| **G-DRV** | `eqo/wave-14/drv` | `eqo-w14-drv` | `runs/wbNN/COR_DRV.md` **DRV section only** where doc-only `DOC_PATCH` / `OWNER_DECISION_RECORD` routes remain open (no product DRV replay) |
| **G-W9** | `eqo/wave-14/w9` | `eqo-w14-w9` | C01–C25 matrix sweep after CLS; still **no** C row `FULLY_MET` |
| **G-INT** | `eqo/wave-14/integ` | `eqo-w14-integ` | `EQO_WAVE14_MANIFEST.md`, README row 40, `T-IGX-046-F.md` checkpoint |

## 2. Task batching (WBS alignment)

| Batch | Tasks | Wave | Notes |
|-------|-------|------|-------|
| **W3 terminal CLS** | `EQO-W3-WBNN-CLS` (×12) | 14 | Compose from `AUD_SESSION_E` + latest EXE + `COR_DRV`; aggregate remains 0 PASS @ pin |
| **Remaining EXE** | WB02,03,05,08,12 `-EXE` → `-EVD` | 13 | WB06 `0` @ wave-10 — EVD refresh optional, not blocking others |
| **DRV doc-only** | `EQO-W3-WBNN-DRV` | 14 | Satisfy WBS with documented deferral to train/operator; no pytest DRV |
| **B-2 / W4-03** | `EQO-W4-03`, W4-CLS follow-up | 13 (doc) | Registry complete; operator packet `WAIT_FOR_OPERATOR` — memo only |
| **EDER no-apply** | `EQO-E-03`…`E-05`, route memos | 13 | E-01 done; E-02 triage exists; **block** register steward PR |
| **C-04 doc** | `EQO-C-04` | 13 | Keep `BLOCKED`; cite existing `c04_targeted_replay.log` exit 1 |

**Out of wave 13–14 scope:** `EQO-W7-*`, `EQO-W8-*`, `EQO-W9-02`, product CL-01, AUD session E re-verdict, `EQO-E-CLS` (needs operator/register).

## 3. Critical path vs parallelizable

```text
[CRITICAL — serial in practice]
  WB08 DES (if missing) → WB08 EXE → WB08 W3_CLS
  pytest lock: EXE lanes queue one process at a time

[CRITICAL — logical]
  Wave-13 EXE merge → Wave-14 CLS lanes (read fresh logs)
  W4-03 terminal W4-CLS row → operator OD (cannot fake in EQO)

[PARALLEL]
  Wave-13: G-E + G-C04 + G-W4 while EXE runs (separate worktrees)
  Wave-14: 12 CLS worktrees (docs only, merge on integ)
  Wave-14: G-W9 matrix edits parallel to CLS batches
```

## 4. Cannot be accelerated (honest blockers)

| Blocker | Owner | Why EQO cannot shorten |
|---------|-------|-------------------------|
| FCS / M3 / RB-2 | Merge train + operator | W7-01..04, W8 Campaign D, C09–C10, C24–C25 |
| #680 / T-IGX-043-A on TRAIN_PIN | Product train | WB01 EXE exit `1`; C16; W2-01 |
| IIAC `IN_FORCE` | Vendor/train | WB01/C19; not docs-fixable |
| Session E `PASS` (0/12 today) | Independent AUD chat | Implementer must not upgrade verdicts |
| C-04 targeted replay green | Product/fixture | Exit `1` logged; doc = BLOCKED only |
| EDER mechanical apply (E-02) | Steward PR / operator | Explicitly blocked in scope |
| W4-03 operator decisions | Operator | `WAIT_FOR_OPERATOR` per wave-11 |
| Product CL-01 routes in `COR_DRV` | Train tickets | Deferred since wave 9 |
| Platform CI green | Infra / pre-existing drift | EQO merges on docs gate pair only |
| `EQO_CLOSED` / G6 / FRZ | W8-CLS + W9-02 post-FCS | Per `EQO_EXECUTION_AGENT_INSTRUCTION.md` |

## 5. Suggested manifests + merge order

1. **Wave 13:** merge `exe-wb*` lanes → `G-E`, `G-C04`, `G-W4` → `eqo/wave-13/integ` → PR → `development`.  
2. **Wave 14:** merge all `cls-wb*` → `G-DRV` → `G-W9` → `eqo/wave-14/integ` → PR → `development`.  
3. **Order vs in-flight:** If any wave-8/11/12 integrator PR not yet @ baseline, merge or rebase **before** wave-13 branch (avoid matrix conflicts).  
4. **AUD:** Schedule separate Session E chats only when operator requests re-audit; not a merge gate for waves 13–14.

## 6. Time / risk tradeoffs

| Strategy | Upside | Risk |
|----------|--------|------|
| **Docs-fast (recommended)** | Maximizes WBS rows closed as `DISPOSITIONED_NOT_CLOSED`; fresh EXE evidence for C08 partial | Operators may misread CLS as “green”; mitigate with wave-5 aggregate language (0 PASS) |
| **Honest-blocker-first** | C01–C25 stay NOT_MET/PARTIALLY_MET; `EQO_NOT_CLOSED_EXACT_BLOCKERS` stable | Slower perceived progress; fewer rework cycles |
| **Skip WB08 EXE** | Saves one heavy slice | Violates pre-FCS instruction §3; leaves WB08 as only missing EXE log |
| **Fake DRV/C-04/E apply** | — | **Reject** — violates ticket write set and E-02 guard |

## Implementation steps (orchestrator)

1. Confirm `EQO_PIN_BIND.json` vs DOC/TRAIN_PIN; RV-01 if drift.  
2. Cut `eqo/wave-13/integ` from `origin/development` @ post-wave-12 HEAD.  
3. Run EXE lanes; record non-zero exits without closure claims.  
4. Land wave 13; cut wave 14 integ.  
5. Publish CLS + DRV doc lanes; refresh matrix + ticket checkpoints.  
6. Escalate `/igx-autonomy` if operator OD or train scope needed for any lane marked `OWNER_DECISION_RECORD`.

## Tests

- Per lane: pytest only on **G-EXE-*** (wave 13).  
- Integrator: `tests/unit/docs/`, `check_docs_domain_pairs.py`, `verify_ticket_approvals` per T-IGX-046-F test contract.

## Risks & escalation triggers

- Pin drift → stop, EQO-RV-01.  
- Attempt to edit product/FRZ/roadmap → BLOCKED, route train ticket.  
- Pressure to mark C01+ `FULLY_MET` → refuse; W9-01 prep only.  
- WB08 EXE env missing deps → document blocker like WB06 wave-6 pattern; do not waive.

**Ref:** `governance/tickets/T-IGX-046-F.md`