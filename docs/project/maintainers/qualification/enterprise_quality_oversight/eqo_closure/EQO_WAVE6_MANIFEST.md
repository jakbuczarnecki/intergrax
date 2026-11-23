# EQO wave 6 — targeted W3 EXE rerun manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/wb04|wb06|wb09/` (logs, `RUN_MANIFEST.json`, disposition touch-ups), this manifest, README index row.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `2061c6d3a417c8a49b5b719e60a9251bb5e57238` (post PR #700 English-only VCS rule).

## Lanes

| Lane | Branch | Worktree | Workstreams | Commit (on integrator) |
|------|--------|----------|-------------|------------------------|
| **G-EXE** | `eqo/wave-6/exe-docs` | `eqo-w6-exe` | WB04, WB06, WB09 pytest EXE @ TRAIN_PIN | `949a76b76`, `2ab004c37`, `a13958b49` |
| **G-INT** | `eqo/wave-6/integ` | `eqo-w6-integ` | Lane merge, manifest, README index, PR → `development` | _(integrator tip)_ |

**Out of scope:** product/FRZ edits, AUD session E verdict changes, terminal `EQO_CLOSED`, G6/IIAC independence claims, W3 `W3_CLS.md` disposition refresh (wave 5 band).

## EXE summary (@ TRAIN_PIN)

| WB | `pytest_exit_code` | Summary | Blockers |
|----|-------------------|---------|----------|
| WB04 | `0` | 13 passed in 79.41s | — |
| WB06 | `1` | 1 failed, 31 passed in 21.09s | `PYTEST_SLICE_NON_ZERO_EXIT` (Ollama adapter env note in manifest) |
| WB09 | `0` | 66 passed in 66.86s | — |

Evidence paths: `runs/wbNN/pytest_wbNN_exe.log`, `runs/wbNN/RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-6/exe-docs`).

## Merge ladder

```text
eqo/wave-6/exe-docs (WB04/06/09) ──► eqo/wave-6/integ ──► development (PR; docs gates)
```

**Docs gates (integrator PR):** `ticket-governance` + `governance-policy-pin` per ticket merge policy; platform CI waiver pattern unchanged.

## Follow-on triggers

- Remaining OPEN W3 packets (WB01–03, WB05, WB07–08, WB10–12) EXE reruns under later waves or operator schedule.
- WB06 slice follow-up only if operator requests env qualification (`llm-langchain-ollama`) — not asserted closed here.
- WB08 `W3_CLS` when DES/EXE evidence exists (deferred from wave 5).

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.