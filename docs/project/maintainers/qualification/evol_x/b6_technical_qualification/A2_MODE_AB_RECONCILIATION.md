# A2 — Mode A/B technical audit reconciliation

**Agent:** A2 · **TES worktree:** `.tmp/session/evol-x-b6-multi/wt-tes` @ `3c4a641b…`

## Mode A @ FINAL (`bbf3e33bc…`)

| Field | Value |
|-------|-------|
| Command | `uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish` |
| Session log | `.tmp/session/evol-x-b6-multi/logs/a2-mode-a.log` |
| Exit / payload | `EVIDENCE_INTEGRITY_OK`, mode A, disposition `final`, evidence_head `bbf3e33bc…` |
| Orchestrator spot-check | Worktree `.tmp/session/evol-x-b6-verify-final` @ FINAL (same command family) |

**G-05 technical disposition:** **PASS** — scoped FINAL bundle verifies; profile `AUTOMATION_FIRST_SCOPED`.

## Mode B — attempt history (G-06)

### Invalid attempt (retained, superseded for evidence)

| Aspect | Detail |
|--------|--------|
| Finding | **F-001** |
| Pytest log | `a2-pytest.log` — widespread FAIL in `test_evol_x_p03_integrity` (nested repo / basetemp) |
| Mode B log | `a2-mode-b.log` — `EVIDENCE_INTEGRITY_FAIL` outcome swaps |
| Root cause | Pytest `--basetemp` under linked worktree `.tmp/…/pytest-tmp` (or default basetemp inside worktree) breaks P03 nested-git fixture tests on **Windows** |
| Evidence value | Procedural negative control; **must not** be erased |

### Valid replay v2 (active G-06 evidence)

| Step | Log / artifact | Result |
|------|----------------|--------|
| VL3 | `a2-vl3-v2.log` | 290 node ids |
| Pytest | `a2-pytest-v2.log` | **290 passed**; JUnit → `wt-tes/.tmp/session/evol-x-b6/junit-v2.xml` |
| Basetemp | Primary repo `build/pytest-b6-tes-replay` (outside nested worktree) | Windows-safe |
| Mode B | `a2-mode-b-v2.log` | `EVIDENCE_INTEGRITY_OK`, mode B, TES `3c4a641b…` |

**G-06 technical disposition:** **PASS** (v2 supersedes invalid attempt for qualification purposes; F-001 remains OPEN as procedural record with **CLOSED** remediation path documented).

## JUnit matching rule

Archived `evol_junit_outcomes.v1.json` @ FINAL binds **canonical node id → outcome**. Local replay JUnit is **not** required to match CI artifact bytes (Linux runner vs Windows replay). Mode B verifies population agreement with manifest and archived outcomes.

## Disposition

**A2 — TECHNICAL PASS** (Mode A + corrected Mode B v2 reconcile to TES/FINAL chain).
