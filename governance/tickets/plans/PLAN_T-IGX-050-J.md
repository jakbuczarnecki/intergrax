# PLAN T-IGX-050-J — ARCH-FREEZE W2 node T-AF2-RED-CLASSES: known-red register and closure-regex lint

Status: DRAFT for Operator review. Drafted by the implementing session. Nothing here is approved, applied or audited.

## 1. Purpose

The freeze qualification at M3 requires that the set of red tests equals a registered set of environment-only (ENV) or out-of-scope (OUT) classes: no red counts as passing and none is hidden (merged plan MEP-AF2 v2.1, known-red policy). This batch builds the register from the W0 CI measurement, the S1 diagnostic and the PDC-X handoff, and adds a lint for the `_CLOSURE` regex of the IIAC activation gate, which flags a stage name followed by the closure keyword even in negated text. It certifies nothing, promotes no FRZ criterion, edits no product path and edits no IIAC script.

## 2. Scope

| Item | Path | Change |
|---|---|---|
| Register | `docs/project/maintainers/qualification/arch_freeze_program/KNOWN_RED_REGISTER.md` | new; one row per known-red class: class, FIX/ENV/OUT/UNKNOWN, evidence, owner node, how to prove ENV |
| Register (machine) | `docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json` | new; schema `af2.known_red_register.v1`, node ids where the sources give them |
| Lint | `scripts/governance/lint_iiac_closure_regex.py` | new; imports `_CLOSURE` from `scripts/governance/iiac_activation_gate.py` (never copies it); exits 1 on a match and prints the line |
| Lint test | `tests/unit/governance/test_lint_iiac_closure_regex.py` | new; negative fixture (a negated closure sentence for the freeze stage) is flagged, clean text is not |
| Workflow comment | `.github/workflows/unit-tests.yml` lines 10-11 | comment only: pull_request runs use the PR branch's workflow file (W0 probe PR #609, run 43858457516) |
| Ticket and plan | `governance/tickets/T-IGX-050-J.md`, this file | new |

Allowed by AR-26/AR-31 under packet 2 (the brief of this node).

## 3. Method

1. Classes come from: the W0 baseline dispatch (run 43858238803; `RESULT.md`, `expected_red_A01.json`, `baseline_failed.log`), the 15 ACT-N11 nodes of probe PR #609, the PDC-X handoff section 5.7 and the PDC debt register, the MCP and ENT-AUDIT handoffs for R-P7-02, AF-05/AF-16, F-W3-001 and F-42, and the S1 diagnostic result.
2. A class is classified only from evidence read. Where the sources do not decide it, it is UNKNOWN. A row is ENV only with a proving run; rows whose cause is visible but unproven carry `env_proof: PENDING`, rows quoting a source carry `QUOTED`.
3. No skip or xfail is added.
4. Every added factual sentence is checked against `origin/development` or the cited log and recorded as `path@sha:line` in `FACT_CHECK.md` (kept under `.tmp/`, not committed).

## 4. Out of scope

Any change under `intergrax/`, `agents/`, `applications/`; `pyproject.toml`, `uv.lock`; the roadmap and the FRZ checklist; any `IIAC_*` file and the five IIAC scripts (`iiac_common.py`, `iiac_activation_gate.py`, the three `verify_iiac_*`); the PDC-X record directories; enumeration of node ids that the sources do not give; installing missing dependencies in CI; assigning owners the plan does not name; merging, approval and certification of any kind.

## 5. Write set (the only paths the PR may touch)

```text
.github/workflows/unit-tests.yml
docs/project/maintainers/qualification/arch_freeze_program/KNOWN_RED_REGISTER.md
docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json
governance/tickets/T-IGX-050-J.md
governance/tickets/plans/PLAN_T-IGX-050-J.md
scripts/governance/lint_iiac_closure_regex.py
tests/unit/governance/test_lint_iiac_closure_regex.py
```

## 6. Verification plan

One process at a time under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`:

- `uv run pytest tests/unit/governance/test_lint_iiac_closure_regex.py`
- `uv run python -m scripts.governance.lint_iiac_closure_regex` once on its default targets; the result is reported as measured (a match on a current record is a finding, not an error of the lint).

## 7. Risks

- The lint's default scan may match wording that exists in current activation documents; that is its purpose and is reported, not edited.
- The register rests partly on quoted counts (marked); an unmarked count would be a defect in this draft.
- `_CLOSURE` is a private name of a frozen module; if it is renamed the lint fails loudly at import.
