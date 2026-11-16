# IMPL-S12 audit handoff - T-IGX-031-A, S12 (lint normalization, Amendment A1)

Implementer session claims no PASS and writes no `Approved:` line.

- S12 SHA: `47c29b4c35e46439b86916405557ac0488344320` (branch `hist-cert/impl-tests`, local only, not pushed)
- PARENT: `217dd739eb51791aa274488b3f02a43be8c8894c` (printed by the step 12 (b) script via `git rev-parse <S12_SHA>^`)
- Commit made WITHOUT `--no-verify`; the pre-commit hook passed (ruff 4 files clean, harness getattr gate OK). Trailers: `Ref: governance/tickets/T-IGX-031-A.md`, `Co-Authored-By: Claude Sonnet 5.5`.
- Plan: `PLAN_T-IGX-031-A.md` digest dc0fcf73...3cb6b, approval cert-031a2.

## Diff (4 files, +17 / -32; only the four existing touched files)
- `test_sec_ent.py`: I001 (blank line after `import pytest` removed); F401 function-local `from intergrax.runtime.events.event_bus import RuntimeEventBus` deleted in `test_defense_middleware_blocks_cross_tenant_scope` (the blank line that followed stays, as the plan says).
- W5 b2: I001 (blank line removed; `testing_support`/`tests.support` imports above `lab_application.*`, which form a trailing block).
- W5 c: I001 as b2; E402 fix by MOVE: `from intergrax.contracts.event_delivery import EventDeliveryPolicy, ObservabilityExportPayload` (late import deleted, no `noqa`); format: `_wired_bus_with_recorder` signature on one line.
- W5 d: I001 as b2 (+ name order `ExporterKind, ExportError`; `ObservabilityExportSinkFactory, OtlpEventExportSink`); format: 4 hunks collapsed to single lines.
- Helper file, tickets, plans, conftest, markers, pyproject, product dirs: untouched. `no_ci` kept; no skip/xfail/noqa added.

## Commands (cwd = worktree; each pytest/ruff process alone under the heavy lock; logs in `impl-s12/`)
Edit sequence (ruff 0.15.20 via `uv run --extra dev-ci`): `ruff check --select F401 --fix SE` (1 fixed); hand edit in C (Python, CRLF preserved); `ruff check --select I --fix SE B2 C D` (4 fixed); `ruff format SE B2 C D` (2 reformatted).

| Step | Command | Result |
|---|---|---|
| 12a | `ruff check` on the five files | exit 0, All checks passed |
| 12a | `ruff format --check` on the five files | exit 0, 5 files already formatted |
| 12b | `python -I impl-s12/s12_equiv.py 47c29b4c3...` (script extracted verbatim from plan section 7) | exit 0, `S12 EQUIVALENT`, all `ALLOWED_NOQA_DELTA` = 0 |
| 1 | pytest SE | exit 0, 14 passed |
| 2 | pytest B2 C D | exit 0, 20 passed |
| 3 | pytest SE B2 C D (one process) | exit 0, 34 passed |
| 4 | pytest D C B2 SE (one process) | exit 0, 34 passed |
| 5 | pytest SE::test_security_spine_counters_increment | exit 0, 1 passed |

All pytest processes exited normally after their summary (no hang).

## Not run (per brief / plan)
27-path run, probes A/B, step 8 ACT-N11 pre-check, step 10 diff guard beyond the AST script's own "only the four files changed" check. Residuals R8 and R9 stay as in the plan.

## Deviations from the plan
None. Notes for the auditor: (1) working tree files are CRLF (index LF, autocrlf); the W5 c hand edit preserved CRLF and the committed blobs are LF. (2) The AST script was run from the evidence dir (outside the repo) with `python -I`. (3) Nothing pushed, merged, rebased, reset or amended; the heavy lock was released after each step.
