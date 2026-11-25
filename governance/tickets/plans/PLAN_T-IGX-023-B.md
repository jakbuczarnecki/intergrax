## DRAFT 2 — `T-IGX-023-B` (working name TLX-CORR-B): make the F3 streamable-HTTP parent-cancel test deterministic (C-2 / F-02)

### Scope
Test-only. Remove the wall-clock race in `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py::test_f3_streamable_http_parent_cancel_during_catalog_child` (`:318-396`): cancel is armed by a hard-coded `asyncio.sleep(0.06)` (`:369-376`) racing the fresh MCP session handshake; when the root unwinds first the server correctly raises `missing_execution_binding` (ToolError; no structured payload), which `_tool_payload()` flattens to `{}`. Classification: **test timing defect, not a product defect** (see root-cause §2).
Secondary: make the test actually exercise "cancel *during* the catalog child" (currently the passing interleaving returns `execution_cancelled` at the pre-admission check, `bridge.py:331-335`, before the child runs).
Depends on **OD-TLX-B** (choose fix vs accept-as-known-limitation vs quarantine).



### Exact files and minimal change
| File | Change |
|---|---|
| `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | In `_RecordingCatalogInvoker` make `started` a `threading.Event` (it is currently an `asyncio.Event` set from a worker thread at `:88` and **never awaited** anywhere in the file — thread-unsafe and unused). In the parent-cancel test replace `_request_parent_cancel`'s `sleep(0.06)` with: `await asyncio.to_thread(invoker.started.wait, 8.0)` (child admitted and running) → then `CancellationCoordinator.request(...)`. Also add `"step_id": _STEP` to the `invoke_catalog_tool` arguments for parity with test 1 (optional; bound step already resolves from metadata). Assert `invoker.results` is non-empty (child actually ran) **before** asserting the outcome. |
| Same file | Add one explicit helper `_is_error_text(result)` (reads `result.isError` and `content[0].text`) so any `ToolError` outcome is asserted by **named fail-closed code** (`missing_execution_binding`, `execution_terminal`) rather than collapsing to `{}`. The accepted structured set stays `{"execution_cancelled","execution_admission_denied"}`. |
| `intergrax/**` | **No change.** |

Implementer note (must verify, not assumed): with cancel triggered mid-child, confirm from a real run which structured outcome the cross-loop path yields (`execution_cancelled` at `bridge.py:405-409`/`:419-423`, or an ingress `execution_admission_denied` from `root_execution_child_work_ingress.py`); the assertion set must match observed deterministic behaviour, not be widened to make it pass.

### New / negative tests
- N1 (`test_f3_streamable_http_parent_cancel_before_admission_is_fail_closed`): cancel the parent and wait until the root deregisters, *then* call `invoke_catalog_tool` → assert `isError` with text starting `missing_execution_binding` (or structured `execution_cancelled`), `invoker.results` empty (child never ran). Deterministic by construction (ordering by events, no sleeps). This converts today's flaky branch into a named, asserted guarantee.
- N2: negative control — a stale/unknown `task_id` yields `missing_execution_binding` ToolError and `invoker.results` empty.

### Test contract
```text
uv run --extra mcp pytest -p no:xdist \
  tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q
```
Run **as the whole module, three consecutive times** (the failing condition is "any earlier test in the same process"), plus once together with the F3 L2 inventory and the L3/TL11 17-module set used by the auditor (`AUD-TOOLLIFE/AUDIT_REPORT.md` §4 row 3), each as a **single** pytest step with the mutex. Acceptance for the ticket: 3/3 module-green + inventory green at the exact SHA; also record that the pytest process exits after the summary (F-07 hygiene) — if it does not, report as a separate finding, do not fix here.

### Out of scope
Any `intergrax/` change (bridge, ingress, registry); changing the fail-closed `missing_execution_binding` behaviour; F-07 thread leak fix; F-09 packaging of the `mcp` extra; FRZ/closure; Docker/vendor.

### Risk
Low–medium: the test is a physical L3 proof used as TL3/TL11 evidence (R4 F3 PASS qual). If the corrected test exposes a real mid-child cancel defect, that becomes a **new finding** (stop and report; do not weaken the assertion). Residual flakiness from the real uvicorn server remains possible → hence the 3× module requirement.

### Rollback
`git revert` of the test commit; returns to known 1-red-in-module baseline.

### Signing-session conditions (binding; part of the approved plan text)
Source: OD-TLX-B signed by SIGN-4_TLX (Option 1). Narrowing only; no scope added.
1. Test-only. `intergrax/**` MUST NOT change (bridge, ingress, registry unchanged).
2. The accepted assertion set must match behaviour OBSERVED in real deterministic runs; never widen it to make the test pass and never weaken an assertion.
3. If the corrected mid-child test exposes a real cancel defect, STOP and report it as a new finding (it is not fixed under this ticket).
4. Acceptance requires the whole module green 3 consecutive times (`uv run --extra mcp`) plus the inventory run at the exact SHA; skip/quarantine/known-limitation is NOT an acceptable closure of C-2 under this ticket.
5. F-07 (non-exiting pytest process), F-09 (`mcp` extra packaging), F-03, F-04 remain out of scope.
6. No PASS, closure, FRZ promotion or stage-parent change is implied by this ticket or its approval.
