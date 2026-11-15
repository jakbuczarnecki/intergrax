# T-IGX-038-F — MCP-RC MRC-6 protocol classification, compatibility record and corrective documentation

## Scope

MRC-6 protocol classification, compatibility record and corrective documentation. Findings: G-PR-01..04, G-DOC-01..12, v1.1 compatibility and migration note. Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-F.md`.

## Plan digest

Plan digest: 4dcc7adc54b5e8a53de0d81ccbfa3c59d86dead13861894953a09d27535c8d4f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-F.md`](plans/PLAN_T-IGX-038-F.md).

## Approved

Approved: operator-delegate @ 2026-11-12T18:21:30+01:00
Delegation: docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-1.md@4ef1e09350c06018ec6beeb46eac645d49b1dc6f

## Test contract

```text
uv run pytest tests/qualification/mcp_server_x/test_mcp_server_x_p7_matrix_consistency.py tests/qualification/mcp_server_x/test_mcp_server_x_p7_l3_protocol_subset.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

**closed**

## Formal scoped closure — operator note and evidence

**Operator instruction (current conversation, verbatim):** "Wykonajmy tę ostatnią folię formalnego close-outu." This authorizes completion of the final formal ticket closeout; it does not issue a new technical audit verdict or waive any residual.

**Publication:** MCP-RC product corrections are on `development` via PR #554 (merge `61531828f5728acc8f2ed357b4b835d7f5dc4ebc`). Independent delta re-audit `PASS_SCOPED`, independent integrated-baseline addendum 4 `PASS_SCOPED`, certifier preflight `CERTIFIER_PREFLIGHT_PASS_SCOPED`, ACT-N11 re-pin 2 `IN_FORCE`, six PR #554 CI checks SUCCESS; terminal verification and 481 passed / 1 strict xfailed documented in `MCP_SERVER_X_RC_CLOSURE_DOSSIER.md` (published PR #555).

**Scope:** `closed (scoped)` under `governance/tickets/README.md` lifecycle rule ("Merged or abandoned with operator note"). This ticket is complete for its MCP-RC deliverable, not an unscoped production-ready certification. No new product change, FRZ promotion, or P8 history revision is claimed.

**Carried outside MCP-RC:** `T-IGX-039-A` remains `draft` for the strict product-scaffold xfail, owned by the operator. IA-D1..IA-D8 are explicitly operator-deferred in PR #556, outside this campaign. PDC-X findings and accepted scoped residuals retain their own owners, dispositions and gates. No historical findings are erased.

**Publication gate:** this ticket-closeout change is effective on `development` only after the closeout PR passes required CI and is merged; independent review of the documentation-only diff must confirm six statuses, this note and the closure matrix.
