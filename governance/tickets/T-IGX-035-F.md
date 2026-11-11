# T-IGX-035-F — MCP-SERVER-X P5: security, execution, adversarial qualification

## Scope

Batch **MCP-P5** only (MCP-P5-01..07): harness principal propagation, tenant isolation, Governance admission, canonical ToolRuntime path, side-effect authorization, cancel/timeout/disconnect/unknown outcomes, adversarial qualification pack, B-P4-TEST-01 disposition.

**Parent:** [`T-IGX-035-E.md`](T-IGX-035-E.md) — P4 closed (scoped) @ `7930c256f` on `origin/development`.

**Baseline SHA:** `7930c256fdbe6bb878d9493670361d3cbe4b07bb`

**Tasks:** MCP-P5-01, MCP-P5-02, MCP-P5-03, MCP-P5-04, MCP-P5-05, MCP-P5-06, MCP-P5-07, B-P4-TEST-01.

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P5-01..07.

## Approved

Approved: operator-delegate @ 2026-11-11T14:25:00+01:00

Delegation: MCP-SERVER-X P5 AGENT mandate (operator chat 2026-11-11 — FULL P5 IMPLEMENTATION AND QUALIFICATION)

## Test contract

```text
.venv/Scripts/python.exe -m pytest tests/qualification/mcp_server_x/test_mcp_server_x_p5_qualification_batch.py -q
```

Expected: exit **0** (consolidated P5 batch).

Supplementary single-suite (ticket parity with P4 contract):

```text
.venv/Scripts/python.exe -m pytest \
  tests/unit/contracts/test_mcp_server_definition.py \
  tests/unit/applications/test_mcp_server_engine.py \
  tests/unit/applications/test_mcp_server_p4_lifecycle.py \
  tests/unit/applications/test_mcp_host_mount.py \
  tests/unit/applications/test_reference_apps_mcp.py -q
```

## Implementation contract

- Invariant: `MCP Client → Transport → Capability Admission → Governance → Canonical Execution → Result`
- No second ToolRuntime; no admission bypass; no fabricated vendor qualification evidence
- Independent G6 review required before ticket `closed` and P5 publication claim

## Acceptance / closure

1. P5 selection batch **PASS** @ recorded `TEST_EXECUTION_SHA`
2. Security matrix MCP-SERVER-X P5 — zero unclassified mandatory mechanisms
3. B-P4-TEST-01: PASS with harness-backed evidence wiring OR documented cross-stage blocker
4. Independent audit **ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING** (separate session; post-publication @ merge SHA)
5. CI green on PR to `development`
6. WBS / gap register / SHA ladder updated
7. Operator OD-MCP-P5-GOV-01..MEM-01 **APPROVED** @ 2026-11-11T16:24:00+01:00

## Out of scope

P6 DX/migration, P7 MCP-ACC evidence @ SHA, P8 independent stage close, FRZ PASS, MCP-SERVER-X stage **CLOSED**, operator IIAC override.

## Lifecycle state

**closed** — scoped P5 acceptance; **F-MCP-P5-GOV-001 preserved** on record.

## Governance note (preserved)

PR #456 merged @ `c59892829` **before** independent G6 was recorded. Post-publication G6 @ same SHA: **ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING**. Operator **OD-MCP-P5-GOV-01 option A** does **not** retroactively approve pre-merge organizational review. See [`GOVERNANCE_RECONCILIATION.md`](../../docs/project/maintainers/qualification/iiac_evidence/MCP-SERVER-X/p5_g6_closeout/GOVERNANCE_RECONCILIATION.md).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch (historical) | `mcp-server-x-p5` → merged PR #456 |
| BASELINE_SHA | `7930c256fdbe6bb878d9493670361d3cbe4b07bb` |
| P5_IMPLEMENTATION_SHA | `8b153422ec1e526c588b7144e934d268055a6df8` |
| P5_PUBLICATION_SHA | `c5989282937f5e592748cb606b2272d04837c601` |
| TEST_EXECUTION_SHA | `c5989282937f5e592748cb606b2272d04837c601` |
| G6_AUDIT_EXECUTION_SHA | `c5989282937f5e592748cb606b2272d04837c601` |
| G6_EVIDENCE_PUBLICATION_SHA | `7fd899aaea873aac6ec9ae3fe6ce5e3ef9ec9c8c` |
| G6_CLOSEOUT_PUBLICATION_SHA | `9ae3fdbe42831153c42846ca8231e4627973d3e3` (PR #457) |
| INDEPENDENT_AUDIT | **ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING** |
| G6 dossier | [`MCP_SERVER_X_P5_INDEPENDENT_AUDIT_DOSSIER.md`](../../docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_P5_INDEPENDENT_AUDIT_DOSSIER.md) |
| Operator OD | **APPROVED** — OD-MCP-P5-GOV-01, TICKET-01, P6-01, MEM-01 @ 2026-11-11T16:24:00+01:00 |
| P6 handoff | **AUTHORIZED** (OD-MCP-P5-P6-01) |
