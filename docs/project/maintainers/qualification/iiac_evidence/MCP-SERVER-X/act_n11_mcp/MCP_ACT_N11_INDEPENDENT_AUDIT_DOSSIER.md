# MCP-SERVER-X — ACT-N11 independent IIIC product audit dossier

| Field | Value |
|---|---|
| Auditor agent role | `independent-quality-auditor` |
| Auditor session id | `a4f8e2c1-9b3d-4e7a-8f62-1c0d5e6a7b8c` |
| Implementer session (evidence only) | `T-IGX-035-E-MCP-P4-IMPLEMENTER` |
| Policy version | `IIAC-2026-11-05.1` |
| Operator decision (canonical) | `D-MCP-IIAC-01` option **(a)** |
| Operator alias (routing chat) | `Approved: D-MCP-IIIC-01=a` |
| Prior `audit_execution_sha` (ACT-N11 pin) | `11ac8ad85f62a6925d25ea840c92d5c6a936cfe4` |
| `origin/development` @ audit | `7172f294fdd44fe35024ba60378fb9f4e0125963` |
| PR #454 `headRefOid` @ audit | `7c401fd98cf2b42b9b2fd5860485bac34a8c2df3` |
| `audited_implementation_sha` (P4 product) | `a21a6629a6f8aa25519d4e25ac0004b38426669b` |
| `audit_execution_sha` | `7c401fd98cf2b42b9b2fd5860485bac34a8c2df3` |
| `test_execution_sha` | `7c401fd98cf2b42b9b2fd5860485bac34a8c2df3` |
| Scoped verdict | **`PARTIAL_EXACT_BLOCKERS`** |

## Scope reconciliation

- `product_paths_changed_between(11ac8ad85f…, 7c401fd98…)` = **16 paths** — matches [`audit_scope.json`](./audit_scope.json) `product_paths_in_scope_full_p4` (0 missing, 0 extra).
- `product_paths_changed_between(a21a6629a…, 7c401fd98…)` = **[]** — governance-only tip commit on PR; auditor N-03 product write-surface **clear**.
- `origin/development` carries **7-path** P2/P3 Δ; PR #454 carries full **16-path** P2–P4 Δ per recovery batch.

## Sixteen-path inventory (purpose)

| # | Path | Stream | Purpose |
|---|------|--------|---------|
| 1 | `intergrax/contracts/mcp_server_definition.py` | P2 | Typed `mcp-server-definition.v1` contract, exposure fingerprint, retired bindings |
| 2 | `intergrax/applications/_shared/mcp_server_engine.py` | P2→P4 | Composition engine, builtins, catalog bridge, custom host tools |
| 3 | `intergrax/applications/_shared/mcp_host_mount.py` | P3 | Harness FastMCP mount |
| 4 | `intergrax/applications/_shared/mcp_import_guard.py` | P3 | Lazy MCP import fail-closed |
| 5 | `intergrax/applications/_shared/mcp_engine_concurrency.py` | P3 | Admission / concurrency guards |
| 6 | `intergrax/applications/_shared/settings_loader.py` | P3 | MCP opt-in settings surface |
| 7 | `intergrax/applications/_shared/mcp_nexus_server.py` | P4 | Nexus MCP façade on engine |
| 8 | `intergrax/applications/_shared/mcp_application_server.py` | P4 | Application façade composer |
| 9–16 | `applications/*/mcp/server.py` (×8) | P4 | Per-app MCP servers on shared pattern |

## Invariant review (read-only)

| Area | Assessment |
|------|------------|
| Tier boundaries | No `intergrax/` → `applications/` imports in MCP shared modules; facades live Tier-3. |
| Contracts vs runtime | Definition contract has no FastMCP imports; engine owns composition. |
| Tenant / principal | Trust-A streamable HTTP tests **PASS**; catalog bridge enforces tenant + principal match. |
| Transport | In-scope HTTP mount / harness path only; stdio/SSE out of ticket scope. |
| Governance | No activation record edit, no gate weakening, no verifier edits in scope. |

## Blocking defect

**F-MCP-ACT-N11-001:** `compose_mcp_server_from_definition` registers `host_custom` tools with an inner `async def _custom_tool(..., **kwargs)` — FastMCP `ParsedFunction` rejects `VAR_KEYWORD`. Breaks **MCP-P4-05** research custom binding at compose time (`test_research_mcp_pipeline_reaches_orchestration_strategy`).

## Commands @ `7c401fd98`

| Command | Exit |
|---------|------|
| `verify_iiac_policy_activation --require-in-force` | **1** (16× ACT-N11; expected pre-activation) |
| Ticket pytest contract (5 modules) | **1** (24 pass, 1 fail — tracked memory admission) |
| Adversarial MCP roll-up | **1** (56 pass, 3 fail — includes F-MCP-ACT-N11-001) |
| `test_mcp_catalog_discovery_qualification` + host mount | **0** (22 pass) |

Raw logs: `auditor_verify_iiac_require_in_force_baseline.log`, `auditor_mcp_ticket_test_contract.log`, `auditor_mcp_adversarial_tests.log`.

## Role separation

- Implementer: PR #454 commits (`jakbuczarnecki` / Cursor agent) — P4 product @ `a21a6629a`.
- Auditor: separate Cursor session; **no** product-path commits between audited implementation and execution SHA.
- Manifest: `iiac_audit_session_manifest.MCP-SERVER-X.act_n11_mcp.json`

## Independent verdict

**`PARTIAL_EXACT_BLOCKERS`** — sixteen-path inventory and ownership model are audit-ready for ACT-N11 re-bind **after** corrective task F-MCP-ACT-N11-001. **Not** eligible for `PASS_SCOPED` or Operator activation until research custom tool registration is fixed and ticket contract is green aside from documented pre-P4 residuals.

## Operator boundary (STOP)

Auditor did **not**: edit `IIAC_POLICY_ACTIVATION_RECORD.json`, fill `operator @` in OD file, merge PR #454, promote FRZ, or claim MCP-SERVER-X CLOSED.

## Corrective candidate (implementer — pending re-audit)

Implementer published product fix for **F-MCP-ACT-N11-001** on `mcp-server-x-p4` / PR #454 — see [`F_MCP_ACT_N11_001_CORRECTIVE_HANDOFF.md`](./F_MCP_ACT_N11_001_CORRECTIVE_HANDOFF.md). Prior **`PARTIAL_EXACT_BLOCKERS`** verdict and logs @ `7c401fd98` remain authoritative until a **new** auditor session re-runs scope.

## Handoff

1. ~~Implementer fixes F-MCP-ACT-N11-001 on `mcp-server-x-p4`.~~ **DONE (candidate)** — awaiting independent re-audit.
2. Re-run this audit scope @ new exact SHA (or delta re-audit if scoped).
3. On `PASS_SCOPED`, certifier validates manifest + `verify_iiac_role_separation --strict-product-diff`.
4. Operator personal line in [`IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md`](../../../IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md).
5. Post-activation: `verify_iiac_policy_activation --require-in-force` exit **0**.
