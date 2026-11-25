# Campaign B alpha — manifest slots without in-repo read-only commands

Diagnostic mapping for P2a (`T-IGX-073-E`). Alpha session commands under `.tmp/session/arch-freeze/` are evidence only; manifest wires **tracked** scripts and pytest targets.

## Undefined slots

| case_id | AB | WP-12 intent | Repository search | Closest in-repo artifact | Why undefined |
| --- | --- | --- | --- | --- | --- |
| R04 | AB-06 | WP-1 AR-3a/3c tenant/LangChain probe pack (`pp_ten.py`) plus scratch `test_documents.py` | `rg pp_ten`, `glob **/pp_ten.py`, `WP-11/AUTHORIZATION_REQUESTS.md` (not in repo) | `tests/qualification/tenant_x/` (suite exists but is not the historical R04 pack) | Probe `b1-alpha/probes/pp_ten.py` and worktree-only compat test live only under `.tmp/` / `_wt/_scratch/`, not in the product tree. |
| R24 | AB-10 | B-2 alpha pack (NT-19 style probe) | `rg pp_nt19`, `glob **/pp_nt19.py` | `tests/unit/campaign_b/_pt_t01_matrix.py` (registry only) | Historical cmd: `pytest …/b2-alpha/probes/pp_nt19.py` — probe not tracked. |
| R25 | AB-10 | B-2 CHANGED-SINCE / 12-SHA impact diagnostic (`R25c`) | `rg r25_impact`, `scripts/enterprise_audit/run_enterprise_audit.py`, `tests/qualification/ent_audit_x` for 12-SHA replay | `scripts/enterprise_audit/run_enterprise_audit.py` (needs `--since-sha`; not equivalent to `r25_impact12.py`) | Historical cmd: `python -I …/b2-alpha/scripts/r25_impact12.py` — session script not in repo; no single tracked pytest node names R25. |

## Defined slots added in P2a (in-repo command summary)

| case_id | AB | In-repo command (manifest) |
| --- | --- | --- |
| R01b | AB-05 | `python scripts/maintenance/check_agent_registry_bypass.py` |
| R06 | AB-08 | MCP security pytest slice (identity fail-closed, MCP-SERVER-X P7 L2/L3) |
| R07 | AB-07 | Four TRACE-X closed-world parity nodes (no `pp_nt07.py`) |
| R08 | AB-07 | `python -m pytest tests/qualification/tenant_x` (no `pp_g09.py`) |
| R10 | AB-08 | TOOL-LIFE-X PT-V02 published node set (side-effect safety, F3 cancel, C0 submit) |
| R12 | AB-07 | `tests/integration/decision_system/test_concurrent_finalization.py` (no `pp_nt09.py`) |
| R13 | AB-07 | ENT/IIAC unit verifiers + `test_n09_*` (no `pp_nt08.py`) |
| R15 | AB-09 | `dup_x` + `test_replay_x_closure_counters_computed` (no `pp_r15_counters.py`) |
| R16 | AB-09 | Full `tests/qualification/trace_x` (canonical R16c-trace in-repo subset) |

**Coverage:** B-1 `manifest_defined_cases` = 17 / 18 (`PARTIAL_DIAGNOSTIC`; R04 open). B-2 = 7 / 9 (`PARTIAL_DIAGNOSTIC`; R24, R25 open).
