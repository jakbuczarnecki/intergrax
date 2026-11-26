# Campaign B V5 — assertion equivalence map (P2 / T-IGX-073-E)

Alpha baseline **B** (`AUDIT_BASELINE_2`): `7960ae7f6c82da60ca9acdc0ceebf63810002a42`. Diagnostic only; does not close alpha.

## Alpha measurement @ B (P2 session `af-v5-p2-cb`)

- **PT-B01:** 19 manifest rows; **8 PASS**, **8 FAIL**, **3 NOT_EXECUTABLE**; **11/19 incomplete** → **`PARTIAL_DIAGNOSTIC`** (runner exit 1). Eight FAIL cause classes recorded in [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md).
- **PT-B02:** 9/9 defined; runner **5 PASS** (`R21`–`R24`, `S02`), **3 FAIL** (`R25`, `S01`, S03 process exit), **1 NOT_EXECUTABLE** (`R20`); **4/9 incomplete** → **`PARTIAL_DIAGNOSTIC`**. **S03** node cited from stdout: **127 passed** (runner exit 4294967295 = termination only). **R24** = recovered NT-19 @ B. **R25** = 9× unmapped `evol_x` / 3× clean (EVOL-X, ENT-AUDIT-X, MCP-SERVER-X).

## Recovered probes (vendor identity)

| case_id | Tracked path | Source | SHA-256 |
| --- | --- | --- | --- |
| R04 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_ten.py` | `.tmp/session/arch-freeze/b1-alpha/probes/pp_ten.py` | `f94ca505365745623ac27743a171ee398d63ccbd4a6cec93559466ec600d9043` |
| R24 | `scripts/qualification/campaign_b_w2_alpha/probes/pp_nt19.py` | `.tmp/session/arch-freeze/af-w2-alpha-evidence-restoration-01/source_snapshot/b2/probes/pp_nt19.py` (adapted: `CAMPAIGN_B_ALPHA_ROOT` / `cwd`) | `837988551770ecdd8f10b6288e02857a8acd03b2550b4a67aeb7bf54eeafc9cd` (pre-adaptation snapshot; tracked file includes `cwd` default) |
| R25 | `scripts/qualification/campaign_b_w2_alpha/scripts/r25_impact12.py` | `.tmp/session/arch-freeze/b2-alpha/scripts/r25_impact12.py` | `d2a8a31264e3ca90b1c4e4ded3bf2b7d4ff6cc08c8c611e5b298cb03c199b21b` |

**B overlay execution (gitignored):** copy tracked probes into `AUDIT_BASELINE_2` worktree `.tmp/campaign-b-overlay/` (same relative layout). Manifest commands use overlay paths so the historical tree at **B** stays clean.

### R04 (`pp_ten.py`)

- **AB-06 / AR-3a+3c:** NT-11..NT-15 + ERL/AHI static scan pack; not the full `tests/qualification/tenant_x` suite.
- **Assert semantics:** PASS = defect refuted; FAIL = defect confirmed; `INCONCLUSIVE` via skip on harness/setup.
- **Worktree guard:** product import path must contain `/_wt/` (scratch/baseline worktree only).

### R24 (`pp_nt19.py`)

- **AB-10 / NT-19 (A5-002):** streaming fault-injection vs bounded retry / `execute_streaming` guard; not `tests/unit/campaign_b/_pt_t01_matrix.py` registry alone.
- **Helpers:** loads `tests/qualification/pdc_x/d3/test_d3_f01_stream_guard.py` from **B** worktree (`CAMPAIGN_B_ALPHA_ROOT` defaults to `cwd`).

### R25 (`r25_impact12.py`)

- **AB-10 / R25c (X-A6-003):** twelve stage-cert `build_impact_map.py --since-sha` children (historical Campaign A / `campaign_a/TOOL_RUN_LOG.md` SHAs), not a generic `run_enterprise_audit.py --mode CHANGED-SINCE` help string.
- **12 SHAs (canonical, from recovered script):**

| stage | since_sha |
| --- | --- |
| TENANT-X | `5ccb72fccd4e3820ebdc9c18008afe6f7d59f558` |
| CONFIG-X | `84b8739b8b6e62c8e580e26ca3011f6aa46d457a` |
| COMPAT-X | `84b8739b8b6e62c8e580e26ca3011f6aa46d457a` |
| STATE-X | `bd54941d933069b8bfb2819bb819c1cbdbe71576` |
| TRACE-X | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` |
| CTRL-X | `bd54941d933069b8bfb2819bb819c1cbdbe71576` |
| GOV-X2 | `318a4d835534ef5c48575f4649551c957dacd8ea` |
| EBH-5 | `33ecf19cc020b112d9e586c87b49e819202f5d6d` |
| DUP-X | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| EVOL-X | `7a4f279fd8c0986431941c77283d872ea5b0c1ba` |
| ENT-AUDIT-X | `5f178174e2462b038d544b61192a666d5b0ca79f` |
| MCP-SERVER-X | `919e34b8f9fbccfc62a16205d8647d6e6070ce45` |

## V4 wired slots → AB contract

| case_id | contract | Manifest command (tracked) | Equivalence to historical alpha contract |
| --- | --- | --- | --- |
| R01b | AB-05 | `python scripts/maintenance/check_agent_registry_bypass.py` | Agent-registry bypass / IIAC negative posture probe (AB-05 governance lane); not a generic lint sweep. |
| R06 | AB-08 | MCP identity + MCP-SERVER-X P7 L2/L3 pytest quartet | MCP security closed-world slice (AB-08); matches WP-3 AR-05 security gate intent. |
| R07 | AB-07 | Four `trace_x` closed-world parity nodes | TRACE-X P5/P6 parity without historical `pp_nt07.py` single-file probe. |
| R08 | AB-07 | `pytest tests/qualification/tenant_x` | TENANT-X qualification surface (AB-07); not the whole pack as R04 substitute. |
| R10 | AB-08 | TOOL-LIFE-X three-node slice (side-effect safety, F3 cancel, C0 submit) | Tool-life operational guards (AB-08 / WP-3). |
| R12 | AB-07 | `tests/integration/decision_system/test_concurrent_finalization.py` | Concurrent finalization probe replacing `pp_nt09.py`. |
| R13 | AB-07 | ENT/IIAC unit verifiers + `test_n09_*` | Enterprise audit / IIAC negative wiring replacing `pp_nt08.py`. |
| R15 | AB-09 | `dup_x` + `test_replay_x_closure_counters_computed` | Replay/dup counters lane (AB-09); not `pp_r15_counters.py` alone. |
| R16 | AB-09 | Full `tests/qualification/trace_x` | R16c-trace canonical in-repo subset (AB-09). |