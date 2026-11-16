# AUDIT_REPORT — HIST product batch (P4), independent auditor `hist-aud-prod`

Auditor: fresh retry session (not implementer `e48820fd-2405-4ac5-84f7-6e089f5c44ab`). Pinned detached checkout `D:/Projekty/_wt/intergrax/hist-aud-prod` @ `427646f41de0df992cd54c0e157de8f36b560f39` (`origin/hist-cert/product-batch` tip verified via `git ls-remote`). Parent integration before doc-only tip: merge `3a6ed857329177cc8e3b7669dbda53be1800c3df`. `START_HEAD` for scope: `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25`.

Read scope: `P4_PRODUCT_AUDIT_HANDOFF.md`, `P3_PRODUCT_IMPL_HANDOFF.md`, tickets/plans C1–C4, SHA ladder. Edit scope: this report, `findings.json`, logs only. No product-path edits. No IIAC activation sign-off. No merge to `development`.

## Batch verdict summary

| Lane | Ticket | Branch tip | Scope guard | Focused pytest | Ruff (scoped) | Verdict |
|---|---|---|---|---|---|---|
| C1 | T-IGX-030-B | `088c9c5db` | OK | 12 passed, exit 0 | check 0 / format 0 | **PASS** |
| C2 | T-IGX-030-C | `30f67c94a` | OK | 6 passed, exit 0 | check 0 / format 0 | **PASS** |
| C3 | T-IGX-032-A | `b6a056e7d` | OK | 9 passed, exit 0 | check 0 / format 0 | **PASS** |
| C4 | T-IGX-033-A | `8bfee643f` | OK | gates 9 passed / CLI 3 failed, 5 passed, 2 errors, exit 1 | check 0 / format 0 | **PASS_SCOPED** |

**Batch (integration):** all four lane implementations hold at `INTEGRATION_SHA`; C4 CLI hardening suite is red for `canonical_governance_identity_required` — same mix documented on clean `START_HEAD` in P3/P4 handoff (out of 033-A defect scope unless proven lane regression). Open campaign residuals unchanged: **HIST-F-DUR-01**, **HIST-F-BND-03B**.

## Per-lane evidence

### Scope (`git diff --name-only fe0937517..<lane-tip>` ⊆ plan)

| Lane | Diff paths | Assessment |
|---|---|---|
| C1 | ticket, 3× `intergrax/`, 1× new architecture test | ⊆ plan §3 (plan file predates lane on branch) |
| C2 | ticket, `invoker.py`, new C0 test file | ⊆ plan (exact two product paths) |
| C3 | ticket, `skill_execution_pinning_wiring.py`, new test | ⊆ plan |
| C4 | ticket, 7× product paths (3.1), `test_bnd03_*`, `test_ebh_5_rpl_g_*`, `ebh7` canonical inputs, regenerated EBH-7 matrix | ⊆ plan §3.1–3.2 (docs-only commit C2 not in lane tip range from `START_HEAD`; expected) |

### Pytest (one process at a time, heavy lock ` .tmp/session/hist-aud-prod/locks/heavy.lock`, `uv run --extra dev-ci pytest -p no:xdist`)

| Step | Command | Result | Log |
|---|---|---|---|
| C1 | `test_runtime_error_code_contract_home.py` + `test_ebh_4_r1_nexus_encapsulation_gate.py` | 12 passed, exit 0 | `c1_pytest.log` |
| C2 | `tests/.../test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` | 6 passed, exit 0 | `c2_pytest.log` |
| C3 | `tests/.../test_skill_execution_pinning_wiring_store_precedence.py` | 9 passed, exit 0 | `c3_pytest.log` |
| C4 gates | `test_bnd03_tier0_upward_edges_gate.py` + `test_ebh_5_rpl_g_regression_gates.py` | 9 passed, exit 0 | `c4_gates_pytest.log` |
| C4 CLI | `applications/.../test_cli_verification_hardening.py` | 3 failed, 5 passed, 2 errors, exit 1 | `c4_cli_pytest.log` |

C4 CLI failure signature: `RuntimeError: create_failed:canonical_governance_identity_required` in offline demo / subprocess paths (`test_demo_then_verify_fresh_process`, signer-failure recovery, retry-invalid). Tier-0 launcher and BND/EBH gates green. Treat as **environment / pre-existing identity fixture gap**, not proof that 033-A moved code incorrectly (handoff § implementer summary).

### Ruff (scoped touched paths per lane; logs `*_ruff_*_v2.log`)

All four lanes: `ruff check` and `ruff format --check` exit 0 on plan-listed paths.

### ACT-N11 (expected red, not fixed)

`uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` → **exit 1**, `IIAC policy activation: FAIL`, lists 12 product paths across C1–C4 (matches batch). Log: `act_n11_verify.log`.

## Independence and limits

This PASS / PASS_SCOPED covers the **product batch integration SHA** and focused contracts above only. It is **not** G6 / IIAC re-activation, **not** FRZ promotion, **not** full QUAL-X Dimension A replay. C4 `PASS_SCOPED` explicitly excludes claiming full CLI host suite green.

Logs: `D:/Projekty/_wt/intergrax/hist-aud-prod/.tmp/session/hist-aud-prod/`. Machine findings: `findings.json`.
