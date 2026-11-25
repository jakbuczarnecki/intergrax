# PT-Q02 wave audit result (MEP W4 / WP-12 `PT-Q02`)

**Banner:** Independent wave-audit replay at the exact RW train head. **Not** G2 PASS, **not** PT-Q02 PASS, **not** RB-2, **not** ARCH-FREEZE CLOSED.

| Field | Value |
|---|---|
| **Verdict** | **CANDIDATE** |
| **Audited SHA (G2 candidate)** | `ea923cf03d88fc9d22637796e9002eb9d68e6e59` |
| **ACT-N11 baseline** | `a73d15ab3` (MEP-AF2 §B / §I) |
| **Worktree** | `D:/Projekty/_wt/intergrax/af-fc-q02` (branch `af-fc/q02`) |
| **Scratch logs** | `D:/Projekty/_wt/intergrax/_scratch/af-fc-q02/` |
| **Product files changed by this session** | **no** (docs-only commit below) |

Authority: MEP-AF2 v2.1 §I (Mutation row: T-AF2-RW-TENANT/SUPPLY); WP-12 `PT-Q02` node (four checks).

## 1. Negative-test mutation (T-AF2-RW-TENANT / T-AF2-RW-SUPPLY)

```text
uv sync --extra dev-ci --extra dev-unit-cert --frozen
uv run pytest <16 files test_rw_tenant*.py test_rw_supply*.py> -q --tb=line
```

| Command | Exit code | Outcome |
|---|---|---|
| `uv sync --extra dev-ci --extra dev-unit-cert --frozen` | 0 | env ready (`00b_uv_sync_unit_cert.log`) |
| `uv run pytest` (109 tests, RW tenant + supply negatives) | **0** | **109 passed** (`01_negative_test_mutation.log`) |

## 2. Fallback census (product delta `a73d15ab3..G2`)

```text
git diff --name-only a73d15ab3..HEAD -- intergrax/ agents/ applications/
git diff a73d15ab3..HEAD -- intergrax/ agents/ applications/
```

| Metric | Value |
|---|---|
| Product paths in delta | **311** (`02_act_n11_paths.txt`; `product_paths_changed_between` → **309**) |
| Pattern hits in diff patch | InMemory **115**; SandboxSessionManager( **1**; return None **71**; getenv **18**; environ.get **58**; or "log" **7**; or Null **1**; fallback or Stub **2** (`02_fallback_census.log`) |

Census completed (accounting only; hits require ticket-level residual mapping, not zero at wave head).

## 3. CONFIG-X delta

```text
uv run pytest tests/qualification/config_x/ -q --tb=line
```

| Exit code | Outcome |
|---|---|
| **0** | **77 passed** (`03_config_x_delta.log`) |

## 4. ACT-N11 accounting (+ governance-fast observation)

```text
uv run python scripts/governance/verify_iiac_policy_activation.py
uv run python scripts/governance/validate_policy_registry.py
python -c "product_paths_changed_between('a73d15ab3','ea923cf03')"  # 309 paths
uv run pytest tests/unit/governance/ -q --tb=line
```

| Command | Exit code | Notes |
|---|---|---|
| `verify_iiac_policy_activation.py` | **1** | **Expected ACT-N11 stale activation** — errors of the form `ACT-N11 product path changed after the audited SHA: <path>` (sample paths in `04_act_n11_accounting.log`; **309** unique product paths vs baseline) |
| `validate_policy_registry.py` | **0** | OK |
| `pytest tests/unit/governance/` (governance-fast) | **1** | **18 failed**, 991 passed, 1 skipped in 1618.83s (`04_act_n11_accounting.log`). Failures include `test_iiac_activation_gate`, `test_iiac_negative_guards`, `test_evol_x_real_repo_fresh_clone` (fresh-clone ACT-N11 guards) and `test_agent_env_sync` (2). **Quoted, not repaired; not treated as a standalone Q02 product defect** per MEP R2/KR-08 (RB-2 rebind pending). |

## Summary table

| Check | Result |
|---|---|
| Negative-test mutation | **PASS** (exit 0) |
| Fallback census | **DONE** (delta accounted) |
| CONFIG-X delta | **PASS** (exit 0) |
| ACT-N11 accounting | **DONE** (verify exit 1 ACT-N11; governance-fast exit 1 — expected class) |

## Explicit non-effects

Does **not** set G2, R, FCS, or M3; does **not** declare ARCH-FREEZE CLOSED or PT-Q02 PASS for RB-2.