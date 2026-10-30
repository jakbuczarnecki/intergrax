# ROADMAP-REPLAY-X — Qualification evidence record (@ IMPLEMENTATION_SHA)

| Field | Value |
| --- | --- |
| **START_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **IMPLEMENTATION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **TEST_EXECUTION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **EVIDENCE_PUBLICATION_SHA (W4)** | `8515728649a280d435b04c23c8700fbd169495f5` |
| **PRE_B6_RECONCILIATION_PUBLICATION_SHA** | **local only @ session** — doc-only pass; not W4 commit |
| **git HEAD @ PRE-B6 preflight** | `8515728649a280d435b04c23c8700fbd169495f5` |
| **Environment** | Windows dev · `uv` · repo root `D:\Projekty\intergrax` |
| **Proof directory** | `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/` |

## L1 — mechanical (@ `TEST_EXECUTION_SHA`)

| Command | Exit | Result | Log |
| --- | ---: | --- | --- |
| `uv run pytest tests/qualification/roadmap_replay_x/ -q` | 0 | **15 passed** | `l1-roadmap_replay_x.log` |

## L2 — parent replay (@ `TEST_EXECUTION_SHA`)

| Parent | Command scope | Exit | Result | Disposition | Log |
| --- | --- | ---: | --- | --- | --- |
| DUP-X | `tests/qualification/dup_x/` | 0 | 44 passed | **PASS** | `l2-dup_x.log` |
| CONFIG-X | `tests/qualification/config_x/` | 0 | 77 passed, 1 warning | **PASS** | `l2-config_x.log` |
| CTRL-X | `tests/qualification/control_planes/ctrl_x/` | 0 | 76 passed | **PASS** | `l2-ctrl_x.log` |
| GOV-X2 | `tests/qualification/governance/gov_x2/` | 0 | 9 passed | **PASS** | `l2-gov_x2.log` |
| EBH-5 | `tests/qualification/ebh_5/` | 0 | 68 passed | **PASS** | `l2-ebh_5.log` |
| EBH-6 | `tests/qualification/ebh_6/` | 0 | 276 passed | **PASS** | `l2-ebh_6.log` |
| COMPAT-X | `tests/qualification/compat_x/` | 0 | 321 passed | **PASS** | `l2-compat_x.log` |
| TENANT-X | `tests/qualification/tenant_x/` | 0 | 194 passed | **PASS** | `l2-tenant_x.log` |
| STATE-X | `tests/qualification/state_x/` | 0 | 655 passed, 5 skipped | **PASS** | `l2-state_x.log` |
| TRACE-X | `tests/qualification/trace_x/` | 0 | 723 passed, 2 skipped | **PASS** | `l2-trace_x.log` |
| NPSC-4.2 | `test_npsc4_2_residual_compatibility_gate.py` | 1 | 26 passed, 1 failed (allowlist) | **EXPECTED_FAIL (SCOPED)** | `l2-npsc.log` |

## L3 — ECAP consolidated batch (@ `TEST_EXECUTION_SHA`)

| Command | Exit | Result | Log |
| --- | ---: | --- | --- |
| `uv run pytest tests/qualification/roadmap_replay_x/ tests/qualification/ebh_6/ tests/qualification/dup_x/ tests/qualification/config_x/ tests/qualification/ebh_5/ -q` | 0 | **480 passed**, 1 warning | `l3-consolidated.log` |

## Log integrity (SHA256)

| Log | SHA256 |
| --- | --- |
| `l1-roadmap_replay_x.log` | `c0ba629543beb415c344e74496a030c1ff17846597df5bf39e27706bb6f4eb6b` |
| `l2-dup_x.log` | `75b47e0273477b47f29689e99e3ad3b37ef84c0c3343ceebd56b9abd23a4b894` |
| `l2-config_x.log` | `af9ac33d3114d7bdd679766629798242240ed092f24f34b607e538623b770055` |
| `l2-ctrl_x.log` | `c9567cffca64f29fee608ddf01c63d7e7bd8c53eb0e18dac7936fd9eaa19413a` |
| `l2-gov_x2.log` | `212a88c1645bcdd9e0442699790fc27e5017e146671e482532630935253c8d60` |
| `l2-ebh_5.log` | `ba139cd80327dca6e7a51e449cf5ddf15bca8ccae0b936a6dec55e5e22cb89e7` |
| `l2-ebh_6.log` | `437c475df73766eeaa2a9a1af9e203cb6b7e950516d7d013efe9877d4e821557` |
| `l2-compat_x.log` | `02f0c851e63294593ea5d91f7ccf9effc748b9346e09f9b86fa4e62cff3e81e0` |
| `l2-tenant_x.log` | `4b16c09b61ca7573f7969fa5e8dfa9899beb153029bb756faec217f7918ccab5` |
| `l2-state_x.log` | `8f7c5b1184b9424961805131c60bf2fb184a45870b03fe8bb512c6d4e645cee3` |
| `l2-trace_x.log` | `968f528e7b40ccef3cb6281990ca298f0822b544ce5c15cce820b354f61c1c20` |
| `l2-npsc.log` | `6d91abd0fe7eb55d3b998b6864041b1f2ffd1d0906096945a3167f8a88727593` |
| `l3-consolidated.log` | `d748903d8a43e3d785ed0c3de2ddfb99a78c0ee50ba59354848dada4f9b404b9` |

## Six closure counters (@ `TEST_EXECUTION_SHA` worktree)

Mechanical via `tests/qualification/roadmap_replay_x/_closure_counters.py` — B6 must recompute independently.

| Counter | Value |
| --- | ---: |
| reopened_historical_stages | 0 |
| unresolved_replay_blockers | 0 |
| roadmap_drift | 0 |
| historical_evidence_invalidated_by_later_changes | 0 |
| unclassified_replay_findings | 0 |
| vendor_neutrality_replay_failures | 0 |

**Production delta:** one **scoped** prod path since DUP-X (`dependency_attempt_execution_boundary.py`); **unscoped** empty → `historical_evidence_invalidated_by_later_changes` **0**; unrelated §5 accepted evidence **not** invalidated.

## Status

**ROADMAP-REPLAY-X — QUALIFICATION PASS (mechanical L1+L2+L3 @ `TEST_EXECUTION_SHA`)** @ W4 **`EVIDENCE_PUBLICATION_SHA` `8515728649a280d435b04c23c8700fbd169495f5`**. **PRE-B6 reconciliation** — doc + 13/13 log hash verify + archive (**local only @ session**). **B6 / Operator OD / parent CLOSED — not asserted.**
