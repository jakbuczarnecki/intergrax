# ROADMAP-REPLAY-X — Qualification manifest (L1 / L2 / L3)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD (baseline)** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **IMPLEMENTATION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **TEST_EXECUTION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **EVIDENCE_PUBLICATION_SHA (W4)** | `8515728649a280d435b04c23c8700fbd169495f5` |
| **PRE_B6_RECONCILIATION_PUBLICATION_SHA** | `31c462dc7c4c4ac03a10ed69254c8822b2322910` |
| **Proof logs** | `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/` · PRE-B6 copy: `.tmp/proof/roadmap-replay-x-pre-b6-archive-8515728/` |

## Gate status

| Gate | Status | Command |
| --- | --- | --- |
| **L1** integrity | **PASS** (15) | `uv run pytest tests/qualification/roadmap_replay_x/ -q` |
| **L2** parent replay | **PASS (SCOPED)** | per table below @ `TEST_EXECUTION_SHA` |
| **L3** ECAP bundle | **PASS** (480) | consolidated qual @ `TEST_EXECUTION_SHA` |

## L2 — mandatory parents (@ `TEST_EXECUTION_SHA`)

| Parent | Path | Exit | Result | Log |
| --- | --- | ---: | --- | --- |
| DUP-X | `tests/qualification/dup_x/` | 0 | 44 passed | `l2-dup_x.log` |
| CONFIG-X | `tests/qualification/config_x/` | 0 | 77 passed | `l2-config_x.log` |
| COMPAT-X | `tests/qualification/compat_x/` | 0 | 321 passed | `l2-compat_x.log` |
| TENANT-X | `tests/qualification/tenant_x/` | 0 | 194 passed | `l2-tenant_x.log` |
| STATE-X | `tests/qualification/state_x/` | 0 | 655 passed, 5 skipped | `l2-state_x.log` |
| TRACE-X | `tests/qualification/trace_x/` | 0 | 723 passed, 2 skipped | `l2-trace_x.log` |
| GOV-X2 | `tests/qualification/governance/gov_x2/` | 0 | 9 passed | `l2-gov_x2.log` |
| CTRL-X | `tests/qualification/control_planes/ctrl_x/` | 0 | 76 passed | `l2-ctrl_x.log` |
| EBH-5 | `tests/qualification/ebh_5/` | 0 | 68 passed | `l2-ebh_5.log` |
| EBH-6 | `tests/qualification/ebh_6/` | 0 | 276 passed | `l2-ebh_6.log` |
| NPSC-4.2 | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` | 1 | **EXPECTED_FAIL** (allowlist) | `l2-npsc.log` |

Remaining §5 category **A** parents: reuse prod-Δ-none + targeted qual per [`ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md`](ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md).

## L3 — consolidated batch (@ `TEST_EXECUTION_SHA`)

`uv run pytest tests/qualification/roadmap_replay_x/ tests/qualification/ebh_6/ tests/qualification/dup_x/ tests/qualification/config_x/ tests/qualification/ebh_5/ -q` → **480 passed**, exit **0**, log `l3-consolidated.log`.

(Full platform PROD-Q **438** — **OUT OF SCOPE** unless `prod_q` in diff.)

## External (@ W10 closure)

| Gate | Status |
| --- | --- |
| B6 Protocol 03 | **ISSUED — PASS (SCOPED)** |
| Operator OD | **APPROVED — OD-REPLAYX-08** |
| Parent | **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)** |
| EBH-7 | **NOT ENTERED** (eligible) |
