# ROADMAP-REPLAY-X — Qualification manifest (L1 / L2 / L3)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD (baseline)** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **TEST_EXECUTION_SHA** | **PENDING** — set @ first pytest on **committed** publication tree |
| **EVIDENCE_PUBLICATION_SHA** | **PENDING** — commit containing this manifest + SSOT |

## Gate status

| Gate | Status | Command |
| --- | --- | --- |
| **L1** integrity | **PRELIMINARY PASS** (dirty tree) | `uv run pytest tests/qualification/roadmap_replay_x/ -q` |
| **L2** parent replay | **OPEN** | per table below @ `TEST_EXECUTION_SHA` |
| **L3** ECAP bundle | **OPEN** | EBH-6 export pattern + consolidated qual @ `TEST_EXECUTION_SHA` |

## L2 — mandatory parents (@ future SHA)

| Parent | Path | Command |
| --- | --- | --- |
| DUP-X | `tests/qualification/dup_x/` | `uv run pytest tests/qualification/dup_x/ -q` |
| CONFIG-X | `tests/qualification/config_x/` | `uv run pytest tests/qualification/config_x/ -q` |
| COMPAT-X | `tests/qualification/compat_x/` | `uv run pytest tests/qualification/compat_x/ -q` |
| TENANT-X | `tests/qualification/tenant_x/` | `uv run pytest tests/qualification/tenant_x/ -q` |
| STATE-X | `tests/qualification/state_x/` | `uv run pytest tests/qualification/state_x/ -q` |
| TRACE-X | `tests/qualification/trace_x/` | `uv run pytest tests/qualification/trace_x/ -q` |
| GOV-X2 | `tests/qualification/governance/gov_x2/` | `uv run pytest tests/qualification/governance/gov_x2/ -q` |
| CTRL-X | `tests/qualification/control_planes/ctrl_x/` | `uv run pytest tests/qualification/control_planes/ctrl_x/ -q` |
| EBH-5 | `tests/qualification/ebh_5/` | `uv run pytest tests/qualification/ebh_5/ -q` |
| EBH-6 | `tests/qualification/ebh_6/` | selective + export per dossier |
| NPSC-4.2 | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` | allowlist test (**expected FAIL**) |

Remaining §5 category **A** parents: reuse prod-Δ-none + targeted qual per [`ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md`](ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md).

## L3 — consolidated batch (OPEN)

Single invocation after L2 green @ `TEST_EXECUTION_SHA` (operator-approved resource budget):

`uv run pytest tests/qualification/roadmap_replay_x/ tests/qualification/dup_x/ tests/qualification/config_x/ tests/qualification/ebh_5/ -q`

(Full platform PROD-Q **438** — **OUT OF SCOPE** unless `prod_q` in diff.)

## Preliminary evidence (dirty @ `a2ed11f`, not publication SHA)

| Run | Result | Log |
| --- | --- | --- |
| L1 replay_x | 13 passed | `.tmp/proof/roadmap-replay-x-w8-l1/l1-roadmap-replay-x.log` |
| L2 dup_x full | 44 passed | `.tmp/proof/roadmap-replay-x-w8-l1/l2-dup-x.log` |
