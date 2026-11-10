# PRE-ENT — post-IIAC gate evidence @ `development`

**Date:** 2026-11-10  
**Verified replay:** `2026-11-10T15:32:00+01:00` (orchestrator `@ AUDIT_HEAD`, exit 0)  
**Campaign:** `pre_ent_enterprise_certification`  
**Ticket:** `governance/tickets/T-IGX-034-A.md`

## Pinned SHAs

| Pin | SHA |
| --- | --- |
| `development` HEAD (`AUDIT_HEAD`) | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| IIAC phase-1 activation | `661978df191d9356058a1312a712fcfa3c890e9e` |
| IIAC phase-2 binding | `661978df191d9356058a1312a712fcfa3c890e9e` (`effective_from_sha`) |
| Product merge (PR #434) | `03aa10959acf6ec3d81a7a0de8ac8224c8126973` |
| Evidence publication (Session B) | `e2bb6e9c0dab0b603f98fb6ffa11ff463da19c0b` |

## G1 — IIAC verify

```text
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
```

| Field | Value |
| --- | --- |
| SHA | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| Exit | **0** |
| Replay @ `2026-11-10T15:32:00+01:00` | **0** — `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from 661978df191d9356058a1312a712fcfa3c890e9e` |
| `origin` contains `AUDIT_HEAD` | **yes** — ancestor of `origin/development` @ `bbf5e4999` (PR #438) |

## G2 — `governance-fast`

| Field | Value |
| --- | --- |
| Run (PR #437) | [42814981916](https://github.com/jakbuczarnecki/intergrax/actions/runs/42814981916) — job `governance-fast` / `129825667490` |
| Head SHA | `4ffa14280761e1d4cd3c5d0a6d8147e7f90beaab` |
| Tree | `ca8f89bae9136144b44291b745e9a07a47c1c974` (= merge commit `1e83ade3f` tree) |
| Result | **SUCCESS** |
| Run (PR #438, Wave 3 docs) | [42817447058](https://github.com/jakbuczarnecki/intergrax/actions/runs/42817447058) — job `governance-fast` / `129834024466` — **SUCCESS** |

`workflow_dispatch` on `development` ([42817064079](https://github.com/jakbuczarnecki/intergrax/actions/runs/42817064079)) does **not** execute `governance-fast` (job `if` = PR \| push only; dispatch skipped).

## G6 — Wave 2 spot replay (docs-only delta since `03aa10959`)

Product diff `03aa10959..1e83ade3f`: qualification/docs only — protocol requires spot replay, not full serial re-run.

| Command bundle | Result |
| --- | --- |
| `test_bnd03b_product_manifest_registry_gate.py` + `test_roadmap_replay_x_mechanical_exit_gates.py` | **18 passed**, exit 0 @ `1e83ade3f` |

## Operator OD

- [`IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md`](../IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md) — **AUTHORIZED** `operator @ 2026-11-10T13:20:00+01:00`
