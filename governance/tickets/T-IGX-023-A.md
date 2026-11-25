# T-IGX-023-A — TOOL-LIFE-X: reconcile four stale Batch B side-effect-safety tests with Batch C ABANDONED semantics (C-1 / F-01)

## Scope

Test-only reconciliation of four stale assertions in `tests/unit/runtime/tools/test_tools_side_effect_safety.py` with the sanctioned `ABANDONED` terminal disposition (roadmap section 3.0.5 "Abandonment"; Batch C `a5f813a2c`):

1. `test_side_effect_timeout_marks_uncertain_and_blocks_replay` (OD-TLX-A)
2. `test_mapped_validation_error_after_executor_marks_uncertain` (OD-TLX-A)
3. `test_close_shuts_down_execution_pool` (OD-TLX-C Option A only)
4. `test_replay_before_close_preserves_exactly_once` (OD-TLX-C Option A only)

Negative tests N1/N2 per the approved plan (N3 is cited, not duplicated). **No change in `intergrax/**`.** No PASS, closure, FRZ promotion or stage-parent change is implied.

Approved plan (byte-unedited copy of the signed plan): [`plans/PLAN_T-IGX-023-A.md`](plans/PLAN_T-IGX-023-A.md).

Signed decisions: `.tmp/session/cert-campaign/SIGNING/SIGN-4_TLX/SIGNED_DECISIONS.md` (OD-TLX-A, OD-TLX-C Option A, OD-TLXCORR-A-PLAN-2026-11-07).

## Plan digest

Plan digest: 7b864ca6c4a9927544c308f0f598691f976dc2e9e472ce9a4232e0f82f4f6012

## Approved

Approved: operator-delegate @ 2026-11-07T18:23:55+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab)

Signature lines (verbatim from the signed record):

```text
Approved OD-TLX-A: operator-delegate @ 2026-11-07T18:23:55+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
Approved OD-TLX-C: operator-delegate @ 2026-11-07T18:23:55+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
Approved OD-TLXCORR-A-PLAN-2026-11-07: operator-delegate @ 2026-11-07T18:23:55+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

## Test contract

Implementer (one pytest per step, heavy-test mutex, `-p no:xdist`):

```text
uv run pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py -q
git diff --name-only 50ddc04fbabd24ac062267d5d33f14ccef89840c..HEAD     # no intergrax/ path
```

Replay of the Batch B L2 set, the F8 67/67 inventory and the L3/TL11 17-module set (`--extra mcp`) at the exact post-fix SHA is performed by the AUDITOR, never by the implementer (OD-TLX-D, plan condition 6).

## Out of scope

Any `intergrax/` edit; FRZ-* promotion; closing TOOL-LIFE-X or any stage parent; IIAC paths; modifying the original acceptance/reconciliation text or audit verdicts; Docker/vendor/prod config (C10/C15); changing `ABANDONED` semantics; anything about C-2 (T-IGX-023-B); ticket T-IGX-023-C; any durable external-operation record left RUNNING after a failed post-close submit (reported as a finding only, not sanctioned).

## Lifecycle state

plan_approved
