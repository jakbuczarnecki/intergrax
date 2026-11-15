# T-IGX-041-L — PDC-X batch B-P7-CX: re-pin of the COMPAT-X counters (debt item Z-D1) at the last product change

## Scope

Re-pin of the COMPAT-X counters named in debt item D1 of `PLAN_T-IGX-036-Z.md`, delegated by `DELEGATION-PDC-P7-REPIN.md` part (a): `ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES` 7356 to 7417, `REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES` 7355 to 7416, the remainder fingerprint, `ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES` 2031 to 2060, `ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES` 1933 to 1963, `ACCEPTED_CURRENT_HEAD_EXCLUSIONS` 5325 to 5357 and four paths of the accepted shim delta fixture. Measured on `63005e290` against the replay lineage `05549f9acb`: 64 raw candidates added and 3 removed, every one attributed to a ticketed change (B-D1, B-D4, B-D5, B-D6, B-D7, B-D8, MCP-SERVER-X and their repairs), none unclassified, no `defect.persisted_without_version`; the four new shim paths are `source.contains:adapter` heuristic hits. Re-pin with classification and an id-exact evidence fixture, plus negative tests; no gate, classifier or heuristic is changed. Tests and fixtures only, no product code; merges before the EBH-7 pin `P`. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-L.md`.

## Plan digest

Plan digest: 92d87572e0683afaccda381e7f497a1fb4c95c2e5ab2bf5dd2dca62be5019003

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-L.md`](plans/PLAN_T-IGX-041-L.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPIN.md`; the implementer never writes the approval line)

## Test contract

```text
uv run pytest tests/qualification/compat_x -q
uv run ruff check tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run pyright tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Any change of the shim heuristic, a classifier, a discovery rule or the meaning of a gate; renaming the pinned constants; the `ACCEPTED_P0_*` historical pins; every file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`.
- The EBH-7 re-pin (T-IGX-041-J), the F10 classifier entries (T-IGX-041-K), the IIAC activation and its rebind, OD signatures, FRZ changes, certification.

## Lifecycle state

plan_pending
