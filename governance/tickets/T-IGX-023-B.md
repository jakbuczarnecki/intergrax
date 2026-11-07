# T-IGX-023-B — F3 streamable-HTTP parent-cancel test: deterministic fix (TEST-ONLY)

## Scope

Make `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py::test_f3_streamable_http_parent_cancel_during_catalog_child` deterministic (audit condition C-2 / F-02): event-driven cancel from `invoker.started` instead of a hard-coded 60 ms timer, named fail-closed assertions, and a named negative test for cancel-before-admission. Test-only; no `intergrax/` change.

Plan (approved, byte-unedited copy): [`plans/PLAN_T-IGX-023-B.md`](plans/PLAN_T-IGX-023-B.md)
Signed record: `.tmp/session/cert-campaign/SIGNING/SIGN-4_TLX/SIGNED_DECISIONS.md` (OD-TLX-B, OD-TLXCORR-B-PLAN-2026-11-07).

## Plan digest

Plan digest: e046a0fc03b2e3d439cc686c00fbd0326a03ed70fca797308930a0815db6ac7c

## Approved

Approved: operator-delegate @ 2026-11-07T18:23:55+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab)

Verbatim signature lines from SIGNED_DECISIONS.md:

```text
Approved OD-TLX-B: operator-delegate @ 2026-11-07T18:23:55+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
Approved OD-TLXCORR-B-PLAN-2026-11-07: operator-delegate @ 2026-11-07T18:23:55+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

## Test contract

```powershell
uv run --extra mcp pytest -p no:xdist tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q
```

Whole module, three consecutive separate single invocations, each under the heavy-test mutex. See the plan for the optional inventory run.

## Out of scope

Any `intergrax/` change; changing fail-closed `missing_execution_binding`; F-07; F-09; F-03; F-04; the side-effect-safety module (T-IGX-023-A); FRZ promotion, closure, PASS; Docker, vendor, production configuration.

## Lifecycle state

in_progress

## Checkpoint

(pending — implementer hands over `READY_FOR_INDEPENDENT_AUDIT`; no PASS claimed.)
