# T-IGX-041-H — PDC-X batch B-P7-D8: repair of the D8 Artifact Lifecycle domain after the independent audit IA-D8 (loop 1)

## Scope

Repair ticket of domain D8 (Artifact Lifecycle) under Operator option B (`DELEGATION-PDC-P7-REPAIR.md`), after the independent audit `PDC_P7_D8_AUDIT_REPORT.md` (verdict PARTIAL). In scope: `RuntimeContext.create_artifact` (F-D8-07: missing tenant, producer and lineage), the retention sweep (F-D8-05: delete conditional on the observed version, no sweep without an audit recorder unless explicitly allowed, positive limit), expiry enforced at read (F-D8-06), the stored workspace required on read, hold, release and delete (F-D8-04, partial), the four raw `storage.*` tools refusing the Collaborative Work content key space (F-D8-03, bounded), the governed port honouring a returned DENY decision and auditing backend failures (F-D8-08, F-D8-12), a hostile-source test (F-D8-16 part) and the documentation of what is library-only (F-D8-17 documentation part). Excluded with reason and reopen condition (exception or debt register of the P9 closure package): F-D8-01 and F-D8-02 (not wired, no evidence producer), the Operator-decision remainders of F-D8-03 and F-D8-04, and the LOW and INFO findings listed in the plan. Product code: yes, no hotspot file (`intergrax/contracts/` and `intergrax/applications/_shared/` are untouched). Seven existing test files are named in the write set for fixture-level edits only. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-H.md`.

## Plan digest

Plan digest: 232913e88f9210eb47b743731601c127e8b396ff1428b984865e9f925c0ed314

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-H.md`](plans/PLAN_T-IGX-041-H.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md`; the implementer never writes the approval line)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/qualification/pdc_x/xd/test_xd_02_agent_context.py tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py tests/qualification/pdc_x/xd/test_xd_11_observability.py tests/qualification/pdc_x/p6_repair/test_p6r_d8_worker_bootstrap.py tests/unit/tools/providers/storage tests/unit/tools/providers/test_t10_catalog_tools.py tests/unit/collaborative_work/test_artifact_content_storage.py -q
uv run ruff check <touched product, script and test files>
uv run pyright <touched product and test files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Every excluded finding of the plan (F-D8-01, F-D8-02, the Operator-decision remainders of F-D8-03 and F-D8-04, F-D8-09 remainder, F-D8-10, F-D8-11, F-D8-13 to F-D8-20); host composition of the artifact store, governance port, audit recorder, sweep trigger and evidence producer; the spill resolver; `queue_worker_wiring.py`, `worker_bootstrap.py`, `intergrax/contracts/`; the requirement matrix; `scripts/governance/`, `.github/`.
- The TENANT-X carrier re-pin (ticket `T-IGX-041-I`), the IIAC activation and its P8-04 rebind (stays stale after these product changes, known red class), the EBH-7 re-pin (stays red until the final re-pin, known red class), OD signatures, FRZ changes, certification, the re-audit.

## Lifecycle state

plan_pending
