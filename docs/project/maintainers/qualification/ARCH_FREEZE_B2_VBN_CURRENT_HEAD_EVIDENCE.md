# ARCH-FREEZE B2 — VBN current-HEAD evidence (T-B2-VBN, best-effort)

**EXECUTION_SHA:** `f7ce9daebad35537da974de0dda3962126c6030e`  
**Not authorized:** FRZ-RPL-01..04 PASS promotion, FIND-VBN-PARTIAL closure as BLOCKER.

## Objective (best-effort)

Capture current-HEAD mechanical VBN qualification inputs per [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2 and ENT VBN acceptance wrappers — **without** claiming full §2.1–2.7 artifact completeness @ HEAD.

## Commands

```bash
uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q
```

Evidence log (gitignored): `.tmp/session/arch-freeze-b2-vbn/vbn-acceptance-pytest.log`

## ENT FULL audit cross-check (T-B4-L3-FULL)

When L3-ENT-FULL runs, VBN-domain outcomes are recorded in:

`.tmp/proof/arch-freeze-full/enterprise_audit_report.json`

## Status

| Artifact (VBN §2.x) | @ EXECUTION_SHA |
| --- | --- |
| §2.1–2.7 full SSOT inventories | **Partial / historical stage evidence** — not re-certified as complete @ HEAD in this wave |
| ENT VBN acceptance pytest (9 modules) | Executed @ EXECUTION_SHA (see log) |
| FIND-VBN-PARTIAL | **Remains BLOCKER** until independent VBN PASS @ HEAD or operator disposition |

**Claimed:** best-effort current-HEAD VBN pytest + L3 report pointers.  
**Not claimed:** prohibited hardcoding=0 globally, FRZ-RPL promotion, ECAP §15 admission.
