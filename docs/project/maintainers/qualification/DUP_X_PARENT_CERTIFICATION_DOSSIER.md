# DUP-X — Parent Certification Dossier

| Field | Value |
| --- | --- |
| **Workstream** | DUP-X duplicate & bypass certification |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **IMPLEMENTATION_SHA** | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` (Commit **A**) — [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) |
| **EVIDENCE_PUBLICATION_SHA** | `f0214533b347f3dfbbd729d96ef860869148dc0f` (Commit **B**) — not manifest Commit **C** |
| **TEST_EXECUTION_SHA** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` · publication replay log |
| **WAVE1_PUBLICATION_START** | `2325460ff7f1f0290f95dae03145e30e20e0d991` · prep replay → `.tmp/proof/dup-x-wave-1/run.log` |
| **WAVE1_IMPLEMENTATION_SHA** | `44759ed4db69a16fc07b837d9ed87d29a235163f` (Wave 1 Commit **A**) — [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) |
| **WAVE1_EVIDENCE_PUBLICATION_SHA** | `3a2d47d2c66fe60a4443d9800678ae2681a7f688` (Wave 1 Commit **B**) — not Wave 1 manifest Commit **C** |
| **WAVE1_TEST_EXECUTION_SHA** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` · post-publish log |
| **WAVE1_FINAL_EVIDENCE_RECONCILIATION_SHA** | `0159c2a18b39547bcd60823489c1a4bea1229b38` (Commit **D**) |
| **BLOCK_I_CURRENT_PUBLICATION_HEAD** | `ffc724c4c2aa325231cd7ac54df5788f66cfc7f7` (Commit **G**) |
| **Publication ladder SSOT** | [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) |
| **ECAP** | ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL — parent B1–B6 handoff |

## Scope

Cross-tier duplicate/bypass census (DUP-01..14), responsibility families (15), EBH-6 class D deferral, COMPAT-R3 lab fence, CONFIG-X qual SSOT (DUP-14).

## Wave summary

| Wave | Outcome |
| --- | --- |
| A | Closed-world inventory + candidate classification |
| B | Qual gates — class D env slug fence, COMPAT-R3 negative import test |
| C | CONFIG-X ∪ EBH-5 hints; tools_base production import gate |
| D | Mechanical exit gates + Protocol 03 package (**FINAL EVIDENCE RECONCILED — not PASS**) |
| **Wave 1** | GAP closure evidence + parent closure mechanical gates (**10/10 PASS** @ reconciliation HEAD) |

## Open items (not parent CLOSED)

| Item | Status |
| --- | --- |
| DUPX-GAP-CLASS-D | **PASS** @ Wave 1 evidence |
| DUPX-GAP-L3-TIER3-DEDUPE | **PASS** @ Wave 1 evidence |
| DUPX-CAND-03 / CAND-09 / CAND-12 | **Closed disposition** @ Wave 1 |
| OD-DUPX-02 / OD-DUPX-01 / OD-DUPX-08 | **OPEN** |

## Evidence commands

```powershell
git checkout development
git rev-parse HEAD
New-Item -ItemType Directory -Force -Path .tmp/proof/dup-x-wave-d
uv run pytest tests/qualification/dup_x/ -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-wave-d/run.log
uv run pytest tests/qualification/ebh_6/test_ebh_6_b2_layer_closed_world_integrity.py -q
# NPSC-4.2 @ HEAD — excluded from DUP-X mandatory (see exclusion record)
uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py -q
```

## FINAL STATUS

**DUP-X B4 — PUBLISHED / READY FOR FINAL INDEPENDENT B6 AUDIT @ `6ccddd3c40a88377e39962a300984d3b81f5c220` — NOT CLOSED**

Agent **forbidden:** Protocol DUP-X-03 PASS · OD-DUPX-08 · roadmap §5 update.
