# DEF-CB-B2-S03-01..03 — PT-WF disposition (Package C)

**Train compared:** `origin/arch-freeze/rw-train` @ `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`  
**Evidence:** [`CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md`](CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md) (defect IDs only; no product path named)

## Defect text (cited)

From `CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md`:

- `DEF-CB-B2-S03-01`, `DEF-CB-B2-S03-02`, `DEF-CB-B2-S03-03` — campaign case **PT-B02 / S03**, owner **PT-WF**, ID-only (no reproduction steps).

From `scripts/qualification/campaign_b_w2_manifest.v1.json` case **S03**:

- Command: `python -m pytest tests/qualification/harness_01 -q --maxfail=3 --tb=line` (contract AB-10).

## Ticket compared

From `governance/tickets/plans/PLAN_T-IGX-043-J.md` §3 (indicative E2–E4 paths):

- E2 → `intergrax/runtime/execution/runtime.py`
- E3 → `intergrax/compat/langchain/documents.py`
- E4 → `intergrax/experiments/store.py`, `intergrax/debug/router.py`, `intergrax/experiments/workflow.py`

From §8.2 and §9: **E2–E4 product edits remain gated on Campaign B-1** (`NT-WF-NEG-E2-01`..`E4-01` **BLOCKED**); §8.1 G-09 harness scope is **`tests/unit/applications/test_harness_evaluation_wiring.py`** (present and green on the train at `e9bebe0ac`).

## Disposition

These three defect IDs **do not activate E2–E4**. They name only the **harness_01** qualification slice (S03), not `runtime.py`, `documents.py`, or the E4 workflow bundle. No product change is authorized under `T-IGX-043-J` for this handoff without B-1–scoped negative node ids and plan revision 2.