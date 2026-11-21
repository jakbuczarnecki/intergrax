# PLAN T-IGX-043-M — ENT-AUDIT Part E scaffold (ARCH-FREEZE development tor)

Status: **plan §7 filled (autonomy fala 4, 2026-11-21)** — minimal development tor; train manifest work remains OUT. SSOT pointer: sibling **T-IGX-043-I** on `arch-freeze/rw-train` (Parts A–D); this ticket is the **ABSENT** Part E recovery named in `EQO_ARTIFACT_RECOVERY_REGISTRY.jsonl`.

## 1. Purpose

Close the ENT-AUDIT-X wave-diff and manifest-consumer gap so Campaign A ENT findings (A6-003..A6-008 family) can complete on the development tor without waiting for train merge.

## 2. Prerequisites

- **T-IGX-043-I** Parts A–D concepts available on train (read-only reference); no requirement to merge train.
- **T-IGX-043-B** `wiring_posture.py` on `development` (FC-2 dependency for unrelated tickets; 043-M does not edit it).
- FC-3 regression batch 78 green for scope in `regression_scope.txt` (fala 2 follow-up).
- RW-PLAN-AUDIT **BLOCKED** — plan does not claim SCOPE closure.

## 3. Write set (indicative — detail in rev 2)

| Area | Paths (indicative) |
|------|---------------------|
| Wave-diff / impact map | `intergrax/platform/enterprise_audit/changed_since.py`, `data/changed_since_impact_map.v1.json` (regenerated) |
| Manifest consumers | modules named by Part E split on train (TBD node ids) |
| Scripts | `scripts/enterprise_audit/build_impact_map.py`, `run_enterprise_audit.py` (Ref trailers) |
| Tests | `tests/qualification/ent_audit_x/unit/` (new / extended; node ids TBD) |

## 4. Test contract (placeholder)

- Command 1: ENT-AUDIT Part E unit package (name TBD in rev 2).
- Command 2: regression slice from `regression_nodes.txt` or EQO split2 list (architect copies exact node ids).
- `uv run python -m scripts.governance.igx_gov ticket` before push.

## 5. Out of scope

Check manifest rewrite, IIAC baseline promotion, train merge, FRZ promotion.

## 6. Next steps (operator / architect)

1. Copy Part E section from train plan `PLAN_T-IGX-043-I.md` (when published on development) into rev 2 with file-level diffs.
2. Record plan digest; `plan_pending` → operator/delegate approval.
3. Implement in worktree `af-dev-043m` after FC-3 regression fala 2 is green.

## 7. Test contract (fala 4 — minimal smoke, development tor)

Part E product/manifest write set stays **BLOCKED** until wave-packet approval (RW-PLAN-AUDIT). Fala 4 records regression smoke only (no manifest edit):

### 7.1 Smoke (verified collect)

| Node id | Assertion | Pytest |
|---------|-----------|--------|
| NT-043M-S1 | impact map invariants at current development head | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py` |

### 7.2 Command

```text
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py -q
```

Logs: `.tmp/session/arch-freeze/REGRESSION-78-fala4/` (with FC-3 core bundle).

### 7.1b Smoke re-run (fala 5)

Same command; **3 passed** on 2026-11-21 @ `af-dev-merge` → ticket lifecycle `in_progress` (manifest Part E unchanged BLOCKED).

### 7.3 Out of scope (unchanged)

Manifest growth 85→91, baseline promotion, train merge — `T-IGX-043-I` Part E on train only until OD allows development product batch.

## 8. Product slice 1 (fala 6 — development tor wave-diff, no manifest)

Manifest Part E remains **BLOCKED**. Slice 1 maps package groups touched on `development` since PDC baseline `a695c9c8…` that had no impact-map rule, using explicit FULL (all 85 manifest ids per prefix). SSOT: `WAVE_DIFF_FULL_IMPACT_PREFIXES` in `impact_map.py`; regenerate `changed_since_impact_map.v1.json`.

| Node id | Assertion | Pytest |
|---------|-----------|--------|
| NT-043M-S2 | development wave-diff prefixes cover remediation-wave modules | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py::test_development_wave_diff_touched_modules_map_to_manifest_ids` |

Command (with NT-043M-S1):

```text
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py -q
```

## 8.2 Product slice 2 (fala B2 — train-parity wave-diff residuals, no manifest)

Manifest Part E remains **BLOCKED**. Slice 2 extends `WAVE_DIFF_FULL_IMPACT_PREFIXES` with train-head parity groups and development residuals (`intergrax/background_tasks`, `intergrax/runtime/sandbox`, agent qualifier packages); regenerate SSOT.

| Node id | Assertion | Pytest |
|---------|-----------|--------|
| NT-043M-S3 | residual / train-parity prefixes escalate FULL | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py::test_development_wave_diff_residual_modules_map_to_manifest_ids` |

Command (with NT-043M-S1..S2):

```text
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py -q
```

Fala 5 close: same command — **9 passed** @ `e61e5136a` (2026-11-21; no §8.4 slice — manifest Part E **BLOCKED**).

## 8.3 Governance slice 3 (fala 5B — development tor wave-diff closure, no manifest)

Manifest Part E remains **BLOCKED**. Slice 3 records CHANGED-SINCE fail-closed cleanliness over the PDC campaign range at the current development worktree head; no further `WAVE_DIFF_FULL_IMPACT_PREFIXES` entries on this tor until OD allows manifest Part E.

| Node id | Assertion | Pytest |
|---------|-----------|--------|
| NT-043M-S4 | PDC-range CHANGED-SINCE does not fail closed | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py::test_development_tor_pdc_changed_since_is_fail_closed_clean` |

Command (with NT-043M-S1..S3):

```text
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py -q
```

## 8.4 Product slice 4 (operator unblock — manifest 85→91, development tor)

**Operator unblock (2026-11-21):** D3 Part E manifest batch allowed on development tor (f-dev-043m-parte). Gate-meaning write set follows ruling Q-RW-04 and sibling split PLAN_T-IGX-043-I.md section 3.3a (train pin c2a88c344, read-only). No baseline promotion, no train merge, no PASS claim.

**Prerequisite on development:** Parts A–D engine fixes of T-IGX-043-I (FRZ attribution / STAGE_GATE_ONLY_FRZ_FAMILIES, runner FAST disclosure) must be present or ported before manifest edits; development tor completed wave-diff slices 1–3 only (sections 8–8.3).

| Phase | Write set (indicative) | Node ids |
|-------|------------------------|----------|
| E0 | Port or reconcile digests.py, 
unner.py, data/schemas/frz_coverage.v1.schema.json, 	est_wd_frz_family_attribution.py from 043-I pin | NT-043M-S5a |
| E1 | enterprise_audit_check_manifest.v1.json: +6 checks (AUD, CTL, DEBT, FRZ, HRN, TRC); manifest.py / evidence.py consumers | NT-043M-S5b |
| E2 | digests.py mapped_check_id rows; impact_map.py final wave-diff + regen JSON; alignment / readiness / qual_wave_lib.py | NT-043M-S5c |
| E3 | Acceptance wrappers + pinned tests per 043-I consumers list; replay section 8.3 command | NT-043M-S5d |

### 8.4.1 Test contract (max 9 nodes per verify step)

With NT-043M-S1..S4:

`	ext
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py -q
`

After E0:

`	ext
uv run python -m pytest tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py -q
`

After E1–E3: extend with 	est_manifest_annex_c_alignment.py / mechanical readiness slices as listed in 043-I section 3.3a (scoped; no FC-3 260 bundle).

uv run python -m scripts.governance.igx_gov ticket before merge.
