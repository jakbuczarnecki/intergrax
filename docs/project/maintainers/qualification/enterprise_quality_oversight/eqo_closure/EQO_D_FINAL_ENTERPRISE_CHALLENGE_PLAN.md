# EQO Campaign D — Final enterprise challenge (post-M3)

**Parent:** `EQO-W8` · **Pin:** candidate **M3** when `T-B7-LOCK-FREEZE-SHA` exists. **Does not run before M3.**

## Preconditions (hard)

- FCS defined and passed MEP check (`EQO-W7-01`).
- B-1/B-2/C/EDER/X disposition registries populated (not necessarily all green).
- M3 commit exists and is the evidence pin for product ≡ FCS ≡ G2.

## W8-01 — Completeness challenge (session E)

Verify **without** re-running all WB EXE:

| Check | Source | Fail → |
|-------|--------|--------|
| 51 B findings dispositioned | B-1 DIS rows | exact blocker |
| 2 C findings dispositioned | C-01/02 | exact blocker |
| X tasks dispositioned | X-01..08 | exact blocker |
| EDER-CLS | E-CLS | exact blocker |
| No open **severity** BLOCKER | FINDINGS on challenge pin | exact blocker |
| impact UNKNOWN = 0 in freeze scope | dossier + EDER | exact blocker |
| Scoped exclusions have OD | SCOPE_EXCLUSIONS | exact blocker |
| Evidence SHAs ancestor of M3 or INVALIDATED | RV registry | exact blocker |

**Output:** challenge report with list of exact blockers **or** auditor verdict “no blocker in EQO scope”.

## W8-02 — Correction loop

If blocker and product fix required: `EQO-CL-01` batch. MEP invalidation may void M3 → new M3. D is **not** pre-FCS audit.

## W8-CLS

D verdict is input to `EQO-W9` and may be read by ARCH-FREEZE `T-AF2-W8-INDEP-CONFIRM`. EQO does **not** sign N1–N13.

## Explicit non-goals

- No FCS before D (acyclic per plan §5).
- No FRZ/roadmap edits (band G).
- No repeat of full W3 EXE matrix.