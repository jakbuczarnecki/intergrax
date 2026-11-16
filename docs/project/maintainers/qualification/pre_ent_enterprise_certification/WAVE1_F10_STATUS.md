# PRE-ENT Wave 1 — EBH-7 B6-F10 status (T-IGX-034-F)

**Evidence HEAD (executed):** `4a932d0cf5487b72e433805e372e3855fcd3d987`  
**Branch:** `development`  
**Date:** 2026-11-10  
**Executor:** T-IGX-034-F (orchestrator / implementer — **no B6 PASS**, no independent acceptance)

**P03 manifest `TEST_EXECUTION_SHA` (SSOT for F10 identity):** `dc18e36d971c93c8f631fbc86f73149d1b81b4d7`  
(`docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json`)

## Recompute command

```text
# Preferred (repo standard)
uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py

# Executed this session (uv absent on PATH; project venv equivalent)
.venv\Scripts\python.exe scripts/qualification/render_ebh7_f10_normative_matrix.py
.venv\Scripts\python.exe scripts/qualification/render_ebh7_f10_normative_matrix.py --check
```

**Render disposition @ HEAD (normative, identity-enforced):**

```text
EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS
C1 BLOCKED  n=32   failed=28
C2 BLOCKED  n=43   failed=43
C3 BLOCKED  n=105  failed=99
C4 BLOCKED  n=9    failed=9
C5 BLOCKED  n=3    failed=3
```

Post-render `--check` → **`IN SYNC`** (matrix + ECAP §14 computed block agree with generator).

Uncommitted artifact from this run: `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` (drift-attested BLOCKED rows).

---

## Tree-drift vs semantic FAIL counts

F10 treats `git diff --name-only TEST_EXECUTION_SHA..HEAD` paths outside the P03 B5 identity allowlist as **`executed tree differs from TEST_EXECUTION_SHA`**. Attestations for bound pytest nodes then **FAIL**, which propagates to gate verdicts.

| Class | Count | Meaning |
|-------|------:|---------|
| **Identity violation paths** | **639** | Files changed between `dc18e36d` and `4a932d0cf` not on allowlist (includes `.claude/`, `.cursor/`, governance/tickets, PRE-ENT docs, P8 integration product paths, etc.) |
| **Obligation FAIL rows (matrix)** | **182** | All classified **tree-drift** via reason prefix / attestation message |
| **Semantic FAIL (same product tree, drift gate off)** | **0** on C1–C4; **1** publication-only probe on C5 | See below — **not** a license to weaken gates |

### Obligation-level attribution (@ HEAD, identity enforced)

- **Tree-drift FAIL:** **182** / **182** obligation results with `ResultStatus.FAIL` (including C5-REPORT-PRESENT, C5-REPORT-COMPUTED, C5-NO-FALSE-PASS gate tests blocked by drift).
- **Semantic FAIL @ HEAD:** **0** among C1–C4 normative obligations when drift messages are excluded from classification.

### Counterfactual probe (analysis only — **does not change gates**)

Clearing `identity_violations` in-process while still evaluating **current tracked tree @ `4a932d0cf`** yields:

| Gate | Verdict | failed |
|------|---------|-------:|
| C1 | PASS | 0 |
| C2 | PASS | 0 |
| C3 | PASS | 0 |
| C4 | PASS | 0 |
| C5 | BLOCKED | 1 (`C5-REPORT-COMPUTED` — on-disk §14 block still reflects drift-attested BLOCKED C1–C4, not counterfactual PASS) |

**Interpretation:** PRE-ENT `development` HEAD is **ahead of** the published `TEST_EXECUTION_SHA`. Normative certification is **correctly BLOCKED** until either:

1. **Clean checkout replay** at `TEST_EXECUTION_SHA` (CI / independent auditor / operator worktree on exact SHA), or  
2. **Controlled P03 advancement** — new B6 corrective `TEST_EXECUTION_SHA` after Wave 1+ batches land, full MANIFEST `pytest_profile` replay, then deliberate matrix render.

**Do not** relax identity allowlist or skip attestations to “green” F10 on a dirty post-P8 tree.

---

## C1–C5 gate verdict summary

| Gate | @ HEAD (enforced identity) | Semantic note (drift excluded) |
|------|----------------------------|----------------------------------|
| **C1** | **BLOCKED** (28 / 32 obligations failing) | PASS — no family/tier semantic gaps on current tree |
| **C2** | **BLOCKED** (43 / 43) | PASS — DUP-X / RF delta bindings attest clean in-process @ HEAD |
| **C3** | **BLOCKED** (99 / 105) | PASS — concern inventory / prohibited surfaces |
| **C4** | **BLOCKED** (9 / 9) | PASS — change-class regression map |
| **C5** | **BLOCKED** (3 / 3) | Publication sync blocked until SHA + render pipeline aligned |

**Campaign verdict:** F10 on `development @ 4a932d0cf` is **BLOCKED by tree-drift**, not by a broad semantic regression in ECAP C1–C4 on the current product tree.

---

## Predecessor-owned families — Wave 1 routing

F10 **does not** emit separate semantic FAILs for **HIST-F-BND-03B**, **HIST-F-DUR-01**, **T-IGX-030-C**, or **032-A** paths at HEAD when drift is stripped (C1–C4 PASS). Wave 1 **implement** routing remains per Wave 0 presence SSOT, not overridden by this matrix:

| Residual | Owner ticket | F10 semantic FAIL? | Wave 1 action |
|----------|--------------|--------------------|---------------|
| HIST-F-BND-03B (`product_manifest_registry` dynamic Tier-0→Tier-3) | **T-IGX-034-B** | No (at HEAD probe) | **implement** |
| HIST-F-DUR-01 (pinning `is_durable` semantics) | **T-IGX-034-C** | No (at HEAD probe) | **conditional implement** |
| T-IGX-030-C / invoker terminalization | **T-IGX-034-D** | No | **qual-only** |
| REPLAY mechanical gates | **T-IGX-034-E** | Not isolated in this slice | per plan |
| QUAL-X Pop B–D | **T-IGX-034-G** | — | serial qual |
| DIST-SCALE Q-03 | **T-IGX-034-H** | — | audit |

---

## Recommended next fix batch

1. **T-IGX-034-F (continuation)** — Operator: choose **replay @ `dc18e36d`** worktree *or* schedule **manifest SHA advance** after Wave 1 product batches; commit deliberate matrix render when policy allows (current matrix diff is drift-true, not semantic-PASS).  
2. **Parallel implement** — **T-IGX-034-B** (BND-03B), **T-IGX-034-C** (DUR-01 if qual confirms); **034-D/E/G/H** qual batches per WBS.  
3. **Do not** claim B6-F10 QUALIFIED / independent acceptance on `4a932d0cf` without P03 identity satisfied.

## Pinned replay — `f10-pin` @ `dc18e36d` (2026-11-10)

Detached worktree `D:/Projekty/_wt/intergrax/f10-pin` @ `dc18e36d971c93c8f631fbc86f73149d1b81b4d7`. F10 reads in-tree `MANIFEST` `TEST_EXECUTION_SHA` **`496016c0`** at this commit (not the PRE-ENT pin field on live `development`). Generator exit **0**; matrix **not** merged to `development`.

| Signal | Value |
|--------|-------|
| **`identity_violations`** | **4 only (not zero)** |
| **TREE_DRIFT obligation FAILs** | **182** |
| **SEMANTIC obligation FAILs** | **0** |
| **Disposition** | C1–C5 **BLOCKED** — **not B6 PASS** |

Violation paths: `scripts/qualification/ebh7_junit_evidence.py`, `tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`, `tests/qualification/ebh_7/_semantic_disposition_registry.py`, `tests/qualification/ebh_7/test_ebh_7_b6_junit_skip_disposition_gates.py`.

Full record: [`WAVE1_F10_PINNED_REPLAY.md`](WAVE1_F10_PINNED_REPLAY.md).

## Cross-links

- Wave 0 presence: [`WAVE0_PRESENCE_RECONCILIATION.md`](WAVE0_PRESENCE_RECONCILIATION.md)  
- Generated matrix: [`../EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](../EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md)  
- Ticket: [`governance/tickets/T-IGX-034-F.md`](../../../governance/tickets/T-IGX-034-F.md)
