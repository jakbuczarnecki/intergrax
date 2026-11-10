# PRE-ENT Wave 1 — EBH-7 B6-F10 status (T-IGX-034-F)

**Campaign:** `pre_ent_enterprise_certification`  
**Date:** 2026-11-10  
**Executor:** T-IGX-034-F (orchestrator/implementer — **no B6 PASS**, no independent acceptance)

## Evidence SHAs

| Role | SHA | Note |
|------|-----|------|
| **Matrix render execution** | `4a932d0cf5487b72e433805e372e3855fcd3d987` | `development` @ double render of `render_ebh7_f10_normative_matrix.py` (~8–10 min each, exit 0) |
| **`development` tip (live)** | `f97d286b89b90d2f58b3c6cd288d629e3237bdaf` | After BND-03B product batch (`01b3e1961`) + local integration note (`f97d286b8`); **not** the render HEAD unless re-run |
| **EBH-7 P03 `TEST_EXECUTION_SHA`** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` | SSOT in `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json`; F10 PROFILE_BOUND attestation anchor |

Re-run F10 generator on **`f97d286b8+`** after Wave 1 product merges if matrix publication must reflect live `development`.

## Generator (usage)

```text
uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py
uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py --check
```

Writes / refreshes:

- `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`
- F10 computed block inside `docs/project/maintainers/qualification/EBH_7_ECAP_EXECUTABLE_CHECKLIST.md` (C5 upstream slice excludes C5 from ECAP block)

## Computed disposition @ render HEAD (`4a932d0cf`)

**`EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS`** — not qualified, not B6 acceptance.

| Gate | Verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |
|------|---------|-------------|------|-------------------|------|
| **C1** | BLOCKED | 32 | 0 | 4 | 28 |
| **C2** | BLOCKED | 43 | 0 | 0 | 43 |
| **C3** | BLOCKED | 105 | 0 | 6 | 99 |
| **C4** | BLOCKED | 9 | 0 | 0 | 9 |
| **C5** | BLOCKED | 3 | 0 | 0 | 3 |

**Total obligation FAIL rows:** 182 (= 28+43+99+9+3).

## TREE_DRIFT vs SEMANTIC classification

### Executable tree identity (P03 / F10 PROFILE_BOUND)

On `development` @ `4a932d0cf`, `build_canonical_f10_inputs()` reported **`identity_violations` = 639** paths changed between `TEST_EXECUTION_SHA` `dc18e36d` and executed HEAD, **outside** the B5 allowlist. Leading buckets: **`.claude/**`, **`.cursor/**`, `agent-env/**`, governance/docs/tickets from post-B6 merges — not a single product-file defect list.

### Obligation-level FAIL attribution @ render

| Class | Count | Basis |
|-------|-------|--------|
| **TREE_DRIFT** | **182** | Every obligation row with `| FAIL |` in the regenerated matrix cites `executed tree differs from TEST_EXECUTION_SHA` (or equivalent attestation) in **Missing / reasons** |
| **SEMANTIC** | **0** | No obligation FAIL row on the regenerated matrix lacks the tree-drift attestation string (automated row scan @ `4a932d0cf`) |

**Interpretation:** On a **dirty / advanced** checkout, F10 correctly **BLOCKS** with PROFILE_BOUND failures. This is **not** evidence that all 182 obligations are semantically missing at `dc18e36d`; it is evidence that **current tree ≠ TEST_EXECUTION_SHA** for attestation purposes.

### Clean checkout requirement (gates unchanged)

- **Do not weaken** F10 gates, allowlists, or `TEST_EXECUTION_SHA` rules for PRE-ENT convenience.
- **Authoritative semantic recompute** requires either:
  1. **Detached clean worktree** at `TEST_EXECUTION_SHA` `dc18e36d` (or CI-identical tree), then `render_ebh7_f10_normative_matrix.py` + MANIFEST-profile replay; or
  2. A **formal P03 SHA ladder advance** after PRE-ENT correctives (separate governed batch — not Wave 1 default).
- Local agent adapter trees (`.claude/`, `.cursor/`) must be absent or unchanged on the execution tree used for attestation.

### Predecessor-owned families (Wave 1 routing if semantic FAIL appears after clean replay)

After clean replay, any remaining **SEMANTIC** FAIL should route by family / gate ownership (from Wave 0 presence SSOT):

| Failure domain | Likely gate | Wave 1 owner ticket |
|----------------|-------------|------------------------|
| BND-03B manifest Tier-0→Tier-3 edge | C1 `FRZ-BND` / tier imports | **T-IGX-034-B** (implemented on `f97d286b8+`; re-validate on clean SHA ladder) |
| DUR-01 / skill pinning durability | C3 concerns + C4 deltas | **T-IGX-034-C** |
| Invoker 030-C / TOOL-LIFE terminalization | C1/C3 product gates | **T-IGX-034-D** (qual-only if already on HEAD) |
| REPLAY / ROADMAP-REPLAY C4 path class | C4 | **T-IGX-034-E** |
| QUAL-X Pop B–D | C1/C3 qualification surfaces | **T-IGX-034-G** |
| DIST-SCALE Q-03 | C3/C2 surfaces | **T-IGX-034-H** |

**At dirty `4a932d0cf` render:** no semantic obligation IDs isolated — routing above is **conditional on clean replay**, not deduced from tree-drift BLOCKED counts.

## C1–C5 gate verdict summary (operator)

| Gate | Verdict | Wave 1 read |
|------|---------|-------------|
| C1 | BLOCKED (28 FAIL, 4 N/A) | PROFILE_BOUND drift; 4 downstream FRZ deferrals unchanged |
| C2 | BLOCKED (43/43 FAIL) | All DUP-X / RF delta rows fail attestation on dirty tree |
| C3 | BLOCKED (99 FAIL, 6 N/A) | Inventory/concern rows fail attestation; 6 N/A dispositions preserved |
| C4 | BLOCKED (9/9 FAIL) | Path-class regression bindings fail attestation |
| C5 | BLOCKED (3/3 FAIL) | Publication/sync obligations fail under drift |
| **B6-F10** | **BLOCKED** | **Not PASS** — independent B6 acceptance not claimed |

## Recommended next fix batch

1. **T-IGX-034-F (this ticket):** clean worktree @ `dc18e36d` (or updated ladder SHA after operator decision); rerun generator; classify **SEMANTIC** FAIL set; refresh matrix only via generator — no manual PASS prose.
2. **Re-render on `f97d286b8+`** once BND-03B and replay batches land, so C4 deltas and tier gates reflect current product tree (still subject to TEST_EXECUTION_SHA attestation rules).
3. **Parallel Wave 1 product/qual:** **034-B** (BND — largely landed), **034-E** (REPLAY), **034-G/H** (QUAL-X / DIST-SCALE) — serial where plan demands; feed semantic failures back into F10 after SHA discipline.
4. **Defer:** ECAP §14 / matrix **publication closure** (**034-I**) until F10 semantic block is understood on clean execution tree.

## Artifacts touched by render @ `4a932d0cf`

- `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` (regenerated; header states **Not B6 PASS**)
- `EBH_7_ECAP_EXECUTABLE_CHECKLIST.md` F10 computed block (regenerated)

## Related SSOT

- Wave 0 presence: [`WAVE0_PRESENCE_RECONCILIATION.md`](WAVE0_PRESENCE_RECONCILIATION.md)
- Ticket: `governance/tickets/T-IGX-034-F.md`
