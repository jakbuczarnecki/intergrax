# EBH-1-R2 — Peer Authority Registry Count & Mechanical Integrity Closure

**Program:** Enterprise Boundary Hardening (EBH)  
**Task:** EBH-1-R2 — Peer Authority Registry Count & Mechanical Integrity Closure  
**Type:** Documentation / qualification integrity closure (no production remediation)  
**Parent baselines:** [EBH-1](EBH-1_CANONICAL_BOUNDARY_BASELINE_RECONCILIATION.md), [EBH-1-R1](EBH-1-R1_CONTEXTVIEW_OWNERSHIP_BASELINE_CORRECTION.md), [EAC-1](ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md)

---

## 1. Scope

Synchronize declared peer authority count with the active §4.1 register after EBH-1-R1 added **PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY**; add mechanical pytest gates parsing EAC-1 SSOT. No ownership, ADR, or production changes.

---

## 2. Git provenance

| Field | SHA / note |
|-------|------------|
| **EBH1_R2_SESSION_START_HEAD** | `95b122c8ec8dce1b280edaa6da09b81df57bc13e` |
| **EBH1_R2_EVIDENCE_HEAD** | `31a51b1dd8c53bf63719f0bc5a02f8d89d97ef2b` |
| **Branch** | `development` |
| **HEAD == origin/development @ session open** | **YES** |
| **Parallel drift** | Uncommitted production edits in `intergrax/collaborative_work/` and `intergrax/contracts/` — **not staged** |

---

## 3. Root cause

EBH-1-R1 correctly split ContextView ownership and added one peer row to §4.1. Active declarations still stated **31** peer types (pre-R1 count), while the register contained **32** rows.

---

## 4. Actual peer authority count

Mechanically parsed from EAC-1 `## 4.1 Strict peer authority register` table data rows (excludes header/separator): **32**.

---

## 5. Declared count correction

| Artifact | Before | After |
|----------|--------|-------|
| EAC-1 §4.2 summary | 31 | 32 + `CURRENT_PEER_AUTHORITY_COUNT = 32` |
| EAC-1 §5.2 metric table | 31 | 32 |
| EAC-1 EAC-1R3 verification V9 | 31 types | 32 types |
| EBH-1 scope line | 31 peer authority types | 32 |

EBH-1-R1 record had no stale numeric peer count (no change).

---

## 6. Peer table parsing method

`tests/unit/docs/_ebh_eac1_peer_authority_registry_support.py` locates §4.1, parses markdown table rows until the next `##` heading, strips formatting, ignores separator rows, and maps Authority Type / Canonical Owner / Competing Peer Owner columns.

---

## 7. Unique authority type proof

Gate: `len(types) == len(set(types))` on parsed §4.1 authority types. **duplicate peer authority names = 0**.

---

## 8. Single canonical owner proof

Each row must have a non-empty canonical owner cell; `canonical_owner_is_singular()` rejects slash/ampersand/`and` multi-owner patterns while allowing scoped labels such as `COLLABORATIVE_WORK (MP-5)`. **ADR REQUIRED** in Competing Peer Owner is not treated as a second canonical owner.

---

## 9. Subordinate authority exclusion

§4.C table parsed separately; **subordinate authority rows = 1** (`INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY`). Must not appear in §4.1 peer set.

---

## 10. Deprecated alias exclusion

Deprecated labels from EAC-1 prose (`LIFECYCLE_AUTHORITY`, `ORCHESTRATION_AUTHORITY`, `CONTEXT_AUTHORITY`, `DISTRIBUTION_AUTHORITY`, `RETRIEVAL_AUTHORITY`, `COMPOSITION_AUTHORITY`, …) must not appear as active §4.1 authority types.

---

## 11. ContextView split regression

Preserved:

- `CONTEXT_ASSEMBLY_AUTHORITY` → `CONTEXT_ENGINEERING` (exactly once)
- `PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY` → `COLLABORATIVE_WORK (MP-5)` (exactly once)

EBH-1-R1 gates remain in `test_ebh1_r1_contextview_ownership_baseline_gates.py`.

---

## 12. Existing HIGH findings preservation

| ID | Status |
|----|--------|
| EBH-F-H-001 | **OPEN** |
| EBH-F-H-002 | **OPEN** |
| EBH-F-H-003 | **OPEN** / ADR REQUIRED |
| EBH-F-H-004 | **RESOLVED** (EBH-1-R1) |

ADR-GOV-01 / GOVERNANCE_AUTHORITY / WORKSPACE_COMPOSITION_AUTHORITY rows unchanged.

---

## 13. Production impact

**production code changed = NO** (docs + qualification tests only).

---

## 14. Tests

```bash
uv run pytest \
  tests/unit/docs/test_ebh1_r1_contextview_ownership_baseline_gates.py \
  tests/unit/docs/test_ebh1_r2_peer_authority_registry_integrity.py \
  -q
```

---

## 15. Final verdict

**EBH-1-R2 — CLOSED / CERTIFIED** (`EBH1_R2_EVIDENCE_HEAD` = `31a51b1dd8c53bf63719f0bc5a02f8d89d97ef2b`).

---

## 16. EBH-2 readiness

**YES** — declared peer count matches §4.1 register with mechanical enforcement; ownership baseline from R1 intact.

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
