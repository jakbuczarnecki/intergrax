# EBH-1-R3 — Derived Authority Metrics & Evidence Provenance Closure

**Program:** Enterprise Boundary Hardening (EBH)  
**Task:** EBH-1-R3 — Derived Authority Metrics & Evidence Provenance Closure  
**Type:** Documentation / qualification closeout (no production remediation)  
**Parent baselines:** [EBH-1](EBH-1_CANONICAL_BOUNDARY_BASELINE_RECONCILIATION.md), [EBH-1-R1](EBH-1-R1_CONTEXTVIEW_OWNERSHIP_BASELINE_CORRECTION.md), [EBH-1-R2](EBH-1-R2_PEER_AUTHORITY_REGISTRY_COUNT_AND_MECHANICAL_INTEGRITY_CLOSURE.md), [EAC-1](ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md)

---

## 1. Scope

Close the last formal inconsistency after EBH-1-R2: §5.2 **Peer types with exactly one canonical owner** must match the mechanical §4.1 register; record immutable **EBH1_R2_EVIDENCE_HEAD**; gate all numeric §5.2 peer metrics from one SSOT parser. No ownership, ADR, or production changes.

---

## 2. Git provenance

| Field | SHA / note |
|-------|------------|
| **EBH1_R3_SESSION_START_HEAD** | `131b5e8fac9e6c05bcc0415677dee91288691500` |
| **EBH1_R3_EVIDENCE_HEAD** | `320083db32eb71195a15d8ef3ad364af840afac5` |
| **Branch** | `development` |
| **HEAD == origin/development @ session open** | **YES** |
| **Parallel drift** | Unrelated untracked paths only — **not staged** |

---

## 3. Root cause

EBH-1-R2 synchronized **peer authority type count** (31 → 32) but left §5.2 **single canonical owner** at **31**, implying one peer row lacked a singular owner although every §4.1 row has exactly one canonical owner cell. R2 qualification also retained placeholder `EBH1_R2_EVIDENCE_HEAD`.

---

## 4. Canonical register state (@ parse)

Mechanical parse of EAC-1 `## 4.1` + `### 4.C`:

| Measure | Value |
|---------|------:|
| Peer authority rows | 32 |
| Unique authority types | 32 |
| Rows with singular canonical owner | 32 |
| Rows with missing canonical owner | 0 |
| Rows with ambiguous canonical owner | 0 |
| Rows with competing canonical owner signal | 0 |
| Subordinate authority rows | 1 |

---

## 5. Derived metrics model

From `tests/unit/docs/_ebh_eac1_peer_authority_registry_support.py`:

| Function | Semantics |
|----------|-----------|
| `count_peer_authorities` | `len(parse_peer_authority_register())` |
| `count_peer_types_with_single_canonical_owner` | Rows with non-empty canonical owner and `canonical_owner_is_singular()` |
| `count_peer_types_with_competing_canonical_owner` | Non-singular canonical owner **or** competing column signals a second canonical owner (excludes `NONE`, `ADR REQUIRED`, `PASS*`) |
| `count_subordinate_authorities` | `len(parse_subordinate_authority_register())` |
| `parse_peer_authority_coverage_metrics` | §5.2 metric table → name → integer |

---

## 6. §5.2 reconciliation

| Metric | Old §5.2 | New §5.2 | Derived from §4.1 |
|--------|---------:|---------:|------------------:|
| Peer authority types (§4.A) | 32 | 32 | 32 |
| Peer types with exactly one canonical owner | **31** | **32** | **32** |
| Peer types with competing canonical owners | 0 | 0 | 0 |
| Subordinate authority types (§4.C) | 1 | 1 | 1 |

---

## 7. Single canonical owner metric

**Definition:** row has exactly one canonical owner value (singular cell; scoped labels such as `COLLABORATIVE_WORK (MP-5)` allowed).

All **32** peer rows satisfy this; §5.2 updated to **32**. Tests derive expected value from the register — no hard-coded count in assertions beyond structural `derived == peer_total`.

---

## 8. Competing owner metric

**Definition:** actual second **canonical** owner — not an architecture conflict note.

`GOVERNANCE_AUTHORITY` and `WORKSPACE_COMPOSITION_AUTHORITY` retain **ADR REQUIRED** in **Competing Peer Owner**; these do **not** increment competing canonical owner count. **0** rows with competing canonical owners.

---

## 9. Subordinate metric

§4.C: **1** row (`INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY`); excluded from peer totals; §5.2 subordinate metric **1**.

---

## 10. R2 evidence SHA closure

| Field | Value |
|-------|-------|
| **EBH1_R2_EVIDENCE_HEAD** | `31a51b1dd8c53bf63719f0bc5a02f8d89d97ef2b` |

Verified via `git show 31a51b1dd8c53bf63719f0bc5a02f8d89d97ef2b` (R2 content commit). R3 records R2 SHA; R3 evidence HEAD recorded separately at R3 commit (no self-reference loop).

---

## 11. Mechanical gates

`tests/unit/docs/test_ebh1_r3_derived_authority_metrics_and_provenance.py` compares §5.2 declared integers to derived §4.1/§4.C values and asserts R2 evidence SHA in the R2 artifact.

---

## 12. Existing findings

| ID | Status |
|----|--------|
| EBH-F-H-001 | **OPEN** |
| EBH-F-H-002 | **OPEN** |
| EBH-F-H-003 | **OPEN** / ADR REQUIRED |
| EBH-F-H-004 | **RESOLVED** (EBH-1-R1) |

---

## 13. Production impact

**production code changed = NO** (docs + qualification tests only).

---

## 14. Tests

```bash
uv run pytest \
  tests/unit/docs/test_ebh1_r1_contextview_ownership_baseline_gates.py \
  tests/unit/docs/test_ebh1_r2_peer_authority_registry_integrity.py \
  tests/unit/docs/test_ebh1_r3_derived_authority_metrics_and_provenance.py \
  -q
```

---

## 15. Final verdict

**EBH-1-R3 — CLOSED / CERTIFIED** when evidence HEAD matches committed gates and `git diff --check` passes.

---

## 16. EBH-2 readiness

**YES** — peer register, derived §5.2 metrics, and R2 provenance are mechanically aligned; ownership baseline from R1 intact.

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
