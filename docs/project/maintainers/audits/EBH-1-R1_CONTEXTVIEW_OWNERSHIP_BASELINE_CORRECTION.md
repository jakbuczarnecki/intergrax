# EBH-1-R1 — ContextView Ownership Baseline Correction

**Program:** Enterprise Boundary Hardening (EBH)  
**Task:** EBH-1-R1 — ContextView Ownership Baseline Correction  
**Type:** Documentation reconciliation only (no production remediation)  
**Parent baseline:** [EBH-1](EBH-1_CANONICAL_BOUNDARY_BASELINE_RECONCILIATION.md)

---

## 1. Scope

Correct the sole semantic inconsistency in EBH-1: conflation of **CONTEXT_ASSEMBLY_AUTHORITY** (Context Engineering) with **Principal-scoped ContextView** authority (Collaborative Work / MP-5). Preserve all other EBH-1 findings (Nexus public leaks, agent imports, ADR-GOV-01).

**Out of scope:** MP-5 runtime, CE runtime, UCL, Memory, RAG implementation changes; new ADRs.

---

## 2. Git provenance

| Field | SHA / note |
|-------|------------|
| **EBH1_R1_SESSION_START_HEAD** | `190af29cc46d358f84c8d66254445284eae0171a` |
| **EBH1_R1_EVIDENCE_HEAD** | *(commit SHA of this correction)* |
| **Branch** | `development` |
| **HEAD == origin/development @ session open** | **YES** (`190af29c…`) |
| **Git safety** | No reset / rebase / stash / clean / amend / force-push |

---

## 3. Original inconsistency

EBH-1 assigned `CONTEXT_VIEW_COMPOSITION_AUTHORITY` → CONTEXT_ENGINEERING and stated CE owns ContextView composition contracts. Canonical MP-5 architecture and ADR freeze **Principal-scoped ContextView** under COLLABORATIVE_WORK / MP-5. Generic CE assembly is a **separate** authority.

**Finding:** [EBH-F-H-004](EBH-1_CANONICAL_BOUNDARY_BASELINE_RECONCILIATION.md#22-findings) — **RESOLVED BY EBH-1-R1**.

---

## 4. Canonical ADR evidence

| ADR | Role |
|-----|------|
| [ADR-MP-006](../../technical/adr/entries/2026-09-17/ADR-MP-006.md) **Accepted** | Principal-scoped ContextView ownership — **FROZEN** |
| [COLLABORATIVE_WORK.md](../../architecture/COLLABORATIVE_WORK.md) § Principal-scoped ContextView (MP-5) | Pipeline, anti-substitution, source ports |

---

## 5. Context ownership matrix

| Semantic responsibility | Canonical owner | Consumers |
| ----------------------- | --------------- | --------- |
| General context assembly | CONTEXT_ENGINEERING | Agents / runtime |
| Context lifecycle optimization | UNIFIED_CONTEXT_LIFECYCLE | CE / runtime |
| Memory records / recall | MEMORY | ContextView / CE / RAG |
| Knowledge retrieval | RAG / KNOWLEDGE | ContextView / CE |
| Principal visibility | COLLABORATIVE_WORK / MP-5 | ContextView pipeline |
| Principal-scoped composition semantics | COLLABORATIVE_WORK / MP-5 | Agents / runtime consumers |
| ContextView source reads | Source-domain contracts | MP-5 ports/adapters |
| ContextView default composition implementation | MP-5 / Collaborative Work | ContextView consumers |

---

## 6. Context Assembly vs ContextView distinction

**Context Engineering** owns model-facing **generic** assembly: collectors, compile plan, budgeting, composition mechanics for CE bundles — **not** Principal-scoped ContextView identity, visibility classes, or MP-5 composition semantics.

**MP-5 ContextView** owns principal visibility, collaborative scope, eligible categories/visibility classes, principal-scoped composition semantics, and ContextView identity/entry semantics. It **consumes** Memory, RAG/Knowledge, UCL, and Collaborative Work via canonical reference read contracts without absorbing their domain ownership.

---

## 7. Source-domain ownership

| Contract | Owner |
|----------|-------|
| `MemoryReferenceReadPort` | MEMORY |
| `KnowledgeReferenceReadPort` | RAG / KNOWLEDGE |
| `UclReferenceReadPort` | UNIFIED_CONTEXT_LIFECYCLE |
| `CollaborativeWorkReferenceReadPort` | COLLABORATIVE_WORK |

ContextView does not own domain truth; MP-5D ports and B5 adapters are consumer-side abstractions.

---

## 8. MP-5 ownership (frozen — not reopened)

Preserved principles:

- Domain owns source read semantics/contracts.
- MP-5D owns consumer-side source ports.
- B5 adapters translate canonical domain refs.
- ContextView does not own domain truth or hydrate source payloads.

Canonical flow unchanged (see COLLABORATIVE_WORK.md MP-5 section).

---

## 9. Updated EBH-1 rows

| EBH-1 section | Change |
|---------------|--------|
| §6 Layer matrix | Split MEMORY / RAG / CE / UCL; CW row names MP-5 ContextView |
| §7 Authority matrix | `CONTEXT_ASSEMBLY_AUTHORITY` + `PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY`; deprecate `CONTEXT_VIEW_COMPOSITION_AUTHORITY` |
| §7.1 | Ownership distinction table |
| §12 | Source-domain ref owners vs MP-5 consumer |
| §15 | CE generic assembly vs MP-5 ContextView |
| §22 | EBH-F-H-004 added / resolved |
| §24 | MP-5-first communication map |

---

## 10. EAC reconciliation

| Artifact | Action | Reason |
|----------|--------|--------|
| **EAC1** | **UPDATED** (minimal) | Add explicit `PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY`; narrow WORKSPACE row |
| **EAC2** | **UPDATED** (minimal) | EAC-CON-024…026 semantic owner → COLLABORATIVE_WORK; pre-R1 CE rows **HISTORICAL** |
| **EBH-1** | **UPDATED** | Active SSOT for boundary baseline |

No duplicate authority registry — EAC1 remains peer-authority register; EBH-1 remains cross-layer baseline SSOT.

---

## 11. Production impact

**production code changed = NO**

Evidence: `intergrax/contracts/context_view*.py`, `intergrax/collaborative_work/context_view*.py` — read-only for ownership proof; no edits in R1.

---

## 12. Validation

- `git diff --check` — required PASS
- `tests/unit/docs/test_ebh1_r1_contextview_ownership_baseline_gates.py` — documentation regression gates

---

## 13. Final verdict

**EBH-1-R1 — CLOSED / CERTIFIED** (documentation baseline correction @ evidence HEAD).

---

## 14. EBH-2 readiness

**YES** — EBH-2 may proceed without active ContextView ownership ambiguity in the EBH-1 baseline (H-001…H-003 remain open remediation items).

---

*Independent verification on GitHub @ `EBH1_R1_EVIDENCE_HEAD` remains required for production certification claims.*
