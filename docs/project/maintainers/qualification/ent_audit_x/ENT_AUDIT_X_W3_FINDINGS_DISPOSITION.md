# ENT-AUDIT-X — W3 findings disposition (orchestrator / W4 gate)

**Scope:** F-W3-001..006 from W3 independent audit dossier @ `dc778d125fd7e7205edd30fbbf144ca12320b0ee`.  
**Authority:** Status overlay only; does not promote FRZ-AUD or close ENT-AUDIT-X.

| ID | Disposition | Rationale |
| --- | --- | --- |
| F-W3-001 | **SCOPED LIMITATION** (not a mandatory closure blocker) | `verify_b5_publication_tree` expects `qualification_at_<TES>` directory prefix; `freeze_normal_at_32a963344…` carries `publication_role: W2_NORMAL_FREEZE_EVIDENCE` with correct bytes. Canonical validator extension deferred unless W4 certifier mandates; no product-path change in W4 reconcile batch. |
| F-W3-002 | **FIXED (SSOT)** | Overlay E-3 and snapshot TES/EVID aligned to W2 ladder rows in `ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md`. |
| F-W3-003 | **FIXED (SSOT)** | `ci_nightly_proof_head_sha` → `32a963344…`; run `42869847981` recorded in closure JSON. |
| F-W3-004 | **FIXED (SSOT)** | Snapshot §1 last-wins TES/EVID supersede stale R-FRESH overlay (historical rows preserved). |
| F-W3-005 | **DEFERRED W4-02** | OD-ENT-02 compensating sufficiency requires independent certifier verdict; not implementer closure. |
| F-W3-006 | **PRESERVED LIMITATION** | W3 audit relied on published L3 bundle; no claim of personal FULL/FREEZE CLI replay in auditor session. |

**F-W3-001 verification (implementer replay @ 2026-11-11):** FULL tree `qualification_at_32a963344…` → 0 errors; FREEZE tree → 1 naming-prefix error only.
