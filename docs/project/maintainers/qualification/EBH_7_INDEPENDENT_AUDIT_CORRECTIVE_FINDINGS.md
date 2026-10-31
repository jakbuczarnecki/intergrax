# EBH-7 — Independent audit corrective findings

**Trigger:** B6 BLOCKED — automatic `proof_module` → EXACT_PROOF promotion @ closure wave (`f913944ff`).

**B6 independent audit (Wave 7 corrective closeout @ audited HEAD `d1bdcfa46`):**

| ID | Defect | Corrective evidence |
| --- | --- | --- |
| **B6-F01** | External-work “dual provider” used two instances of one in-memory class | `QualExternalWorkAlphaIntegration` / `QualExternalWorkBetaIntegration`; `prove_external_work_dual_provider_replaceability` + profile fail-closed |
| **B6-F02** | PROD-Q negative was a source-string prefix on `factory.py` | `_internal_rpl_behavioral_proofs.prove_prodq_runtime_adapter_ownership_*` via `_require_venv_production_adapter` |
| **Sweep** | FIXED memory proof exposed SQLite store API drift | `ConversationalMemory.max_messages` + `SQLiteConversationalMemoryStore.save_memory` uses `get_all()` / `str(role)` |

**Correction @ semantic disposition SSOT:**

| Finding | Resolution |
| --- | --- |
| Auto-EXACT in `_semantic_proof_binding_audit.py` | Removed; `_semantic_disposition_registry.py` authoritative |
| ECAP 26/26 + 34/34 false PASS | Restored **14+16 EXACT**, **12+18 OPEN** |
| Platform proof delegations | Documented PARTIAL with assertion inspection notes |
| W6/W7 pending==0 gates | Replaced with open-register counts + B6 XFAIL exact coverage |
| RPL 56/56 without semantic review | `_rpl_h_semantic_review.py` — 51× PARTIAL Package C |

**Remaining OPEN for B6:** independent re-audit @ new TEST_EXECUTION_SHA; OD-EBH7-08.

**Not claimed:** B6 PASS, EBH-7 CLOSED, global FRZ PASS.
