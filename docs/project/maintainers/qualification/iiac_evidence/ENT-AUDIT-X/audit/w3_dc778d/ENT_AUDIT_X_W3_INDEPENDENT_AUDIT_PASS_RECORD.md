# ENT-AUDIT-X — W3 independent audit PASS record (Batch A)

**Verdict:** `PASS_SCOPED`  
**Record type:** independent auditor publication under `IIAC-2026-11-05.1` (`IN_FORCE`); **not** formal stage closure, **not** `OD-ENT-08`, **not** FRZ-AUD promotion.

## 1. Identity

| Field | Value |
| --- | --- |
| Stage | `ENT-AUDIT-X` |
| Wave | W3 independent audit (Session D) |
| Profile | `iiac-v1` |
| Auditor role | `independent-quality-auditor` |
| Auditor harness session (publication) | `7c58500c-17c9-42a8-acbc-4b0e05487751` |
| Prior dossier session (adopted evidence) | `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` |
| Assurance level | `process_enforced` |

## 2. Exact SHAs

| Role | SHA |
| --- | --- |
| `audited_implementation_sha` = `test_execution_sha` | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| `evidence_publication_sha` (B5 FULL) | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| W2 normal FREEZE publication | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| `audit_execution_sha` | `dc778d125fd7e7205edd30fbbf144ca12320b0ee` |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | unset (W4/W5 ladder) |
| `ACCEPTED_BASELINE_SHA` | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` (pin; no new promotion) |

## 3. Result (summary)

| Check | Result |
| --- | --- |
| B5 FULL @ TES′ | `verify_b5_publication_tree` → 0 errors |
| B5 FREEZE normal @ TES′ | 1 naming-prefix error; bytes OK (F-W3-001) |
| L3 FULL / FREEZE | 85/85/0 certifying @ published bundle |
| Product delta TES′→`dc778d12…` | empty on `intergrax/`, `agents/`, `applications/` |
| IIAC policy | `IN_FORCE` @ `IIAC-2026-11-05.1` |

## 4. Mandatory limitations

1. **LIM-dossier-adoption:** operational verdict adopted from session `c8e4a91f…`; publication session `7c58500c…` performs docs-only IIAC binding.  
2. **LIM-SCOPED:** F-W3-001, F-W3-005, F-W3-006 remain scoped until W4-02 / certifier.  
3. **LIM-no-closure:** no `OD-ENT-08`, no `frz_aud_status=CLOSED`, no FRZ-AUD row promotion.  
4. **LIM-B6-pin:** `W3_INDEPENDENT_AUDIT_PUBLICATION_SHA` set via ladder after this commit.

## 5. Explicit non-actions

No roadmap CLOSED · no accepted-baseline promotion · no certifier W4 verdict in this batch.

## 6. Evidence directory

`docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/`
