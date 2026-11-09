# ENT-AUDIT-X — IIAC independent audit PASS record

**Verdict:** `ENT-AUDIT-X — INDEPENDENT IIAC AUDIT PASS (SCOPED)` · manifest verdict `PASS_SCOPED`  
**Record type:** independent auditor record under `IIAC-2026-11-05.1` (`IN_FORCE`). Parallel to legacy Protocol 03 records; **no human signature block**.  
**Status of this record:** certifier-published auditor deliverable (W2-04 phase 1). It is **not** formal stage closure, **not** `OD-ENT-08`, and **not** a roadmap CLOSED headline update.

## 1. Identity

| Field | Value |
|---|---|
| Stage | `ENT-AUDIT-X` |
| Profile | `iiac-v1` |
| Auditor role | `independent-quality-auditor` |
| Auditor harness session | `631309c1-023b-47e2-ac78-f7c104011033` |
| Audit round | Session D, round 3 (R-AUD @ `ee5470f543d0c1494a7dd5febc41e5ff968ade72`) |
| Assurance level | `process_enforced` |
| Certifier publication | W2-04; certifier harness `d90d76e6-b899-4ad8-910e-701d53cb5d56` |

## 2. Exact SHAs

| Role | SHA |
|---|---|
| `audited_implementation_sha` = `test_execution_sha` (TES′) | `b5c2481d79f955fa5f483a9288aee8de4581f953` |
| `evidence_publication_sha` (EVIDENCE′) | `63ff80819da5a7324257ea598e342810c1838199` |
| `audit_execution_sha` | `ee5470f543d0c1494a7dd5febc41e5ff968ade72` |
| IIAC activation | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | unset until W2-05 ladder pin |
| `ACCEPTED_BASELINE_SHA` | unset (W3-02) |
| `FORMAL_CLOSURE_PUBLICATION_SHA` | unset (W5) |

## 3. Result (summary)

| Check | Result |
|---|---|
| B5 tree @ TES′ | `verify_b5_publication_tree` → 0 errors @ `autocrlf=false` |
| Qual wave / nightly | PASS; nightly run `42593448899` @ TES′ |
| Product delta TES′→audit HEAD | empty on `intergrax/`, `agents/`, `applications/` |
| Loop-1 HIGH (F-01, F-02) | FIXED_VERIFIED |
| New HIGH/BLOCKER @ TES′ | none |

## 4. Mandatory limitations (travel with verdict)

1. **LIM-process:** `process_enforced`; auditor sub-agent under orchestrator (`631309c1-023b-47e2-ac78-f7c104011033`).  
2. **LIM-carried:** F-03..F-08, F-12..F-20 remain scoped acceptance limits until W3–W5 closure objects exist.  
3. **LIM-OD-ENT-01:** promote trust boundary procedural; Matrix C rows C001,C002,C003,C042,C057,C065 stay NOT_QUALIFIED until W3-02.  
4. **LIM-OPEN-BP-01:** `verify_ent_audit_branch_protection.py` FAIL on live `development`; COMPENSATING route (F-19).  
5. **LIM-B6-pin:** formal ladder `B6_ACCEPTANCE_PUBLICATION_SHA` is pinned in W2-05; this record alone does not close ENT-AUDIT-X.

## 5. Explicit non-actions

No `OD-ENT-08` · no accepted-baseline pin · no formal CLOSED headline · no FRZ-AUD promotion · no `ACCEPTED_BASELINE_SHA`.

## 6. Evidence

Directory `docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/audit/`: dossier `IIAC_ENT_AUDIT_REPORT.md`, findings, scope, test matrix, `EVIDENCE_INDEX.json` (digests).

## 7. Certification binding

Independent audit at `audit_execution_sha` **`ee5470f543d0c1494a7dd5febc41e5ff968ade72`** with verdict **`PASS_SCOPED`** is accepted for IIAC B6 evidence publication only. W3–W6 remain open per recovery plan.
