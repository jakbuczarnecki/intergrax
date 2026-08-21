# LKW_PRODUCT_PROOF — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** LKW_PRODUCT_PROOF
- **Owning architecture/program:** LKW product proof / current product claim · shared public proof and evidence (`PROOFS.md`, `PROOF_RECEIPTS`)
- **Tier(s):** Tier-3 `applications/local_workspace_application/` (product proof docs, certification matrix, quickstart); Tier-0/Tier-1 `scripts/proof/`, `intergrax/proofs/receipts/` (canonical manifest, suite runner, ProofReceipt)
- **audited_sha:** `563076c553fd7b9d2611b71fd4137b8164a58d81`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `applications/local_workspace_application/docs/ARCHITECTURE.md`
  - `docs/project/architecture/PROOF_RECEIPTS.md`
- **Plan doc(s):**
  - `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`
  - `docs/project/maintainers/plans/PROOF_RECEIPTS.md`
- **Scope in:**
  - LKW product proof claim ownership and public proof narrative honesty
  - canonical ProofManifest / suite runner governance
  - ProofReceipt execution provenance and certification freshness
  - public proof profile semantics (`quick` vs Product Quick Start)
  - historical vs current evidence distinction
  - positive controls: LKW as primary PRODUCT proof; Product Quick Start on real LKW path; central ProofManifest authority; provider-neutral ProofReceipt/DocumentStore; no second proof framework
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - proof reruns
  - creating a second LKW proof framework or new platform proof runtime
  - weakening bounded historical PASS evidence
- **Prior audit reference(s):** [`PLATFORM_FOUNDATION`](PLATFORM_FOUNDATION.md) (PF-PROOF-INTEGRITY); [`PROOF_RECEIPTS`](../project/architecture/PROOF_RECEIPTS.md) closed wave PROOF-RECEIPTS-1A–1E
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `—`

## Scope / ownership mapping

| Concept | Canonical ownership |
|---------|---------------------|
| Audit unit (Protocol v2 layer code) | **LKW_PRODUCT_PROOF** |
| LKW product proof / current product claim | `applications/local_workspace_application/docs/ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, `proof/LKW_PLATFORM_PROOF.md` |
| Shared public proof / evidence | `docs/project/proofs/PROOFS.md`, `docs/project/architecture/PROOF_RECEIPTS.md`, `docs/project/maintainers/plans/PROOF_RECEIPTS.md` |
| Canonical proof manifest / suite runner | `scripts/proof/intergrax_proof_manifest.py`, `scripts/proof/intergrax_proof_runner.py`, `scripts/proof/run-intergrax-proof-suite.py` |
| ProofReceipt contract / store | `intergrax/proofs/receipts/contracts.py`, `store.py`, `document_store.py` |
| Per-layer report | `docs/audit_results/2026-08-18/LKW_PRODUCT_PROOF.md` |
| LKW arch target invariants | `applications/local_workspace_application/docs/ARCHITECTURE.md` — [Protocol v2 LKW product proof target invariants (2026-08-18)](#protocol-v2-lkw-product-proof-target-invariants-2026-08-18) |
| Receipt arch target invariants | `docs/project/architecture/PROOF_RECEIPTS.md` — [Protocol v2 proof receipt target invariants (2026-08-18)](#protocol-v2-proof-receipt-target-invariants-2026-08-18) |

## Executive summary

**Verdict: FAIL.** Five accepted HIGH and one accepted MEDIUM finding show that public-evidence-eligible proof can execute modified/uncommitted code while displaying HEAD SHA as apparent source identity; canonical `ProofReceipt` lacks mandatory execution provenance; checked-in certification matrix is historical without mechanical invalidation when relevant source changes; `--profile live` can return shell success via `PASS_WITH_BLOCKED` when required live proofs did not execute; Governed Evidence Decision Proof (`advanced_flagship_proof`) sits outside the canonical manifest and public reference validation; and `--profile quick` does not execute the primary LKW Product Quick Start despite public naming ambiguity. Positive controls: LKW remains the primary PRODUCT proof; Product Quick Start uses the real LKW application path without `lab_application` / `echo.basic` prerequisite; implementation ≠ proof ≠ production ≠ user validation ≠ commercial validation is preserved; product status remains Backend Product Alpha / MVP; complete indexed + authorized-live Hybrid Ask, real-user validation, and commercial validation remain explicitly unclaimed; OS matrix correctly states native Linux, macOS, and full Linux multi-phase Core proof are not certified; historical proof artifacts remain valid historical evidence; central ProofManifest and provider-neutral ProofReceipt abstraction are the correct future authority. Remediation is **ACCEPTED / PLANNED**, not implemented.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-LKW_PRODUCT_PROOF-01 (LKW-PROOF-01)

- **Severity:** HIGH
- **Category:** PROOF PROVENANCE / SOURCE IDENTITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY
- **Claim falsified:** Public-evidence-eligible proof requires exact source identity; dirty worktree disqualifies public evidence promotion.
- **Observation:** Master proof runner reads HEAD and git dirty status. `SuiteReceipt` records both `git_commit_sha` and `git_dirty`. But dirty worktree does not disqualify PASS or public-evidence eligibility. `suite_exit_code` returns success for PASS regardless of `git_dirty`. A proof can therefore execute modified/uncommitted code while still displaying the current HEAD SHA as the apparent source commit.
- **Location:**
  - `scripts/proof/intergrax_proof_runner.py` — HEAD/dirty capture; `suite_exit_code`; public-evidence eligibility
  - `scripts/proof/run-intergrax-proof-suite.py` — suite orchestration
- **Impact:** Public proof evidence can misrepresent the exact code tree executed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LKW_PRODUCT_PROOF-02 (LKW-PROOF-02)

- **Severity:** HIGH
- **Category:** PROOF RECEIPT / EXECUTION PROVENANCE
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY
- **Claim falsified:** Canonical `ProofReceipt` includes mandatory execution provenance: source revision/tree identity, dirty/exact-source posture, proof contract/version, application/build identity, environment/profile fingerprint.
- **Observation:** LKW platform proof declares persisted `ProofReceipt` documents authoritative. Canonical `ProofReceipt` has `proof_id`, `proof_kind`, `application_id`, `result`, `recorded_at`, `run_id`, `correlation_id`, `task_id`, `provider_evidence`, `domain_evidence`, `guardrails`, `metadata` — but no mandatory source revision, source tree digest, dirty state, application/build identity, image digest, proof contract version beyond receipt schema, or environment/profile fingerprint. `ProofReceiptStore` lookup is `application + proof_kind + run_id`.
- **Location:**
  - `intergrax/proofs/receipts/contracts.py` — `ProofReceipt` model
  - `intergrax/proofs/receipts/store.py` — `ProofReceiptStore` lookup key
  - `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` — receipt authority claims
- **Impact:** Receipts cannot mechanically bind recorded outcomes to the code/build/environment actually executed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LKW_PRODUCT_PROOF-03 (LKW-PROOF-03)

- **Severity:** HIGH
- **Category:** CERTIFICATION FRESHNESS / CLAIM VALIDITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY
- **Claim falsified:** Current certification requires a validity envelope matching present source/dependency closure; relevant change marks certification `STALE_REVALIDATION_REQUIRED` without rewriting historical PASS.
- **Observation:** Current checked-in LKW platform certification matrix was generated 2026-07-21 from commit `4847e957...`. Windows certification source is `6b71a841...`; Linux Docker source is `40a73fbb...`. Audited current source is `563076c553...`. Public LKW platform proof still presents Windows/native and Linux-Docker bounded live certification. No mechanical invalidation/revalidation rule binds current certification state to changes in the proof's relevant source/dependency closure.
- **Location:**
  - `applications/local_workspace_application/docs/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json`
  - `applications/local_workspace_application/docs/proof/LKW_PLATFORM_CERTIFICATION_MATRIX.md`
  - `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` — certification presentation
- **Impact:** Historical certification can be read as silently current after relevant code changes.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LKW_PRODUCT_PROOF-04 (LKW-PROOF-04)

- **Severity:** HIGH
- **Category:** PROOF SUITE / FALSE SUCCESS
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-EXECUTION-QUALIFICATION-INTEGRITY
- **Claim falsified:** For `--profile live`, missing required live proof yields non-success/incomplete exit; optional proofs skip only when manifest marks them optional.
- **Observation:** Proof runner maps unsatisfied requirements to `BLOCKED_ENVIRONMENT`. `aggregate_overall_status` maps blocked environment in LIVE profile to `PASS_WITH_BLOCKED`. `suite_exit_code` maps `PASS_WITH_BLOCKED` to `0`. An explicitly requested live suite can return shell success while one or more selected live proofs did not execute because required provider/env requirements were missing.
- **Location:**
  - `scripts/proof/intergrax_proof_runner.py` — `aggregate_overall_status`, `suite_exit_code`, `BLOCKED_ENVIRONMENT`
  - `scripts/proof/run-intergrax-proof-suite.py`
  - `scripts/proof/intergrax_proof_manifest.py` — required vs optional membership
- **Impact:** Process exit 0 can be inferred as complete live certification when required proofs were blocked.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LKW_PRODUCT_PROOF-05 (LKW-PROOF-05)

- **Severity:** HIGH
- **Category:** PROOF GOVERNANCE / SECOND PUBLIC PROOF PATH
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-EXECUTION-QUALIFICATION-INTEGRITY
- **Claim falsified:** Every publicly executable proof has one canonical `ProofManifestEntry` with canonical `proof_id`, command, profile, safety class, requirements, timeout, and `public_evidence_eligible`.
- **Observation:** `PROOFS.md` states canonical proof membership lives in `scripts/proof/intergrax_proof_manifest.py`. Public LKW proof exposes Governed Evidence Decision Proof with execution identity `advanced_flagship_proof` and direct module command. That proof is not registered in the canonical manifest. Public proof-reference checker scans `**Proof:**` / `**Proofs:**` lines and accepts only uppercase-hyphen IDs matching `[A-Z][A-Z0-9-]*`. Lowercase underscore `advanced_flagship_proof` sits outside canonical public-proof reference validation.
- **Location:**
  - `docs/project/proofs/PROOFS.md` — Governed Evidence Decision Proof entry
  - `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` — `advanced_flagship_proof` command
  - `scripts/proof/intergrax_proof_manifest.py` — manifest membership
  - `scripts/proof/public_proof_references.py` — reference validation pattern
- **Impact:** Second public flagship proof path bypasses manifest governance and reference checking.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LKW_PRODUCT_PROOF-06 (LKW-PROOF-06)

- **Severity:** MEDIUM
- **Category:** PROOF PROFILE / REVIEWER SEMANTICS
- **Status at publication:** ACCEPTED
- **Remediation block:** LKW-PROOF-REVIEWER-SEMANTICS-INTEGRITY
- **Claim falsified:** Proof profile semantics are explicit: `QUICK` either includes bounded flagship LKW product smoke or cannot be interpreted as Product Quick Start coverage.
- **Observation:** Public Product Quick Start is the primary supported product-evaluation proof. Manifest registers all three `LKW-PRODUCT-QUICKSTART-*` entries only for FULL/LIVE. `--profile quick` therefore does not execute the primary LKW product quickstart. Public proof dashboard separately advertises the repository quick suite.
- **Location:**
  - `scripts/proof/intergrax_proof_manifest.py` — `LKW-PRODUCT-QUICKSTART-*` profile membership
  - `docs/project/proofs/PROOFS.md` — quick suite advertisement
  - `applications/local_workspace_application/scripts/run-lkw-product-quickstart.py` — Product Quick Start entry
- **Impact:** Reviewers can infer Product Quick Start coverage from `--profile quick` when it is not executed.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| LKW remains primary PRODUCT proof, not merely platform demo | NOT falsified |
| Product Quick Start uses real LKW application path | NOT falsified |
| No `lab_application` / `echo.basic` prerequisite | NOT falsified |
| implementation ≠ proof ≠ production ≠ user validation ≠ commercial validation | NOT falsified |
| Product status remains Backend Product Alpha / MVP | NOT falsified |
| Complete indexed + authorized-live Hybrid Ask explicitly unclaimed | NOT falsified |
| Real-user and commercial validation explicitly unclaimed | NOT falsified |
| OS matrix: native Linux not certified; macOS not certified; full Linux multi-phase Core proof not certified | NOT falsified |
| Historical proof artifacts remain historical evidence | NOT falsified |
| Central ProofManifest is correct future authority | NOT falsified |
| Provider-neutral ProofReceipt/DocumentStore abstraction remains correct | NOT falsified |
| No second proof framework required | NOT falsified |

## Historical vs current evidence distinction

Historical PASS receipts, certification JSON artifacts, and reviewer evidence from earlier source revisions remain **valid historical evidence**. Protocol v2 findings require explicit **current vs historical** semantics: a historical PASS does not automatically qualify as **current** certification or public-evidence eligibility until a validity envelope (source revision/tree, dependency closure, proof contract version, environment profile) matches the present tree. Remediation must preserve historical truth — never rewrite old evidence as false.

## Duplicate ownership / cross-links

| Existing finding / domain | Relationship |
|---------------------------|--------------|
| **PLATFORM_FOUNDATION / PF-PROOF-INTEGRITY** | Tier-boundary and foundation proof integrity — coordinate; LKW-PROOF owns product/public proof claim honesty |
| **PROOF_RECEIPTS closed wave** | Provider-neutral receipt storage — extend with provenance invariants; LKW must not invent private receipt contract |
| **OBSERVABILITY_EVIDENCE** | Evidence durability — coordinate on public-evidence promotion semantics |

## Root-cause remediation grouping

### LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY — exact source/build/environment binding

**Priority:** P0  
**Findings:** LKW-PROOF-01, LKW-PROOF-02, LKW-PROOF-03  
**Owner:** `PROOF_RECEIPTS` + shared proof infrastructure; LKW consumes that authority  

Every promoted/current product proof must be mechanically tied to the code/build/environment it actually executed. Historical evidence becomes stale rather than silently current when relevant code changes. Primary owner: `docs/project/architecture/PROOF_RECEIPTS.md` and `docs/project/maintainers/plans/PROOF_RECEIPTS.md`.

### LKW-PROOF-EXECUTION-QUALIFICATION-INTEGRITY — manifest-owned suite success

**Priority:** P0/P1  
**Findings:** LKW-PROOF-04, LKW-PROOF-05  
**Owner:** shared proof manifest/runner (`scripts/proof/`)  

Canonical proof manifest owns every public executable proof. Suite success means all required requested evidence actually executed. No second flagship proof path. Fold Governed Evidence Decision Proof into existing manifest/reference governance.

### LKW-PROOF-REVIEWER-SEMANTICS-INTEGRITY — profile naming honesty

**Priority:** P2  
**Findings:** LKW-PROOF-06  
**Owner:** LKW plan + `PROOFS.md`  

Product Quick Start and repository proof-profile naming must communicate actual proof coverage without ambiguity.

## Architecture / plan sync state

| Doc | Section | Status |
|-----|---------|--------|
| `applications/local_workspace_application/docs/ARCHITECTURE.md` | Protocol v2 LKW product proof target invariants | SYNCED |
| `docs/project/architecture/PROOF_RECEIPTS.md` | Protocol v2 proof receipt target invariants | SYNCED |
| `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md` | LKW-PROOF-EXECUTION-QUALIFICATION-INTEGRITY, LKW-PROOF-REVIEWER-SEMANTICS-INTEGRITY | SYNCED |
| `docs/project/maintainers/plans/PROOF_RECEIPTS.md` | LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY | SYNCED |
| `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` | historical vs current certification honesty | SYNCED |
| `docs/project/proofs/PROOFS.md` | profile semantics and manifest-governance gap | SYNCED |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `563076c553fd7b9d2611b71fd4137b8164a58d81`; current `development` HEAD was not re-audited beyond persistence sync.
- Remediation not performed in this task.
- No proof reruns were executed for this persistence task.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-LKW_PRODUCT_PROOF-01` … `06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
