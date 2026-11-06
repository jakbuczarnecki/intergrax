# EVOL-X RA-B4-01..04 — consolidated corrective: independent delta re-audit handoff

**Ticket:** [`T-IGX-017-B`](../../../../../governance/tickets/T-IGX-017-B.md) (RA-B4 proof) · [`T-IGX-018-A`](../../../../../governance/tickets/T-IGX-018-A.md) (separate governance test hotfix)
**Policy:** `IIAC-2026-11-05.1` (`IN_FORCE`) · **Role:** implementer (`may_commit: true`; cannot certify its own correction)
**Baseline:** `9a8fa3634713f69b56ce87ceaf9ed319fad27dae` · **Audited predecessor:** `23feb2bbb017b1beb63f9368b5978384093ff924`
**`RA_B4_CORRECTIVE_IMPLEMENTATION_SHA`:** `6cf8c5a2247465a1321c239d0559d751383daec2` (bound in the record commit that adds §9; hotfix commit `1e28b9e2b77bbc44171f80017fd56b4032aa196e` is separate)
**EVOL-X status:** `NOT_CLOSED`. No B6 acceptance, no FRZ promotion, no operator decision, no roadmap update was made.

This handoff is evidence for an **independent delta re-audit**. It is not a PASS record and does not close B-4.

## 1. Original auditor evidence — preservation (mission §2)

| Item | Result |
|---|---|
| Source | `docs/project/maintainers/qualification/iiac_evidence/EVOL-X/b4_reaudit/` (untracked, 10 indexed files, **not modified**) |
| Digest verification | all 10 SHA-256 digests and byte counts in `EVIDENCE_INDEX.json` matched the files before any work started |
| Immutable copy | read-only byte-identical copy and zip outside the repository: `D:\Projekty\_intergrax_evidence_archive\EVOL-X_b4_reaudit_20261106\` (+ `.zip`, with `ARCHIVE_SHA256.json`); every copy file was compared byte-for-byte with its source |
| Auditor session id | The auditor draft records `IIAC-C-EVOL-X-B4-REAUDIT-2026-11-05__claude-code-cf31a7e9-08ee-47f4-84ca-5b420d9a377d`. A Claude Code session with the id `cf31a7e9-08ee-47f4-84ca-5b420d9a377d` exists in the local project session store, which is **consistent with** — but not proof of — the claim. The operator attestation checkbox/signature in the auditor handoff record remains **unsigned and is not completed here**. The implementation session id of T-IGX-017-A is likewise not recorded in the repository and was not invented. |
| Survives checkout/reset/clean | The tracked tree never contained the package; `git clean`/`reset`/worktree operations on the repository can still remove it from the working copy, therefore the external archive is the custody copy. |
| Publication | **Not performed.** Publishing the original evidence needs its own authorised evidence-publication path (operator-attested handoff, byte-preserving commit). The verdict belongs to the independent auditor, not to this implementer. |

An unrelated, pre-existing uncommitted change to `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` (a regenerated inventory) was present in the working tree at session start; it is not part of this change and was not staged.

## 2. Findings → changes

Changed code: [`scripts/qualification/evol_x_closure_publication_preflight.py`](../../../../../scripts/qualification/evol_x_closure_publication_preflight.py)
Changed config: [`.gitattributes`](../../../../../.gitattributes) (one rule)
Changed tests: [`tests/unit/qualification/test_evol_x_closure_preflight_b4.py`](../../../../../tests/unit/qualification/test_evol_x_closure_preflight_b4.py)

### RA-B4-01 — fresh checkout / CRLF (A-14)

*Reproduced* in a fresh `git worktree` checkout of `9a8fa3634` with `core.autocrlf=true`: `git ls-files --eol` reports `i/lf w/crlf attr/` for `evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` while the auditor copy (under `iiac_evidence/**`) is `-text`; Gate A emitted the three A-14 errors (differs from auditor copy; digest mismatch; differs from HEAD).

Bytes compared: the working-tree pass record vs (a) the byte-preserved auditor copy, (b) `pass_record_digest` of the certification record, (c) the `HEAD` blob. The comparison was evidence-independent: only the checkout configuration decided it.

Contract now (documented in the module docstring):

1. **Canonical bytes = the Git blob at `HEAD`.** `_read_bytes` returns the blob when the working-tree file is byte-identical, or — only where `git check-attr text` is not `unset` and the blob is pure LF — when it is exactly `blob.replace(LF, CRLF)` (the transformation of a converting checkout). Everything else (edited content, partial conversion, CRLF in a `-text` path, uncommitted edits) is returned as-is and fails the digest/blob comparisons.
2. **`.gitattributes`**: `docs/.../evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md -text` (exactly one path; global conversion is untouched). Fresh checkouts now materialise the pass record at its blob bytes.
3. No integrity check was removed; no auditor/indexed artefact was normalised or rewritten.

One-time note for **stale** pre-existing clones: Git now treats a CRLF pass record as modified (`-text`); refresh it with `git checkout -- <path>` (or `git add --renormalize`).

### RA-B4-02 — evidence-bound blocker closure (A-18 / B-09)

* The immutable baseline is the `open_blockers_for_closure` list **at `certification_publication_sha`** (read from Git). Stable ids: `B-<n>`; `B6-P3` for the unlabeled P3 blocker; `TXT-<sha256[:12]>` otherwise. No new source of truth: the certification record stays the register.
* `open_blockers_for_closure` is required (missing/null/non-list = FAIL). Every baseline blocker must be **still open** or have a `resolved_blockers_for_closure` entry; removal, renaming, omission, duplicate ids and open+resolved-at-once all fail with `A-18 blocker lifecycle: …` (and `B-09` in Gate B). Status text alone (`RESOLVED`) never resolves.
* A resolution entry (`status: RESOLVED_WITH_EVIDENCE`) is re-verified independently:
  * `baseline_text_sha256`, `opened_at_sha` = certification publication, `resolution_kind` by blocker (`B-4` → `INDEPENDENT_REAUDIT`; `B-5`, `B6-P3` → `OPERATOR_AUTHORIZATION`), `closure_publication_sha` a commit strictly after certification and a strict ancestor of `HEAD`.
  * **`INDEPENDENT_REAUDIT`**: a dedicated, indexed evidence directory (not the Session C audit/certification dirs) verified by the existing index routine at the closure SHA; re-audit manifest digest-bound; canonical `verify_iiac_audit_manifest` + `verify_iiac_role_separation --strict-product-diff` run on it; stage/policy/profile/assurance; verdict `PASS`/`PASS_SCOPED`, not a draft; three distinct non-PENDING sessions and distinct agents; auditor = `independent-quality-auditor`, certifier = `certification-governance-agent`; `audited_implementation_sha` contains `resolved_implementation_sha` (itself a strict descendant of certification) and precedes the closure; report/findings digest-bound; no unresolved blocking finding (class/severity or any finding with a non-empty `blocks`); operator-attested handoff; an explicit `certifier_disposition` (`ACCEPTED`, matching certifier session/agent/policy and the audit digests).
  * **`OPERATOR_AUTHORIZATION`**: operator decision records under `iiac_evidence/EVOL-X/operator_decisions/`, present at the closure SHA, digest-bound, verified by the record contract of §RA-B4-03. `B-5` needs both OD-EVOL-12 and OD-EVOLX-08; the P3 authorization must be a distinct decision whose scope contains `B6 acceptance` and `EVOL-X`.
* Gate A verifies resolutions of every non-Gate-B-only blocker; Gate B verifies `B-5`. Gate A does not read the B6 SHA.

### RA-B4-03 — operator decision parsing (B-07)

* Allow-list (whole normalised status): `SIGNED / APPROVED`, `SIGNED/APPROVED`, `OPERATOR APPROVED`, `SIGNED`, `APPROVED`, `CONFIRMED` (markdown emphasis stripped, NFKC, zero-width characters removed; homoglyphs fail closed). Anything else — `NOT APPROVED`, `UNAPPROVED`, `PENDING`, `REJECTED`, `DRAFT`, `NOT SIGNED`, empty/dash, `REVOKED (previously APPROVED)`, `APPROVED - WITHDRAWN`, `CONFIRMED? NO`, scope-limited approvals such as `OPERATOR APPROVED — DIRECTION AND IMPLEMENTATION ONLY` — is rejected. The date must be a real ISO calendar date, not in the future (1-day clock skew).
* Rows are read **only from the register status table** (header `ID | Status | Signed date`); rows of the decision-text table and narrative sentences never count; zero or duplicate rows are errors.
* OD-EVOL-12 and OD-EVOLX-08 additionally need a decision record: the register notes cite exactly one record path and `sha256:<digest>`; the record (column-0 fields, fenced/quoted/mid-line text ignored) must state the matching `Decision ID`, `Stage`, `Policy version`, a `Decision scope` containing the required tokens (OD-EVOL-12: FRZ-EVOL-07/08/10; OD-EVOLX-08: `EVOL-X`, `formal closure`), `Decision status: AUTHORIZED`, exactly one `Approved <id>: operator @ <ISO>` line for the same id, no placeholder, and be committed at `HEAD`.
* **Provenance limitation (unchanged and stated in the gate docstring):** the operator signature line is process-enforced text, not a cryptographic signature. A real OD-EVOL-12 / OD-EVOLX-08 authorization remains a separate operator act; none was fabricated.

### RA-B4-04 — evidence-bound B6 acceptance (B-01..B-05)

New control **B-05** (this also fills the documented B-05 gap of RA-B4-10). In addition to the existing flag/ladder/two-phase checks (B-01..B-04, B-03 historical-SHA rejection), Gate B now requires, at the pinned `B6_ACCEPTANCE_PUBLICATION_SHA`:

* the record `iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md` **present at that commit**, not already present at the certification publication, identical at `HEAD`, indexed (`B6_ACCEPTANCE_EVIDENCE_INDEX.json`), and its SHA-256 recorded in the certification record (`b6_acceptance_record_path` / `b6_acceptance_record_digest`);
* the phase-1 certification record at the B6 SHA already issues the acceptance, binds the digest and leaves `b6_acceptance_publication_sha` empty (non-self-referential two-phase binding); the IIAC manifest is unchanged since;
* the record states the verified audit: accepted verdict, **exactly** the certified limitation ids, manifest/report/findings digests, audited implementation SHA, certification publication SHA, certifier session and agent;
* a valid operator record (`AUTHORIZED`, signature line) whose signature is neither older than the certification publication nor later than the B6 commit (+15 min), and B6 follows the P3 operator authorization.

Flags alone, a fabricated SHA, a record absent at the pinned SHA, a historical/internal B6 path or SHA, widened verdict/limitations and digest mismatches all fail.

## 3. Verification campaign

Executed on the implementation tree (Windows, `core.autocrlf=true`):

| Check | Result |
|---|---|
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py` (existing B-4 matrix + RA-B4 adversarial matrix) | 209 tests: full run 208 passed + 1 fixture-collision error in a test written in this session; the test was fixed and re-run with the other fixture-sensitive/canonical tests (25 passed) |
| `test_evol_x_b4_ruff_evidence_exemption.py`, `test_evol_x_frz_evol_canonical_matrix.py` | 10 passed |
| `verify_iiac_policy_activation.py --require-in-force` | PASS |
| `verify_iiac_audit_manifest.py` / `verify_iiac_role_separation.py --strict-product-diff` (EVOL-X certification manifest) | PASS / PASS |
| Canonical `--gate b6-readiness` | FAIL with exactly two `A-18` (open B-4 and P3 blockers) — expected |
| Canonical formal-closure gate | FAIL: Gate A as above; Gate B: B-01/B-02 (no B6), B-06 (FRZ rows OPEN), B-07 (OD-EVOL-12, OD-EVOLX-08 `PENDING`), B-09 (B-4, B-5, B6-P3 open) — expected |
| Original auditor evidence (`b4_reaudit/`) | all 10 indexed digests re-verified after the work; unchanged |
| Fresh-checkout proof | see §9 |

Adversarial coverage by finding (each negative asserts the rejecting control; each positive asserts `[]`):
RA-B4-01 — LF clone PASS, CRLF clone with the `-text` rule PASS, CRLF clone *without* the rule PASS (pass record verified CRLF on disk), committed tamper FAIL (LF and CRLF), uncommitted tamper FAIL, CRLF inside a `-text` path FAIL, wrong digest FAIL, unit semantics of the byte contract, canonical attribute/narrowness checks.
RA-B4-02 — retained / open+resolved / deleted (B-4, P3, B-5) / all deleted / missing or null field / renamed / status-text-only / reclassified text all FAIL; 15 forged-evidence variants and 12 forged independent-acceptance variants (verdict, draft, sessions, agents, stage, policy, audited SHA, unresolved findings, unsigned handoff, tampered/unindexed evidence) FAIL; 5 forged P3 operator records FAIL; genuine resolutions PASS.
RA-B4-03 — 27 negated/ambiguous statuses, 7 positive states, 7 invalid dates; Gate B negated closing decisions, narrative approvals, duplicate rows, renamed status table, wrong scope, 7 unverifiable-record variants, 3 register-binding variants, uncommitted record FAIL; genuine decisions PASS.
RA-B4-04 — flags only, flag flip by an arbitrary actor, fabricated SHA, record absent at the pinned SHA, historical path and historical SHA, record edited after publication, manifest changed after publication, 16 record-content mismatches, post-dated signature, B6 before P3 FAIL; correctly published B6 with verified evidence PASS; Gate A PASS never implies Gate B PASS; scoped limitations remain binding; no self-certification.

## 4. Governance test hotfix (separate: T-IGX-018-A)

Baseline comparison (executed here in an isolated worktree at `9a8fa3634`, 14 min): `tests/unit/governance` → **4 failed, 121 passed, 1 skipped** — `test_act_n01_inactive_policy_fails_default_and_require`, `test_positive_fixture_passes_without_activating_canonical_policy`, `test_n09_implementer_must_not_publish_pass_filename_glob`, `test_n12_session_a_policy_not_in_force_by_default`. All assert pre-activation state. The same run also reproduced **HF-02**: afterwards `git status` of that worktree showed `IIAC_POLICY_ACTIVATION_RECORD.json` modified (blob `c067ebc4…` instead of `c505dd5b…`). The hotfix changes only tests and one workflow line, keeps every negative control (inactive policy, forged `IN_FORCE`, unbound/invalid-manifest/tampered pass record) and adds a canonical-record hash guard. It is a separate commit and is **not** part of the RA-B4 proof.

## 5. Residual risks (explicit)

1. **RA-B4-05 not addressed**: gate PASS is still not a semantic scope guarantee (phrase presence, no widening deny-list, certifier-authored limitations not compared with the P2 blob). Out of the ordered scope.
2. Signature lines and session ids are process-enforced text (`assurance_level: process_enforced`); no cryptographic proof exists.
3. The lifecycle and B6 record formats are new contracts defined here (field names, record layout, directories). The re-auditor should confirm they are the intended canonical shape; no real instance exists yet, only isolated fixtures.
4. Stale clones need the one-time refresh described under RA-B4-01.
5. The timestamp checks use the committer date and the local clock (1-day / 15-minute tolerance).
6. HF-03 (CI `fetch-depth: 0`) is unverified in GitHub Actions at the time of writing.
7. The full platform suite was not rerun; the campaign is targeted.

## 6. What this change deliberately does not do

No B6 acceptance, no `B6_ACCEPTANCE_PUBLICATION_SHA`, no FRZ-EVOL promotion, no OD-EVOL-12 / OD-EVOLX-08 or P3 authorization, no resolution entry for B-4/B-5/P3 in the canonical certification record, no change to Session C / P2 / original B-4 re-audit evidence, no activation-record edit, no EVOL-X closure.

## 9. Binding and fresh-checkout proof

| Role | SHA |
|---|---|
| `RA_B4_CORRECTIVE_IMPLEMENTATION_SHA` (preflight, `.gitattributes`, tests, ticket T-IGX-017-B, this handoff) | `6cf8c5a2247465a1321c239d0559d751383daec2` |
| Governance test hotfix (T-IGX-018-A; not part of the RA-B4 proof) | `1e28b9e2b77bbc44171f80017fd56b4032aa196e` |
| Baseline / audited predecessor | `9a8fa3634713f69b56ce87ceaf9ed319fad27dae` / `23feb2bbb017b1beb63f9368b5978384093ff924` |

The binding is non-self-referential: the SHA above is recorded in a later documentation-only commit.

Fresh-checkout proof (`git worktree add` of the hotfix commit, `core.autocrlf=true`): `git ls-files --eol` shows the pass record `w/lf attr/-text` and the unprotected ladder `w/crlf`; `--gate b6-readiness` reports **only the two expected `A-18` errors** (no `A-14`), and the canonical preflight tests pass (13 passed). The same checkout procedure at `9a8fa3634` reproduced three `A-14` errors. A checkout *updated in place* from an older commit keeps its stale CRLF pass record and still fails `A-14` until the file is refreshed (see §2, RA-B4-01) — observed and documented.

Governance suite hotfix verification (isolated worktree, never the canonical tree): the two modified governance modules pass (18 passed in the targeted rerun) and the activation record blob stayed `c505dd5b…` (pre-hotfix, the same suite left it at `c067ebc4…`). A full `tests/unit/governance` rerun on the final tree and the GitHub Actions `governance-fast` result were **not** executed here.

