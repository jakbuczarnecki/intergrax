# PLAN T-IGX-046-A — EQO-A remediation node PT-D03 (batch DB-J): byte-preserving publication of locally held audit artefacts

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementation session from the WP-3 plan (finding K4), the WP-9 batch DB-J planning manifest and the Operator decisions DP-07 and DP-44. Nothing here is approved, accepted or independently reviewed.

## 1. Purpose

Several statements in tracked records rest on audit reports and signing records that exist only in gitignored session directories on the Operator machine (EQO-A findings A2-026 and A7-026). This ticket adds byte-preserving copies of those files, with sha256 values, a secret/PII scan result and a custody statement, under `docs/project/maintainers/qualification/iiac_evidence/`. It is a custody and traceability action only. It certifies nothing: it does not make any audit independent, does not verify any stage, does not close or promote any roadmap row or FRZ criterion, and does not accept any verdict a published file carries.

## 2. Scope

| Item | Content |
|---|---|
| Source | 33 files listed in the WP-9 DB-J planning manifest (24 digest-pinned by the tracked record `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md`, 9 not pinned) under `.tmp/session/cert-campaign/` and `.tmp/session/arch-freeze-g6-wave1-audit/` |
| Scan | Pattern scan of every file (private-key headers, cloud and token patterns, bearer and JWT strings, connection strings, e-mail and IP addresses, local profile paths, weak secret words, personal-name markers) plus a targeted read of lines carrying URLs, `@`, key or cookie words |
| Published | Files with no scan hit, copied byte-for-byte to `iiac_evidence/<stage>/...` (`.gitattributes` line 1 sets `-text` for that tree) |
| Held back | Any file with a scan hit: recorded by path, line and kind in the manifest, not published |
| Manifest | `iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md`: path, sha256, size, origin, scan result, custody statement |
| Excluded | The optional ARCH-FREEZE G6 wave-1 report (finding A2-015, owned by another batch) |

Decisions applied: DP-07 (publish the delegation master record; the Operator's re-affirmation in a tracked signature is separate and is NOT part of this ticket), DP-44 (publish AUD-SKILLX byte-preserving with digest and custody statement; settling G-SKILL is NOT part of this ticket). Authority: AR-20.

## 3. Explicitly out of scope

- Any edit of an existing tracked file, including `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` (the citing-record correction is a separate documentation batch), the roadmap, the FRZ checklist, tickets other than this one, `scripts/`, `tests/`, `.github/`, `intergrax/`, `agents/`, `applications/`.
- Re-running, re-verifying or re-auditing anything recorded in a published file.
- The Operator re-affirmation of the delegation, any approval line, any FRZ promotion, any stage closure.
- Files whose scan hit is not released by the Operator (see the manifest section "Held back").

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/iiac_evidence/QUAL-X/**
docs/project/maintainers/qualification/iiac_evidence/QUAL-X-P0/**
docs/project/maintainers/qualification/iiac_evidence/SKILL-X/**
docs/project/maintainers/qualification/iiac_evidence/TOOL-LIFE-X/**
docs/project/maintainers/qualification/iiac_evidence/CERT-CAMPAIGN-2026-11-07/**
docs/project/maintainers/qualification/iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md
governance/tickets/T-IGX-046-A.md
governance/tickets/plans/PLAN_T-IGX-046-A.md
```

All paths are new files. No existing tracked file is modified.

## 5. Method

1. Worktree outside the repository (`<parent>/_wt/intergrax/eqo-pt-d03`) on a new branch `docs/eqo-a-pt-d03-evidence-publication` cut from a freshly fetched `origin/development`.
2. Recompute size and sha256 of every source file and compare with the planning manifest; stop on any difference.
3. Scan every file; a file with any hit is not copied.
4. Byte copy of the clean files; compare each copy with its source.
5. Write the manifest; commit with the `Ref:` trailer and no tool attribution.
6. After the commit, read each blob back with `git cat-file blob` and compare its sha256 with the manifest (detects any git-side normalisation).

## 6. Test contract

One process at a time under the shared pytest lock (`agent-env/rules/intergrax-test-process-budget.md`); documentation checks only:

- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/unit/docs/ -p no:cacheprovider`
- Reviewer check: `git diff --stat origin/development...HEAD` lists only the paths of section 4 and shows added files only.

## 7. EBH-7 F10 and digest-pinned artefact rules (checked for the target directory)

- `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` classifies `docs/` as `DOCUMENTATION_ONLY` (C4 rule at line 238) and `governance/tickets/` as `DOCUMENTATION_ONLY` (line 321); no rule names `iiac_evidence/`.
- `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json` pins named files in the qualification directory and does not mention `iiac_evidence/`; this batch adds no file to, and edits no file of, `ebh_7_p03_independent_replay/`.
- The executable-identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py`, `_path_allowed`, prefixes at lines 19-23) allows every path under `docs/project/maintainers/qualification/`, which covers all evidence files and the manifest of this batch. It does NOT list `governance/tickets/`, so this ticket and plan file (like the ticket files of other documentation batches) are outside that allowlist; whether they matter for the TEST_EXECUTION_SHA identity check depends on the EBH-7 pin timing and is for the Operator or the PDC-X session to confirm before merge.
- `iiac_evidence/` is excluded from ruff by `pyproject.toml` (`extend-exclude`) and the IIAC activation gate refers to it only for ACT-N14 artefacts; a pointer to the gate files read is in the pull request body.

## 8. Risks

- A published copy with a digest is a custody anchor, not verification of the verdict it carries; the manifest says so in its first section.
- The scan is a pattern scan plus a targeted read, not a full content review. The Operator or a commissioned reviewer should read the files before merge.
- The directory name `CERT-CAMPAIGN-2026-11-07` is a campaign name, not a roadmap stage; unconfirmed by the Operator.
- Held-back files stay only on the Operator machine until released.

## 9. Rollback

Revert the single PR. Only new files are added; no existing file, code path, configuration or gate input changes.

## 10. Closure condition

PR merged after review; the ticket records the merge SHA; findings A2-026 and A7-026 are marked `DOC_PATCH_APPLIED` (partially: held-back files and the citing-record correction remain) in the Enterprise Debt & Exceptions Register when it exists. No finding is marked closed or verified by this ticket.
