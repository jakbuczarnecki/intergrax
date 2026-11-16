# PLAN T-IGX-044-A - EQO-A remediation node PT-E00 (T-ENF-0): enforcement documentation records

Status: DRAFT for Operator review. Drafted 2026-11-16 from the EQO-A remediation master plan (node PT-E00). Nothing here is approved, applied or audited.

## 1. Purpose

Record three documentation items of the enforcement workstream, with no change to any status, script, setting or product path:

1. **EQOA-A2-006** - three of the 40 IIAC audit-session manifests under `iiac_evidence/` fail `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py` (39 manifests at the EQO-A baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`). A point-in-time record states the facts and keeps the files byte-identical. It declares no supersession because no tracked record does.
2. **EQOA-A0-006** - the qualification index already lists the entrypoints for the four ECAP modes and says no single ECAP executable exists at the baseline. This batch adds only the statement that whether a standalone executable is required is an open Operator decision (DP-54), and takes no position.
3. **FRZ-AUD-10** - a clearly labelled DRAFT of the future live-enforcement amendment (Operator decisions OD-REM-07, C9, DP-64). It does not change the FRZ-AUD-10 disposition and edits neither the freeze checklist nor the roadmap (Operator-only files).

## 2. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md   (new)
docs/project/maintainers/qualification/ent_audit_x/FRZ_AUD_10_LIVE_ENFORCEMENT_AMENDMENT_DRAFT.md            (new)
docs/project/maintainers/qualification/README.md                                                              (section 7 closing sentence, new section 8, header comment)
docs/project/maintainers/qualification/enterprise_debt_register/README.md                                     (one pointer paragraph at the end of section 9.6)
governance/tickets/T-IGX-044-A.md
governance/tickets/plans/PLAN_T-IGX-044-A.md
```

## 3. Explicitly out of scope

- `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, `pyproject.toml`, `uv.lock`.
- The roadmap `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`, the FRZ checklist, the ECAP protocol, the PDC-X record directories, every `IIAC_*` file, every manifest and evidence index, the IIAC protocol, the five IIAC scripts and the six material governance paths.
- Any F10 canonical input or digest-pinned artefact of EBH-7 (checked: none of the files above is an input of `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`; the EBH-7 executable-identity allowlist covers the whole `docs/project/maintainers/qualification/` prefix; `ARTIFACT_DIGESTS.json` and `MANIFEST.json` of the EBH-7 replay do not name any file of the write set).
- A corpus-verifier allowlist, any validator hardening, any ruleset or GitHub setting (separate authorisations).
- Any statement of PASS, closure, FRZ promotion, stage status or independent verification.

## 4. Method

1. Worktree `<parent>/_wt/intergrax/eqo-pt-e00` from `origin/development` (`2ab57e36d94eae98c962b3d7b09e5a540d8c70d2`), branch `docs/eqo-a-pt-e00-enf0-docs`.
2. Every factual sentence is checked against the head with a `path@sha:line` source (validator runs logged under `.tmp/session/eqo-a-remediation/IMPL/pt-e00/`).
3. One commit with the `Ref: governance/tickets/T-IGX-044-A.md` trailer; no AI-tool attribution; upstream set to `origin/development` before the explicit-refspec push; draft PR against `development`.

## 5. Test contract

One process at a time, under the shared pytest lock (`agent-env/rules/intergrax-test-process-budget.md`):

- `uv run pytest tests/unit/docs/ -o cache_dir=<scratch dir outside the repository>` (`-p no:cacheprovider` is not usable here: the repository sets the `cache_dir` ini option, which then fails with a usage error)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the six files of section 2.

## 6. Risks

- **Content drift.** The manifest counts and the live GitHub state are point-in-time; both are dated and tied to the head SHA.
- **EDER and index overlap.** The two README edits are append-only pointers in documents owned by T-IGX-042-D (merged); another open batch that edits the same READMEs would conflict textually only.
- **EBH-7 pin.** Only files inside the allowlisted qualification prefix change; no F10 input changes.

## 7. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 8. Closure condition

PR merged after review; the ticket records the merge SHA. Finding A2-006 is marked as covered by a point-in-time record (not closed) and A0-006 as indexed with the open question stated, in the debt register when its rows exist. The FRZ-AUD-10 draft stays a draft until the controlled activation plan, its trials and an independent audit at an exact SHA exist.
