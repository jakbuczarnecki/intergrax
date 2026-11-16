# PLAN T-IGX-042-D — EQO-A remediation batch DB-G: Enterprise Debt & Exceptions Register structure, qualification index, documentation map link

Status: DRAFT for Operator review. Nothing here is approved, applied or audited. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 1. Purpose

Implement the structure part of Operator decision OD-REM-04 / condition C4 of the EQO-A remediation planning (Campaign A dossier, PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`): the governing document and the row schema of the Enterprise Debt & Exceptions Register (EDER), an index of `docs/project/maintainers/qualification/`, and the inbound links in `docs/project/technical/DOCUMENTATION_MAP.md`. Findings addressed: EQOA-A0-001, -002, -003, -004, -006, EQOA-A2-006, EQOA-A4-028, -033, -046, EQOA-A7-034, -035, -037.

Operator decisions applied (2026-11-16): DP-46 (location and index path), DP-47 (one program/stage owner per row from a vocabulary the Operator confirms once), DP-02 (owner-rated triage, no mechanical ratings), DP-50/DP-51/DP-53 (scope of the initial load, category rows for skips), DP-52 (EDER is the debt register named "P9 debt register" in the dossier). DP-54 (whether a standalone ECAP executable is required) is still open: no document in this change states a position.

## 2. Scope (6 hunks, 4 files)

| Hunk | File | Change |
|---|---|---|
| H01 | `docs/project/maintainers/qualification/enterprise_debt_register/README.md` (new) | EDER governing document: purpose, non-duplication rule, identity, fields, lifecycle, authority matrix, maintenance, initial-load scope, standing categories (known-red tests, N/A decisions, skip categories, severity and stale snapshots, superseded IIAC manifest drafts) |
| H02 | `docs/project/maintainers/qualification/enterprise_debt_register/register_schema.json` (new) | Row schema (JSON Schema 2020-12) |
| H03 | `docs/project/maintainers/qualification/README.md` (new) | Qualification directory index (map only, no status claims) with a section mapping the four ECAP modes to existing entrypoints |
| H04 | `docs/project/technical/DOCUMENTATION_MAP.md` | Quick routing row naming the CONFIG-X, TENANT-X and STATE-X certification records as canonical owners (A4-028) |
| H05 | same | Security owner chain row (A4-033) |
| H06 | same | Vendor/backend neutrality owner chain row (A4-046) and the qualification index pointer row (A0-004) |

Source: `.tmp/session/eqo-a-remediation/WP-9/patches/DB-G/PATCH.diff`, applied with `git apply` against `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42`, then adjusted as listed in section 3.

## 3. Adjustments to the drafted patch

- The operator-placeholder sentence in the superseded-manifest section of H01 is removed; the section states only that the manifests are to be recorded as superseded drafts and that the superseding manifest is not stated here.
- H01 and H03: the HTML draft comments no longer refer to a phase-1 `.tmp` draft; they point to this ticket.
- H01 section 8 now follows DP-50/DP-51 (accepted entries as exception rows and per-manifest pointer rows), and states DP-52, DP-47 and DP-02.
- H03: the sentence that called the standalone ECAP executable an open owner decision is replaced by "This index does not decide whether a standalone ECAP executable is required" (DP-54 stays open); the reference to a not-yet-existing `INDEX.json` is rephrased.

## 4. Explicitly out of scope

- Register data rows (`register.jsonl`), `reconciliation.json`, the owner vocabulary file and `excluded_non_entries.jsonl`: rows name owners and the vocabulary is not yet confirmed by the Operator (DP-47). They follow after that confirmation, under their own ticket.
- The consistency script and its tests, any file under `scripts/`, `tests/`, `.github/`, `intergrax/`, `agents/`, `applications/`, the roadmap, the FRZ checklist, `governance/` other than this ticket and plan.
- The superseding-manifest names for the three draft IIAC manifests (needs an Operator or owner fact).
- A position on DP-54.
- The PDC-P9-01 reference amendment of DP-52 (a separate PDC amendment).
- Any certification statement: no sentence asserts or upgrades independent verification, PASS or closure; roadmap labels are quoted and attributed.

## 5. Write set

```text
docs/project/maintainers/qualification/README.md
docs/project/maintainers/qualification/enterprise_debt_register/README.md
docs/project/maintainers/qualification/enterprise_debt_register/register_schema.json
docs/project/technical/DOCUMENTATION_MAP.md
governance/tickets/T-IGX-042-D.md
governance/tickets/plans/PLAN_T-IGX-042-D.md
```

None is under the IIAC product-path prefixes (`intergrax/`, `agents/`, `applications/`). A static read of `scripts/governance/iiac_activation_gate.py` (`MATERIAL_GOVERNANCE_PATHS`, exact-path membership) found none of the four documentation paths in that list; the gate itself was not run.

## 6. Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`: that flag conflicts with the repo pytest temp plugin, which sets `cache_dir`, and the run aborts with exit 4)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- The register's own read-only checker `check_register.py` (standard library, kept outside the repository) is run once against the draft data to confirm the schema enums it relies on; its result is informative only because no data rows are committed.
- Reviewer checks: `git diff --name-only origin/development...HEAD` lists exactly the Write set; every link in the two READMEs and the three map rows resolves; every count is reproducible with `git ls-tree` at the stated SHA.

## 7. Risks

- Counts in the qualification index describe the baseline and drift with the head (they are labelled "at baseline"); a later regeneration changes them.
- `DOCUMENTATION_MAP.md` contains two existing links to `../architecture/CRITIC_VERIFICATION.md`, which does not exist at `origin/development`. They are not touched by this change.
- Data rows and the owner vocabulary are a separate batch; until then EDER is a structure without rows.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review; the ticket records the merge SHA. No finding is marked closed or verified by this ticket; the findings above are covered as far as the structure of the register and the index are concerned.
