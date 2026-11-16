# PLAN T-IGX-047-A — EQO-A remediation node PT-R01 (EDER batch B): register data load, reconciliation, exclusions, proposed owner vocabulary

Status: DRAFT for Operator review. Nothing here is approved, applied or audited. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status, triages no row, closes no row and edits no product path.

## 1. Purpose

Load the data of the Enterprise Debt & Exceptions Register (EDER) whose structure was merged by T-IGX-042-D (PR #607, `origin/development` `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2`): one row per active EQO-A finding and per ingested dossier entry, the exception rows, the per-manifest pointer rows and the category rows, plus the reconciliation and the exclusions list required by the register README (sections 7 and 8). Master-plan node PT-R01 (EQO-A remediation, `WP-12/TICKETS_DRAFT.md`). Baseline of the dossier: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589).

Operator decisions applied (2026-11-16): DP-02 (owners rate, no mechanical rating), DP-46 (location), DP-47 (one program/stage-level `owner_ref` per row from a 55-ID vocabulary that the Operator confirms once), DP-48 (ownerless rows get an owner after classification), DP-50 (155 accepted entries as EXCEPTION rows, 39 per-manifest pointer rows for the 301 IIAC limitations), DP-51 (26 DEFERRED entries stay; 155 non-entries excluded; 605 + 155 = 760), DP-52 (EDER is the debt register named "P9 debt register" in the dossier), DP-53 (skip markers as category rows only). DP-54 (standalone ECAP executable) is open: no row, file or sentence here states a position.

## 2. Scope: four data files

| File (under `docs/project/maintainers/qualification/enterprise_debt_register/`) | Content |
|---|---|
| `register.jsonl` | 1069 rows, all `lifecycle_state` `INGESTED` |
| `reconciliation.json` | counts per source and per exclusion class, the exact equations, per-class distributions |
| `exclusions.jsonl` | the 155 `NOT_AN_ENTRY` rows with currency OPEN or UNKNOWN, excluded (DP-51) |
| `owner_vocabulary.proposed.json` | the 55 proposed owner IDs, status `PROPOSED_NOT_CANONICAL`, confirmation `UNCONFIRMED` (DP-47) |

## 3. Composition and exact reconciliation (dossier counts at the baseline)

Rows: 1069 = 239 findings + 605 OPEN/UNKNOWN entries + 26 DEFERRED entries + 155 accepted A2/register EXCEPTION rows + 39 IIAC pointer rows + 5 category rows (4 skip categories, 1 N/A category).

- Findings: 240 in `FINDINGS.jsonl`, 1 merged (EQOA-A2-025), 239 OPEN, one row each (`EDER-F-0001..0239`).
- Dossier entries: 1480 = 605 ingested OPEN/UNKNOWN + 26 ingested DEFERRED + 456 accepted (155 A2/register rows ingested as EXCEPTION rows; 301 IIAC limitations covered by 39 pointer rows, not listed per entry) + 211 resolved/closed/superseded (99/84/28, not ingested) + 182 not-an-entry rows (155 excluded because their currency is OPEN or UNKNOWN, 27 others with currency RESOLVED or ACCEPTED, not ingested).
- `605 + 155 = 760`: the OPEN or UNKNOWN currency rows of the dossier (real entries plus not-an-entry rows).
- Row IDs `EDER-D-0001..0631` keep the numbering of the WP-6 draft that the master plan cites (for example `EDER-D-0167`); accepted A2/register rows continue as `EDER-D-0632..0786`; pointer and category rows are `EDER-X-0001..0044`.
- The 33 snapshot-contradicted entries (30 OPEN, 2 UNKNOWN, 1 DEFERRED; EQOA-A7-035) carry `disposition = STALE_SNAPSHOT_REVERIFY`, keep their stated currency, and are not closed. Four more accepted entries are stale-flagged in the dossier; their currency is already corrected there, they are EXCEPTION rows with the stale flag recorded and are not counted in the 33.

## 4. Rules applied

1. Owners (DP-47): `owner_ref` is a proposed ID from the proposed vocabulary. The merged schema has no `owner_status` value `PROPOSED_UNCONFIRMED`; the schema's own proposed values are used (`SINGLE_PROPOSED`, `SHARED_UNRESOLVED`, `OPERATOR_ONLY`; `MISSING`, `ROLE_NOT_OWNER`, `UNMAPPED`, `GENERIC_NOT_AN_OWNER` where no owner can be proposed, with `owner_ref` null), and every row also carries `owner_vocabulary_status = PROPOSED_UNCONFIRMED`. No row is `CONFIRMED`; `accountable_person` is empty. The schema is not changed.
2. Severity and impact (DP-02): severity of findings is the audit finding's own severity; severity of entries is the source register's token, otherwise `UNRATED` with `severity_source NONE`. `triage_priority` is an ordering aid only. Impact of findings is the WP-12 effective class (strictest of the finding-level class, the dossier class, the worst linked FRZ criterion and recorded floors; finding-level and dossier values are kept in `frz_impact_finding_level` and `frz_impact_dossier`); it is not confirmed by an owner or the ARCH-FREEZE owner. Rows keep `UNKNOWN` where no class exists.
3. Finding rows (finding X-14 of the master plan): the draft is regenerated with the WP-12 `master_rows.jsonl` values for track, effective impact, finding-level impact, closure condition, audit need, corrected `iiac_sensitive`, FRZ rows, plan nodes and closure wave. Owners are referenced by ID; the WP-12 owner text is kept in `plan_owner_text`, not copied into `owner_ref`. One severity proposal of WP-12 (EQOA-A4-016, HIGH) is kept as `severity_proposal`, not as the rating.
4. Skip markers (DP-53): four category rows over the static scan (no_ci, importorskip, conditional skip, unconditional skip and xfail) and one N/A category row; no per-marker row. No `KNOWN_RED_TEST` row exists (no authorised bounded run exists).
5. Evidence: every cited path is pinned `path@sha40`; rows point to their dossier source by path and do not copy canonical prose. `current_head_status` is computed at `2ab57e36d` and means only that the cited tracked paths are byte-identical, not that the claim was re-verified.

## 5. Explicitly out of scope

Owner confirmation (DP-47), severity ratings, impact confirmation, triage, any lifecycle state beyond `INGESTED`, `EXCEPTION_ACCEPTED` records, the consistency script and its CI step (PT-R02), the README and schema (merged), source-register header labels, the PDC-P9-01 reference amendment of DP-52, the DP-54 decision, every file outside the write set, `scripts/`, `tests/`, `.github/`, product paths, the roadmap and the FRZ checklist.

## 6. Write set

```text
docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl
docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json
docs/project/maintainers/qualification/enterprise_debt_register/exclusions.jsonl
docs/project/maintainers/qualification/enterprise_debt_register/owner_vocabulary.proposed.json
governance/tickets/T-IGX-047-A.md
governance/tickets/plans/PLAN_T-IGX-047-A.md
```

None is under the IIAC product-path prefixes (`intergrax/`, `agents/`, `applications/`).

## 7. Generation and reproducibility

The files are generated by a stdlib Python script kept outside the repository (`.tmp/session/eqo-a-remediation/IMPL/PT-R01/build_pt_r01.py`, sha256 `6d519f13adadbd36031232abb03d5aa3861c3949641e4370b779fdc5351ca14e`) from the dossier at `origin/development`, the WP-12 master rows (`.tmp/session/eqo-a-remediation/WP-12/master_rows.jsonl`, sha256 `afbe369646dd92942654b94a85a1fc693743ad325cfc85f49527aedf4393e769`, not tracked) and two triage files. The master rows are workspace-only; the finding rows carry the copied values and cite the master row by `plan_ref`.

## 8. Test contract

One process at a time, under the shared pytest lock, held at most 15 minutes:

- Register checker `check_register.py` (standard library, read-only, kept outside the repository; the run copy adds the schema subset check and the reconciliation, pointer, owner, stale-set, placeholder and category checks listed in the PR body) over the committed data. Result reported as is.
- `uv run pytest tests/unit/docs/`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer checks: `git diff --name-only origin/development...HEAD` lists exactly the Write set; the counts of section 3 are reproducible from `reconciliation.json`; no operator or owner placeholder marker (double-angle-bracket form) in any committed file.

## 9. Risks

- The checker's GATE classes are non-zero by design at `INGESTED` (unconfirmed owners, unrated severities, unclassified impact); they are readiness gaps, not defects of this load.
- The register file is about 3.4 MB; row prose is limited to labels, the cited closure text and short basis strings.
- Owner proposals come from keyword mapping of the dossier's free-text owner field; 14 rows are unmapped, 15 rows have no owner, 12 name a role and 4 a generic owner. They are not owners until the Operator confirms the vocabulary and assigns the rest.
- `reconciliation.json` and the findings' plan fields cite workspace-only WP-12 files; a reviewer without that workspace verifies the counts against the dossier only.

## 10. Rollback

Revert the single PR. No code, configuration or other document depends on these files.

## 11. Closure condition

PR merged after review; the ticket records the merge SHA. No row, finding or FRZ criterion is closed, accepted, verified or promoted by this ticket.
