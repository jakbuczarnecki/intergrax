# PLAN T-IGX-042-E — EQO-A remediation: publication of the Operator decision record

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementation session as a draft pull request, at the request of the orchestrating session. Nothing here is approved or audited.

## 1. Purpose

Publish, as repository documents, the record of the Operator decisions made in chat on 2026-11-16 about the EQO-A remediation program, because the delegate review of the code-wave tickets (its finding X-8) pointed out that approvals must not rely on a gitignored record. The Operator authorised publishing the record as a docs-only pull request. Documentation only. This ticket certifies nothing, decides nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 2. Scope (3 new files)

| File | Content |
|---|---|
| `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md` | OD-REM-01..08, conditions C1-C10, decision sessions 1-6 (DP-01..DP-68), summary of the code-wave acceptances |
| `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` | The Operator's messages on the code-wave tickets T-IGX-043-A..F, how each item was applied, the addenda for messages 4 and 5 with their interpretation marks |
| `docs/project/maintainers/qualification/eqo_a_remediation/README.md` | Short index |

Both records are the orchestrating session's transcription (IIAC Amendment 01 section 1.2). The Operator's chat is the original and was in Polish; the repository holds only English renderings, each labelled as the orchestrator's rendering of a chat message of the given date. The Polish text is not reproduced.

## 3. Content rules

- Faithful to the sources: no decision is added, no interpretation line is turned into a decision; the marks of the addendum (interpretation, not decided) are kept.
- No claim that the record is signed: it is a transcription and publication does not sign it.
- No closure, PASS or independent-verification claim.
- Session-local paths are replaced by repository-relative references where the target exists in the repository, otherwise by a plain description.
- Pull request numbers #605..#608 are stated only as draft pull requests at the time of writing.
- Two verification notes are added to the files because the check found them: the SHA-256 values of the four question files recorded at transcription could not be reproduced (FC-1 matches a preserved accepted copy only), and the ticket ids `T-IGX-043-A` and `T-IGX-043-C` also appear on the open ARCH-FREEZE wave 2 branches.

## 4. Explicitly out of scope

Any change to the decisions themselves; the approval lines of any ticket; the Enterprise Debt & Exceptions Register and the qualification index (batch DB-G); the roadmap, the FRZ checklist, the PDC-X records and `PDC_INDEX.md`; any file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`; GitHub settings.

## 5. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md
docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md
docs/project/maintainers/qualification/eqo_a_remediation/README.md
governance/tickets/T-IGX-042-E.md
governance/tickets/plans/PLAN_T-IGX-042-E.md
```

## 6. Test contract

One process at a time, under the shared pytest lock:
- `uv run python scripts/docs/check_docs_domain_pairs.py` (the new directory is not a domain-layer architecture or plan directory, so no change in its result is expected).
- EBH-7 / B6-F10 classification: static read only of `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`; no qualification suite is run. Finding (`classify_change` in `tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py` picks the longest matching prefix): the rules contain `docs/` (`DOCUMENTATION_ONLY`) and `governance/tickets/` (`DOCUMENTATION_ONLY`), and no rule names `docs/project/maintainers/qualification/`; the new files fall under `docs/`. The owner-delta bindings (`RF_OWNER_DELTA`) name five specific owner documents under `docs/project/architecture/`, none of them in the new directory. The class regression for `DOCUMENTATION_ONLY` is one test, the B6-F10 matrix markdown test.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the five files above.

## 7. Risks

- **Rendering fidelity.** The English text is a rendering by the orchestrator; the Operator may correct any entry. Mitigation: the status section of each file says so.
- **Misreading as a signature.** Mitigation: each file states that it is not signed and that publication does not sign it.
- **EBH-7 pin interplay.** The path is documentation-only by the C4 rules; whether a pin is needed for a docs-only change is for the PDC-X and ARCH-FREEZE owners to confirm, and this ticket makes no claim about it.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review; the ticket records the merge SHA. Merging does not sign the record, close any finding or confirm any milestone.
