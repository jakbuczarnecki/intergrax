# T-IGX-046-F - EQO-A remediation: release of the nine held DB-C hunks as dated gap and custody notes

Implementer session: local_398ae9d6-d95a-48cb-9957-e428fed258e5

## Scope

Documentation-only follow-up to T-IGX-046-E (PR #656, merged as `cb60f683f409baf2b87c27e31902a14d918e1e0b`), which held nine of the sixteen prepared DB-C hunks. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589); re-verified at `origin/development` `44d1e5d105beb9b62d0a12e693066b0b26c125eb`.

- Nine append-only dated notes in nine stage records (98 added lines, no line changed or deleted): five gap notes (H03, H06, H07, H08, H12), one gap note on the SHA ladder (H11), two custody notes for replay logs (H04, H05) and one custody note with a decision cross-reference (H09). Findings: EQOA-A2-007, EQOA-A2-036, EQOA-A2-040, EQOA-A2-044, EQOA-A2-046, EQOA-A3-017.
- Publication of three local logs (W5 batch A and B, PRE-ENT W6 bounded regression) under `iiac_evidence/EQO-A-DBC-HELD/` with a manifest, sizes and sha256. The two W4 replay logs (T11-A, T11-B) are not published because the pre-publication scan found user-profile paths; the notes record their local digests.
- The G-SKILL note quotes only the published decision DP-44.

Out of scope: the roadmap, the freeze checklist, the FRZ matrices, `PDC_*` records, generated registers and register data, the PRE-ENT final matrix, `scripts/`, `tests/`, `.github/`, every product path, redaction of any log, supply of any missing owner fact, person assignments, row-level owner confirmation. No PASS, FRZ promotion, stage closure, independent-verification claim or independence claim.

## Plan digest

Plan digest: a8eef929756b7089ce4fe35dbd416780d4b4269d4ddb007d697c8f67c707fab9

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-F.md`](plans/PLAN_T-IGX-046-F.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W7_HOST_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md
docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/EVIDENCE_MANIFEST.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v11-batch-a.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v12-batch-b.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/pre-ent-w6-cross-audit/parent_regression_5gate.log
```

## Approved

Approved: operator-delegate @ 2026-11-17T15:11:25+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@cd93d7e2b519bc2d894e2da82f4f8156d281af2d
Delegate session: local_7b06d036-e322-483f-8e3f-8d2010ae9e29

## Test contract

One process at a time, under the shared pytest lock; logs under `.tmp/session/dbc-held/`:

- `uv run pytest tests/unit/docs/`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `python -m scripts.governance.verify_ticket_approvals --since origin/development`
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the write set plus this ticket and its plan; `git diff --numstat` shows 0 deleted lines for the nine records; a read-only reviewer confirms that no added sentence adds an independence claim without a preserved artefact, that every cited SHA resolves and that every cited line reads as cited (AR-9-07).

## Out of scope

Edits of the roadmap, the freeze checklist, PDC-owned records, generated registers, `scripts/`, `tests/`, product paths, person assignments, row-level owner confirmation, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
