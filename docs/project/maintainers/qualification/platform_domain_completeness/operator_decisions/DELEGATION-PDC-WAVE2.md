# DELEGATION-PDC-WAVE2 — Operator decisions for the closing wave of PDC-X

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE2 |
| **Stage** | PDC-X P6 remainder, P7, P8, P9 preparation |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-P6-REPIN.md`; `DELEGATION-PDC-P6-ECAPFIX.md` |
| **Date** | 2026-11-14 |

## Commission (transcription)

The Operator asked to stop taking decisions step by step and proposed two waves: wave 1 settles every open decision with a recommendation, wave 2 runs to the closure of PDC-X without further questions. The orchestrator listed nine decisions with a recommendation each. The Operator answered in English rendering: **"I accept everything"** (original wording in Polish: "Akceptuję wszystko"). Every decision below is therefore the recommendation as stated.

| # | Decision | Answer |
| --- | --- | --- |
| 1 | Dispatch the EBH-7 replay at the pin `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (both SHA inputs) | **Yes** |
| 2 | Ratify retroactively the two test files changed in PR 538 (option B of the re-pin: `_ebh7_p03_evidence_integrity.py`, `test_ebh_7_b6_f10_f11_corrective_gates.py`) although they are outside the signed write set of T-IGX-036-W | **Yes** |
| 3 | B6-F10 certification (6 red `ebh_7` tests, unclassified changed paths) | **Option A:** fix in a separate ticket, plan signed by a delegate session |
| 4 | `dist_scale_x` and Docker (C10/C15) | **Option B:** the scope is excluded from PDC-X as a declared, labelled exception ("not qualified, deferred"); the freeze is not lifted |
| 5 | Remaining debt | One omnibus ticket (see scope); the rest recorded as documented debt |
| 6 | Audit and repair loops | Up to 2 loops per domain pre-authorized; one re-pin after all P7 repairs and one after P8 if product paths changed |
| 7 | Dependency-bot (`uv.lock`) pull requests | Held until the final re-pin is recorded, then merged in one batch |
| 8 | Merging pull requests | **Option A:** the implementer merges pull requests covered by authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` with `gh pr merge` after green CI and a head match |
| 9 | P9 | The orchestrator prepares the whole closure package; the Operator signs once at the end |

## Scope of the decisions

1. **Replay (1).** `unit-tests.yml` with `profile=ebh7_p3_l3`, `ebh7_test_execution_sha` and `ebh7_evidence_publication_sha` both `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec`; the SHA of the W1 merge commit cannot be pinned inside its own manifest. The result is recorded in a docs-only PR (W2) and is **not** a B6 or certification verdict.
2. **Ratification (2).** The write set of T-IGX-036-W lists `tests/qualification/ebh_7/_constants.py` only. PR 538 also changed the two files above (decouple `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` from the test pin: it must equal or be an ancestor of `TEST_EXECUTION_SHA`; a `PENDING` `last_replay` carries no counts). The Operator chose this option in the implementer pane and ratifies it here. The ticket and plan records of T-IGX-036-W are not edited, because `governance/tickets/` is outside the executable-identity allowlist; this record is the amendment. No verifier allowlist changed and no gate was weakened.
3. **F10 (3).** A new ticket extends the F10 path classifier, the handling of closed downstream stages (ENT-AUDIT-X, EVOL-X) and the MANIFEST pytest profile. Unknown paths keep the BLOCKER classification; a gate may be strengthened or receive classification entries, never weakened. A delegate session may sign that plan under the checks of `DELEGATION-PDC-STANDING.md` answer 4 (no hotspot, no new authority, no gate weakened, plan matches the WBS batch).
4. **`dist_scale_x` (4).** P4 and P6 of `dist_scale_x` stay blocked. The P9 closure package carries an exception register that names this scope, its reason (C10/C15) and the condition to reopen it. The C10/C15 restriction is unchanged.
5. **Debt (5).** One omnibus ticket repairs: the `pdc_x` pytest process that does not exit after its summary line (R-05), tests that are red on the baseline for environment reasons (for example the missing `INTERGRAX_PGVECTOR_TENANT_ID`), the unused `embedding_provider/ollama` manifest and the `compat_x` counters. Recorded as documented debt without a ticket: Celery, Kafka and Rabbit identity and DLQ, strict-host `plugin_provenance`, the `ent_audit_x` `launcher_05` FREEZE timeout. `roadmap_replay_x` dispositions stay with MCP-SERVER-X P7. The omnibus plan is signed by a delegate session under the same limits as 3.
6. **Loops (6).** A finding of a P7 or P8 audit is repaired in a ticket of the owning domain; a delegate session may sign the plan and write-set amendments of such repair tickets under the limits of answer 4 of `DELEGATION-PDC-STANDING.md`. A fresh auditor re-audits the corrected SHA. A third loop on one domain, a finding of a new kind, a hotspot path, a gate whose meaning would change or a new authority returns to the Operator. Product paths changed by repairs are re-pinned once after all P7 repairs and once after P8 if needed (the Operator already accepts further re-pins).
7. **Dependency bot (7).** The hold of `DELEGATION-PDC-P6-REPIN.md` decision 2 is extended to the final re-pin; the held pull requests are merged afterwards in one batch under authorization 1.
8. **Merges (8).** Applies to documentation-only pull requests (authorization 1) and to product pull requests that satisfy authorization 3 (green CI, known red classes only: IIAC activation, until PDC-P7-00; head equals the verified SHA). The implementer never merges its own change to IIAC protocol or activation files, an OD signature, a FRZ line or a closure line.
9. **P9 (9).** The orchestrator prepares the FRZ promotion proposal, the roadmap closure text, the exception register (decision 4) and the debt register (decision 5). The Operator signs once at the end; the orchestrator asks no other question before then, except for the returns named in 6.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure line (PDC-P9-03, PDC-P9-04); lifting C10/C15 (Docker, new vendor integrations, production configuration); a change of the meaning of a gate beyond decision 3; edits to hotspot paths `intergrax/contracts/` and `intergrax/applications/_shared/` not owned by the batch; anything outside these limits.

## Separation

The implementer session and its implementer subagents never write `Approved:` or activation lines. A delegate session does not implement or audit what it signs. An auditor never certifies and never fixes what it audits.
