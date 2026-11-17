# Operator answers on the code-wave tickets T-IGX-043-A..F: transcription by the orchestrator

## Status of this record

- **Nature.** Transcribed on 2026-11-16 by the orchestrating session under IIAC Amendment 01 section 1.2 (the orchestrator MAY transcribe a chat commission or decision; see [`IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md`](../IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md)). It is a transcription, not an Operator signature. Publishing it in the repository does not sign it.
- **Original and language.** The Operator's chat is the original; the messages were in Polish. This file contains only the orchestrator's English rendering of each message, labelled with its date. It is not a verbatim quote and the Polish text is not reproduced.
- **Purpose of publication.** Finding X-8 of the independent delegate review of the code-wave tickets was that approvals must not rely on a gitignored record. The Operator authorised publishing this transcription as a docs-only pull request (ticket `T-IGX-042-E`).
- **No closure claim.** Nothing here is a PASS, a closure or an independent verification, and no approval line of any ticket is written by this file.
- **Ticket ids.** The ticket ids `T-IGX-043-A..F` are used as they appear in the Operator's messages and in the drafts. The ids `T-IGX-043-A` and `T-IGX-043-C` also appear, for different work, on the branches of the open ARCH-FREEZE wave 2 pull requests #611 and #610 at the time of writing; this transcription does not resolve that overlap. The batch tickets `T-IGX-042-A..D` are in draft pull requests #605..#608 at the time of writing.

Companion record: [`OPERATOR_DECISIONS_EQOA_REMEDIATION.md`](OPERATOR_DECISIONS_EQOA_REMEDIATION.md) (index: [`README.md`](README.md)).

## Operator messages (chat, 2026-11-16; orchestrator's English rendering of each)

1. Orchestrator's English rendering of a chat message of 2026-11-16: "I accept the recommended answers to all questions of FC-1..4."
2. Orchestrator's English rendering of a chat message of 2026-11-16, given as the answer to the question whether the last correction loop should be ordered, with FC-3 restoring option 1 of Q-FC3-02: "Yes, the last loop."
3. Orchestrator's English rendering of a chat message of 2026-11-16, given after the delegate's final list of what the Operator must still do (items 1-8 of section 5 of the delegate's second delta review of the code-wave tickets; a session artefact, not published): "I accept the recommended answers to all items."

## What message 1 covers (records by content hash)

The recorded questions and the recommended option of each are in four session files (not published; one open-questions file per ticket). The table gives the SHA-256 that the orchestrator recorded for each file when it made this transcription.

| Ticket | Open-questions file | sha256 as recorded at transcription |
|---|---|---|
| T-IGX-043-A (FC-1) | FC-1 open questions | `544894ba6131415c1e11d5a4773466fd4b6c24f031c601537562f0b090841f33` |
| T-IGX-043-B (FC-2) | FC-2 open questions | `3f031b396f65639f4ba5e2ee3f39726215443e6bcfbb920a25dcbd14e3502a72` |
| T-IGX-043-C (FC-3) | FC-3 open questions | `9c0b21ab796ca7d1d73cb2876d427836c10686ebe49eeb3d27cb9048513c35db` |
| T-IGX-043-D (FC-4) | FC-4 open questions | `f3e0ef73fbef69bf7a152a653bd3a27693f01d05ae85006cfe7e4bc3fada03d0` |

Verification note added at publication: a re-computation on 2026-11-16 of the SHA-256 of the current question files gave different values for all four tickets. For FC-1 the recorded value matches a preserved copy of the question file as accepted (the copy is a session artefact). For FC-2, FC-3 and FC-4 the recorded values could not be reproduced, and the cause is not established. The recorded values are therefore reproduced here as recorded, and are not verified.

The Operator decision record is [`OPERATOR_DECISIONS_EQOA_REMEDIATION.md`](OPERATOR_DECISIONS_EQOA_REMEDIATION.md).

## What message 3 covers: items 1-8 and how each was applied

Where the delegate's list had a stated recommendation, that option is recorded. Where it had none, the orchestrator applied the answer the ticket's own plan drafts, and says so. The Operator may change any entry.

| # | Item | Recorded answer | Basis |
|---|---|---|---|
| 1 | Publish or sign the answers to the FC questions | This transcription is the publication candidate; publication as a docs-only PR or the Operator's signature is still an act to be performed | stated recommendation |
| 2 | D-FC1-03 and the write-set ceiling of A (S-08) | D-FC1-03 = option D (hosts with HITL inject a store built from the existing SQLite persistence composition; hosts and workers without HITL pass the typed acknowledgement `disable_human_decision_persistence=True`; the check is presence plus not in-memory). The ceiling of up to 21 conditional product paths is accepted | stated recommendation (D) and plan as drafted (ceiling) |
| 3 | C open items O-08, O-09, L-1 | O-09: option 1, accept 62 product files. O-08: the plan as drafted (option 1: late refusal by the coordinator guard, class BLOCKS kept, the unrefused paths named in the audit; options 2 and 3 not chosen). L-1: the plan as drafted | O-09 stated recommendation; O-08 and L-1 had no stated recommendation, plan as drafted applied |
| 4 | OPEN-ATT-1 for D (who may attest Redis durability) | Fallback (b): the `SKILLS.md` procedure states the attestation shape only; who may attest stays an open item; D is approvable under (b). No new authority grant is recorded | no stated recommendation; the conservative option applied because it is an authority grant |
| 5 | Personal rulings on E and F | E: both parts (capacity gate repair, sandbox gate) accepted as drafted; the sandbox part may be ruled out separately without affecting B. F: extend the CONFIG-X scan, scoped to the eight FC-3 paths with a recorded finding per hit outside them (3a), same wave behind T-IGX-043-C, and the scratch measurement F-A2 (non-certifying, one process, read-only) is authorised, to be run before F's approval line. The approval lines of E and F remain the Operator's own and are NOT written by this transcription | F items 3 and 4 stated recommendation; E and F item 1 as drafted |
| 6 | Merge order | B, then A, then C, then F; D at any time; W-A/W-B before A on `host_orchestration_environment_spec_builder.py`, `harness_host_runtime.py` and `nexus_loop.py` | stated recommendation |
| 7 | Owners and EDER rows for the accepted residuals; freeze-document exception statements | To be assigned in the Enterprise Debt & Exceptions Register; the owner of each residual is the owner named in the plan until the Operator confirms the 55-ID vocabulary (DP-47); the exception statements (A S-04, B DP-30 text) are written with the freeze documents | stated recommendation |
| 8 | Milestone M1 and the product-path go | The Operator's go for product-path work (AR-39..AR-44; wave package W-C covers FC-1..FC-4) is recorded as: GRANTED, EFFECTIVE ONLY WHEN M1 (PDC P9 closed) HOLDS. M1 itself is NOT confirmed by this transcription: [`PDC_INDEX.md`](../platform_domain_completeness/PDC_INDEX.md) at `7960ae7f6` lists the P9 artefacts as awaiting the Operator's final confirmation. The approval lines of A-D are to be written through a commissioned non-implementing session (the commission is still to be given in chat with scope and `Expires`) | stated recommendation, M1 left open |

## What is not decided by this file

M1 (PDC P9 closure), the commission of a delegate for the approval lines of A-D, the approval lines of E and F, the publication of this record, the owner vocabulary (DP-47), DP-54, and every authorisation not named above (tests, probes, S1 diagnostics beyond F-A2, GitHub settings).

## Addendum: message 4 and the commission (2026-11-16, after the table above)

4. Orchestrator's English rendering of a chat message of 2026-11-16: "I grant all consents; carry out the full implementation." Given after the orchestrator listed what was still open (M1, the commission for the approval lines, the approval lines of E and F, publication of this record, the scratch measurement F-A2, DP-47, DP-54, GitHub settings).

How the orchestrator applies message 4 (an interpretation; the Operator may correct it at any time; `Expires`: none stated). The rows below are the orchestrator's interpretation of message 4, not decisions stated by the Operator, except where a row says the item is not decided:

| Item | Applied as |
|---|---|
| M1 (PDC P9 closed) | Treated as CONFIRMED by the Operator for the purposes of this program. The PDC-owned P9 records and `PDC_INDEX.md` are NOT edited by this program (they remain the PDC session's). Observed at `origin/development` `7960ae7f6`: the index still shows the P9 artefacts as awaiting confirmation; open PRs #610 and #611 (ARCH-FREEZE wave 2, branches `af2/*`) show that the ARCH-FREEZE wave 2 effort is running in parallel |
| Product-path go (AR-39..AR-44, wave package W-C covers FC-1..FC-4) | GRANTED. Implementation of a ticket still starts only after its plan has an approval line and its ticket is in the repository |
| Commission under IIAC Amendment 01 (R-1..R-6, chat is sufficient) | GRANTED to ONE new session dedicated solely to writing the approval lines of T-IGX-042-A..D (draft PRs #605..#608 at the time of writing) and, once they are in the repository, T-IGX-043-A..D. It is not an implementation session, not a continuation of any other session, and it writes no line for T-IGX-043-E or T-IGX-043-F |
| Approval lines of E and F | NOT commissioned: gate-meaning changes are the Operator's personal act under the standing rules; the Operator writes them or states a commission for them |
| Scratch measurement F-A2 | AUTHORISED (non-certifying, one process, read-only, scratch worktree at the baseline) |
| Publication of this record | AUTHORISED as a docs-only PR (ticket T-IGX-042-E) |
| GitHub settings (AR-45..AR-53 rulesets) | Treated as covered in principle, but NOT executed by the orchestrator without a named go for the first step, because required checks on `development` can block merges of the concurrent ARCH-FREEZE sessions (#610, #611) |
| DP-47 owner vocabulary, DP-54 ECAP executable | NOT decided by message 4 |

## Addendum 2: message 5 and the extension of the commission (2026-11-16, later the same day)

5. Orchestrator's English rendering of a chat message of 2026-11-16: "Yes, extend the delegate's commission." Given in answer to the orchestrator's question whether the commission under IIAC Amendment 01 should be extended to the plan approval of the new EQO-A tickets.

How the orchestrator applies message 5 (an interpretation; `Expires`: none stated; the commission stays with the same new, dedicated signing session, for the duration of the delegation). The rows below are the orchestrator's interpretation of message 5, not decisions stated by the Operator:

| Item | Applied as |
|---|---|
| Added scope | Approval lines of the tickets T-IGX-042-E, T-IGX-044-A, T-IGX-044-B, T-IGX-044-E, T-IGX-046-A, T-IGX-046-B, T-IGX-046-C and T-IGX-047-A, each only when its plan passes the checks of the delegate checklists and only on the PR branch that carries it |
| Still excluded | Tickets that change a gate's meaning (T-IGX-043-E, T-IGX-043-F and every ticket of the nodes PT-G01, PT-G02, PT-E03, PT-E02 when it rewrites a validator or a gate) and tickets of the material governance paths (PT-E05, PT-D08): the Operator's personal act or a separate commission |
| Not changed | The approval of a plan is not an implementation go; product paths stay held until the ARCH-FREEZE gate SCOPE; merges need the Operator's separate order per PR |

Note added at publication: of the ticket ids named in the added scope, only `T-IGX-042-E` is the ticket of the pull request that carries this file; the `T-IGX-044-*`, `T-IGX-046-*` and `T-IGX-047-A` tickets are not on `development` at the time of writing. This file writes no approval line for any of them.

## Addendum 3: message 6, the pending decisions on three pull requests (2026-11-17)

6. Orchestrator's English rendering of a chat message of 2026-11-17: "Go through these decisions and set the recommended ones." Given after the orchestrator reported that three items waited on the Operator: pull request 616 (ticket T-IGX-044-B, three workflow files under `.github/`), pull request 623 (ticket T-IGX-044-E, new read-only IIAC verifiers under `scripts/` and `tests/`), and the node PT-E02 (ticket T-IGX-044-C, ticket-approval validators, a new workflow, held locally). The Polish original is in the chat of that date and is not reproduced here. This section is a transcription by the orchestrator under IIAC Amendment 01 section 1.2, not an Operator signature; the Operator may correct any row. `Expires`: none stated.

How the orchestrator applies message 6 (an interpretation limited to the three items above; the recommended option of each is the one the orchestrator stated in its reports):

| Item | Recommended option, applied as |
|---|---|
| PR 616 (T-IGX-044-B, PT-E01) | Merge after an independent read-only review at the exact head SHA reports PASS and the CI of that head is green. DP-65 ("if stable") is applied: `governance-policy-pin` becomes blocking. Two side questions of the PR body are answered "not now": `harness-release.yml` and `sync-repository-metadata.yml` stay on `ubuntu-latest` and the stale "ADVISORY" step label stays; both may be taken up in a later ticket. No GitHub setting, branch protection or ruleset is changed; the trial ruleset (AR-45/AR-50) still needs its own named go after this PR is merged |
| PR 623 (T-IGX-044-E, PT-E04) | Design point (a): the strict SHA chain (audited <= test_execution <= audit_execution <= evidence <= certification) is the rule for new manifests, enforced in profile `final` only; `legacy` keeps it a warning. (b): the known-draft allowlist stays at `scripts/governance/iiac_known_draft_manifests.v1.json`. (c): the delegation verifier stays separate from the PT-E02 validators; folding them is decided together with PT-E05. The approval line of T-IGX-044-E is written by the signing delegate (already in its scope, addendum 2). Merge after the independent audit AUD-ENF-2 at the exact head SHA reports PASS and the CI of that head is green. The verifiers stay unwired until PT-E05 |
| Node PT-E02 (T-IGX-044-C) | The validator-code change is authorised for this ticket only (the separate authorisation required by OD-REM-07), including the new workflow `ticket-governance.yml` as drafted; the job is not made a required check and no branch setting is changed. The delegate's commission (addendum 2) is extended to the approval line of T-IGX-044-C, which addendum 2 had excluded as a validator-rewriting ticket, because the Operator has now decided the matter. The approval line is written only when the plan passes the delegate checklist and the line cites this addendum. Merge after an independent review at the exact head SHA reports PASS and the CI of that head is green |
| Not changed | Nothing here covers T-IGX-043-E, T-IGX-043-F, T-IGX-050-*, T-IGX-051-*, PT-E03, PT-E05, PT-D08, PT-G01, PT-G02, DP-47, DP-54, the scope ruling on A4-040, or any GitHub setting; those stay as recorded in addenda 1 and 2 and in the ARCH-FREEZE decision register. A merge still needs the order of the Operator, and this message is that order only for the three items above |
