# IIAC operator signing delegation record - MCP-SERVER-X P6 (ACT-N11 P6 activation + OD-MCP-P6-GOV-01)

**Status of the governing rule:** `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` is IN FORCE as operator-adopted policy text (revised 2026-11-08). This record is a transcription of a commission given in chat, not a condition of validity (S-1 struck). It is written by the designated signing session itself, not by the operator; no operator signature line is claimed.

**Delegation ID:** `DELEG-MCP-P6-GOV-2026-11-12`
**Policy version:** `IIAC-2026-11-05.1`
**Date (ISO-8601):** `2026-11-12` (record written on 2026-11-12 between 06:30 and 07:00 +01:00, system clock of the designated session)

## Commission (transcription)

The operator, in the Claude Code chat on 2026-11-12, chose a separate designated signing session instead of signing personally. The text below is what the orchestrator session passed to the designated session; the Polish original is quoted verbatim, the English line is a labelled rendering and is **not** an operator-written line.

- Preceding operator instruction (original, Polish): `Sesja podpisująca`
- Operator commission (original, Polish): `Uruchom w moim imieniu pod sesję i zleć certyfikację.`
- English rendering by the orchestrator session (not an operator-written line): "Start, in my name, a sub-session and commission certification."
- Operator answers recorded earlier in the chat (as relayed by the orchestrator session): OD-MCP-P6-GOV-01 -> option A; Expires -> 24 hours after the commission.
- Orchestrator session (drafted the decisions, relayed the commission): `local_4dfbaa01-6b9e-4e2a-ab65-2d4c81e01bd8`.

Follow-up operator instruction given directly to the designated session in chat on 2026-11-12, after the designated session stopped because PR #460 had already been merged (original, Polish): `Nowa gałąź z development`. English rendering by the designated session (not an operator-written line): "A new branch from development." Effect: the signatures are committed on the new branch `mcp-p6-gov-signing`, created from `origin/development` at `bc9ef52bd690738e71638a599eec64c76ac356c9`, instead of on `mcp-p6-gov-closeout`. The scope, decisions and `Expires` of the commission are unchanged.

## Delegated acts

Stage: MCP-SERVER-X P6 governance closeout (ticket `governance/tickets/T-IGX-035-G.md`).

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `bc9ef52bd`) | Delegated act |
|---|---|---|---|---|
| 1 | `IIAC-OD-MCP-ACT-N11-P6-2026-11-11` | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` | `203798c085260914d5450efd8930512e42b644582266d75a41389ef8b9549261` | authorize: `Decision status: AUTHORIZED` and the gate line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` (plain form, because the IIAC activation gate `_SIGNATURE` accepts only that form; it is not claimed to be operator-written) |
| 2 | `OD-MCP-P6-GOV-01` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P6-GOV-01.md` | `a52e08020292881d864eac22ff555f36564338742412efb49a73e151004b9bf9` | option **A**: status `APPROVED — Option A` and the line `Approved: OD-MCP-P6-GOV-01 option A @ <timestamp>` |

The digests above are of the unsigned drafts as checked out at `bc9ef52bd`; the signed text necessarily differs (signature lines and provenance notes). Scope text, digests and SHAs inside the records are not changed by the signing.

Out of scope: `OD-MCP-P6-TICKET-01`, `OD-MCP-P6-P7-01`, any merge, any edit of `IIAC_POLICY_ACTIVATION_RECORD.json` (W4, a separate certifier session), auditor artifacts, product paths, the SHA ladder and the ticket.

## Designated signing session

- Designated session id: `82703e33-e552-43b7-94c2-94e5944cfd2d` (the session identifier shown by the Claude desktop app in this session's scratchpad path; the designated session has no other identifier it can verify and does not invent one)
- Designated agent id: Claude Code, model `claude-sonnet-5-5`
- Dedication statement (operator): the commission requires that the designated session is a NEW session, dedicated solely to the certification task named above, NOT a continuation of any other session and NOT an implementation session. The operator's confirmation is the commission above, relayed by the orchestrator session; no `[x]` is ticked on the operator's behalf.
- Dedication statement (designated session, recorded in this record and in its own messages): this session (a) is a new session, started for this task by the orchestrator session on the operator's commission; (b) is dedicated solely to writing the two signatures named above and this record; (c) is not a continuation of any other session; its only inputs are its brief, the shared notes folder (empty at start), the mission file `.tmp/session/mcp-p6-closeout/SIGNING_SESSION_MISSION.md` and repository files; (d) is not an implementation session: it has written no product-path code and no auditor or certifier artifact, and it does not perform W4 certifier work.
- Implementer session id(s) of this stage (must differ from the designated session id): `T-IGX-035-G-MCP-P6-IMPLEMENTER` (label recorded in the decision file; no UUID is recorded in the repository and none is invented). The orchestrator session that drafted the decisions is `local_4dfbaa01-6b9e-4e2a-ab65-2d4c81e01bd8`. The designated session id differs from both.

## Limits

- Expires: 24 hours after the commission (operator answer relayed by the orchestrator session). The exact commission time is not stated in the brief and is not invented here; the commission is dated 2026-11-12, so `Expires` is not earlier than `2026-11-13T00:00:00+01:00`. The signing timestamps are written from the system clock on 2026-11-12 before 08:00+01:00, inside that window on any reading (R-5: act immediately).
- Single use: the delegation covers only the two Decision IDs and the digests above. Confirmed by the designated session's scope: no other record is signed.

## Operator signature line

Not required under Amendment 01 (S-1 struck) and not filled. This record carries no operator-written line.

## Addendum 1 (2026-11-12): third delegated act, OD-MCP-P6-GOV-02

Commit D2. This addendum extends the delegated acts above with one further decision. It does not alter rows 1 and 2, their signatures (commit `4bf5a62ec`) or the `Expires` value, which is unchanged.

Relay chain. The commission extension below was relayed to the designated session by the orchestrator session `local_4dfbaa01-6b9e-4e2a-ab65-2d4c81e01bd8` (this id is verified: it is the sender identifier shown by the app on the cross-session message and in the designated session's own system context; the earlier statement of this record that the orchestrator id is unverified does not apply). The operator's answers below were relayed by the orchestrator session; their original Polish text is quoted verbatim as relayed, and the English renderings are by the orchestrator session and are not operator-written lines.

- Operator answer, branch question (original, Polish, as relayed): `mcp-p6-gov-signing (Recommended)`. The operator also told the designated session directly in chat: `Zostaw gov-signing` (English rendering by the designated session: "Leave gov-signing"). Effect: `mcp-p6-gov-signing` is the canonical branch for the signatures; the branch `mcp-p6-activation` is unused.
- Operator answer, F-MCP-P6-GOV-003 question (original, Polish, as relayed): `Osobny OD-MCP-P6-GOV-02 (Recommended)`. English rendering by the orchestrator session: "Separate OD-MCP-P6-GOV-02 (recommended)". The option offered to the operator was described as a separate short decision about F-MCP-P6-GOV-003 in the same pattern as OD-MCP-P6-GOV-01 (option A: accept with the finding), signed by the signing session under the operator's commission (valid until 2026-11-13).
- Direct operator confirmation to the designated session in chat, after the designated session asked whether the operator personally confirms the signature of `OD-MCP-P6-GOV-02` option A (original, Polish): `Tak, podpisz`. English rendering by the designated session (not an operator-written line): "Yes, sign."

History. The first attempt of this designated session stopped on the merged PR #460 and wrote no signature; the signatures of rows 1 and 2 were then written on the new branch. The same designated session (id above) signs row 3; it is not a new session and it is not a continuation of any implementation session. A later cross-session message that said the session had signed nothing yet was incorrect and was not acted on.

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `2923b49df`) | Delegated act |
|---|---|---|---|---|
| 3 | `OD-MCP-P6-GOV-02` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P6-GOV-02.md` | `12a6b6fcf581b627a5933b46d4d5ceaaf075e1b82e5c9bcd4ec929d76068a173` | option **A**: status `APPROVED — Option A` and the line `Approved: OD-MCP-P6-GOV-02 option A @ <timestamp>` |

Single use for row 3: the extension covers only `OD-MCP-P6-GOV-02` and the digest above. The same out-of-scope list applies (no merge, no activation-record edit, no signature of `OD-MCP-P6-TICKET-01` or `OD-MCP-P6-P7-01`). The assurance level stays `process_enforced`.

## Addendum 2 (2026-11-12): fourth and fifth delegated acts, OD-MCP-P6-TICKET-01 and OD-MCP-P6-P7-01

Commit D3. This addendum extends the delegated acts with two further decisions. It does not alter rows 1-3, their signatures (commits `4bf5a62ec`, `d7451dff0`) or the `Expires` value, which is unchanged (not earlier than `2026-11-13T00:00:00+01:00`; the signing timestamps are on 2026-11-12).

Branch. The activation PR #464 is merged (`66dcd1bfe260b33a41881c7b9911b2a2abb752f9`); the branch `mcp-p6-gov-signing` is merged and obsolete. The signatures of rows 4 and 5 are written on `mcp-p6-closure` (cut from `origin/development` @ `dcdb35601305a2ad9a3ee22894b26c29800a9640`, drafts at `73d71c5dc31ec72a891d481d83ced1dfa93b77da`). The designated session works in its own detached worktree at that branch.

Relay chain. The commission extension was relayed by the orchestrator session `local_4dfbaa01-6b9e-4e2a-ab65-2d4c81e01bd8` (id verified as in addendum 1). Original Polish operator answers as relayed, with English renderings by the orchestrator session (not operator-written lines):

- Question "Kto podpisze OD-MCP-P6-TICKET-01 i OD-MCP-P6-P7-01?" -> `Sesja podpisująca (Recommended)` ("Signing session (recommended)").
- Question about creating the branch -> `Tak, mcp-p6-closure (Recommended)` ("Yes, mcp-p6-closure").
- Instruction `Najpierw podpis, potem kolejne rzeczy.` ("First the signature, then the next things.").

Direct operator confirmation to the designated session in chat, after the designated session reported its read-only condition checks and asked whether the operator personally commissions both signatures (original, Polish): `Tak, podpisz oba`. English rendering by the designated session (not an operator-written line): "Yes, sign both."

Same designated session. Rows 4 and 5 are signed by the same designated session as rows 1-3 (id above); it is not a new session and not a continuation of any implementation session.

Conditions of `OD-MCP-P6-TICKET-01`, verified by the designated session before signing (read-only):

1. G6 `ACCEPT_SCOPED` / IIIC `PASS_SCOPED` evidence is published on `origin/development` (evidence published at `2d3eceb6c`, in `origin/development` history; this session did not re-audit it).
2. `verify_iiac_policy_activation --require-in-force` exits 0 on this tree: `IN_FORCE @ IIAC-2026-11-05.1 from 92ad3555775649742299826e3680ec8ce703cc8a`; the activation record binds `audit_execution_sha` `2d3eceb6c`.
3. The `Regression gate` push run on `origin/development` @ `dcdb35601` (run `43167849565`) concluded `success`. The earlier publication merge of PR #460 was red; this is F-MCP-P6-GOV-003, preserved and covered by `OD-MCP-P6-GOV-02`, not waived.
4. The activation OD, `OD-MCP-P6-GOV-01` and `OD-MCP-P6-GOV-02` are signed (option A) on `origin/development`/this branch.
5. PR #464 is merged as `66dcd1bfe260b33a41881c7b9911b2a2abb752f9`; the same run shows `governance-fast`, `ci-smoke` and `evol-x-b4-preflight` each `completed/success`.

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `73d71c5dc`) | Delegated act |
|---|---|---|---|---|
| 4 | `OD-MCP-P6-TICKET-01` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P6-TICKET-01.md` | `6a10bc6bdc6e071b3a0bda3101de09e238de92b5a1c15aa8a21d72c9a666b051` | status `APPROVED` and the line `Authorized: operator @ <timestamp>` (second line of the block unchanged) |
| 5 | `OD-MCP-P6-P7-01` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P6-P7-01.md` | `a61dca4d9611c6d9dd8f55e657e72af97fac4b5d00d4e43923cc9153f154ebde` | status `APPROVED` and the line `Authorized: operator @ <timestamp>` (second line of the block unchanged) |

Single use for rows 4 and 5: the extension covers only these two Decision IDs and the digests above. Still out of scope: any merge, any edit of `IIAC_POLICY_ACTIVATION_RECORD.json`, any edit of the ticket file itself (the lifecycle transition the TICKET-01 decision authorizes is performed by a separate step, not by the signing session), any other decision. The effect of each decision is only what its own text states; `OD-MCP-P6-P7-01` takes effect only at the origin/development SHA where the P6 terminal verification PASS is published. The "Non-actions" section below applies to rows 1-3 and does not restate or widen the effect of rows 4 and 5.

## Non-actions

This delegation and the signatures it authorizes do not retroactively approve PR #459, do not waive any governance control, do not promote FRZ, do not declare MCP-SERVER-X closed and do not authorize P7. Finding F-MCP-P6-GOV-001 / F-MCP-P6-GOV-002 wording is unchanged. Assurance level: `process_enforced` (operator attestation in chat plus the designated session's own statement; no cryptographic proof of operator identity or session novelty).
