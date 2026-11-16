# IIAC operator signing delegation record - MCP-SERVER-X P7/P8 authorization round (tickets H and I, OD-MCP-P7-P8-AUTH-01, OD-MCP-P8-CLOSE-01)

**Status of the governing rule:** `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` is IN FORCE as operator-adopted policy text (revised 2026-11-08). This record is a transcription of a commission given in chat, not a condition of validity (S-1 struck). It is written by the designated signing session itself, not by the operator; no operator signature line is claimed.

**Delegation ID:** `DELEG-MCP-P7-P8-AUTH-2026-11-12`
**Policy version:** `IIAC-2026-11-05.1`
**Date (ISO-8601):** `2026-11-12` (record written on 2026-11-12 at about 09:07 +01:00, system clock of the designated session)

## Commission (transcription)

The operator, in the Claude Code chat on 2026-11-12, approved the P7/P8 authorization pack and the start of the signing round. The text below is what the orchestrator session passed to the designated session (mission file `.tmp/session/mcp-p7p8-auth/SIGNING_ROUND_MISSION.md`). Polish originals are quoted verbatim as relayed; each English line is a labelled rendering by the orchestrator session and is **not** an operator-written line.

Operator decisions given in chat for the P7/P8 campaign:

- Original (Polish): `Residual bez zmiany produktu (Recommended)`. Rendering: "Residual without a product change (recommended)". Meaning: `REQ-MCP-OBS-01` / MCP-ACC-17 trace id stays a documented residual, no product change.
- Original (Polish): `Tak, po zielonych checkach (Recommended)`. Rendering: "Yes, after green checks (recommended)". Meaning: the orchestrator may merge PRs with a merge commit after all required checks finished green.
- Original (Polish): `Tak, warunkowa decyzja teraz (Recommended)`. Rendering: "Yes, conditional decision now (recommended)". Meaning: prepare the conditional closure decision for signature now.
- Original (Polish): `Tak, ograniczone pakiety (Recommended)`. Rendering: "Yes, bounded packages (recommended)". Meaning: bounded qualification and replay packages only.
- Approval of the pack, in answer to the question "Zatwierdzasz pakiet i start rundy podpisu?" (original, Polish): `Zatwierdzam pakiet`. Rendering: "I approve the pack".

Scope of the commission: sign exactly the four documents below, nothing else. Act immediately (R-5).

Orchestrator session (drafted the pack, relayed the commission): `local_4dfbaa01-6b9e-4e2a-ab65-2d4c81e01bd8` (the sender identifier shown by the app on the brief of the designated session).

## Delegated acts

Stage: MCP-SERVER-X P7 (exact-SHA qualification package) and P8 (independent audit, IIAC certification, stage closure) authorization round (tickets `governance/tickets/T-IGX-035-H.md` and `governance/tickets/T-IGX-035-I.md`).

Unsigned drafts: commit `a7204c98c9dbe6979f31137af8f9dfdb74501aca` on `mcp-p7-authorization` (verified contained in `origin/mcp-p7-authorization` before signing; the branch tip equalled that commit). The digests below are SHA-256 of the file bytes at that commit; the signed text necessarily differs (signature lines, state change, provenance notes). Scope text, plan digests and conditions inside the records are not changed by the signing.

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `a7204c98c`) | Delegated act |
|---|---|---|---|---|
| 1 | `T-IGX-035-H` (plan approval, P7) | `governance/tickets/T-IGX-035-H.md` | `6c44aaebe9cc94311af5cf96c0b0f11c73951ef7cb84d27acfcfd51b1afc2329` | replace the `PENDING` line under `## Approved` by `Approved: operator-delegate @ <ISO-8601>` and `Delegation: <this record path>@<commit D sha>`; set `## Lifecycle state` to `**plan_approved**`; nothing else |
| 2 | `T-IGX-035-I` (plan approval, P8) | `governance/tickets/T-IGX-035-I.md` | `ef9c6c7dd4cde7b1d0ff83e50ba258df1b855c6580e1d21db9762c8b50ef57c3` | the same act as row 1 |
| 3 | `OD-MCP-P7-P8-AUTH-01` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P7-P8-AUTH-01.md` | `074f65c0e105af0395981e00276cf8fee94aebaf09fa556048ade0ffa9407700` | status `APPROVED` and the line `Approved: OD-MCP-P7-P8-AUTH-01 @ <ISO-8601>` (plain form), plus a provenance note (written by a designated signing session under this record; not claimed to be operator-written) |
| 4 | `OD-MCP-P8-CLOSE-01` | `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/OD-MCP-P8-CLOSE-01.md` | `c3bfa15e615d8c3f181b580fa68193fdca2583e79b0f580cc42bff6a03bc1970` | status `APPROVED (conditional: void unless C1-C10 hold)` and the line `Approved: OD-MCP-P8-CLOSE-01 @ <ISO-8601>` (plain form), plus the same provenance note and one sentence that the decision takes effect only after an independent report with the verdict `P8_TERMINAL_VERIFIED` |

All four signature timestamps use one style, taken once from the system clock at signing; none is composed.

Out of scope: any other decision, any product-path file, any merge or pull request, FRZ promotion, the SHA ladder, auditor or certifier artifacts, the plan file, and any statement that MCP-SERVER-X is closed.

## Designated signing session

- Designated session id: `972cc2df-3310-42b8-960a-ff67886b05f8` (the session identifier shown by the Claude desktop app in this session's scratchpad path; the designated session has no other identifier it can verify and does not invent one)
- Designated agent id: Claude Code, model `claude-sonnet-5-5`
- Dedication statement (operator): the commission requires that the designated session is a NEW session, dedicated solely to the signing task named above, NOT a continuation of any other session and NOT an implementation session. The operator's confirmation is the commission above, relayed by the orchestrator session; no `[x]` is ticked on the operator's behalf.
- Dedication statement (designated session, recorded in this record and in its own messages): this session (a) is a new session, started for this task by the orchestrator session on the operator's commission; (b) is dedicated solely to writing the four signatures named above and this record; (c) is not a continuation of any other session; its only inputs are its brief, the shared notes folder, the mission file named above and repository files (the notes folder holds hand-over notes of earlier signing and certifier sessions, read as data only); it is a different session from the earlier P6 signing session `82703e33-e552-43b7-94c2-94e5944cfd2d` and from the linked P6 certifier and terminal-verifier sessions; (d) is not an implementation session: it has written no product-path code and no tester, auditor or certifier artifact.
- Implementer session id(s) of this stage (must differ from the designated session id): the P7 and P8 implementer, tester, auditor, certifier and verifier sessions do not exist yet (they are started after this signature, under `OD-MCP-P7-P8-AUTH-01` SA-2). The P6 implementer label recorded in the repository is `T-IGX-035-G-MCP-P6-IMPLEMENTER` (no UUID recorded; none invented). The designated session id differs from it and from the orchestrator session id above.

## Limits

- Expires: 24 hours after the commission (operator answer relayed by the orchestrator session). The exact commission time is not stated in the brief and is not invented here; the commission is dated 2026-11-12, so `Expires` is not earlier than `2026-11-13T00:00:00+01:00`. The signing timestamps are written from the system clock on 2026-11-12, inside that window on any reading (R-5: act immediately).
- Single use: the delegation covers only the four Decision IDs and the digests above. Confirmed by the designated session's scope: no other record is signed.

## Operator signature line

Not required under Amendment 01 (S-1 struck) and not filled. This record carries no operator-written line.

## Non-actions

This delegation and the signatures it authorizes do not close MCP-SERVER-X (the closure decision is conditional and takes effect only after an independent `P8_TERMINAL_VERIFIED` report), do not close ARCH-FREEZE, do not promote any FRZ criterion, do not authorize any product-path change, production configuration, Docker or new vendors, do not waive OPEN-BP-01 or any finding, and do not retroactively approve any earlier work. Assurance level: `process_enforced` (operator attestation in chat plus the designated session's own statement; no cryptographic proof of operator identity or session novelty).
