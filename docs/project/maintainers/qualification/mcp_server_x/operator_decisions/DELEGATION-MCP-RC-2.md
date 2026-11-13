# DELEGATION-MCP-RC-2 — IIAC operator signing delegation record for the MCP-RC activation line

**Status of the governing rule:** `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` is IN FORCE as operator-adopted policy text (revised 2026-11-08). This record is a transcription of a commission given in chat, not a condition of validity (S-1 struck). It is drafted by the activation-package drafting session, not by the operator and not by the designated signing session; it carries no operator signature line and no signature of any kind. Fields marked `PLACEHOLDER` are filled only by the designated signing session, from facts it can verify. Nothing in them is invented here.

| Field | Value |
| --- | --- |
| **Delegation ID** | `DELEGATION-MCP-RC-2` |
| **Policy version** | `IIAC-2026-11-05.1` |
| **Stage** | MCP-SERVER-X corrective campaign (MCP-RC), ticket `governance/tickets/T-IGX-038-F.md` |
| **Date (ISO-8601)** | `2026-11-13` (record drafted; commission given in chat the same day) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6; `OD-MCP-RC-01` (D-3); precedent `docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_ACT_N11_P6_2026-11-12.md` |
| **Not the same as** | `DELEGATION-MCP-RC-1` (plan approvals of `T-IGX-038-A..F`; it expressly excludes IIAC activation lines) |

## Commission (transcription)

The operator answered two questions in the Claude Code chat on 2026-11-13. The answers were relayed to the drafting session by the orchestrator session `local_329f8bde-1685-4aee-abe0-78d0ef7304ef` (the session that started the drafting session; its id is the one the app gave in the drafting session's brief). The drafting session did not see the operator's chat itself. The Polish option labels are quoted as relayed; every English line is a labelled rendering and is **not** an operator-written line.

1. Question (original, Polish, as relayed): `Kto podpisuje linię Approved IIAC policy activation po tych certyfikacjach?`
   - Operator answer (original, Polish): `Nowy delegat (Recommended)`
   - English rendering by the orchestrator session (not an operator-written line): "New delegate (recommended)".
   - Description of the option given to the operator, as relayed: "narrow new delegation for a separate session; scope only this line, after the delta re-audit PASS and the preflight, no rebind and no merge".
2. Question (original, Polish, as relayed): `Czy upoważniasz do dwufazowego rebindu ACT-N11 przez osobną sesję certyfikatora aktywacji, gdy PDC-X otworzy okno PDC-P7-00?`
   - Operator answer (original, Polish): `Tak, gdy okno otwarte (Recommended)`
   - English rendering by the orchestrator session (not an operator-written line): "Yes, when the window is open (recommended)".
   - Effect: the two-phase rebind is a different act, performed by a different session (the activation certifier), only inside the PDC-P7-00 window. It is **not** part of this delegation and the designated signing session has no role in it. It is transcribed here only so that the two commissions are not confused.

Recording of the operator's direct confirmation to the designated signing session (the P6 designated session asked the operator to confirm the signature personally in chat before writing it): original text: `PLACEHOLDER (verbatim, to be filled by the designated signing session)`; English rendering by the designated session, labelled as not operator-written: `PLACEHOLDER`.

## Delegated act

Stage: MCP-RC ACT-N11 activation decision (ticket `governance/tickets/T-IGX-038-F.md`).

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `83e587553e163c2e401e21a1947f8615368ea316`) | Delegated act |
|---|---|---|---|---|
| 1 | `IIAC-OD-MCP-RC-ACT-N11-2026-11-13` | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` | `f5116d96e030d93ca5122f5e481d04718e90e2a1f949f1e916533733c7650873` | (a) replace the standalone placeholder line `________________` by the plain line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` (plain form, because the IIAC activation gate accepts only that form; it is not claimed to be operator-written); (b) change `Decision status: DRAFT` to `Decision status: AUTHORIZED`; (c) replace the `Drafting note` paragraph by a `Provenance note` paragraph in the P6 wording (the line was written by an operator-designated signing session under this record and the operator's chat commission of 2026-11-13; the plain form is not claimed to be operator-written; the timestamp is the designated session's system clock at signing; this record named with its commit SHA, read from `git log`). Nothing else. |

The digest is computed over the committed (LF, `core.autocrlf=false`) blob of the unsigned draft at the commit named in the column heading. It was recomputed from that commit by the drafting session. The signed text necessarily differs (signature line, status, provenance note); scope text, digests and SHAs inside the record are not changed by the signing. The timestamp is ISO-8601 with offset, taken from the designated session's own system clock at the moment of signing (R-5: act immediately); it is never composed in advance and never copied from another record.

### What the designated session verifies before it signs (read-only)

The operator's conditions are: the delta re-audit is PASS and the certifier preflight is passed. Both are recorded on the branch that carries this record:

1. Delta re-audit dossier verdict `PASS_SCOPED` (`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md`; manifest `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json`, field `dossier_verdict`).
2. Certifier preflight verdict `CERTIFIER_PREFLIGHT_PASS_SCOPED` (`iiac_evidence/PDC-X/MCP-RC/certifier/certifier_preflight_result.json`, field `verdict`).
3. The unsigned draft at the checkout it signs in hashes to the digest in the table above, and the digests and SHAs quoted inside that draft match the files and git (spot-check at least the dossier, findings, scope and matrix digests of both audits and the evidence publication SHA `78d4c496071141ca4092b926b838fb80016f1b91`).
4. The draft still contains no `Approved IIAC policy activation` line, and nothing in `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json` has changed.

5. **Widened scope.** The draft's section "Widened scope" must no longer say that scope item 2 is not covered: a revision of the draft must already cite, from published evidence, the integrated baseline SHA, the path count and audit-scope file, and the independent audit dossier and certifier preflight covering the merged `development` paths. That revision changes the digest; its new digest and commit must be recorded in an addendum to this record (Addendum 1, written by the orchestrator or drafting session, not by the designated session) before it signs. Until then the designated session must not sign.

If any of these fails, the designated session writes nothing and reports to the orchestrator session.

The draft lists operator-owned open items (named owner of the `[product]` xfail via `governance/tickets/T-IGX-039-A.md`, acceptance of RA-02..RA-04 and AF-05, and the treatment of the 125 non-MCP product paths on `development`). This delegation neither requires nor implies their resolution, and the signed line does not resolve them. The designated session lists them as still open in its report and does not claim otherwise.

Out of scope (the designated session does none of these): any IIAC rebind or edit of `IIAC_POLICY_ACTIVATION_RECORD.json` (phase 1 and phase 2 belong to the activation-certifier session, see `mcp_server_x/MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md`), any merge, any pull request, any ticket approval or lifecycle change (including `T-IGX-039-A.md`), any FRZ change (counter stays 225), any P8 edit, any edit of auditor or certifier artifacts, product paths, tests, the SHA ladder, any other operator decision (including `OD-MCP-RC-01` and `OD-MCP-RC-02`), and any acceptance of an audit finding.

## Designated signing session

- Designated session id: `PLACEHOLDER` (the identifier the Claude desktop app shows for that session; if it has no verifiable identifier it says so and invents none)
- Designated agent id: `PLACEHOLDER`
- Name proposed by the commission: `DELEGATION-MCP-RC-2` signing session (a new, narrow session).
- Dedication statement (operator): the commission requires that the designated session is a NEW session, dedicated solely to the single signature named above, NOT a continuation of any other session and NOT an implementation, audit or certification session. The operator's confirmation is the commission above, relayed by the orchestrator session; no `[x]` is ticked on the operator's behalf.
- Dedication statement (designated session, to be recorded in its first message): the same points, restated by the session itself. Message reference: `PLACEHOLDER`.
- Session ids that the designated session must differ from (it records each comparison):
  - implementer: `T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (label as recorded in the manifests; no UUID is recorded in the repository and none is invented);
  - first G6 auditor: `MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2`;
  - delta re-audit auditor: `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48`;
  - preflight certifier designation: `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13`;
  - orchestrator session: `local_329f8bde-1685-4aee-abe0-78d0ef7304ef`;
  - drafting session of this record and of the draft decision: the session shown as `86c8cab4-342b-4dd6-9568-44ce7852a5d4` in its scratchpad path (the only identifier that session can verify);
  - the later activation certifier (new dedicated session, id recorded in its binding manifest).
- Branch for the signature commit: `PLACEHOLDER` (cut by the designated session from the commit that carries this record; the orchestrator names it in the commission). The designated session pushes only that branch.

## Limits

- Expires: 24 hours after the operator's commission answer (precedent: P6 delegation). The exact commission time is not in the drafting brief and is not invented here. The commission is dated 2026-11-13, so `Expires` is not earlier than `2026-11-14T00:00:00+01:00`. The designated session signs on 2026-11-13 if it can, and in any case before `2026-11-14T00:00:00+01:00`; if it is later than that, it stops and asks the orchestrator session to obtain a renewed commission.
- Single use: the delegation covers only the Decision ID, the record path and the SHA-256 above. If the unsigned draft is edited after the digest was recorded, this delegation does not cover the edited text; a new digest and a renewed commission are needed. `[ ] confirmed by the designated session at signing` (left unticked here).

## Addendum 1 (2026-11-13): widened scope of the decision

Operator answers relayed in the chat with the drafting session on 2026-11-13 (original, Polish, as typed; English renderings by the drafting session, not operator-written lines): order `Rebind po merge'u` ("Rebind after the merge"); and, to the question whether to widen the scope of the unsigned decision before the delegate signs, `Tak, rozszerz` ("Yes, widen it"). The decision text was widened accordingly (commit `83e587553e163c2e401e21a1947f8615368ea316`; digest in the table above is of that text). The widened text is not signable yet (check 5). The delegated act, the single-use limit and the out-of-scope list are unchanged; only the digest and the precondition list changed. The 24 h expiry runs from the original commission and may lapse before the revision exists; if it does, a renewed commission is needed (see Limits).

## Operator signature line

Not required under Amendment 01 (S-1 struck) and not filled. This record carries no operator-written line.

## Non-actions

This delegation and the line it authorizes do not rebind the activation record, do not merge anything, do not approve any ticket, do not accept any audit finding, do not waive any governance control, do not promote FRZ, do not close the MCP-SERVER-X stage and do not claim production readiness. The line has no effect on `verify_iiac_policy_activation --require-in-force` until the separate two-phase rebind is committed; that command continues to fail with 23 ACT-N11 paths until then. Assurance level: `process_enforced` (operator attestation in chat, relayed, plus the designated session's own statement; no cryptographic proof of operator identity or session novelty).
