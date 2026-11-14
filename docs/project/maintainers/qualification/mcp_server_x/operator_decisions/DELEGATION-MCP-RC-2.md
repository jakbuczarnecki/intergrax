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

| # | Decision ID | Record path (repo-relative) | SHA-256 of the unsigned text (file bytes at `16938dae801bf480b5cc8c04b8b6b3941e9c2716`) | Delegated act |
|---|---|---|---|---|
| 1 | `IIAC-OD-MCP-RC-ACT-N11-2026-11-13` | `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` | `a4deb39c67cede23a1d858c6c442df81511ee39819074a883242e9093f992e72` | (a) replace the standalone placeholder line `________________` by the plain line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` (plain form, because the IIAC activation gate accepts only that form; it is not claimed to be operator-written); (b) change `Decision status: DRAFT` to `Decision status: AUTHORIZED`; (c) replace the `Drafting note` paragraph by a `Provenance note` paragraph in the P6 wording (the line was written by an operator-designated signing session under this record and the operator's chat commission of 2026-11-13; the plain form is not claimed to be operator-written; the timestamp is the designated session's system clock at signing; this record named with its commit SHA, read from `git log`). Nothing else. |

The digest is computed over the committed (LF, `core.autocrlf=false`) blob of the unsigned draft at the commit named in the column heading. It was recomputed from that commit by the drafting session. The signed text necessarily differs (signature line, status, provenance note); scope text, digests and SHAs inside the record are not changed by the signing. The timestamp is ISO-8601 with offset, taken from the designated session's own system clock at the moment of signing (R-5: act immediately); it is never composed in advance and never copied from another record.

### What the designated session verifies before it signs (read-only)

The operator's conditions are: the delta re-audit is PASS and the certifier preflight is passed. For the integrated baseline (Addendum 2) that means the MCP-RC and the PDC-P7-00 audits and both preflights. All are recorded on the branch that carries this record:

1. Delta re-audit dossier verdict `PASS_SCOPED` (`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md`; manifest `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json`, field `dossier_verdict`).
2. Certifier preflight verdict `CERTIFIER_PREFLIGHT_PASS_SCOPED` (`iiac_evidence/PDC-X/MCP-RC/certifier/certifier_preflight_result.json`, field `verdict`), and for the integrated baseline the PDC-P7-00 audit dossiers (baseline, addendum 1, addendum 2: `PASS_SCOPED`, manifest field `dossier_verdict`) and both PDC-P7-00 preflights (`platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT.md` and `..._ADDENDUM_1.md`: `CERTIFIER_PREFLIGHT_PASS_SCOPED`).
3. The unsigned draft at the checkout it signs in hashes to the digest in the table above, and the digests and SHAs quoted inside that draft match the files and git (spot-check at least the dossier, findings, scope and matrix digests of the audits and the evidence publication SHAs `78d4c496071141ca4092b926b838fb80016f1b91`, `6b195980d20067449ade0d2c2fd2280781c5a25d` and `29d66ae3fcc7a06a919d7e07541740841e8abfe8`).
4. The draft still contains no `Approved IIAC policy activation` line, and nothing in `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json` has changed.

5. **Draft is current.** The checkout the designated session signs in has an empty product diff to the audited tree (`git diff --name-only 324d9dba37cb54e1b74a79a05148d2dcc05e6d09 HEAD -- intergrax agents applications` is empty, or to the later audited tree that a newer draft names), and the draft's open condition 1 (F-41) still describes the state of `development`. If the impact-map rule has landed on `development`, was merged into `mcp-rc` and an audit addendum 3 exists, the draft is stale: it must be revised and re-digested first and this record updated by an addendum with the new digest and commit; the designated session does not sign a stale draft.
6. The designated session records its own start time (system clock, ISO-8601 with offset) before it signs; the 24 hour limit of Addendum 2 runs from it.

If any of these fails, the designated session writes nothing and reports to the orchestrator session.

The draft lists operator-owned open items (F-41 pending change, F-42, F-35, F-09, the CI contexts that are not verified, the named owner of the `[product]` xfail via `governance/tickets/T-IGX-039-A.md`, acceptance of RA-02..RA-04 and AF-05, and the audits' coverage limits). This delegation neither requires nor implies their resolution, and the signed line does not resolve them. The designated session lists them as still open in its report and does not claim otherwise.

Out of scope (the designated session does none of these): any IIAC rebind or edit of `IIAC_POLICY_ACTIVATION_RECORD.json` (phase 1 and phase 2 belong to the activation-certifier session, see `mcp_server_x/MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md`), any merge, any pull request, any ticket approval or lifecycle change (including `T-IGX-039-A.md`), any FRZ change (counter stays 225), any P8 edit, any edit of auditor or certifier artifacts, product paths, tests, the SHA ladder, any other operator decision (including `OD-MCP-RC-01` to `OD-MCP-RC-04`), and any acceptance of an audit finding.

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
  - preflight certifier designations: `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` and `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`;
  - PDC-P7-00 baseline auditor (three rounds): `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27` and its `-ADDENDUM-1` / `-ADDENDUM-2` suffixed ids;
  - PDC-X implementer designations `T-IGX-036-J-T-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` and `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`;
  - orchestrator session: `local_329f8bde-1685-4aee-abe0-78d0ef7304ef`;
  - drafting session of this record and of the draft decision: the session shown as `86c8cab4-342b-4dd6-9568-44ce7852a5d4` in its scratchpad path (the only identifier that session can verify);
  - the later activation certifier (new dedicated session, id recorded in its binding manifest).
- Branch for the signature commit: `PLACEHOLDER` (cut by the designated session from the commit that carries this record; the orchestrator names it in the commission). The designated session pushes only that branch.

## Limits

- Expires (original rule, superseded by Addendum 2): 24 hours after the operator's commission answer (precedent: P6 delegation). The exact commission time is not in the drafting brief and is not invented here. The commission is dated 2026-11-13, so `Expires` is not earlier than `2026-11-14T00:00:00+01:00`. The designated session signs on 2026-11-13 if it can, and in any case before `2026-11-14T00:00:00+01:00`; if it is later than that, it stops and asks the orchestrator session to obtain a renewed commission.
- Single use: the delegation covers only the Decision ID, the record path and the SHA-256 above. If the unsigned draft is edited after the digest was recorded, this delegation does not cover the edited text; a new digest and a renewed commission are needed. `[ ] confirmed by the designated session at signing` (left unticked here).

## Addendum 1 (2026-11-13): widened scope of the decision

Operator answers relayed in the chat with the drafting session on 2026-11-13 (original, Polish, as typed; English renderings by the drafting session, not operator-written lines): order `Rebind po merge'u` ("Rebind after the merge"); and, to the question whether to widen the scope of the unsigned decision before the delegate signs, `Tak, rozszerz` ("Yes, widen it"). The decision text was widened accordingly (commit `83e587553e163c2e401e21a1947f8615368ea316`; digest in the table above is of that text). The widened text is not signable yet (check 5). The delegated act, the single-use limit and the out-of-scope list are unchanged; only the digest and the precondition list changed. The 24 h expiry runs from the original commission and may lapse before the revision exists; if it does, a renewed commission is needed (see Limits).

## Addendum 2 (2026-11-14): integrated baseline, renewal and expiry rule

Source: `mcp_server_x/operator_decisions/OD-MCP-RC-03.md` and `OD-MCP-RC-04.md` (recorded chat answers of 2026-11-13 and 2026-11-14, relayed by the orchestrator session `local_329f8bde-1685-4aee-abe0-78d0ef7304ef`). The Polish option labels are quoted as selected; every English line is a labelled rendering, not an operator-written line. This addendum does not alter the delegated act, the single-use limit or the out-of-scope list; it changes the digest and the commit named in the table (the decision was rewritten for the integrated baseline, commit `16938dae801bf480b5cc8c04b8b6b3941e9c2716`), the preconditions, the expiry rule and the list of ids the designated session must differ from.

| Operator answer (Polish label as selected) | English rendering (not operator-written) | Effect on this delegation |
| --- | --- | --- |
| `Tak, czekaj na okno` (`OD-MCP-RC-03`) | "Yes, wait for the window." | The line is not signed before the PDC-P7-00 integrated baseline exists. |
| `Okno PDC-P7-00 jest otwarte, wznów kroki` (`OD-MCP-RC-04` section 1) | "The PDC-P7-00 window is open, resume the steps." | The window is declared open by the operator. |
| `Przepnij i zamroź (Recommended)` (`OD-MCP-RC-04` section 1) | "Re-pin and freeze." | Integrated baseline is `development` `a156cbbd5`, merged into `mcp-rc` as `324d9dba3`; PDC-X declared frozen on product paths for the rest of the window (a declaration, not an enforced control). |
| `Tak, wzór P6 (Recommended)` (`OD-MCP-RC-04` section 3) | "Yes, the P6 pattern." | The activation certifier authors a new binding manifest with `verdict` = `PASS`, with the scope limits, open findings and accepted conditions stated; the decision text says so (its section "What a signature on this record means"). |
| `Tak, odnawiam (Recommended)` (`OD-MCP-RC-04` section 4) | "Yes, I renew." | Renewal for 24 hours from the start of the signing session, same scope (activation line only, no rebind, no merge). |
| `PDC-X dodaje regułę na development (Recommended)` (`OD-MCP-RC-04` section 5) | "PDC-X adds the rule on development." | Open condition F-41: the rule may land after this addendum and make the draft stale (check 5). |
| `Rebind po merge'u`, `Tak, rozszerz`, `Orkiestrator zleca` (chat with the drafting session, 2026-11-13, see Addendum 1 and the runbook) | "Rebind after the merge"; "Yes, widen it"; "The orchestrator commissions it" | Order and scope as in Addendum 1; the integrated-baseline audit evidence now exists (decision section "Independent audits and preflights"). |

**Expiry rule (replaces the original rule above).** The renewal runs 24 hours from the start of the designated signing session. The start time is not known today and is not invented: it is `PLACEHOLDER` (ISO-8601 with offset, from the designated session's own system clock, recorded by that session before it signs). `Expires` is therefore `PLACEHOLDER` plus 24 hours. A signature written after that instant is not covered and is not written. As `OD-MCP-RC-04` section 4 says, the renewal is transcribed into this record by the signing session; this addendum prepares that transcription and does not claim the renewal has started.

**Single use, digest.** The unsigned text covered is the one with the SHA-256 in the delegated-act table (commit `16938dae801bf480b5cc8c04b8b6b3941e9c2716`). If the decision is revised afterwards (for example because of F-41 and an addendum 3), a new addendum records the new digest and commit; until then the designated session does not sign a stale text.

## Operator signature line

Not required under Amendment 01 (S-1 struck) and not filled. This record carries no operator-written line.

## Non-actions

This delegation and the line it authorizes do not rebind the activation record, do not merge anything, do not approve any ticket, do not accept any audit finding, do not waive any governance control, do not promote FRZ, do not close the MCP-SERVER-X stage and do not claim production readiness. The line has no effect on `verify_iiac_policy_activation --require-in-force` until the separate two-phase rebind is committed; that command continues to fail with the ACT-N11 product-path errors (158 at the audited integrated tree) until then. Assurance level: `process_enforced` (operator attestation in chat, relayed, plus the designated session's own statement; no cryptographic proof of operator identity or session novelty).
