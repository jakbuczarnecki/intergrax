# DELEGATION-PDC-STANDING — Operator standing authorizations for the remainder of PDC-X

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-STANDING |
| **Stage** | PDC-X (P5 remainder, P6, P7, P8, P9 preparation) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-1.md` .. `DELEGATION-PDC-WAVE-3C.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

The Operator asked to go through all pending consents at once, so that the implementer session could then work autonomously where possible. The Operator answered eight questions in chat. Answers are given in English rendering; the original wording was Polish.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Implement batch B-REG-P (`ollama`, ticket T-IGX-036-M) now in a worktree with a draft PR, merge after MCP-SERVER-X P8? | **Yes, now** |
| 2 | May the implementer merge docs-only PRs after green CI and a head that equals the signed SHA? | **Yes, after green CI** |
| 3 | May the implementer merge the nine product PRs in DAG order after MCP-SERVER-X P8 is published, after green CI and conflict reconciliation? | **Yes, after P8 and green CI** |
| 4 | Pre-approve delegate signing of plans M and S and of write-set amendments that close findings of a batch's own work, within stated limits? | **Yes, within those limits** |
| 5 | Who corrects the IIAC protocol header wording (finding PDC-BL-F01)? | **In the PDC activation (PDC-P7-00)** |
| 6 | Operator-only IIAC activation lines (PDC-P7-00 and PDC-P8-04): sign personally or delegate? | **Delegate to a delegate session** |
| 7 | May the implementer start the independent P7 and P8 auditor sessions autonomously? | **Yes, autonomously** |
| 8 | May the implementer dispatch the GitHub Actions replay workflows needed for the EBH-7 re-pin after P6? | **Yes, after P6** |

## Scope of the authorizations

1. **B-REG-P (answer 1).** The implementer plans and implements ticket T-IGX-036-M in an isolated worktree; the PR stays draft and is not merged before MCP-SERVER-X P8 (OD-PDC-SEQ-01). The decision of 2026-11-12 ("wait for MCP P8") now applies to the merge only, not to the implementation.
2. **Docs-only merges (answer 2).** The implementer may merge a PR that changes no product path (`intergrax/`, `agents/`, `applications/`) and no material governance path, only after: all required checks are green, and `headRefOid` equals the SHA that was verified. The implementer records each merge on the board. A PR that touches `scripts/` or `governance/` only for planning artifacts (tickets, plans, delegation records, the WBS generator and its output) counts as docs-only; a change to IIAC protocol or activation files does not.
3. **Product merges (answer 3).** Starts only when the roadmap shows MCP-SERVER-X as closed and its P8 certification as published. Order follows `PDC_DEPENDENCY_DAG.md`. Each merge needs green CI; conflicts are reconciled on the PR branch by a normal merge, never by rewriting published history. Each merge invalidates the IIAC activation (ACT-N11), which the Operator accepted in OD-PDC-SEQ-01; PDC-P7-00 re-activates it. The implementer stops and asks if a merge would require changing a gate's meaning.
4. **Delegate signing (answer 4).** A delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <record>@<sha>` for:
   - the plan of **T-IGX-036-M** (B-REG-P) and the plan of **T-IGX-036-S** (P6 integration), after the same checks as in the wave records (digest, WBS and ADR match, write set, no new authority, no merge before P8 where applicable);
   - a **write-set amendment** of a batch whose own implementation found the need, only when all of these hold: the added paths are required to close a finding of that same batch; they are not under the hotspots `intergrax/contracts/` or `intergrax/applications/_shared/` unless that batch already owns them; no gate is weakened (a gate may be strengthened or given one classification entry); no new registry, scheduler, engine, trust, policy or audit authority is introduced; the change is not a new functional scope. Anything outside these limits returns to the Operator.
5. **IIAC header (answer 5).** The wording `NOT_IN_FORCE` in the IIAC protocol header (lines 4 and 91, finding PDC-BL-F01) is corrected in the same batch as the PDC activation (PDC-P7-00), so that one rebind covers both.
6. **IIAC activation lines (answer 6).** A dedicated delegate session, separate from every implementer and auditor, may write the activation lines for **PDC-P7-00** (at the integrated baseline) and **PDC-P8-04** (rebind at the global audit SHA). Before writing a line the delegate confirms: no product delta between the audited SHA and the activation HEAD (ACT-N11); `verify_iiac_policy_activation.py` preconditions and `verify_iiac_role_separation.py` pass; the P6 evidence is complete; for P8-04 the global audit is published. **The delegate must first confirm that the IIAC protocol and Amendment 01 permit delegated activation lines. If they do not, the delegate writes nothing and reports to the Operator.**
7. **Auditors (answer 7).** The implementer may start the independent auditor sessions for PDC-P7 (one per domain, at most three at once, each a clean session, never an implementer of that domain's batch, never fixing what it audits) and the global auditor session for PDC-P8, once the gates of those phases are met. Findings are routed to the implementer lane and re-audited by a fresh auditor on the corrected SHA. Verdicts and certification remain with the certifier and the Operator (see below).
8. **GitHub Actions (answer 8).** After P6, the implementer may dispatch only the replay workflows named in the EBH-7 re-pin dossier, at the integrated baseline SHA, and records run IDs and results. Nothing else is sent to an external service.

## Not delegated (unchanged)

- OD signatures (new ones).
- FRZ promotion and the roadmap closure line of PDC-X (PDC-P9-03 and PDC-P9-04): certification by the certifier agent and the Operator's closure decision.
- Any merge before MCP-SERVER-X P8 is published.
- Lifting the C10/C15 restriction (Docker, new vendor integrations, production configuration). Tests that need Docker (`dist_scale_x` P4 and P6) stay blocked.
- Any action outside the limits stated above.

## Separation

- The implementer session and its implementer subagents never write `Approved:` or activation lines.
- A delegate session does not implement or audit the tickets it signs.
- An auditor never certifies and never fixes what it audits.
