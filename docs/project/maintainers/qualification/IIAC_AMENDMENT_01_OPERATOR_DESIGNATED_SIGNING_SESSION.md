# IIAC Amendment 01 - operator-designated signing session

**Status:** PROPOSED - NOT IN FORCE
**Ticket:** `T-IGX-021-A` (specification batch; no code, no material governance path edited)
**Applies to policy:** nothing yet. This text is a proposal. It amends no document, activates nothing and signs nothing. The in-force text remains `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` as activated by `IIAC_POLICY_ACTIVATION_RECORD.json`, unchanged.
**Authored by:** implementer session (agent). This document is a draft for operator decision and for a later independent audit (`T-IGX-021-C`). The author does not certify it and claims no PASS.

## 1. Operator requirement (normative core)

Operator statement of 2026-11-07 (chat), verbatim (Polish, ASCII transliteration as given to the implementer):

> operator ma mozliwosc wyznaczenia osobnego niezaleznego agenta do podpisywania i certyfikacji w jego imieniu. Warunkiem jest jedynie to, ze musi byc to nowa sesja, ktora jest dedykowana tylko do tego zadania certyfikacyjnego i nie moze ona byc kontynuacja innej sesji ani sesja implementacyjna

English rendering (agent translation; the Polish text above governs):

> The operator may designate a separate, independent agent to sign and certify in the operator's name. The only condition is that it must be a new session, dedicated solely to that certification task, and it may not be a continuation of any other session nor an implementation session.

### 1.1 Rule text (proposed)

R-1. The operator may designate a separate independent agent session (the "designated signing session") to sign and certify in the operator's name.

R-2. The only condition on the designated signing session is that it is:

- (a) a **new** session;
- (b) **dedicated solely** to the certification task for which it is designated;
- (c) **not a continuation** of any other session;
- (d) **not an implementation session**.

R-3. Nothing in this amendment relaxes the independence already required between the implementer, the independent auditor and the certification agent. The designated signing session replaces only the act of writing the operator signature line, and only as described in section 2.

## 2. Agent-proposed safeguards (strikeable by the operator)

The operator's condition in section 1 is the normative core. The safeguards below were added by the implementer agent to keep the rule auditable. They are NOT the operator's words. The operator may strike any of them (S-1, S-2, S-3) in `governance/tickets/T-IGX-021-A.md` before audit; a struck safeguard is then removed from this document by a follow-up edit and does not apply.

- **S-1 (agent-proposed).** The delegating record is signed by the operator once per delegation. It fixes: decision id, record path, SHA-256 of the text to be signed, designated session id, and expiry. Template: `IIAC_OPERATOR_SIGNING_DELEGATION.template.md`.
- **S-2 (agent-proposed).** A delegated signature line is distinguishable from an operator-written one: `Approved <Decision ID>: operator-delegate @ <ISO-8601>`, plus a line `Delegation: <path>@<sha>` that points to the delegating record.
- **S-3 (agent-proposed).** Not delegable: the delegation record itself, IIAC policy activation, and any amendment of this rule. These remain written by the operator personally.

## 3. Scope and non-effects

- **Not retroactive.** The rule would apply only to certifications started after the amendment has been activated (section 6). It does not re-characterise any signature, record or decision that exists today.
- **Does not unblock the running EVOL-X P3/B6 chain.** The current EVOL-X B6 acceptance, FRZ promotion, OD-EVOL-12 and OD-EVOLX-08 remain as they are and still require the operator-written signature line `Approved <id>: operator @ <time>` defined in `evol_x/EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md`.
- **Not in force.** Until sections 5 and 6 are completed, any record carrying `operator-delegate` must be treated as invalid by every existing validator; no agent may write one.
- This batch changes no protocol text, validator, preflight, test, activation record, FRZ checklist or OD register.

## 4. Specification of required changes (text only, no code)

The rule can take effect only after the following changes are implemented, audited and activated. They are split into the follow-up tickets named in `T-IGX-021-A`. Nothing below is implemented by this batch.

### 4.1 T-IGX-021-B - code (implementation ticket)

**B-1 Signature grammar.** The signature-line grammar of the B6 record (`_RECORD_SIGNATURE` in `scripts/qualification/evol_x_closure_publication_preflight.py`) today accepts exactly `Approved <Decision ID>: operator @ <ISO-8601 timestamp>`. Specify an additional accepted form `Approved <Decision ID>: operator-delegate @ <ISO-8601 timestamp>`. A record with the delegate form must contain exactly one further allow-listed line `Delegation: <repo-relative path>@<40-hex sha>`. Records with the delegate form but no `Delegation:` line, with a `Delegation:` line but the plain operator form, or with both signature forms, are rejected. The B6 allow-list of fields and the format document `EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` are extended in the same change; the existing test `test_w401_ra_06_format_document_matches_the_code` keeps format document and code in step.

**B-2 Delegation-record validation.** A new validator step (in the preflight and, where relevant, in a `verify_iiac_*` verifier) loads the record named by `Delegation:` at the pinned sha and checks:

1. it follows `IIAC_OPERATOR_SIGNING_DELEGATION.template.md` field-for-field and carries the operator-written line `Approved delegation <Delegation ID>: operator @ <ISO-8601>` (S-1);
2. its `Stage`, `Decision ID` and `Record path` equal those of the record being signed;
3. its `SHA-256 of text to be signed` equals the SHA-256 of the record's signed text as computed by a single documented normalisation (the same normalisation the preflight already applies to bound digests);
4. its `Designated session id` equals the session id that wrote the delegate signature (see B-3);
5. the delegate timestamp is not after the record's `Expires` value and not before the delegation timestamp;
6. the delegated decision is not one of the S-3 non-delegable items: the delegation record itself, IIAC policy activation (`IIAC_OD_ACTIVATION_*.md`), or an amendment of this rule. A delegate signature on any such decision is rejected fail-closed.

**B-3 Role separation for the designated session.** Extend `verify_iiac_role_separation` (and the session-id inequality of `verify_iiac_audit_manifest`) so that the designated signing session id must differ from the `implementation_session_id`, from every session recorded as an implementer in the stage's ticket and handoff records, and must be dedicated: it may appear as signer of only the single delegation it is named in, may not equal the `auditor_session_id` or `certifier_session_id` of the same stage unless the operator's delegation record explicitly names that role as the dedicated task (open question OQ-2), and must not have written any product-path commit (reuse the auditor product write-surface check). This operationalises R-2(a)-(d) as far as is mechanically checkable; "new session" and "not a continuation" remain `process_enforced` assurance (see 4.4).

**B-4 Preflight Gate B.** Gate B (the B6-record exactness check, including finding `B-11`) accepts both signature forms under B-1/B-2 and still rejects every other deviation. Specify negative cases: delegate form without delegation record; delegation expired; digest mismatch; wrong stage; wrong decision id; designated session id equals an implementer session; S-3 decision delegated; operator-form line written by a session that is registered as designated (an operator line must not be written by an agent at all).

**B-5 Tests.** Extend `tests/unit/qualification/` with the negative and positive cases of B-4 plus a round-trip test for the format document. Test files are in the protected set of this batch and are edited only in `T-IGX-021-B`.

**Consequence for B-4 acceptance (important).** Editing `evol_x_closure_publication_preflight.py` or its tests voids OD-3 condition 3 (the pinned B-4 corrective). `T-IGX-021-B` therefore carries a mandatory, separate independent audit of the B-4 delta (section 5.2).

### 4.2 Policy and protocol text (operator-activated, after 021-B)

**P-1 New policy version.** Because `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` is a material governance path, the changes below are made as a new policy version (for example `IIAC-<date>.2`; the actual identifier is chosen at activation) and not as an in-place edit of `IIAC-2026-11-05.1`:

- Roles table: add the role "OPERATOR-DESIGNATED SIGNING SESSION" with Allowed = write the delegate signature line for the single delegated decision; Forbidden = any implementation, any audit of the same stage, any S-3 decision, any continuation of another session.
- "B6 under IIAC" and "Handoff record" sections: reference this amendment and the delegation record template; the handoff record template gains a line for the designated session id (changed only inside the new policy version).
- "Policy activation" section: state that policy activation, the delegation record and amendments of this rule are written by the operator personally (S-3).

**P-2 Audit manifest.** If the operator keeps S-1, the audit manifest schema gains optional fields `delegation_record_path` and `delegation_record_sha`; this is a schema version change and is listed here only so that it is decided in the same activation (open question OQ-3).

### 4.3 Operator actions (not agent work)

- Decide on this proposal and on S-1..S-3 (strike or keep) in the ticket.
- After `T-IGX-021-B` and `T-IGX-021-C` (independent audit): write the activation decision `IIAC_OD_ACTIVATION_*.md` for the new policy version with the exact line the protocol requires, and perform the two-phase SHA publication (phase 1 `IN_FORCE` with empty `effective_from_sha`; phase 2 sets `effective_from_sha` and `policy_activation_sha`). Until then `verify_iiac_policy_activation --require-in-force` remains tied to the current activation record.

### 4.4 Assurance level

Mechanically checkable: field equality, digests, expiry, session-id inequality, absence of product-path commits from the designated session, non-delegable list. Not mechanically provable: that the session really is new, dedicated and not a continuation. These remain `process_enforced` (same assurance level as the existing handoff record: operator attestation, no cryptographic session proof). The delegation record therefore carries a dedication statement made by the operator and, in addition, the designated session states it in its own first message in the record (see template).

### 4.5 Follow-up ticket map

| Ticket | Content | Executor |
|--------|---------|----------|
| `T-IGX-021-A` | This specification, template, impact analysis (this batch) | implementer (agent) |
| `T-IGX-021-B` | Code for 4.1 (B-1..B-5) | implementer (agent), different session from 021-A is recommended but not required by this document |
| `T-IGX-021-C` | Independent audit of 021-A and 021-B, including the B-4 delta | separate independent auditor session, never the implementer |
| operator | Policy version P-1/P-2, `IIAC_OD_ACTIVATION_*.md`, two-phase SHA publication | operator personally |

## 5. Impact analysis

This analysis is consistent with the Constraints section of `T-IGX-021-A`.

### 5.1 Material governance paths

`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `scripts/governance/iiac_activation_gate.py`, `scripts/governance/iiac_common.py` and the `verify_iiac_*` scripts are `MATERIAL_GOVERNANCE_PATHS`. Editing any of them makes `verify_iiac_policy_activation --require-in-force` fail (stale audit) until a new audited activation is performed. The real change (4.1 B-2/B-3 touch `verify_iiac_*`; 4.2 P-1 touches the protocol) therefore cannot be done silently: it ends the current activation's validity until a new audited activation. This batch edits none of these paths, and the required `git diff --name-only` guard of the ticket's test contract must print nothing for them.

### 5.2 OD-3 condition 3 (B-4 corrective)

`scripts/qualification/evol_x_closure_publication_preflight.py` and its tests are the pinned B-4 corrective (OD-3 condition 3). Any change (4.1 B-1, B-4, B-5) voids the B-4 acceptance. It needs its own ticket (`T-IGX-021-B`) and its own independent audit (`T-IGX-021-C`) of the B-4 delta before the corrective is again relied upon. This batch edits neither the preflight nor `tests/unit/qualification/**`.

### 5.3 Activation staleness

Until a new policy version is audited and activated (4.3), the current activation state is the only authority: the amendment is `PROPOSED - NOT IN FORCE`, and `verify_iiac_policy_activation --require-in-force` must continue to exit 0 on the unchanged activation record. After 021-B lands, that command is expected to fail until the new activation completes; that is the intended fail-closed behaviour, not a defect, and the operator should plan the sequence accordingly (code, audit, activation in one controlled window).

### 5.4 Signature rule that stays in force

`EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` and `_RECORD_SIGNATURE` require `Approved <id>: operator @ <time>` "written by the operator, never by an agent". That remains true for every certification started before activation and for everything in S-3 after activation. The rule cannot take effect before the follow-up code and the new policy version are audited and activated.

### 5.5 Running chain

The running EVOL-X P3/B6 chain is unaffected. FRZ promotion, OD-EVOL-12, OD-EVOLX-08 and EVOL-X P3/B6 publication are out of scope of this batch and are not unblocked by it.

## 6. Activation preconditions (summary)

1. Operator decision on this proposal and on S-1..S-3.
2. `T-IGX-021-B` implemented.
3. `T-IGX-021-C` independent audit by a separate session, with verdict PASS (or scoped), including the B-4 delta.
4. New policy version with the protocol changes of P-1/P-2.
5. Operator-written `IIAC_OD_ACTIVATION_*.md` and two-phase SHA publication.
6. `verify_iiac_policy_activation --require-in-force` exit 0 on the new activation.

Only after step 6 may this document's status change from PROPOSED - NOT IN FORCE, and that change is itself operator-written (S-3).

## 7. Open questions for the operator

- OQ-1: Keep, strike or amend each of S-1, S-2, S-3.
- OQ-2: May the designated signing session also be the certification agent of the same stage, or must it be a third session? (The operator condition only requires: new, dedicated, not a continuation, not an implementation session. B-3 is written so that the stricter reading can be chosen.)
- OQ-3: Should the audit manifest record delegation fields (P-2)?
- OQ-4: Default maximum validity of a delegation (template `Expires` is a mandatory field; the maximum is a policy choice).
