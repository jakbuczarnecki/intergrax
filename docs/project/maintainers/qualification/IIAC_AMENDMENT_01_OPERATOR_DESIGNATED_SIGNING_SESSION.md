# IIAC Amendment 01 - operator-designated signing session

**Status:** IN FORCE AS OPERATOR-ADOPTED POLICY TEXT (2026-11-07), REVISED 2026-11-08 on the Operator's direct instruction (chat, given personally; section 0). Safeguards **S-1 and S-3 are STRUCK**; S-2 is kept as a strikeable default. Not independently audited, not mechanically enforced; assurance level `process_enforced`.
**Ticket:** `T-IGX-021-A` (specification batch; no code, no material governance path edited)
**Applies to policy:** rules R-1..R-6 of section 1.1 are adopted and apply to every certification started after commit d82f8b722 (2026-11-07T15:37:50+01:00), on the operator's direct instruction (chat, 2026-11-07). The 2026-11-08 revision (section 0) changes the safeguards only; it does not alter R-1..R-6. The agent recorded these instructions; the text below is not an operator signature line and no signature line is required for it to apply.

**Status note (2026-11-08, written by the designated certifying session `cert-w0` under delegation `historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md`; a delegate note, not an Operator-written line; decision DR-2 / OD-HIST04-B, option (c)).** (a) This amendment is in force as process-enforced policy for certifications started after commit d82f8b722; the header, section 3, section 5.3 and section 6 state the same status, and no "PROPOSED - NOT IN FORCE" wording remains in the text merged by PR 406. S-1 and S-3 are struck; S-2 is kept as the default. (b) Mechanical recognition of delegated lines (T-IGX-021-B code, T-IGX-021-C independent audit, a new policy version and its activation) is a separate stage that has not started; it is not a condition of validity (section 6). Rules R-1..R-6 and the safeguards are not changed by this note.

**What "in force" means here, exactly.** (1) The operator's right to commission a separate, new, dedicated session to sign, certify and validate in the operator's name is policy from d82f8b722. (2) The amendment was NOT independently audited (T-IGX-021-C waived by the operator's instruction, not performed). (3) It is NOT mechanically enforced: `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, the IIAC verifiers, the activation record and the EVOL-X closure preflight were deliberately left unchanged, because editing them makes `verify_iiac_policy_activation --require-in-force` fail (stale audit) and voids the B-4 acceptance. Consequence: those validators do not recognise the form `operator-delegate`; a record that must pass a validator which accepts only the literal line `Approved <id>: operator @ <time>` may carry that form if, and only if, the commission names it (section 2, S-2). Which of these statements is a statement of the Operator's will and which a description of tooling is marked in each section.
**Authored by:** implementer session (agent); section 0 and the 2026-11-08 revision are a candidate text prepared for the Operator and take effect when committed on the Operator's instruction. The author does not certify them and claims no PASS.

## 0. Revision of 2026-11-08 (Operator instruction)

Operator statement of 2026-11-08 (chat, given personally; the original was given in Polish; English rendering by the orchestrator session). The English text governs:

> I do not agree that anywhere it should be enforced that I must sign anything personally. If the rules on this are unclear, they should be updated. I can authorize any certification, validation and signature, and I can commission, in my own name, a separate session that does not implement the current functionality. That is how the rules should have been written.

Effect of this revision:

- **S-1 is STRUCK** (operator-signed delegation record). **S-3 is STRUCK** (non-delegable list, "written by the operator personally"). No decision class is excluded from R-1..R-6 by this amendment, including the delegation record, IIAC policy activation and amendments of this rule; the limits that remain are those of R-2 and R-3 and the tooling limits in section 6a. This applies the operator's "any certification, validation and signature" to the delegation record, IIAC policy activation and amendments of this rule; it is the orchestrator's reading and the operator may narrow it.
- **S-2 is kept as the default and remains strikeable** by the operator: a delegated line discloses itself as delegated (`operator-delegate` plus `Delegation:`). Where a validator accepts only the plain form, the disclosure is carried by the delegation record and the commit message, not by the line.
- A commission given in chat is sufficient (section 1.2).
- R-1..R-6 are unchanged. Wherever an older passage of this document says that something must be "written by the operator", "operator personally" or "operator-written", read it as: written by the operator **or by a session commissioned under R-6**, unless the passage describes what an existing validator accepts (those passages are marked as tooling descriptions).
- **Stated reading of R-6 (R-6 itself is not edited).** The closing clause of R-6 ("... and the activation preconditions of section 6") is read as: activation concerns only mechanical recognition by tooling; the validity of a commission does not depend on it (section 6, last paragraph).

## 1. Operator requirement (normative core)

Operator statement of 2026-11-07 (chat). The original was given in Polish; the English rendering below was recorded by the implementer session on 2026-11-07, and **the English text governs** in this document (the Polish original is not reproduced; the repository language is English).

> The operator may designate a separate, independent agent to sign and certify in the operator's name. The only condition is that it must be a new session, dedicated solely to that certification task, and it may not be a continuation of any other session nor an implementation session.

### 1.1 Rule text (adopted)

R-1. The operator may designate a separate independent agent session (the "designated signing session") to sign and certify in the operator's name.

R-2. The only condition on the designated signing session is that it is:

- (a) a **new** session;
- (b) **dedicated solely** to the certification task for which it is designated;
- (c) **not a continuation** of any other session;
- (d) **not an implementation session**.

R-4 (operator answer 2026-11-07, OQ-2). The designated signing session may also be the certification agent of the same stage. This holds for the duration of the delegation only (R-5).

R-5 (operator answer 2026-11-07, OQ-4). The designated session must act immediately after it is commissioned. The delegation is valid until the `Expires` value of its delegation record at the latest; the operator sets `Expires` in each order. The operator did not fix a numeric maximum validity; none is invented here.

R-6 (procedure, operator wording 2026-11-07). When a stage reaches certification and confirmation, the operator, or another user acting as the operator, may explicitly and on request commission a separate agent session to certify in the operator's name. The commission automatically transfers to that session the certification scope to the extent that the session itself would certify. Statement of the operator's will; it takes effect only through R-1..R-5 and the activation preconditions of section 6.

R-3. Nothing in this amendment relaxes the independence already required between the implementer, the independent auditor and the certification agent. The designated signing session replaces only the act of writing the operator signature line, and only as described in section 2.

### 1.2 How a commission is given and recorded (adopted 2026-11-08)

- **Chat is sufficient.** A commission given by the operator in chat (R-6) is a valid commission. No operator-signed document, ticket line or commit is required for it to be valid (S-1 struck).
- **Transcription.** The orchestrator session MAY transcribe the commission into a delegation record: verbatim quotation of the operator's chat message, its date, the designated session(s) or their mission files, the scope (decision ids, records), `Expires` exactly as the operator stated it (or "none stated"; a bound proposed by the orchestrator is marked as such and is not the operator's), and the dedication statements of R-2. Template: `IIAC_OPERATOR_SIGNING_DELEGATION.template.md` (fields may be filled by the orchestrator; the operator signature line of the template is not required). The record is a transcription, not a condition of validity. The operator signature line of the template is not required by this rule, but it is required by the OD-ENT-01 validator (section 3).
- **Assurance level.** `process_enforced`: operator attestation in chat plus the designated session's own statement. There is no cryptographic proof of operator identity, of the session's novelty or of its dedication.
- **Who is "the operator".** R-6 allows the operator "or another user acting as the operator" to commission. This amendment does not define how that user is identified.
- **Earlier chains.** This amendment rewrites no earlier record. The operator may confirm by a commission (R-6, chat) that an earlier delegated line was written on the operator's authority; the confirmation is transcribed as above and does not change the original timestamps.

## 2. Safeguards (agent-proposed; S-1 and S-3 STRUCK, S-2 kept as strikeable default)

The operator's condition in section 1 is the normative core. The safeguards below were added by the implementer agent to keep the rule auditable. They are NOT the operator's words. The operator struck S-1 and S-3 on 2026-11-08 (section 0); S-2 remains as a default and may be struck by a further instruction.

- **S-1 - STRUCK by the operator's instruction of 2026-11-08.** ~~The delegating record is signed by the operator once per delegation. It fixes: decision id, record path, SHA-256 of the text to be signed, designated session id, and expiry.~~ Replaced by section 1.2: the commission is given in chat; a delegation record, where made, is a transcription. The fields (decision id, record path, digest of the text to be signed, designated session id, `Expires`) remain good practice and are what a future validator can check; they are not a condition of validity.
- **S-2 (agent-proposed; kept as the default, strikeable).** A delegated signature line is distinguishable from an operator-written one: `Approved <Decision ID>: operator-delegate @ <ISO-8601>`, plus a line `Delegation: <reference>` that points to the commission or its transcription (a path@sha, or a path@sha256 for a local session record). Where a validator that exists today accepts only the plain form `Approved <id>: operator @ <time>` (the EVOL-X closure preflight B6 allow-list), a designated session may write the plain form if the commission names it; the delegation reference is then kept in the delegation record and in the commit message, and the line is not claimed to be operator-written.
- **S-3 - STRUCK by the operator's instruction of 2026-11-08.** ~~Not delegable: the delegation record itself, IIAC policy activation, and any amendment of this rule. These remain written by the operator personally.~~ Nothing is non-delegable under this amendment. See section 6a for the tooling that still reads "operator" literally.

## 3. Scope and non-effects

- **Not retroactive.** The rule applies to certifications started after d82f8b722. It does not re-characterise any signature, record or decision that exists today. Section 1.2 ("Earlier chains") lets the operator confirm earlier delegated lines by commission.
- **EVOL-X P3/B6 chain.** This amendment does not by itself re-characterise the EVOL-X B6 acceptance, FRZ promotion, OD-EVOL-12 and OD-EVOLX-08 lines. Those carry the plain form `Approved <id>: operator @ <time>`; whether they were written by sessions designated by chat commission is not derivable from the records and is a matter for the operator's confirmation under section 1.2. Whether this chain counts as started before or after d82f8b722 is a judgement, not a repository fact (the B6 acceptance line is timestamped 2026-11-07T15:41:06+01:00, after d82f8b722); this document does not decide it.
- **Validator recognition.** The EVOL-X closure preflight and the IIAC verifiers do not recognise `operator-delegate` (tooling fact: `scripts/qualification/evol_x_closure_publication_preflight.py:368-369`). The ENT-AUDIT-X OD-ENT-01 baseline promotion (`scripts/enterprise_audit/iiac_baseline_promotion.py:62,77-78`) does recognise it, but it additionally requires a delegation record that carries an operator-written line `Approved delegation <id>: operator @ <ISO-8601>`, published under `iiac_evidence/ENT-AUDIT-X/operator_decisions/` (`iiac_baseline_promotion.py:79-81,207-226`). That requirement conflicts with S-1 being struck (sections 0 and 1.2): until the validator is changed, a chat-only commission is not sufficient for OD-ENT-01 promotion. A record meant for any validator follows S-2 (plain form only if the commission names it). Nothing in this amendment changes a validator.
- This document changes no protocol text, validator, preflight, test, activation record, FRZ checklist or OD register.

## 4. Specification of required changes (text only, no code)

The rule is in force as process-enforced policy (sections 0 and 3). The changes below are the specification for **mechanical** recognition of delegated lines; they are optional with respect to the validity of the rule. They are split into the follow-up tickets named in `T-IGX-021-A`. Nothing below is implemented by this document.

### 4.1 T-IGX-021-B - code (implementation ticket)

**B-1 Signature grammar.** The signature-line grammar of the B6 record (`_RECORD_SIGNATURE` in `scripts/qualification/evol_x_closure_publication_preflight.py`) today accepts exactly `Approved <Decision ID>: operator @ <ISO-8601 timestamp>`. Specify an additional accepted form `Approved <Decision ID>: operator-delegate @ <ISO-8601 timestamp>`. A record with the delegate form must contain exactly one further allow-listed line `Delegation: <repo-relative path>@<40-hex sha>`. Records with the delegate form but no `Delegation:` line, with a `Delegation:` line but the plain operator form, or with both signature forms, are rejected. The B6 allow-list of fields and the format document `EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` are extended in the same change; the existing test `test_w401_ra_06_format_document_matches_the_code` keeps format document and code in step. The delegation reference of this grammar is `Delegation: <repo-relative path>@<40-hex sha>` (as in `iiac_baseline_promotion.py:78`); the sha256 form mentioned in S-2 is not accepted by it, so a local session record must first be published and referenced by commit.

**B-2 Delegation-record validation.** A new validator step (in the preflight and, where relevant, in a `verify_iiac_*` verifier) loads the record named by `Delegation:` at the pinned sha and checks:

1. it follows `IIAC_OPERATOR_SIGNING_DELEGATION.template.md` field-for-field (as a transcription of the commission; S-1 is struck, so an operator-written signature line is not required and the validator must not demand one);
2. its `Stage`, `Decision ID` and `Record path` equal those of the record being signed;
3. its `SHA-256 of text to be signed` equals the SHA-256 of the record's signed text as computed by a single documented normalisation (the same normalisation the preflight already applies to bound digests);
4. its `Designated session id` equals the session id that wrote the delegate signature (see B-3);
5. the delegate timestamp is not after the record's `Expires` value and not before the delegation timestamp;
6. (S-3 is struck: the validator rejects no decision class on the ground that it is delegated; it checks the fields above only.)

**B-3 Role separation for the designated session.** Extend `verify_iiac_role_separation` (and the session-id inequality of `verify_iiac_audit_manifest`) so that the designated signing session id must differ from the `implementation_session_id`, from every session recorded as an implementer in the stage's ticket and handoff records, and must be dedicated: it may appear as signer of only the single delegation it is named in, may not equal the `auditor_session_id` or `certifier_session_id` of the same stage unless the commission names that role as the dedicated task (R-4, permissive), and must not have written any product-path commit (reuse the auditor product write-surface check). This operationalises R-2(a)-(d) as far as is mechanically checkable; "new session" and "not a continuation" remain `process_enforced` assurance (see 4.4).

**B-4 Preflight Gate B.** Gate B (the B6-record exactness check, including finding `B-11`) accepts both signature forms under B-1/B-2 and still rejects every other deviation. Specify negative cases: delegate form without a delegation reference; delegation expired; digest mismatch; wrong stage; wrong decision id; designated session id equals an implementer session; plain-form line written by a registered designated session where the commission does not name the plain form.

**B-5 Tests.** Extend `tests/unit/qualification/` with the negative and positive cases of B-4 plus a round-trip test for the format document. Test files are in the protected set of this batch and are edited only in `T-IGX-021-B`.

**Consequence for B-4 acceptance (important).** Editing `evol_x_closure_publication_preflight.py` or its tests voids OD-3 condition 3 (the pinned B-4 corrective). `T-IGX-021-B` therefore carries a mandatory, separate independent audit of the B-4 delta (section 5.2).

### 4.2 Policy and protocol text (operator-activated, after 021-B)

**P-1 New policy version.** Because `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` is a material governance path, the changes below are made as a new policy version (for example `IIAC-<date>.2`; the actual identifier is chosen at activation) and not as an in-place edit of `IIAC-2026-11-05.1`:

- Roles table: add the role "OPERATOR-DESIGNATED SIGNING SESSION" with Allowed = write the delegate signature line for the decision(s) named in the commission; Forbidden = any implementation, any audit of the same stage, any continuation of another session.
- "B6 under IIAC" and "Handoff record" sections: reference this amendment and the delegation record template; the handoff record template gains a line for the designated session id (changed only inside the new policy version).
- "Policy activation" section: state that policy activation, the delegation record and amendments of this rule may be commissioned under R-6 like any other certification act (S-3 struck), and replace the literal "operator decision file ... exact line `Approved ... : operator @`" requirement by the form the activation gate then accepts.

**P-2 Audit manifest.** If the operator keeps S-1, the audit manifest schema gains optional fields `delegation_record_path` and `delegation_record_sha`; this is a schema version change and is listed here only so that it is decided in the same activation (open question OQ-3).

### 4.3 Operator actions (not agent work)

- DONE 2026-11-08 (section 0): decision on S-1..S-3 (S-1 and S-3 struck, S-2 kept as strikeable default).
- Only if mechanical enforcement is wanted: after `T-IGX-021-B` and `T-IGX-021-C` (independent audit), the operator, or a session commissioned under R-6, writes the activation decision `IIAC_OD_ACTIVATION_*.md` for the new policy version in the form the gate accepts and performs the two-phase SHA publication (phase 1 `IN_FORCE` with empty `effective_from_sha`; phase 2 sets `effective_from_sha` and `policy_activation_sha`). Until then `verify_iiac_policy_activation --require-in-force` remains tied to the current activation record.

### 4.4 Assurance level

Mechanically checkable: field equality, digests, expiry, session-id inequality, absence of product-path commits from the designated session. Not mechanically provable: that the session really is new, dedicated and not a continuation, and who the operator in a chat is. These remain `process_enforced` (operator attestation in chat, no cryptographic session proof). The delegation record, where made, transcribes the operator's chat statement; the designated session states R-2 in its own first message in the record.

### 4.5 Follow-up ticket map

| Ticket | Content | Executor |
|--------|---------|----------|
| `T-IGX-021-A` | This specification, template, impact analysis (this batch) | implementer (agent) |
| `T-IGX-021-B` | Code for 4.1 (B-1..B-5) | implementer (agent), different session from 021-A is recommended but not required by this document |
| `T-IGX-021-C` | Independent audit of 021-A and 021-B, including the B-4 delta | separate independent auditor session, never the implementer |
| operator or commissioned session | Policy version P-1/P-2, `IIAC_OD_ACTIVATION_*.md`, two-phase SHA publication (only if mechanical enforcement is chosen) | commissioned under R-6 |

## 5. Impact analysis

This analysis is consistent with the Constraints section of `T-IGX-021-A`.

### 5.1 Material governance paths

`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `scripts/governance/iiac_activation_gate.py`, `scripts/governance/iiac_common.py`, `verify_iiac_policy_activation.py`, `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py` are the `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-47`). Editing any of them makes `verify_iiac_policy_activation --require-in-force` fail (stale audit) until a new audited activation is performed. The real change (4.1 B-2/B-3 touch `verify_iiac_*`; 4.2 P-1 touches the protocol) therefore cannot be done silently: it ends the current activation's validity until a new audited activation. This batch edits none of these paths, and the required `git diff --name-only` guard of the ticket's test contract must print nothing for them.

### 5.2 OD-3 condition 3 (B-4 corrective)

`scripts/qualification/evol_x_closure_publication_preflight.py` and its tests are the pinned B-4 corrective (OD-3 condition 3). Any change (4.1 B-1, B-4, B-5) voids the B-4 acceptance. It needs its own ticket (`T-IGX-021-B`) and its own independent audit (`T-IGX-021-C`) of the B-4 delta before the corrective is again relied upon. This batch edits neither the preflight nor `tests/unit/qualification/**`.

### 5.3 Activation state and staleness

The amendment is in force as process-enforced operator policy (sections 0 and 3). The IIAC activation record is unchanged and `verify_iiac_policy_activation --require-in-force` is expected to continue to exit 0 on it, because this document edits no material or product path (not executed when this text was written). If the operator later chooses mechanical enforcement, then after 021-B lands that command is expected to fail until the new activation completes; that is the intended fail-closed behaviour, not a defect, and the sequence (code, audit, activation) should be planned as one controlled window.

### 5.4 Signature forms accepted by existing tooling

`EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` and `_RECORD_SIGNATURE` (tooling description) accept the line `Approved <id>: operator @ <time>` and say it is "written by the operator, never by an agent". That sentence describes the format document, not the operator's rule: under R-1..R-6 a session commissioned by the operator may write the line when the commission names the plain form (S-2). The sentence in the format document is out of date and is to be updated in a documentation change; the validator cannot distinguish authors either way.

### 5.5 Running chain

The EVOL-X P3/B6 chain (section 3, bullet 2) is not re-characterised by this document. FRZ promotion, OD-EVOL-12, OD-EVOLX-08 and EVOL-X P3/B6 publication keep the standing they have today; the operator may confirm them under section 1.2.

## 6. Preconditions for MECHANICAL enforcement (summary; not preconditions of validity)

1. Operator decision to pursue mechanical enforcement (the decision on S-1..S-3 was taken on 2026-11-08).
2. `T-IGX-021-B` implemented.
3. `T-IGX-021-C` independent audit by a separate session, with verdict PASS (or scoped), including the B-4 delta.
4. New policy version with the protocol changes of P-1/P-2.
5. `IIAC_OD_ACTIVATION_*.md` (operator or commissioned session) and two-phase SHA publication.
6. `verify_iiac_policy_activation --require-in-force` exit 0 on the new activation.

The rule of section 1 applies from d82f8b722 regardless of steps 1-6; they concern only whether tooling recognises and enforces it.

### 6a. What this text cannot change by itself (material governance paths)

The protocol `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` ("Policy activation", item 3: operator decision file with the exact line `Approved IIAC policy activation ...: operator @ <timestamp>`) and the activation gate `scripts/governance/iiac_activation_gate.py` are material governance paths. This amendment does not edit them and cannot do so without ending the validity of the current activation (ACT-N11 / stale audit). Consequently: (a) a re-activation or a new policy version is, for the gate, an operator-form line; whether the operator writes it or commissions a session to write it under R-6 is invisible to the gate; (b) the sentence of the protocol that reads "operator" literally is to be reconciled only inside a new audited policy version (P-1). (c) The OD-ENT-01 baseline promotion validator (`scripts/enterprise_audit/iiac_baseline_promotion.py`) reads the template's operator signature line literally and refuses a delegation without it; this enforces S-1, which is struck, so a chat-only commission does not satisfy that validator until it is changed (a change under scripts/enterprise_audit/, which is outside the ACT-N11 product-path prefixes and outside the material governance paths (iiac_common.py:22, iiac_activation_gate.py:40-47), so it does not by itself invalidate the IIAC activation; its other consequences for the ENT-AUDIT-X evidence are not assessed here). This section states the limits; it does not choose between options.

## 7. Open questions for the operator

- OQ-1: ANSWERED by the operator on 2026-11-08 (section 0): S-1 struck, S-3 struck, S-2 kept as a default that may be struck.
- OQ-2: ANSWERED by the operator on 2026-11-07 (R-4): the designated signing session may also be the certification agent of the same stage, for the duration of the delegation. B-3 therefore uses the permissive reading.
- OQ-3: Should the audit manifest record delegation fields (P-2)? Open; relevant only for mechanical enforcement.
- OQ-4: PARTLY ANSWERED (R-5): act immediately after commissioning; validity ends at the delegation record `Expires`. No numeric maximum was set by the operator; the maximum remains a policy choice if one is wanted.
