<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# SCENARIO-FACTORY-X (SFX) - usability test protocol

**Ticket:** [`T-IGX-037-D`](../../../../../../governance/tickets/T-IGX-037-D.md) (WBS unit SFX-P3-01; rounds 1-3 run in SFX-P12-05)  
**Companion:** [`SFX_UX_DESIGN.md`](SFX_UX_DESIGN.md) (personas, screens `SCR-xx`, interaction specs)  
**Sources:** SFX master plan R1 §8 (UX, generalization test, G-1..G-5), §12 (REF changes); R0 §8 (usability and proposed metrics) and §12 (REF-1..3).  
**Status:** PROTOCOL ONLY. No round has been run. Every result field is `NOT_RUN`. All targets are PROPOSED and are decided under **OD-SFX-07** (operator, at P3 exit).

---

## 1. Rounds

| Round | When | Subject | Scenario | Participants | Purpose |
|---|---|---|---|---|---|
| R0 | P3 (SFX-P3-01), before W3 | Static prototype Artifact (link pending, lane L3b) | REF-1 text walk-through | 5 | Validate journeys and IA; record TTFS **proxy** baseline; find critical concept defects before any Studio code |
| R1 | P12 (SFX-P12-05) | Studio MVP + IDE toolkit v1 at a recorded SHA | REF-1 Invoice automation | 5 | First end-to-end measurement |
| R2 | P12, after R1 fixes | Same, new SHA | REF-2 API-to-API workflow | 5 | Engineer-heavy flows, drift and retries |
| R3 | P12, after R2 fixes | Same, new SHA | REF-3 RAG decision support + HITL | 5 | Approval and evidence flows |

Rules:
- Each round uses **new participants** to avoid learning effects. Repeat participants, if any, are reported separately and excluded from TTFS and SUS aggregates.
- If OD-SFX-08 excludes REF-2 or REF-3, the round keeps the same structure and runs on REF-1 with the replaced persona mix; the substitution is recorded.
- Usability tasks use **REF-1..3 only**. The sealed GEN set (SFX-P3-02) is never used, shown or discussed in usability sessions; moderators and note-takers must not have access to GEN content.
- R0 timings come from a static prototype and are **not comparable** with R1..R3 timings. They are reported as a separate baseline series.

---

## 2. Recruitment profiles

| Profile | Persona | Inclusion | Exclusion |
|---|---|---|---|
| RP-BU | P-BU Business user | Owns or operates a business process (finance, operations, customer service, procurement); no programming in daily work; has described a process to IT at least once | Developers; anyone involved in SFX planning, design, prototype or implementation |
| RP-ENG | P-ENG Engineer | Writes Python professionally; has used an AI coding assistant (Claude Code or Cursor) in the last 3 months; comfortable with a CLI | SFX implementers, architects, reviewers; GEN set authors; anyone with GEN access |
| RP-APR | P-APR Approver | Approves changes, releases, vendor access or process changes as part of the role (team lead, risk, compliance, delivery owner) | Anyone involved in SFX design or implementation |

Persona mix per round (PROPOSED):

| Round | RP-BU | RP-ENG | RP-APR |
|---|---|---|---|
| R0 | 2 | 2 | 1 |
| R1 (REF-1) | 2 | 2 | 1 |
| R2 (REF-2) | 1 | 3 | 1 |
| R3 (REF-3) | 2 | 1 | 2 |

Screener records only the profile, years in role band and tool familiarity band. No names or employer data are stored in repository artifacts.

---

## 3. Tasks and success criteria

Each task has a start state, an end state and binary success criteria. A task is a **success** only if all criteria hold without a moderator rescue (§5). Persona codes: BU, ENG, APR.

### 3.1 REF-1 Invoice automation (R0 on prototype, R1 on Studio/IDE)

| Task | Persona | Instruction (given to participant) | Success criteria |
|---|---|---|---|
| T1-01 | BU | "Describe this invoice process so the system can automate it." (scenario card with synthetic process description) | Problem statement saved; state DRAFT (SCR-02) |
| T1-02 | BU | "Answer the system's questions until it says it has enough." | Interview ends with no blocking gap; at least one "Why is this asked?" opened or participant states they understood why questions were asked; state DISCOVERED (SCR-03) |
| T1-03 | BU | "Check what the system understood and fix anything wrong." (one planted wrong inferred field, e.g. wrong match rule) | Planted error found and corrected through the field control; spec confirmed; state SPECIFIED (SCR-04) |
| T1-04 | BU | "Find out which parts of the solution reuse existing capabilities, and route exceptions to a person." | Participant reads reuse class for at least one node; HITL routing set through the property form, not by wiring; state BLUEPRINTED (SCR-05) |
| T1-05 | BU | "Connect the invoice mailbox." (fixture adapter) | Fixture adapter bound; participant states that no password was shown or asked for; no attempt to enter a secret value succeeds (SCR-06) |
| T1-06 | BU | "Ask for approval to try it." | Approval request submitted for the current digest (SCR-07) |
| T1-07 | APR | "Decide whether to allow this test." | Participant identifies what is unlocked (L2 sandbox, this version only) and that integrations are fixture-only; records approve or reject with reason (SCR-08) |
| T1-08 | BU | "Build it and test it with the sample invoices." | Build reaches BUILT; sandbox test reaches SANDBOX_TESTED; participant can say which negative cases (duplicate, wrong VAT, missing PO) were handled (SCR-09, SCR-10) |
| T1-09 | BU | "Tell me how much time this would save and where that number comes from." | Participant opens an evidence chip for a measured figure and distinguishes at least one assumption from a measured value (SCR-12) |
| T1-10 | BU | "Change the routing threshold and see what changes." | New version created; participant identifies from the diff that the approval must be repeated (SCR-13) |
| T1-11 | ENG | "Using your IDE, get this scenario to a validated blueprint." | `sfx validate` passes on spec; `sfx compose` produces a blueprint with `canonical_digest`; state BLUEPRINTED via authenticated door (IDE steps 1-4) |
| T1-12 | ENG | "Your session was interrupted. Continue in the Studio (or the IDE) where you left off." | Session resumed in the other door at the correct step without re-entering data (SCR-14) |
| T1-13 | ENG | "Make an invalid change to the spec in the file and find the same error in the Studio." | Participant matches the same `sfx.error` code and JSON Pointer in both doors (PC-2) |

R0 note: on the static prototype, T1-08 and T1-11..T1-13 are walked as screen sequences; success means the participant chooses the correct path and can explain the result shown.

### 3.2 REF-2 API-to-API workflow (R2)

| Task | Persona | Instruction | Success criteria |
|---|---|---|---|
| T2-01 | ENG | "Describe and specify an order-sync workflow from system A to system B." | Spec confirmed with typed contracts for both systems; state SPECIFIED |
| T2-02 | ENG | "Make sure the same event is never delivered twice." | Idempotency and retry/DLQ settings located and set through forms or files; participant explains where they are set |
| T2-03 | ENG | "Bind both systems using the stub adapters." | Stub HTTP adapters bound; no live endpoint entered; secrets referenced by handle only |
| T2-04 | BU | "Check that the mapping between the two systems is what you expect." | Participant finds the mapping in SCR-04/SCR-05 and corrects one planted mapping error |
| T2-05 | APR | "Decide whether to allow this test." | Decision recorded with reason; participant notes fixture/stub-only scope |
| T2-06 | ENG | "Run the test and find out what happened when the schema of system A changed." | Drift alert located in test results; participant links it to the drift test case |
| T2-07 | ENG | "Find out whether any event was duplicated under retries." | Participant reads the duplicate count and its evidence |
| T2-08 | ENG | "Start this in the IDE and finish the configuration in the Studio." | Same artifact IDs and digest visible in both doors (PC-4) |

### 3.3 REF-3 RAG decision support with HITL (R3)

| Task | Persona | Instruction | Success criteria |
|---|---|---|---|
| T3-01 | BU | "Describe a policy question-answering assistant for your team." | Spec confirmed including mandatory citations and approval before any effect |
| T3-02 | BU | "Make sure a person approves every decision before anything happens." | HITL point visible in the blueprint; participant confirms no path bypasses it |
| T3-03 | ENG | "Configure the document corpus." | Local vector store per profile; tenant scope visible |
| T3-04 | APR | "Decide whether to allow this test." | Decision recorded with reason |
| T3-05 | BU | "Run the sample questions and check that every answer shows its sources." | Participant verifies citations on at least three answers |
| T3-06 | BU | "One of the documents contains hidden instructions. Find out what the system did." | Participant finds the injection test result and states the outcome |
| T3-07 | APR | "A draft decision is waiting for you. Approve or reject it." | Decision recorded against the specific draft; participant identifies cited evidence |
| T3-08 | APR | "Is this ready for production?" | Participant states that it is sandbox-tested, not certified and not admitted, and that admission is outside SFX |

---

## 4. Session plan and think-aloud script outline

Session length: 75 minutes (R0: 60 minutes). One moderator, one note-taker. Remote or in person.

1. **Welcome (5 min).** Purpose: "We are testing the product, not you." Explain think-aloud. Confirm consent (§8). Confirm that all data is synthetic.
2. **Background (5 min).** Role, tools used, how they describe processes or approve changes today. No employer-identifying details are recorded.
3. **Practice (3 min).** Think-aloud warm-up on a neutral task.
4. **Tasks (45-55 min).** Tasks for the participant's persona in order. For each task:
   - Read the scenario card; participant restates the goal.
   - Participant works and thinks aloud. The moderator uses only neutral prompts: "What are you thinking?", "What do you expect to happen?", "What are you looking for?"
   - Moderator does not explain the UI. Assistance follows the levels in §5.
   - After the task: Single Ease Question (1-7) and "What, if anything, surprised you?"
5. **Trust probes (5 min).** "What did you just allow?", "Where would you check a number?", "Could this have touched a real system?", "Where is your password or key stored?"
6. **SUS (3 min).** Standard 10-item SUS on the front door they used most.
7. **Debrief (5 min).** Best and worst moment; one thing to change.

---

## 5. Metrics

All targets are **PROPOSED** (R0 §8) and are decided under **OD-SFX-07**.

| ID | Metric | Definition | Proposed target |
|---|---|---|---|
| UM-01 | TTFS, business user | Clock time from first keystroke on SCR-02 (Describe) to the first sandbox test results displayed on SCR-10, for REF-1, excluding approver wait time (approval decision time is measured separately in UM-08) | ≤ 45 min |
| UM-02 | TTFS, engineer | Clock time from first IDE prompt (`/sfx-session` start) to a validated blueprint (state BLUEPRINTED with `canonical_digest`) | ≤ 20 min |
| UM-03 | Task success | Share of attempted tasks meeting all success criteria without rescue | ≥ 80% |
| UM-04 | SUS | Mean System Usability Scale score per round | ≥ 75 |
| UM-05 | Error count | Observed user errors per task: slips (wrong control) and mistakes (wrong mental model), recorded separately; plus system errors shown (`sfx.error` codes) | Reported; no target |
| UM-06 | Manual interventions | (a) G-2: human edits outside the designed approval and answer steps, counted from lifecycle-record actor entries; (b) moderator assists by level | Reported; G-2 target per OD-SFX-07 |
| UM-07 | Secret values exposed | Any secret value visible in UI, artifact, log, diff or report during sessions | 0 |
| UM-08 | Approval decision time | Time from opening a request on SCR-08 to a recorded decision | Reported; no target |
| UM-09 | Single Ease Question | Mean SEQ per task | Reported; no target |
| UM-10 | Critical defects open at P12 exit | Count of open findings with severity critical | 0 |

**Moderator assist levels:** A0 none · A1 neutral prompt · A2 hint (points to an area) · A3 rescue (moderator gives the answer or performs the step). A3 makes the task a failure. A2 is counted under UM-06(b).

**R0 measurement note:** UM-01 and UM-02 on the prototype are proxies (time to reach the equivalent screen in the prototype). They are labelled `PROXY` and form the TTFS baseline required by SFX-P3-01.

---

## 6. Accessibility checks within sessions

Rounds R1..R3 include at least one participant who uses keyboard-only navigation for the whole session, when recruitment allows; it is recorded if not achieved. Findings against WCAG 2.2 AA (SFX_UX_DESIGN §8) are logged with the same severity scale.

---

## 7. Severity scale

| Severity | Definition | Handling |
|---|---|---|
| Critical | Prevents task completion with no workaround, **or** causes a wrong or unsafe outcome without the user noticing: secret value exposed, action taken beyond the shown authority, approval of a different version than intended, a number shown without evidence, the user believes sandbox means production | Must be fixed and re-tested before the next round; blocks P12 exit |
| Major | Task completed only with A2 hint or significant delay; repeated across 2 or more participants; or misleads but the user notices and recovers | Fixed before the next round, or explicitly deferred by operator with reason |
| Minor | Causes hesitation or a recoverable error for one participant | Backlog, prioritized in SFX-P12-05 |
| Cosmetic | Visual or wording issue with no effect on completion | Backlog, batched |

---

## 8. Recording and consent

- **Synthetic data only.** All scenarios use the REF synthetic datasets (synthetic invoices, fixture APIs, synthetic policy corpus). **No real customer data**, no participant's own company data, no real documents.
- **No real credentials.** Participants never enter a real password, API key or token into the product under test. Test accounts and secret references are prepared by the team; secret values are never shown.
- **Sandbox only.** Sessions run at L2 in LAB with fixture or stub adapters and egress deny. No live vendor is reachable (C10/C15).
- **Informed consent** before recording, covering purpose, what is recorded (screen and audio; camera off by default), storage, retention and the right to stop or withdraw at any time without giving a reason.
- **Storage.** Recordings are never committed to the repository or placed in product paths. They are kept outside the repository with access limited to the moderator and note-taker, for a retention period set by the operator (PROPOSED: deleted after the round report is accepted).
- **Repository artifacts** contain only anonymized notes, participant codes (`P-R<round>-<nn>`), metrics and findings. No names, faces, voices, employers or contact details.
- **Session artifacts** produced while running a round (exports, logs) go under `.tmp/session/sfx-usability/<round>/` per the session-artifacts rule and are not committed.

---

## 9. Results templates

### 9.1 Round summary

| Round | SHA / prototype version | Date | Participants | UM-01 TTFS BU | UM-02 TTFS ENG | UM-03 Task success | UM-04 SUS | UM-07 Secrets exposed | Critical | Major | Minor | Cosmetic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN (PROXY) | NOT_RUN (PROXY) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| R1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| R2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| R3 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

### 9.2 Per-participant, per-task (copy per round)

| Participant | Profile | Task | Success | Time | Assist level | Slips | Mistakes | SEQ | Notes |
|---|---|---|---|---|---|---|---|---|---|
| P-R0-01 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| P-R0-02 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| P-R0-03 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| P-R0-04 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| P-R0-05 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

### 9.3 Findings log (copy per round)

| Finding | Severity | Screen | Step | Participants affected | Observation | Contract or UI | Recommendation | Backlog item | Status |
|---|---|---|---|---|---|---|---|---|---|
| UXF-R0-01 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

---

## 10. From findings to the refinement backlog (SFX-P12-05)

1. **Log.** Every observation becomes a finding `UXF-R<round>-<nn>` with severity (§7), screen ID (`SCR-xx`), lifecycle step, affected participant count and an anonymized observation.
2. **Classify the fix type.**
   - **UI-only:** layout, wording, navigation, accessibility. Fixed in the Studio or IDE skill.
   - **Contract change request:** the fix needs a new or changed field, error code, operation or state. Per SFX_UX_DESIGN §0 ("UI must not define the architecture"), it is routed to the Tier-2 schema owner and, where needed, to the ADR owner. It is never fixed by adding client-only logic.
   - **Parity defect:** the doors behave differently. Routed to the parity suite owner (SFX-P10-03) with a failing parity case.
3. **Backlog entry.** Each finding gets a refinement backlog item under SFX-P12-05 (`SFX-P12-05-BL-<nn>`), linked from the findings log.
4. **Prioritize.** Critical first, then major. Critical findings must be fixed and re-tested before the next round. Major findings are fixed before the next round or deferred by the operator with a recorded reason.
5. **Verify.** A fix is closed only after the next round (or a targeted re-test with at least two participants of the affected profile) shows the problem does not recur.
6. **Report.** Each round produces a round report with the §9 tables filled in at a recorded SHA. R0 findings feed the Studio build plan (SFX-P10-05/06); R1..R3 findings feed the SFX-P12-05 burn-down. Exit condition for P12: UM-10 = 0 critical open (target PROPOSED, OD-SFX-07).
