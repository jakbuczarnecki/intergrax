<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# SCENARIO-FACTORY-X (SFX) - UX design

**Ticket:** [`T-IGX-037-D`](../../../../../governance/tickets/T-IGX-037-D.md) (WBS unit SFX-P3-01, wave W2, docs only)  
**Companion:** [`SFX_USABILITY_PROTOCOL.md`](SFX_USABILITY_PROTOCOL.md)  
**Sources:** SFX master plan R1 §7.4 (SFX-CON-01..17), §7.7 (PC-1..6), §7.8 (stack), §7.9 (FSM), §7.10 (authority levels), §8 (UX and generalization test), §13 (security qualification); R0 §8 (full UX) and §12 (REF-1..3).  
**Status:** PROPOSED design. No Studio code exists, and no usability round has been run. All contract names are PROPOSED (`sfx.<name>.v1`) until the P2-02 schemas and ADR-SFX-001..006 are Accepted (OD-SFX-06).

---

## 0. Platform UI constraints (binding on this document)

This design follows two platform rules. Where they conflict with UX preference, they win.

1. [`PLATFORM_FOUNDATION_extended_depth.md`](../../../architecture/satellites/PLATFORM_FOUNDATION_extended_depth.md) §8.6 "UI Is Optional" states: "Intergrax is not frontend-first." and "UI must not define the architecture."
2. [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §39.7 (DX-MAINT-04): "product UI belongs to Tier-3 hosts or Phase K, not the DX control plane."

Consequences for this document:

- **Contract-first.** Every screen is a view over SFX-CON contracts and service operations owned by Tier-2 `agents/scenario_factory`. No screen introduces a field, state, transition or rule that the contracts do not already define. A UX need that requires a new field is raised as a schema change request against P2-02, never solved in the client.
- **Tier-3 placement.** The Studio is a React + TypeScript SPA under the Tier-3 host `applications/scenario_studio_application/web/` (OD-SFX-02). It talks to the Studio API v1 with a JWT bearer token and an explicit CORS origin list; there are no cookies and no cookie session (§7.8 S-2, ADR-SFX-003).
- **The Studio is one front door, not the product.** The IDE front door (skills plus the `sfx` CLI and the MCP live mode) must be able to complete every lifecycle step that its authority level allows, with identical artifacts (§7.7, §8 below).
- **No visual decision drives architecture.** §9 (visual direction) is limited to tokens and principles.

---

## 1. Personas

| | **P-BU Business user** ("Process owner") | **P-ENG Engineer** ("Scenario engineer") | **P-APR Approver** ("Risk / delivery approver") |
|---|---|---|---|
| Typical role | Operations, finance or customer-service process owner; no coding | Python engineer who works in Claude Code or Cursor | Team lead, risk or compliance officer, delivery owner |
| Primary front door | Studio | IDE (`/sfx-session`, `sfx` CLI, MCP live mode); Studio for review | Studio approval inbox |
| Goals | Turn a business problem into a working sandbox scenario without writing code; see what it costs and saves; stay in control of what it does | Get a validated blueprint fast; reproduce every step from the command line; review and extend generated code; keep files and Studio in sync | Understand exactly what is being approved; approve or reject with a recorded reason; be sure nothing changes after approval without coming back |
| Fears | Being asked questions they cannot answer; jargon; "the AI did something I didn't ask for"; numbers they cannot defend to their manager | Black-box generation; drift between Studio and files; losing work on interruption; hidden validation in the UI; being blocked by UI-only steps | Approving a moving target; hidden secrets or live vendor access; being accountable for production effects; approval fatigue |
| Authority ceiling | L0 read, L1 proposal; triggers L2 sandbox actions only on an approved version | L0, L1 (offline CLI is limited to L0/L1 on local files, PC-3); L2 only through an authenticated door (API or MCP) on an approved version | L0 read plus the human decision binding (SFX-CON-13) that unlocks L2 for one version digest |
| Never | L3 | L3 | L3. **L3 is never granted by SFX** (§7.10) |
| Success signal | First sandbox run and a report they understand | Validated blueprint and reproducible handoff | Decision recorded against a digest, with evidence |

A fourth role exists in the FSM but is not a design persona: the **independent certifier** who moves `SANDBOX_TESTED → SANDBOX_CERTIFIED` (not the implementer). The Studio shows that transition as an external event and offers no control for it.

---

## 2. Design principles

| ID | Principle | What it means on screen |
|---|---|---|
| DP-1 | **Contract-first UI generated from JSON Schema** | Forms, validation hints, enums and field help are generated from `agents/scenario_factory/schemas/*.schema.json` (PC-1). Client-side checks are transport parsing and convenience only; the authoritative result is the Tier-2 `sfx.error` code and JSON Pointer (PC-2). |
| DP-2 | **Progressive disclosure** | Ask only unresolved fields. Show summaries first and detail on demand. Advanced fields (budgets, retries, egress) are collapsed but always reachable, and their resolved value is always visible. |
| DP-3 | **Explain why, on every question** | Every question and every required field has a "Why is this asked?" affordance showing: the spec field it fills (JSON Pointer), the downstream contract or gate that depends on it, and the effect of skipping or defaulting it. |
| DP-4 | **Never show secret values** | Integrations show secret references and binding status only. There is no reveal control, no copy-value control and no value echo in errors, diffs, logs or reports. Secret values in artifacts: target 0 (R0 §8). |
| DP-5 | **Authority badge on every action** | Every control that calls a service operation carries an L0/L1/L2/L3 badge with text, not colour alone. Controls above the current principal's authority are shown disabled with the reason, not hidden. L3 never appears as an enabled control. |
| DP-6 | **Nothing irreversible without HITL** | Any transition gated in §7.9 (approval, handoff, dossier export) requires an explicit human decision bound to a version digest. There is no auto-advance across a gated transition and no "approve all". |
| DP-7 | **Generated is not certified** | Build, test and run screens carry the SFX-INV-01 banner: generated code is never a certified scenario; SFX ends at SANDBOX_CERTIFIED / ADMISSION_DOSSIER_READY. |
| DP-8 | **Every number traced to evidence** | Metrics, costs and ROI figures render only with an evidence reference. A number without evidence is not shown; an assumption is labelled as an assumption. |
| DP-9 | **Same operation, same result, any door** | The UI never offers an operation that the CLI or MCP cannot express (PC-4, PC-6), except where PC-3 deliberately restricts the offline CLI. |
| DP-10 | **Recoverable by default** | Work survives a closed tab, an expired token or a switch of front door, through `SfxSessionState` (SFX-CON-12), not browser storage. |

---

## 3. Lifecycle, contracts and states (reference)

Lifecycle steps (R0 §8): **Describe → Discover → Understand → Design → Configure → Approve → Build → Test → Run → Present → Improve.**

Authority levels (§7.10): **L0** read-only · **L1** proposal-only · **L2** sandbox-executable (LAB, fixture adapters, budgets, egress deny) · **L3** production, never granted by SFX.

FSM (§7.9): `DRAFT → DISCOVERED → SPECIFIED → BLUEPRINTED → APPROVED → BUILDING → BUILT → SANDBOX_TESTED → SANDBOX_CERTIFIED → ADMISSION_DOSSIER_READY`, with `BUILDING → APPROVED` on job failure and `SANDBOX_TESTED → BLUEPRINTED` on improve. Configure, Run and Present cause no state transition; they produce new artifact versions within the current state.

Every transition is persisted as a new `ScenarioLifecycleRecord` (SFX-CON-14) version with `{from, to, actor, evidence_refs}`. The UI reads the current state from that record and never from markdown frontmatter (§7.9).

---

## 4. Journeys

Column legend: **Contract** = SFX-CON IDs read or written; **Auth** = authority level of the step's main action; **FSM** = transition caused (— = none).

### 4.1 P-BU Business user (Studio)

| # | Step | Screen | User action | System response | Contract | Auth | FSM |
|---|---|---|---|---|---|---|---|
| 1 | Describe | SCR-02 | Types the problem in free text; optionally uploads a document | Creates the scenario container and session; stores the statement; shows a short echo of what was understood and the next step | 01, 12, 14 | L1 | `[*] → DRAFT` |
| 2 | Discover | SCR-03 | Answers adaptive interview questions one at a time; asks "why is this asked?"; skips or defers | Computes spec gaps from the schema, picks the next question, records answers with provenance and confidence; ends when no blocking gap remains | 02, 12 | L1 | `DRAFT → DISCOVERED` |
| 3 | Understand | SCR-04 | Reviews the "Understanding" summary; edits statements inline; confirms | Re-validates the specification on every edit; shows field-level errors; publishes a spec version | 03, 12 | L1 | `DISCOVERED → SPECIFIED` |
| 4 | Design | SCR-05 | Reviews the generated blueprint canvas; edits node properties in the side form | Runs capability resolution and composition; shows the reuse class per binding and the reuse ratio (G-1); publishes a blueprint version | 04, 05 | L1 | `SPECIFIED → BLUEPRINTED` |
| 5 | Configure | SCR-06 | Picks a profile, maps integrations to fixture adapters, sees secret placeholders | Validates the integration plan; shows unresolved bindings; never shows secret values | 06, 05 | L1 | — (new versions in BLUEPRINTED) |
| 6 | Approve | SCR-07 | Reviews the approval package; submits the approval request with a note | Creates an approval request bound to the blueprint version digest; routes it to the approver inbox; locks that version against edits | 13 | L1 (request) | — (pending decision) |
| 7 | Build | SCR-09 | Starts the build in the available mode; watches progress | Issues the engineering handoff for the approved digest; streams job state | 08, 15 | L2 | `APPROVED → BUILDING → BUILT` (or `→ APPROVED` on failure) |
| 8 | Test | SCR-10 | Starts the sandbox test with the synthetic dataset; reviews results | Runs the test plan at L2 (fixture adapters, egress deny, budgets); records qualification results | 07, 16 | L2 | `BUILT → SANDBOX_TESTED` |
| 9 | Run | SCR-11 | Runs the scenario on the dataset; inspects outputs and exceptions | Produces a run report with evidence references | 09 | L2 | — |
| 10 | Present | SCR-12 | Opens the business report and ROI; adjusts labelled ROI assumptions; exports | Renders the value report with every figure linked to evidence; marks assumptions | 09, 10 | L0 / L1 | — |
| 11 | Improve | SCR-13 | Chooses "Improve"; changes the blueprint; reviews the semantic diff | Opens a new blueprint version; shows which approval and tests are invalidated | 05, 14 | L1 | `SANDBOX_TESTED → BLUEPRINTED` |

### 4.2 P-ENG Engineer (IDE, CLI, MCP; Studio for review)

IDE skills come from SFX-P4-A01 (v0, L1 output to `.tmp/sfx/<slug>/`) and SFX-P10-08 (v1, skills call CLI/MCP). CLI commands are `sfx validate|resolve|compose|handoff|test|report` (R0 §8).

| # | Step | IDE step | User action | System response | Contract | Auth | FSM |
|---|---|---|---|---|---|---|---|
| 1 | Describe | `/sfx-session` (start) | States the problem in chat or points at a file | Writes the problem statement file; `sfx validate` checks it | 01, 12 | L1 (local) | `[*] → DRAFT` only when synced through an authenticated door |
| 2 | Discover | `/sfx-discover` | Answers the same gap-driven questions conversationally | Writes the discovery record; shows remaining gaps | 02 | L1 | `DRAFT → DISCOVERED` (authenticated door) |
| 3 | Understand | `/sfx-specify` + `sfx validate` | Edits the spec file directly or through the skill | Validates; prints `sfx.error` code and JSON Pointer per error | 03 | L1 | `DISCOVERED → SPECIFIED` (authenticated door) |
| 4 | Design | `/sfx-resolve`, `/sfx-blueprint` + `sfx resolve`, `sfx compose` | Runs resolution and composition; edits the blueprint file | Writes the resolution report and blueprint with `canonical_digest` | 04, 05 | L1 | `SPECIFIED → BLUEPRINTED` (authenticated door) |
| 5 | Configure | `/sfx-integration-setup` + `sfx validate` | Edits the integration plan; references secrets by handle | Validates bindings; rejects any literal secret value | 06 | L1 | — |
| 6 | Approve | No offline path (PC-3). MCP live mode or Studio SCR-07 | Submits the approval request through MCP live mode, or hands over to the Studio | Same as Studio: request bound to the digest | 13 | L1 (request) | — |
| 7 | Build | `sfx handoff --mode …` (MCP live for the shared transition) | Starts the handoff for the approved digest | Offline: local scaffold preview only, no transition. Live: same job as Studio | 08, 15 | L2 (live) / L1 (offline preview) | `APPROVED → BUILDING → BUILT` (live only) |
| 8 | Test | `/sfx-test-plan` + `sfx test` (MCP live for L2) | Drafts the test plan; runs it in the sandbox through MCP live mode | Offline: test plan validation only. Live: L2 run with results | 07, 16 | L2 (live) | `BUILT → SANDBOX_TESTED` (live only) |
| 9 | Run | MCP live mode | Triggers a sandbox run | Run report with evidence references | 09 | L2 | — |
| 10 | Present | `/sfx-present` + `sfx report` | Generates the report | Same report artifact as SCR-12 (same digest) | 09, 10 | L0 / L1 | — |
| 11 | Improve | `/sfx-blueprint` + `sfx compose` | Edits the blueprint; reviews the diff | New version; diff output (see UX-GAP-02) | 05, 14 | L1 | `SANDBOX_TESTED → BLUEPRINTED` (authenticated door) |

### 4.3 P-APR Approver (Studio)

| # | Step | Screen | User action | System response | Contract | Auth | FSM |
|---|---|---|---|---|---|---|---|
| 1 | Describe | SCR-17 (timeline) | Optional: reads the problem statement | Read-only view | 01 | L0 | — |
| 2 | Discover | SCR-17 | Optional: reads the interview record and its provenance | Read-only view | 02 | L0 | — |
| 3 | Understand | SCR-04 (read-only) | Reads the confirmed specification | Read-only view | 03 | L0 | — |
| 4 | Design | SCR-05 (read-only) | Inspects the blueprint, reuse classes, HITL points | Read-only canvas | 04, 05 | L0 | — |
| 5 | Configure | SCR-06 (read-only) | Checks that integrations are fixture-only and secrets are references | Read-only inspector | 06 | L0 | — |
| 6 | Approve | SCR-08 | Opens the request; reviews the package and diff since last approval; approves or rejects with reason | Records the decision binding against the exact version digest; on approve unlocks L2 for that digest only | 13, 14 | Human decision (unlocks L2 for one digest) | `BLUEPRINTED → APPROVED` (approve); — (reject) |
| 7 | Build | SCR-09 (read-only) | Optional: monitors the job | Read-only | 15 | L0 | — |
| 8 | Test | SCR-10 (read-only) | Reviews test and security-gate results | Read-only | 16 | L0 | — |
| 9 | Run | SCR-11 (read-only) | Reviews outputs and exceptions | Read-only | 09 | L0 | — |
| 10 | Present | SCR-12 / SCR-15 | Reviews report and ROI; after independent certification, reviews the dossier export request | Read-only report; dossier export is a gated transition requiring a decision (UX-Q-03) | 10, 17 | L0 / human decision | `SANDBOX_CERTIFIED → ADMISSION_DOSSIER_READY` |
| 11 | Improve | SCR-08 (re-approval) | Receives a new request for the improved version; reviews the semantic diff | Earlier approval does not carry over; new decision required | 13, 14 | Human decision | `BLUEPRINTED → APPROVED` |

---

## 5. Information architecture

### 5.1 Navigation

- **Global bar:** workspace and tenant indicator (always visible), scenario switcher, approval inbox with count, catalog, help, principal and current authority ceiling.
- **Scenario shell:** a horizontal **lifecycle stepper** of the 11 steps. Each step shows its status (not started, in progress, done, blocked, invalidated). The current FSM state is shown as a separate state chip, because steps and states are not 1:1.
- **Right rail (contextual):** "Why?" explanations, validation issues list, evidence drawer, version picker.
- **Stepper rules:** a user can always go back to read any step (L0). Going forward past a gated transition requires the gate. Editing an earlier step after approval opens a new version (Improve semantics), never an in-place edit.
- **Deep links:** every screen is addressable by scenario ID, step and artifact version, so the IDE can print a Studio link and vice versa.

### 5.2 Screen inventory

| ID | Screen | Purpose | Key components | Contracts |
|---|---|---|---|---|
| SCR-01 | Scenario home | List scenarios in the workspace; start new; resume | Scenario table (name, FSM state, last step, last actor, updated), "Resume" banner, filters, "New scenario" | 12, 14 |
| SCR-02 | Describe | Capture the business problem | Free-text editor, document upload (classified, tenant-scoped), example prompts, data-handling notice (no real customer data in sandbox) | 01 |
| SCR-03 | Discover - adaptive interview | Fill spec gaps by asking only what is missing | Question card, "Why is this asked?", answer controls generated from schema, skip or defer, gap progress meter, confidence chips, transcript | 02, 12 |
| SCR-04 | Understand - summary | Confirm what the system understood | Sectioned summary (goal, inputs, outputs, rules, exceptions, HITL points, data classes, success metrics), inline edit, provenance markers, validation list, "Confirm" | 03 |
| SCR-05 | Design - blueprint canvas | Show the composed solution and let the user adjust it through forms | Read-only graph, outline/table alternative, node property form, reuse class badges, G-1 reuse ratio, unresolved capability list | 04, 05 |
| SCR-06 | Configure - configuration inspector | Choose profile and integrations; see secret placeholders | Profile selector, integration list with adapter type (fixture/stub), secret reference rows, budget and retry settings, egress summary, validation | 06, 05 |
| SCR-07 | Approve - request | Assemble and submit an approval request | Approval package summary, digest, authority being unlocked, risk checklist, diff since last approved version, note, "Submit request" | 13 |
| SCR-08 | Approval inbox | Decide on pending requests | Inbox list (pending, decided, superseded), request detail, evidence links, approve / reject with mandatory reason, decision history | 13, 14 |
| SCR-09 | Build progress | Follow the engineering job | Mode indicator, step list or job state timeline, logs (redacted), SEF job reference, failure panel, retry (policy-bounded) | 08, 15 |
| SCR-10 | Test - sandbox results | Run and inspect the test plan | Test plan summary, case results, negative tests, security gates, budget use, evidence links, SFX-INV-01 banner | 07, 16 |
| SCR-11 | Run results | Inspect a sandbox run | Run summary, output records, exception queue, timeline, cost and latency with evidence | 09 |
| SCR-12 | Present - report and ROI | Business report | Narrative summary, KPI tiles with evidence chips, ROI model with labelled assumptions, export | 09, 10 |
| SCR-13 | Improve - version history and diff | Compare versions and start an improvement | Version list with digests, semantic diff, impact panel (invalidated approvals and tests), "Start new version" | 05, 14 |
| SCR-14 | Resume / interruption recovery | Restore a session in any door | Resume dialog, last step and pending item, conflict notice, IDE resume command | 12 |
| SCR-15 | Qualification and dossier | Show certification status and export the admission dossier | Qualification record, independent certification status (read-only), dossier contents list, export request | 16, 17 |
| SCR-16 | Catalog | Browse catalog sidecar entries | Entry list (package ID, version, checksum, listed/deprecated/retired), "metadata only, never execution truth" notice | 11 |
| SCR-17 | Lifecycle timeline and audit | Show every transition with actor and evidence | Transition list from the lifecycle record, actor, evidence refs, decision bindings | 14, 13 |

Error states are a cross-cutting pattern (§6.11), not a separate screen.

---

## 6. Key interaction specs

### 6.1 Adaptive interview (SCR-03)

**Next-question selection.**
1. Gap computation is **deterministic**. The service validates the current draft `BusinessScenarioSpecification` against its schema and lists unresolved JSON Pointers. The UI renders that list; it does not compute gaps.
2. Gaps are ranked by (a) blocking for downstream contracts (resolver, blueprint, test plan), (b) safety-relevant fields (tenant scope, data classes, HITL points, external effects, egress), (c) expected reduction in capability-resolution ambiguity, (d) user effort. The ranking rule lives in Tier-2 and is the same in the IDE.
3. One question per card. Closely related fields can be grouped on one card (for example, the three match keys of a 3-way match in REF-1).
4. The LLM only phrases the question and parses the answer. Parsed values are shown back as structured fields before they are stored, so the user can correct a misparse. LLM turns are recorded for replay (PC-5).
5. Already answered or inferred fields are never asked again (WCAG 2.2 SC 3.3.7 redundant entry). An inferred field appears in SCR-04 for confirmation instead.
6. The interview ends when no blocking gap remains. The user can continue optional questions or go to Understand.

**Confidence display.**
- Each captured field carries **provenance** (stated by user, extracted from upload, inferred, default) and a **categorical confidence** (high, medium, low). The UI shows no fake-precision percentage.
- Low-confidence and inferred fields are listed first in SCR-04.
- Confidence and provenance must exist in SFX-CON-02/03. If the P2-02 schema does not carry them, this is a schema change request (UX-DEP-01), not a client feature.

**Controls.** Answer (schema-generated control), "Why is this asked?", "I don't know" (records an explicit unknown; may block), "Skip for now" (non-blocking fields only), "Show transcript". Each carries the L1 badge.

### 6.2 "Understanding" summary editing (SCR-04)

- The summary is a **presentation** of SFX-CON-03, grouped into business-readable sections. Each statement maps to one or more JSON Pointers.
- Inline edit opens the schema-generated control for that field, not free text over the whole summary. Free-text corrections are allowed only for free-text fields.
- Every edit triggers server validation. Errors show inline at the field with the `sfx.error` code available in a details disclosure.
- Provenance markers: "you said", "from your document", "inferred - please confirm". Confirming an inferred field changes its provenance to stated.
- "Confirm understanding" publishes a spec version and performs `DISCOVERED → SPECIFIED`. It is disabled while any blocking error exists, with the reason shown.

### 6.3 Blueprint canvas (SCR-05)

- **Read-only graph** generated from SFX-CON-05: nodes for capabilities, agents, tools, integrations, data stores and HITL points; edges for data and control flow. Layout is automatic and deterministic for a given digest, so the same version always looks the same.
- **No free wiring.** Users cannot draw, drag-connect or delete edges. All changes go through the **node property form** (schema-generated), which may add, remove or replace bindings. The graph re-renders from the recomposed blueprint. This also removes any need for drag interactions (WCAG 2.2 SC 2.5.7).
- **Equivalent outline view**: a keyboard- and screen-reader-friendly tree or table listing the same nodes and edges. It is not a fallback; it is a first-class view.
- Node badges: resolution class from SFX-CON-04 (for example REUSE_AS_IS, REUSE_WITH_CONFIGURATION), HITL marker, external-effect marker, authority needed at run time (L2).
- Header shows G-1 reuse ratio, the count of unresolved capabilities, and the blueprint digest (short form, full on hover/focus).
- Unresolved capabilities appear as a list with "why unresolved" from the resolution report and the options the resolver offers. The UI does not invent options.

### 6.4 Configuration inspector (SCR-06)

- **Profiles:** the user chooses from the profiles the service lists for the scenario. L2 sandbox implies LAB, fixture adapters, budgets and egress deny; these are shown as fixed facts, not toggles.
- **Integrations:** each row shows integration name, adapter type (fixture or stub HTTP only under C10/C15), binding status (bound, unbound, invalid) and the related blueprint node.
- **Secret placeholders:** a secret is shown as its reference name, scope and binding status. There is no input that accepts a secret value in the Studio, no reveal, no copy. If the schema marks a field as a secret reference and the user pastes something that looks like a secret into any free-text field, the service rejection is shown without echoing the value.
- **Budgets and retries:** shown with their resolved values and source (default, profile, user). Changing them is L1 and creates a new version.
- **Egress summary:** "Outbound network: denied (empty allowlist)" in L2. Any attempt to add egress is out of SFX scope and shown as unavailable with the reason.

### 6.5 Approval inbox and decision binding (SCR-07, SCR-08)

- A request (SFX-CON-13) binds to **one artifact version digest**. The request card shows scenario, version, short digest, requester, requested authority ("L2 sandbox build and test for this version"), and age.
- Detail view shows: what will happen if approved (build mode, test plan, budgets), integrations (fixture-only check), secret handling (references only), HITL points, diff since the last approved version, and links to evidence.
- **Approve** and **Reject** are explicit. Reject requires a reason. Approve may take an optional note. Both record a decision binding and a lifecycle record entry.
- **Superseded:** if the bound artifact gets a new version after the request, the request moves to "Superseded" and cannot be approved (the digest no longer matches).
- **No bulk approval.** One decision per request.
- **Self-approval:** PROPOSED rule that the requester cannot approve their own request, aligned with the "independent check (not implementer)" principle. Needs confirmation (UX-Q-02).

### 6.6 Build progress (SCR-09), both modes

| | DETERMINISTIC_SCAFFOLD | AGENTIC_ENGINEERING |
|---|---|---|
| Availability | Available after APPROVED | Shown disabled until GATE-SEF-CONSUME is met, with the reason "SEF-X contracts not yet consumable" |
| Display | Ordered deterministic step list (validate, compose, canonicalize, scaffold), each with status | Job state timeline: QUEUED, RUNNING, WAITING, SUCCEEDED, FAILED, DEAD_LETTERED; SEF `job_id` shown by reference (SFX-CON-15) |
| Waiting | — | WAITING states that need a human show the reason and link to where the decision is made |
| Failure | Failed step, `sfx.error` code, redacted log excerpt | Job failure summary and SEF reference |
| After failure | State returns to APPROVED. "Retry" is available only within policy; no auto-retry beyond policy | Same |
| Live region | Polite announcements on step completion; no announcement per log line | Same, per state change |

Both modes show the SFX-INV-01 banner and the workspace isolation fact ("writes confined to the job workspace").

### 6.7 Sandbox test results (SCR-10)

- Summary: cases passed/failed, negative tests passed/failed, security gates (injection, secret values in artifacts, egress, tenant crossover), budget used versus limit.
- Case table: case ID, intent, expected, actual, status, evidence link. Filters by status and type.
- Negative tests from REF scenarios (malformed PDF, duplicate invoice, 5xx storm, poisoned document and so on) are grouped and labelled as expected failures that the scenario must handle.
- Banner: "Sandbox tested. Not certified. Certification is an independent check outside this screen."
- "Improve" and "Run" are the next actions.

### 6.8 Run report and ROI (SCR-11, SCR-12)

- **Evidence rule:** every number (accuracy, counts, tokens, cost, latency, time saved) has an evidence chip that opens the evidence reference from SFX-CON-09/10. A figure with no evidence ref is **not rendered**; the slot shows "No evidence - not shown".
- **Measured vs assumed:** ROI inputs such as hourly cost, volume per month or manual handling time are user-entered **assumptions**, visibly labelled, versioned with the report, and never mixed with measured values without the label.
- **Formula visible:** ROI shows its formula and each term's source.
- **Scope label:** "Sandbox run on synthetic data". Production obligations (admission outside SFX, PROD-Q) are listed, not hidden.
- Export produces the same report artifact that `sfx report` produces (same digest).

### 6.9 Version history and semantic diff (SCR-13)

- Version list: version number, short digest, author, door (Studio, IDE, MCP), time, FSM state at creation, approval status.
- **Semantic diff** compares canonical models (PC-5), not YAML or pretty JSON text. Changes are grouped by section and shown as added, removed or changed fields with JSON Pointers and human labels.
- **Impact panel:** which approvals the change invalidates, which tests must re-run, whether reuse ratio changed.
- Diffing any two versions is L0. Starting a new version is L1.

### 6.10 Interruption recovery and resume (SCR-14)

- `SfxSessionState` (SFX-CON-12) is saved by the service at each step boundary and after each answered question, as an append-only version with a CAS current pointer.
- On opening a scenario with an unfinished step, SCR-14 shows: last step, pending question or draft, last door used, time, and a "Resume" action. The IDE equivalent is printed (`/sfx-session` resume for that scenario).
- **Token expiry:** with JWT bearer and no cookies, a session can expire mid-step. The UI keeps the unsent input visible, asks the user to sign in again, and then retries the save. It does not use browser storage as the source of truth.
- **Cross-door conflict:** if another door advanced the state (CAS mismatch), the UI shows "Updated elsewhere (IDE / Studio) at <time>" with a diff and the choice to reload. It never merges silently.
- **Offline IDE work:** offline CLI output is local L1 files. Bringing them into the shared store goes through an authenticated door and validation (UX-Q-04).

### 6.11 Error states (cross-cutting)

| Class | Example | UI behaviour |
|---|---|---|
| Validation | Missing required field | Inline at field via JSON Pointer; summary list in the right rail; `sfx.error` code in details |
| Authority | Action needs L2 but version not approved | Control disabled with reason and the path to unlock it (request approval) |
| Gate not met | AGENTIC_ENGINEERING before GATE-SEF-CONSUME | Option disabled with a reason; no workaround offered |
| Conflict | CAS mismatch from another door | Conflict notice with diff and reload |
| Policy denied | PolicyEngine denies a tool or effect | Plain-language reason, policy reference, no retry button |
| Budget exceeded | Token or cost budget hit during test | Run stopped; partial results labelled partial; budget shown |
| Dependency unavailable | Backend or MCP live mode unreachable | Read-only mode for loaded data; retry with backoff; no data loss |
| Session expired | JWT expired | Re-authentication with input preserved (§6.10) |
| Unknown | Unexpected server error | Correlation ID, no stack trace, no secret or payload echo |

Error copy rules: say what happened, why, and what the user can do; never blame the user; never include secret values or raw payloads.

---

## 7. IDE parity mapping

### 7.1 Screen to IDE skill / CLI / MCP

| Screen | IDE skill | CLI command (offline, L0/L1 local) | MCP live / API (authenticated) |
|---|---|---|---|
| SCR-01 Scenario home | `/sfx-session` (list, resume) | — | list scenarios |
| SCR-02 Describe | `/sfx-session` | `sfx validate` (statement) | create scenario |
| SCR-03 Discover | `/sfx-discover` | `sfx validate` (discovery record) | interview operations |
| SCR-04 Understand | `/sfx-specify` | `sfx validate` (spec) | publish spec version |
| SCR-05 Design | `/sfx-resolve`, `/sfx-blueprint` | `sfx resolve`, `sfx compose` | publish blueprint version |
| SCR-06 Configure | `/sfx-integration-setup` | `sfx validate` (integration plan) | publish integration plan |
| SCR-07 Approve request | — (no offline path, PC-3) | — | submit approval request |
| SCR-08 Approval inbox | — (Studio only; UX-Q-01) | — | list and decide requests (UX-Q-01) |
| SCR-09 Build | — (v1: skill calls CLI/MCP) | `sfx handoff` (local preview only) | handoff, job status |
| SCR-10 Test | `/sfx-test-plan` | `sfx test` (plan validation only) | sandbox test run (L2) |
| SCR-11 Run | — | — | sandbox run (L2) |
| SCR-12 Present | `/sfx-present` | `sfx report` | report |
| SCR-13 Improve / diff | `/sfx-blueprint` | `sfx compose` (diff: UX-GAP-02) | new version, diff |
| SCR-14 Resume | `/sfx-session` | — | read `SfxSessionState` |
| SCR-15 Qualification / dossier | — | — | qualification record, dossier export (gated) |
| SCR-16 Catalog | — | — | catalog read |
| SCR-17 Timeline | — | — | lifecycle record read |

Gaps in this table are recorded as UX-GAP items (§11), not filled by inventing commands. The final operation names come from the PC-6 operation map (ADR-SFX-004).

### 7.2 Parity rules applied to the UX

| Rule | UX consequence |
|---|---|
| PC-1 Schemas | Studio forms and MCP tool schemas render from the same committed schemas. A form field that has no schema field is a defect. |
| PC-2 Validation | The Studio shows the same `sfx.error` code and JSON Pointer that the CLI prints for the same invalid input. Client-only validation messages are not allowed to differ in meaning. |
| PC-3 Permissions | Authority badges and disabled states come from one authorization path. The offline CLI never shows or performs L2 or shared-store writes; the Studio and MCP do so only for an approved digest. |
| PC-4 Semantics | An operation started in the Studio and the same operation in the CLI/MCP produce the same `artifact_id`, version, `canonical_digest` and lifecycle transition. The version list shows the door used, not a different artifact. |
| PC-5 Canonicalization | Digests shown in the UI are always `canonical_digest` (RFC 8785, sha256). YAML or pretty JSON exports are labelled "presentation". Diffs are computed on the canonical model. |
| PC-6 Versioning | The Studio displays the single API/CLI/MCP version constant in Help/About. A screen that needs an operation not in the op map is blocked until the op map has it. |

---

## 8. Accessibility

Target: **WCAG 2.2 level AA** for every Studio screen. Verified in SFX-P10-06 e2e and in usability rounds (protocol §6).

- **Perceivable:** text contrast at least 4.5:1 (3:1 for large text and UI component boundaries); authority badges, states and severity never rely on colour alone (text plus icon); the blueprint graph has the outline view as an equivalent (§6.3); charts in reports have a data table.
- **Operable:** every action reachable by keyboard in a logical order; visible focus that is not obscured by sticky bars (SC 2.4.11); targets at least 24 by 24 CSS px (SC 2.5.8); no drag-only interaction (SC 2.5.7); skip link to main content; documented shortcuts for next question, confirm, open "Why?", and open evidence, none of which use single printable keys without a modifier.
- **Understandable:** consistent help location for "Why?" on every step (SC 3.2.6); no re-entry of already provided information (SC 3.3.7); sign-in without cognitive function tests (SC 3.3.8); error messages with field reference and suggestion.
- **Robust:** semantic HTML first, ARIA only where needed; live regions are polite for progress and assertive only for blocking errors.
- **Reduced motion:** honour `prefers-reduced-motion`. No animated graph layout, no auto-scrolling log, no pulsing indicators when it is set. Progress uses static state changes. No content flashes more than three times per second at any setting.
- **Time limits:** no step has a time limit. Token expiry preserves input (§6.10).

---

## 9. Visual direction (tokens and principles only)

This section defines no component library, layout engine or architecture. The design system is built in SFX-P10-04.

- **Tone:** calm, factual, auditable. Evidence and state are more prominent than decoration.
- **Colour tokens (roles, not values):** `surface`, `surface-raised`, `text`, `text-muted`, `border`, `accent`, `focus-ring`; status roles `ok`, `warning`, `danger`, `info`; authority roles `auth-l0`, `auth-l1`, `auth-l2`, `auth-l3` (L3 role used only for the disabled "never granted" state). Light and dark themes both meet the contrast targets in §8.
- **Typography tokens:** one sans-serif UI family and one monospace family for digests, JSON Pointers and IDs; a type scale of at most six steps.
- **Spacing and density:** 4 px base grid; comfortable density by default, compact density optional for engineers.
- **Shape and elevation:** small radius; elevation only for overlays and the right rail.
- **Iconography:** one outline icon set; every icon that carries meaning has a text label.
- **Motion:** short and functional only; disabled under reduced motion.
- **Data display:** digests shortened with full value on focus; numbers always with unit and evidence chip.

---

## 10. Prototype

Prototype Artifact (private claude.ai artifact, static clickable prototype, example data only, version 1 published 2026-11-12, republished as version 2 with figures relabelled as illustrative example values): https://claude.ai/artifact/R5fc5PLDQ768fSTcNbwZYJ. Source kept outside the product paths in the gitignored `.tmp/session/sfx-w2/prototype/`; prototype-only controls (persona switch, jump-to-state) are labelled in the page.

All figures shown in the prototype (accuracy, decision counts, cost, latency, ROI inputs) are illustrative example values, labelled as such on the Run and Present screens; they are not measured results and must not be quoted as such.

The prototype is a static mock-up outside repository product paths (SFX-P3-01). It is the subject of usability round 0 (see [`SFX_USABILITY_PROTOCOL.md`](SFX_USABILITY_PROTOCOL.md)).

---

## 11. Open UX questions, gaps and dependencies

### 11.1 Open questions for the operator

| ID | Question |
|---|---|
| UX-Q-01 | Can approvers decide through MCP live mode or the IDE, or is the approval inbox Studio-only? PC-6 requires a 1:1 op map; a Studio-only decision would need an explicit exception. |
| UX-Q-02 | Is self-approval (requester approves own request) forbidden? Proposed: forbidden. |
| UX-Q-03 | Is the dossier export (`SANDBOX_CERTIFIED → ADMISSION_DOSSIER_READY`) a human-decision gate, and who decides: the approver persona or another role? |
| UX-Q-04 | How do offline L1 files from the IDE enter the shared store (import operation, or always live mode)? |
| UX-Q-05 | Do configuration changes after APPROVED always invalidate the approval (proposed: yes, any digest change), given the FSM has no `APPROVED → BLUEPRINTED` edge? |
| UX-Q-06 | Is document upload on Describe in scope for v1, and what data classes are allowed in the sandbox? |
| UX-Q-07 | TTFS, task success and SUS targets (OD-SFX-07, at P3 exit). |
| UX-Q-08 | REF-2 and REF-3 inclusion (OD-SFX-08) changes usability rounds 2 and 3. |

### 11.2 Gaps found while mapping

| ID | Gap |
|---|---|
| UX-GAP-01 | No IDE skill in v0 for Approve, Build, Run, catalog and timeline; v1 (SFX-P10-08) must cover them through CLI/MCP. |
| UX-GAP-02 | No `sfx diff` command in the R0 CLI list; the semantic diff needs an operation in the PC-6 op map. |
| UX-GAP-03 | The R0 CLI list has no command for create, interview, approval request or run; these exist only as authenticated operations, which is consistent with PC-3 but must be named in the op map. |

### 11.3 Dependencies

| ID | Depends on | Needed for |
|---|---|---|
| UX-DEP-01 | P2-02 schemas: provenance and categorical confidence per field in SFX-CON-02/03 | §6.1, §6.2 |
| UX-DEP-02 | P2-02 schemas: secret-reference type in SFX-CON-06 and evidence-ref type in SFX-CON-09/10 | §6.4, §6.8 |
| UX-DEP-03 | P2-02 schemas: approval request with bound digest and superseded status in SFX-CON-13 | §6.5 |
| UX-DEP-04 | ADR-SFX-002: canonicalization and persistence mapping (§7.5) | §6.9, §6.10 |
| UX-DEP-05 | ADR-SFX-003: Studio stack, JWT bearer, CORS, token handling | §0, §6.10 |
| UX-DEP-06 | ADR-SFX-004: PC-1..6 op map and version constant | §7 |
| UX-DEP-07 | ADR-SFX-006: authority levels, engineering modes, admission separation | §2 DP-5/6/7, §6.6, SCR-15 |
| UX-DEP-08 | GATE-SEF-CONSUME and OD-SFX-10 | AGENTIC_ENGINEERING in §6.6 |
| UX-DEP-09 | OD-SFX-07, OD-SFX-08 | usability targets and rounds |
