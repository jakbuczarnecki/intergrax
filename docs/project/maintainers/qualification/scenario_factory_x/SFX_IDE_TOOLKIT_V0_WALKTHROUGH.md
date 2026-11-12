# SFX IDE toolkit v0 - REF-1 dry-run walkthrough record

> **Date:** 2026-11-12 · **Ticket:** T-IGX-037-E · **Unit context:** SFX-P4-A01 (operator walkthrough still pending)
> **Executor:** orchestrator implementer subagent (simulated dry run by text-following). **This is NOT an operator walkthrough and NOT a certification.** No claim of approval, qualification or admission is made.
> **Status:** `DRY_RUN_COMPLETE_WITH_FINDINGS`

## 1. Purpose and method

Check that the SFX IDE toolkit v0 texts (rule `agent-env/rules/intergrax-scenario-factory.md`, skills `sfx-session`, `sfx-discover`, `sfx-specify`, `sfx-resolve`, `sfx-blueprint`, `sfx-integration-setup`, `sfx-test-plan`, `sfx-present`, and the six persona briefs in `agent-env/agents/sfx-*.md`) can be followed literally to produce schema-conformant drafts for the REF-1 problem statement (R0 §12), and list every mismatch between skill text, schemas and inventory.

Method: the executor read the texts, produced each step's drafts under `.tmp/sfx/ref1_invoice_automation/` (gitignored), answered the skill questions itself from R0 §12 (section 5 below), and validated each draft against its schema by reading. Deviations from "by reading only":

- Two short Node scripts (outside the repository, in the session scratchpad) generated the canonical-ready JSON (sorted keys) and ran a mechanical check of `required`, `additionalProperties`, enums, patterns and integer bounds as an aid. They are not a deliverable and replace nothing in the toolkit; real tool validation arrives with SFX-P5-07. A v0 user would not have them (WT-25).
- The `sfx-critic` review was run as a separate subagent (it did not write the drafts), as `sfx-present` requires.
- No `uv`, `python` or `pytest` ran; no git write command ran; nothing outside `.tmp/sfx/ref1_invoice_automation/` and this record was written in the repository (`git status` is clean because `.tmp/` is gitignored).

## 2. Draft files

Slug `ref1_invoice_automation`. Schema directory: `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/`. All `canonical_digest` values are 64 zeros (placeholder).

| draft (under `.tmp/sfx/ref1_invoice_automation/`) | schema `$id` | sha256 of the file | validation by reading | deviations |
|---|---|---|---|---|
| `business_problem_statement.v1.json` | `sfx.business_problem_statement.v1` | `52ef67655b5694ddb120ab04751baa70e752acca3af7b2d73d9e066a6176039c` | conforms | `source_door` = `CLI` (no `IDE` value, per skill; WT-17) |
| `discovery_record.v1.json` | `sfx.discovery_record.v1` | `2beafb0dce3d999b70dd135388216dfca365399b6f6ff198839fdde0e53aa96c` | conforms | none; 10 turns, 6 facts, `completeness_bp` 7500 by my convention (WT-08) |
| `business_scenario_specification.v1.json` | `sfx.business_scenario_specification.v1` | `93c02770cac69a020bc5c3a79b8c5672ea298e5051ff90503222d12afcce10f2` | conforms (24 required fields present) | `latency_p95_target_ms` null (Q-05); volume converted from a month to a day (WT-11) |
| `capability_resolution_report.v1.json` | `sfx.capability_resolution_report.v1` | `74c6f976daf474bad2e314c1c73f8f6eac32957653744027d93aa8923a42dc23` | conforms; `unresolved_count` 5 = count of UNRESOLVED; sorted by `requirement_id` | `inventory_version` composed from `schema_version:base_sha`, digest zero (WT-02); `evidence_ref` null on a row with a `capability_id` (WT-03) |
| `scenario_blueprint.v1.json` | `sfx.scenario_blueprint.v1` | `96d402cc4e9164fe131e884a6c268f3f07e600c2985c75b8f2b24c70f08878b8` | conforms; 24 bindings match the report row for row; `environment_profile` `LAB`; `workspace_path` matches the pattern | plan refs null by rule |
| `integration_setup_plan.v1.json` | `sfx.integration_setup_plan.v1` | `2aea0c54ffb8913f8921f59fd423513a9ddeee272003f916c257eb9ce4c50f70` | conforms; `DENY_ALL` with empty allowlists; settings keys clean; `secret_ref` pattern matches | `adapter_ref` points at a registry row (WT-15) |
| `scenario_test_plan.v1.json` | `sfx.scenario_test_plan.v1` | `137332539c69e8c44f62f64cba6aae3ec1efbe4ba64f6a1695cc58a905fce2ca` | conforms; 18 cases; all criterion ids exist in the specification | none |
| `sfx_session_state.v1.json` | `sfx.sfx_session_state.v1` | `ea7148b1d02fb2f3766910a383de5e56a27366459912cfc28acdf9cb54689510` | conforms; `door` `IDE`, stage `PRESENT`, snapshot `DRAFT`, refs sorted by `schema_id` | version 7 simulates one save per step (WT-19) |

Companion notes (not contracts): `open_questions.md` (`d9e5429945408e9058cd1a09dd73d58d5084196390873c11967c6d3af92ad428`), `resolution_notes.md` (`11d714d7bda0b22f9b1ee7bea28a9b83790e0f75fcf51b30f5ff546092457518`), `blueprint_notes.md` (`56261b8c1dfde1a4dc8f7227466214d5a3e1aed9ad00342e061be2899aa66ff2`), `integration_notes.md` (`6fe1b5c882e75ad289914c7a4dccf7cd92c7c19e53da46bb215357eaab233dc8`), `test_plan_notes.md` (`2ae487966bbe8d245801e739970004ae3b96e3c438bd7fe24c7b95a702a3af26`), `presentation.md` (`dd2e786aaaead148c2dd4f261aa93a4fc62170c68e894e131280c7b35a3e7703`), `critic_findings.md` (`b164b6788b00fb8b3a246be803b4572c268fe8f2c438983c892f00456969f138`).

Headline numbers: 29 requirements, 10 `REUSE_AS_IS`, 14 `REUSE_WITH_CONFIGURATION`, 5 `UNRESOLVED`, reuse ratio 8275 bp; 18 test cases (3 POSITIVE, 8 NEGATIVE, 5 SECURITY, 1 PERFORMANCE, 1 EVAL). Independent critic: BLOCKER 0, MAJOR 7, MINOR 8, INFO 4 (REF-1 content, section 4).

Result of the literal run: every draft validates against its schema. The toolkit texts are followable end to end; the findings below are about places where following them literally needed an invented convention or produced a misleading result.

## 3. Findings WT-nn (toolkit texts, schemas, inventory)

Severity: MAJOR = a literal reader produces a wrong or misleading draft, or cannot proceed without inventing a rule; MINOR = ambiguity with an obvious workaround; INFO = observation. The skills and schemas were not edited. Counts: **MAJOR 6, MINOR 15, INFO 5 (26 total).**

| id | sev | affected | finding | suggested fix |
|---|---|---|---|---|
| WT-01 | MAJOR | `sfx-resolve` mapping table; `sfx-capability-resolver` brief; inventory | The inventory uses `REQUALIFICATION_REQUIRED` as a `reuse_class` value (3 rows: CAP-15, 19, 30). The skill and brief treat it as a qualifier and the mapping table has no row for it. I held CAP-30 (the natural store for invoice decision records) at `UNRESOLVED` to avoid upgrading it. | Add a table row (for example `REQUALIFICATION_REQUIRED` -> `UNRESOLVED`, target class in the rationale) or define the rule. |
| WT-02 | MAJOR | `sfx-resolve` step 6; `capability_resolution_report` `inventory_ref` | The skill says to use `inventory_version` and `inventory_digest` "as stated in the inventory file". The file has only `schema_version` and `base_sha`. I composed `sfx-capability-inventory.v1:<base_sha>` and used the zero digest. | State the mapping (which fields form `inventory_version`) and when a real digest appears. |
| WT-03 | MAJOR | rule invariant 8; `sfx-resolve` step 3; inventory | The inventory has no owner-path field. `code_path` is prose with several paths, line numbers, class names, docs paths (CAP-05) or "No code" (CAP-26..29). `evidence_ref` is an `id` (no spaces, no `;`), so the path had to be extracted by hand (first path, line number dropped). The skill also says `evidence_ref` is null when unresolved, while `capability_id` is set for an unresolved row with a match (OUT-01), so the row is half filled. | Add `owner_paths[]` to the inventory or state the extraction rule; say what `evidence_ref` is for a matched but unresolved row. |
| WT-04 | MAJOR | `sfx-resolve` steps 1 and 3; G-1 | Every id (inputs, outputs, criteria, rules) must map to a platform capability. Scenario-specific deterministic logic (VAT check, 3-way match) and data inputs have no honest class. Binding rules to SFX-CAP-03 as `REUSE_AS_IS` made 13 of 24 resolved rows hosting rows and gave 8275 bp; `UNRESOLVED` would understate. The critic raised the same inflation (M-03). | Add a class or flag "hosted scenario logic" that does not count in G-1, or restrict requirements to capability-bearing ids; define the G-1 numerator and rounding. |
| WT-05 | MINOR | `sfx-discover` steps 5-6; `discovery_record` `turns` | The reason for each question must be stated, but `turns` has no field for it. I appended the reason to `question_text`. | Say where the reason is recorded. |
| WT-06 | MINOR | `sfx-discover` step 6 | `SKIPPED` is undefined: verbatim "I don't know" or null answer, and who is `answered_by`. | Define it. |
| WT-07 | MINOR | `sfx-discover` step 7 | `confidence_bp` for `ANSWERED` and `DOCUMENT` facts is undefined (I used 10000). | State the values. |
| WT-08 | MINOR | `sfx-discover` steps 3 and 10; `sfx-specify` validation | `completeness_bp` counts "required fields", defined as 8 specification groups in step 3 and as 24 schema fields in the specify validation text. Treatment of `partial` is undefined (I counted 0): 6/8 = 7500. | Define the denominator and `partial` weight. |
| WT-09 | MINOR | rule draft conventions | No clock source for `created_at` when no runtime may run. I chose session-date times by hand. | Prescribe a source (for example the session date plus a counter) or allow `date -u`. |
| WT-10 | MINOR | `sfx-specify` step 10 | "One entry per top-level field" is ambiguous (I covered the 11 content fields, not envelope fields). There is no way to record that the user confirmed an `INFERRED` field. | Name the fields; say how a confirmation is stored. |
| WT-11 | MINOR | `sfx-specify` step 8; `sfx-resolve` step 1 | The schema stores volume per day; the user said per month, so the conversion is an inference with no field for the basis (20 working days). A null target (latency) still produces a requirement `nfr:latency` that can only be `UNRESOLVED`. | Allow a note field or a rule for derived values; say whether null targets are requirements. |
| WT-12 | MINOR | schemas; CF-7 | CF-7 says each array is declared ordered or set with a sort key, but the descriptions name one only for `inferred_facts`, `resolutions`, `capability_bindings` and `tool_allowlist`. Not declared: `actors`, `inputs`, `outputs`, `business_rules`, `decision_points`, `integrations_required`, `acceptance_criteria`, `stakeholders`, `constraints`, `agents`, `integrations`, `test_cases`, `eval_suites`, `active_artifact_refs`. The skills improvise (`integrations` in spec order, `active_artifact_refs` by `schema_id`). I sorted `test_cases` by id. | Declare each array. |
| WT-13 | MINOR | `sfx-resolve` step 4 and 7 | The mapping from inventory class to UCA gap class is undefined except for `missing_capability` and `incomplete`; the critic found my own convention applied unevenly (m-02). Rounding of the G-1 ratio is undefined (I rounded down). | Give a mapping table and the rounding. |
| WT-14 | MINOR | `sfx-capability-resolver` brief vs `sfx-resolve` | The brief says open inventory questions are "listed inside the report"; the schema is closed and the skill puts them in `resolution_notes.md`. | Align the brief. |
| WT-15 | MINOR | `sfx-integration-setup` step 3; inventory | (a) "Take the scheme from the secret-store row": CAP-09 names none, so the fallback `secretref:` was used. (b) `adapter_ref` is "the adapter capability from the resolution report", but the inventory has no fixture or stub adapter row; R0 REF-1 assumes one. I pointed at the registry row SFX-CAP-10. | Add adapter rows or allow `adapter_ref` null plus a note. |
| WT-16 | INFO | `integration_setup_plan` schema | The forbidden-word pattern is unanchored on keys only: it blocks benign keys such as `max_tokens` and does not look at values. The skill's credential scan by reading is the only value check. | Document, or add value checks in tooling. |
| WT-17 | MINOR | `business_problem_statement` schema; `actors` | `source_door` has no `IDE` (the skill's `CLI` workaround is disclosed but inaccurate). `actors.kind` has no external organisation (the supplier was left out). | Add `IDE`; consider an external-party kind. |
| WT-18 | MINOR | `sfx-session` step 6 | `resume_cursor` precedence is unclear when only non-blocking questions are pending and the next step is outside v0. I used the first pending id `Q-05`. | State the precedence. |
| WT-19 | MINOR | `sfx-session` step 6 | "Save after every step and every answered question" means ~16 rewrites; in-place rewrite keeps no history, so `version` carries no meaning. I simulated one save per step (version 7). | Save per step; say what `version` and `created_at` mean for the cursor. |
| WT-20 | INFO | `sfx-specify`, `sfx-blueprint` | Later steps append questions to `open_questions.md` only (Q-14), so `discovery_record.open_questions` goes stale by design; blueprint plan refs are null by design. Both are documented but readers may miss them. | Mention in the present skill. |
| WT-21 | MINOR | `sfx-test-plan` steps 2-3 | "At least one POSITIVE and one NEGATIVE per criterion" cannot be met by SECURITY-kind cases (AC-04, AC-05 have SECURITY but no NEGATIVE-kind case). `EVAL` is defined as LLM-judged, but REF-1 accuracy on a truth set is deterministic. | Say whether SECURITY counts as negative; redefine `EVAL`. |
| WT-22 | MAJOR | `sfx-session` step 4; `sfx-blueprint` step 1; `sfx-test-plan` | "Blocking" is yes or no on a Q id. Gaps that stop a later build (IQ-01 no PDF capability) have no Q id, so they are in neither `pending_question_ids` nor `resume_cursor`, and the flow continued to Present with 5 UNRESOLVED requirements. No rule says what the blueprint and test plan do for an UNRESOLVED requirement; I allowlisted a WRITE tool for an unresolved store and wrote five cases against it (critic M-01, M-02). | Add "blocking for" (draft, build) and give inventory gaps an id; say that tools and cases for UNRESOLVED needs are conditional or omitted. |
| WT-23 | MAJOR | `sfx-integration-setup`; `sfx-resolve` | The LLM provider is a dependency, but only `integrations_required` entries become integrations, so the model endpoint is in no plan, egress rule or fixture, while L2 is egress DENY_ALL and the extraction metric needs a model (critic M-07). | Treat the model provider as an integration (local or stub model kind) or say where it is declared. |
| WT-24 | INFO | `sfx-session` step 3, persona section | The critic ran once, at the end, as a separate subagent from inside a subagent session and worked; it found 7 MAJOR content issues the drafting steps did not (section 4). The session skill says to run it after every drafting step; no rule says how to handle a MAJOR finding (loop, stop, or record). | Add a disposition rule. |
| WT-25 | INFO | rule "Validation by reading" | Checking a 29-row report and 18 cases by reading is error-prone; I used an ad hoc script to be safe. | Tool validation (SFX-P5-07) is the fix; until then note the cost. |
| WT-26 | INFO | REF-1 content | The critic's MAJOR and MINOR findings are about the REF-1 drafts, not the toolkit (examples: DP-02 "accept as matched" against AC-03; "payment-processing queue" not in discovery; dead-letter expectation with no capability, since inventory CAP-33 says no DLQ exists). They were NOT remediated: the dry run stops at Present. | None for the toolkit; the specification and discovery skills could ask for a consistency pass (decisions against criteria). |

## 4. Critic result on the REF-1 drafts (separate subagent, read-only)

BLOCKER 0, MAJOR 7, MINOR 8, INFO 4; full text in `.tmp/sfx/ref1_invoice_automation/critic_findings.md`. The critic found no schema, authority, secret, claim or citation violation. MAJOR: M-01 unresolved decision-record store with a WRITE tool; M-02 unresolved PDF and VAT-table needs assumed downstream; M-03 adapter rows and reuse ratio inflated; M-04 enforcement of AC-04 and AC-05 unbound; M-05 DP-02 vs AC-03; M-06 payment-processing queue untraced; M-07 LLM provider absent from the integration plan. This is not an approval, and "no BLOCKER" is not one either.

## 5. Simulated answers (all by the executor, from R0 §12 REF-1; none was given by a person)

| question | simulated answer | basis |
|---|---|---|
| Slug | `ref1_invoice_automation` | derived from the REF-1 name |
| Q-01 fields | Typed invoice schema: number, supplier, date, currency, lines, net, VAT, gross, PO reference | R0 "Typed invoice schema" plus my list |
| Q-02 sources and match | PO and goods receipt from the purchasing system, read-only; 3-way match; fixture adapter in the sandbox | R0 "3-way match", "Fixture mailbox/folder adapter" |
| Q-03 exceptions | Exception queue for an AP clerk; nothing auto-rejected or auto-approved; 100% routed | R0 acceptance; "nothing auto-" is my addition |
| Q-04 volume | About 60 per working day (1,200 a month, 20 days); confirmed | inferred |
| Q-05 latency | "I don't know" (SKIPPED) | chosen to exercise the null path |
| Q-06 duplicates | Duplicate goes to the exception queue | R0 dataset lists duplicates |
| Q-07 VAT | Rate must match the expected rate for supplier country and goods; no table yet | R0 "wrong VAT"; table gap invented |
| Q-08 access | Mailbox read; purchasing read-only; no writes | R0 "No live vendors" plus my choice |
| Q-09 classification | CONFIDENTIAL; synthetic data in the sandbox | my choice |
| Q-10 play-back | No corrections | chosen |
| Confirmations of INFERRED facts (volume, working days) | Confirmed | chosen |

No credential was offered in the simulation, so the refuse-a-secret path (`sfx-discover` step 8) was not exercised.

## 6. Checks requested in the task

- **Steps without a v0 skill (Approve, Build, Run, Catalog, Timeline):** confirmed. `sfx-session` step 3 marks APPROVE, BUILD and RUN "none in v0"; step 5 has a table of the five steps with the reason, the planned door (approval request API or MCP, SFX-P10-01/02; CLI `sfx handoff` and MCP handoff, SFX-P8-01..03; sandbox run API or MCP, SFX-P9-01; MCP catalog read, SFX-P11-03; MCP lifecycle record read, SFX-P5-05, SFX-P6-03) and an explicit STOP, with a ban on emulating them or writing their contracts. Rule invariant 9 and `sfx-present` say the same. Nit: the planned operations are named by door, not by final operation name (the skill says names are not final).
- **"Must never" lists and the rule:** the drafts contain no secret value (only `secret_ref` strings; a scan for credential-shaped strings found none); nothing was written outside `.tmp/sfx/ref1_invoice_automation/` apart from this record (`git status` clean); no draft claims approval, certification, qualification, a passed test or admission, and `lifecycle_state_snapshot` is `DRAFT`. The word "approval" appears only as scenario content (`requires_human_approval`, DP-02, "auto-approved" in a rule); the critic found no claim. `Run`, `Approve` and the other contracts (run report, approval ref, qualification record, dossier, catalog entry, value report) were not written. Exception to literal compliance: the Node helper scripts (section 1).

## 7. Not exercised

- Real invocation of the skills in Claude Code or Cursor (slash invocation, skill discovery, skill loading, `.claude/` and `.cursor/` adapters): not exercised; the texts were followed by hand.
- Cursor duplicate skill discovery (F-02, each skill listed twice): not exercised.
- MCP live mode and any MCP tool: not exercised.
- Improve (new blueprint version, `versions/`, `blueprint_diff.md`): not exercised.
- Resume from a saved session state in a new session: not exercised (only the file was written).
- Persona subagents other than `sfx-critic` were not spawned (followed in this session).
- A person answering the questions: all answers simulated.

**The operator walkthrough (acceptance of unit SFX-P4-A01) is still pending.** Nothing here substitutes for it, and this record does not change any unit, gate or lifecycle status.
