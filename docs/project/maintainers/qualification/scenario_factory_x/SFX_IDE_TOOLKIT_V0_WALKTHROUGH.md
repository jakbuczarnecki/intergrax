# SFX IDE toolkit v0 - REF-1 dry-run walkthrough record

> **Date:** 2026-11-12 · **Ticket:** T-IGX-037-E · **Unit context:** SFX-P4-A01 (operator walkthrough still pending)
> **Executor:** orchestrator implementer subagent (simulated dry run by text-following). **This is NOT an operator walkthrough and NOT a certification.** No claim of approval, qualification or admission is made.
> **Status:** `DRY_RUN_COMPLETE_WITH_FINDINGS` · fix round 1 applied 2026-11-12 (section "Fix round 1"); the operator walkthrough is still pending

## 1. Purpose and method

Check that the SFX IDE toolkit v0 texts (rule `agent-env/rules/intergrax-scenario-factory.md`, skills `sfx-session`, `sfx-discover`, `sfx-specify`, `sfx-resolve`, `sfx-blueprint`, `sfx-integration-setup`, `sfx-test-plan`, `sfx-present`, and the six persona briefs in `agent-env/agents/sfx-*.md`) can be followed literally to produce schema-conformant drafts for the REF-1 problem statement (R0 §12), and list every mismatch between skill text, schemas and inventory.

Method: the executor read the texts, produced each step's drafts under `.tmp/sfx/ref1_invoice_automation/` (gitignored), answered the skill questions itself from R0 §12 (section 5 below), and validated each draft against its schema by reading. Deviations from "by reading only":

- Two short Node scripts (outside the repository, in the session scratchpad) generated the canonical-ready JSON (sorted keys) and ran a mechanical check of `required`, `additionalProperties`, enums, patterns and integer bounds as an aid. They are not a deliverable and replace nothing in the toolkit; real tool validation arrives with SFX-P5-07. A v0 user would not have them (WT-25).
- The `sfx-critic` review was run as a separate subagent (it did not write the drafts), as `sfx-present` requires.
- No `uv`, `python` or `pytest` ran; no git write command ran; nothing outside `.tmp/sfx/ref1_invoice_automation/` and this record was written in the repository (`git status` is clean because `.tmp/` is gitignored).

## 2. Draft files

Slug `ref1_invoice_automation`. Schema directory: `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/`. All `canonical_digest` values are 64 zeros (placeholder).

| draft (under `.tmp/sfx/ref1_invoice_automation/`) | schema `$id` | sha256 of the file | validation by reading | deviations |
|---|---|---|---|---|---|
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

Severity: MAJOR = a literal reader produces a wrong or misleading draft, or cannot proceed without inventing a rule; MINOR = ambiguity with an obvious workaround; INFO = observation. At the time of the dry run the skills and schemas were not edited; fix round 1 (section "Fix round 1") edited them, and the last column gives the outcome per finding. Counts: **MAJOR 6, MINOR 15, INFO 5 (26 total).**

| id | sev | affected | finding | suggested fix | Status (fix round 1) |
|---|---|---|---|---|---|
| WT-01 | MAJOR | `sfx-resolve` mapping table; `sfx-capability-resolver` brief; inventory | The inventory uses `REQUALIFICATION_REQUIRED` as a `reuse_class` value (3 rows: CAP-15, 19, 30). The skill and brief treat it as a qualifier and the mapping table has no row for it. I held CAP-30 (the natural store for invoice decision records) at `UNRESOLVED` to avoid upgrading it. | Add a table row (for example `REQUALIFICATION_REQUIRED` -> `UNRESOLVED`, target class in the rationale) or define the rule. | FIXED. `sfx-resolve` step 3: new mapping row `REQUALIFICATION_REQUIRED` -> `UNRESOLVED`; the row id and owner path are kept, the rationale starts `qualification_state=REQUALIFICATION_REQUIRED; target_class=<class from the row limitations>`, repeated in `resolution_notes.md`. Resolver brief aligned. (Inventory r1 now has four such rows: CAP-08, 15, 30, 31.) |
| WT-02 | MAJOR | `sfx-resolve` step 6; `capability_resolution_report` `inventory_ref` | The skill says to use `inventory_version` and `inventory_digest` "as stated in the inventory file". The file has only `schema_version` and `base_sha`. I composed `sfx-capability-inventory.v1:<base_sha>` and used the zero digest. | State the mapping (which fields form `inventory_version`) and when a real digest appears. | FIXED. `sfx-resolve` step 7: `inventory_version` = the top-level `inventory_version` string of the inventory JSON, verbatim (`sfx-capability-inventory.v1+r1`); `inventory_digest` = `sha256:` + 64 zeros until tooling computes it. Schema change: the report `inventory_version` pattern now allows `+`. |
| WT-03 | MAJOR | rule invariant 8; `sfx-resolve` step 3; inventory | The inventory has no owner-path field. `code_path` is prose with several paths, line numbers, class names, docs paths (CAP-05) or "No code" (CAP-26..29). `evidence_ref` is an `id` (no spaces, no `;`), so the path had to be extracted by hand (first path, line number dropped). The skill also says `evidence_ref` is null when unresolved, while `capability_id` is set for an unresolved row with a match (OUT-01), so the row is half filled. | Add `owner_paths[]` to the inventory or state the extraction rule; say what `evidence_ref` is for a matched but unresolved row. | FIXED. `sfx-resolve` step 3: exact owner-path rule (first word beginning `intergrax/`, `agents/`, `applications/`, `scripts/` or `testing_support/`, with `:line`; none for `No code`/`Verified absent`/`None` rows); a matched `UNRESOLVED` row keeps its owner path in `evidence_ref`. Rule invariant 8 points to it; report schema descriptions added. |
| WT-04 | MAJOR | `sfx-resolve` steps 1 and 3; G-1 | Every id (inputs, outputs, criteria, rules) must map to a platform capability. Scenario-specific deterministic logic (VAT check, 3-way match) and data inputs have no honest class. Binding rules to SFX-CAP-03 as `REUSE_AS_IS` made 13 of 24 resolved rows hosting rows and gave 8275 bp; `UNRESOLVED` would understate. The critic raised the same inflation (M-03). | Add a class or flag "hosted scenario logic" that does not count in G-1, or restrict requirements to capability-bearing ids; define the G-1 numerator and rounding. | FIXED. `sfx-resolve` step 2: scenario-specific logic or data is never reuse; it resolves as `NEW_REQUIRED` (hosting row as `capability_id`) or `UNRESOLVED`. Step 8: G-1 = floor(10000 x (AS_IS + WITH_CONFIGURATION) / requirements), `NEW_REQUIRED` printed on its own line. Resolver brief, architect brief and critic check aligned. REF-1 effect in Fix round 1. |
| WT-05 | MINOR | `sfx-discover` steps 5-6; `discovery_record` `turns` | The reason for each question must be stated, but `turns` has no field for it. I appended the reason to `question_text`. | Say where the reason is recorded. | FIXED. `sfx-discover` step 6: the reason is a final `Why: ...` sentence in `question_text`; analyst brief aligned. |
| WT-06 | MINOR | `sfx-discover` step 6 | `SKIPPED` is undefined: verbatim "I don't know" or null answer, and who is `answered_by`. | Define it. | FIXED. `sfx-discover` step 6: `SKIPPED` = user declined or did not know; `answer_text_untrusted` verbatim or null, `answered_by` the human principal if the user replied, else null. |
| WT-07 | MINOR | `sfx-discover` step 7 | `confidence_bp` for `ANSWERED` and `DOCUMENT` facts is undefined (I used 10000). | State the values. | FIXED. `sfx-discover` step 7: `confidence_bp` 10000 for `ANSWERED` and confirmed facts, 9000 for `DOCUMENT`, 1..9999 (default 5000) for unconfirmed `INFERRED`; confirmation changes the fact to `ANSWERED`, 10000. |
| WT-08 | MINOR | `sfx-discover` steps 3 and 10; `sfx-specify` validation | `completeness_bp` counts "required fields", defined as 8 specification groups in step 3 and as 24 schema fields in the specify validation text. Treatment of `partial` is undefined (I counted 0): 6/8 = 7500. | Define the denominator and `partial` weight. | FIXED. `sfx-discover` steps 3 and 10: denominator = the 8 gap groups, `partial` counts 0, floor(10000 x answered / 8). `sfx-specify` validation says the 24 is the schema field count and unrelated. |
| WT-09 | MINOR | rule draft conventions | No clock source for `created_at` when no runtime may run. I chose session-date times by hand. | Prescribe a source (for example the session date plus a counter) or allow `date -u`. | FIXED. Rule, "Clock": current UTC from the tool context or the read-only `date -u`, never invented, no going backwards across drafts. |
| WT-10 | MINOR | `sfx-specify` step 10 | "One entry per top-level field" is ambiguous (I covered the 11 content fields, not envelope fields). There is no way to record that the user confirmed an `INFERRED` field. | Name the fields; say how a confirmation is stored. | FIXED. `sfx-specify` step 10: provenance for the 11 named content fields; `INFERRED` = unconfirmed; a confirmed field becomes `ANSWERED` and is listed under "Confirmations" in `open_questions.md`; a user-changed field is `MANUAL_EDIT`. |
| WT-11 | MINOR | `sfx-specify` step 8; `sfx-resolve` step 1 | The schema stores volume per day; the user said per month, so the conversion is an inference with no field for the basis (20 working days). A null target (latency) still produces a requirement `nfr:latency` that can only be `UNRESOLVED`. | Allow a note field or a rule for derived values; say whether null targets are requirements. | FIXED. `sfx-specify` step 8: derived values allowed (20 working days default), marked `INFERRED`, arithmetic under "Derived values"; a null target creates no `nfr:` requirement (`sfx-resolve` step 1). |
| WT-12 | MINOR | schemas; CF-7 | CF-7 says each array is declared ordered or set with a sort key, but the descriptions name one only for `inferred_facts`, `resolutions`, `capability_bindings` and `tool_allowlist`. Not declared: `actors`, `inputs`, `outputs`, `business_rules`, `decision_points`, `integrations_required`, `acceptance_criteria`, `stakeholders`, `constraints`, `agents`, `integrations`, `test_cases`, `eval_suites`, `active_artifact_refs`. The skills improvise (`integrations` in spec order, `active_artifact_refs` by `schema_id`). I sorted `test_cases` by id. | Declare each array. | NOT FIXED (v0 workaround only). The rule now states how a v0 reader treats undeclared arrays ("Arrays": a named sort key means set, every other array is ordered and keeps its written order). The per-schema CF-7 declarations are a schema design change for the design owner and were not made. |
| WT-13 | MINOR | `sfx-resolve` step 4 and 7 | The mapping from inventory class to UCA gap class is undefined except for `missing_capability` and `incomplete`; the critic found my own convention applied unevenly (m-02). Rounding of the G-1 ratio is undefined (I rounded down). | Give a mapping table and the rounding. | FIXED. `sfx-resolve` step 4: table from inventory class and situation to UCA gap class (v0 convention); step 8: G-1 rounded down. |
| WT-14 | MINOR | `sfx-capability-resolver` brief vs `sfx-resolve` | The brief says open inventory questions are "listed inside the report"; the schema is closed and the skill puts them in `resolution_notes.md`. | Align the brief. | FIXED. Resolver brief step 6 and outputs: open gaps go to `resolution_notes.md` and `open_questions.md` (as `IQ` rows), never into the report; the rationale names the `IQ` id. |
| WT-15 | MINOR | `sfx-integration-setup` step 3; inventory | (a) "Take the scheme from the secret-store row": CAP-09 names none, so the fallback `secretref:` was used. (b) `adapter_ref` is "the adapter capability from the resolution report", but the inventory has no fixture or stub adapter row; R0 REF-1 assumes one. I pointed at the registry row SFX-CAP-10. | Add adapter rows or allow `adapter_ref` null plus a note. | FIXED. `sfx-integration-setup` step 3: `adapter_ref` rule (registry row plus an `IQ` when the inventory has no adapter row; `UNRESOLVED:<IQ id>` when the report row has no `capability_id`); `secret_ref` scheme fallback now records an `IQ`. No inventory field or row was added. |
| WT-16 | INFO | `integration_setup_plan` schema | The forbidden-word pattern is unanchored on keys only: it blocks benign keys such as `max_tokens` and does not look at values. The skill's credential scan by reading is the only value check. | Document, or add value checks in tooling. | UNCHANGED (INFO). The integration skill now states that a key such as `max_tokens` is rejected and budgets go to the notes. |
| WT-17 | MINOR | `business_problem_statement` schema; `actors` | `source_door` has no `IDE` (the skill's `CLI` workaround is disclosed but inaccurate). `actors.kind` has no external organisation (the supplier was left out). | Add `IDE`; consider an external-party kind. | FIXED. Schema change (additive): `IDE` added to `source_door`; `sfx-discover` uses `IDE`. External organisation: `sfx-specify` step 4 says `SYSTEM` actor with the party in `role` (no new actor kind). |
| WT-18 | MINOR | `sfx-session` step 6 | `resume_cursor` precedence is unclear when only non-blocking questions are pending and the next step is outside v0. I used the first pending id `Q-05`. | State the precedence. | FIXED. `sfx-session` step 6: `resume_cursor` precedence: first open id with `blocking_for: draft`; else next step with a v0 skill; else first open id with `blocking_for: build`; else null. |
| WT-19 | MINOR | `sfx-session` step 6 | "Save after every step and every answered question" means ~16 rewrites; in-place rewrite keeps no history, so `version` carries no meaning. I simulated one save per step (version 7). | Save per step; say what `version` and `created_at` mean for the cursor. | FIXED. `sfx-session` step 6: save once per completed step and once on stop; `version` counts saves, `created_at` is the latest save. |
| WT-20 | INFO | `sfx-specify`, `sfx-blueprint` | Later steps append questions to `open_questions.md` only (Q-14), so `discovery_record.open_questions` goes stale by design; blueprint plan refs are null by design. Both are documented but readers may miss them. | Mention in the present skill. | UNCHANGED (INFO). |
| WT-21 | MINOR | `sfx-test-plan` steps 2-3 | "At least one POSITIVE and one NEGATIVE per criterion" cannot be met by SECURITY-kind cases (AC-04, AC-05 have SECURITY but no NEGATIVE-kind case). `EVAL` is defined as LLM-judged, but REF-1 accuracy on a truth set is deterministic. | Say whether SECURITY counts as negative; redefine `EVAL`. | FIXED. `sfx-test-plan` step 2: a `SECURITY` case counts as negative coverage; step 3: `EVAL` = scoring over a labelled dataset (deterministic, or LLM-judged with rubric). |
| WT-22 | MAJOR | `sfx-session` step 4; `sfx-blueprint` step 1; `sfx-test-plan` | "Blocking" is yes or no on a Q id. Gaps that stop a later build (IQ-01 no PDF capability) have no Q id, so they are in neither `pending_question_ids` nor `resume_cursor`, and the flow continued to Present with 5 UNRESOLVED requirements. No rule says what the blueprint and test plan do for an UNRESOLVED requirement; I allowlisted a WRITE tool for an unresolved store and wrote five cases against it (critic M-01, M-02). | Add "blocking for" (draft, build) and give inventory gaps an id; say that tools and cases for UNRESOLVED needs are conditional or omitted. | FIXED, no new schema field. Rule "Blocking": `Q-`/`IQ-` ids with `blocking_for` draft, build or none; every `UNRESOLVED` requirement gets an `IQ` row (`sfx-resolve` step 5); all open ids go to `pending_question_ids` (closest existing field; schema description added). Blueprint: no binding, step, tool (above all `WRITE`) or agent for an unresolved requirement. Test plan: cases marked `CONDITIONAL on IQ-nn` and `runner: unresolved`. Present and critic check it. |
| WT-23 | MAJOR | `sfx-integration-setup`; `sfx-resolve` | The LLM provider is a dependency, but only `integrations_required` entries become integrations, so the model endpoint is in no plan, egress rule or fixture, while L2 is egress DENY_ALL and the extraction metric needs a model (critic M-07). | Treat the model provider as an integration (local or stub model kind) or say where it is declared. | FIXED, no schema change. `sfx-specify` step 7 adds `INT-LLM` to `integrations_required` when any rule is non-deterministic or an agent exists; it resolves like any integration (UNRESOLVED if the inventory has no model gateway row, which holds at r1); `sfx-integration-setup` plans it as `STUB_HTTP` or `FIXTURE_ADAPTER` with egress DENY_ALL, a `model_access` `secret_ref`, model failure modes and PROD-Q provider approval; blueprint notes and critic check it. |
| WT-24 | INFO | `sfx-session` step 3, persona section | The critic ran once, at the end, as a separate subagent from inside a subagent session and worked; it found 7 MAJOR content issues the drafting steps did not (section 4). The session skill says to run it after every drafting step; no rule says how to handle a MAJOR finding (loop, stop, or record). | Add a disposition rule. | UNCHANGED (INFO). |
| WT-25 | INFO | rule "Validation by reading" | Checking a 29-row report and 18 cases by reading is error-prone; I used an ad hoc script to be safe. | Tool validation (SFX-P5-07) is the fix; until then note the cost. | UNCHANGED (INFO). |
| WT-26 | INFO | REF-1 content | The critic's MAJOR and MINOR findings are about the REF-1 drafts, not the toolkit (examples: DP-02 "accept as matched" against AC-03; "payment-processing queue" not in discovery; dead-letter expectation with no capability, since inventory CAP-33 says no DLQ exists). They were NOT remediated: the dry run stops at Present. | None for the toolkit; the specification and discovery skills could ask for a consistency pass (decisions against criteria). | UNCHANGED (INFO). |

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

## Fix round 1

- **Date:** 2026-11-12. **Executor:** orchestrator subagent (implementer). **Scope:** ticket T-IGX-037-E; the texts in `agent-env/rules/intergrax-scenario-factory.md`, `agent-env/skills/sfx-*/SKILL.md`, `agent-env/agents/sfx-*.md`, four design schemas, and the gitignored REF-1 drafts.
- **Verification:** the fixes were verified **by reading** and by **one node JSON check** (all 17 schemas parse, every integer has I-JSON bounds, and the regenerated REF-1 drafts pass a subset validator for `required`, closed properties, enums, patterns, integer bounds plus cross checks: report sorted, `unresolved_count`, bindings equal to the resolved rows, integration ids equal to the specification, every pending id present in `open_questions.md`). It was **not** a new dry run, and it was **not** the operator walkthrough, which is still pending. No claim of approval, qualification or admission is made.
- **Counts:** MAJOR 6 of 6 FIXED. MINOR 15: 14 FIXED, 1 NOT FIXED (WT-12). INFO 5 UNCHANGED (WT-16, WT-20, WT-24, WT-25, WT-26).

### Schema changes (all keep `$id`, `additionalProperties: false`, secrets by reference and I-JSON integer bounds)

| schema | change | kind |
|---|---|---|
| `sfx.business_problem_statement.v1` | `source_door` enum gains `IDE`; description added | additive |
| `sfx.capability_resolution_report.v1` | `inventory_ref.inventory_version` is now an inline string with pattern `^[A-Za-z0-9][A-Za-z0-9._:+/-]*$` (the opaque id pattern plus `+`), because the inventory's own string `sfx-capability-inventory.v1+r1` contains `+`; descriptions added to `inventory_version`, `inventory_digest`, `capability_id`, `evidence_ref` | one character allowed on one field (a loosening of that field's character class; flagged for the design owner) |
| `sfx.sfx_session_state.v1` | descriptions of `pending_question_ids` and `resume_cursor` | documentation |
| `sfx.integration_setup_plan.v1` | I-JSON bounds added to the integer branch of `settings` values (found missing by the JSON check) | tightening |

### Re-check of the REF-1 drafts (by reading, plus the node check)

The drafts under `.tmp/sfx/ref1_invoice_automation/` were regenerated in place at version 1, as if run with the fixed texts (a simulation, not a user revision; the first-pass files are overwritten). Changed: problem statement (`source_door` `IDE`), specification (actor ACT-05, `INT-LLM`, provenance of `integrations_required` `INFERRED`), resolution report, blueprint, integration plan, test plan, session state, and the notes `open_questions.md`, `resolution_notes.md`, `blueprint_notes.md`, `integration_notes.md`, `test_plan_notes.md`, `presentation.md`. Unchanged: discovery record, `critic_findings.md` (first-pass review; not re-run).

Inventory note: the inventory file moved to r1 after the dry run (it now carries `inventory_version`, and CAP-08 and CAP-31 are `REQUALIFICATION_REQUIRED`), so part of the numbers below comes from the newer inventory and not only from the text fixes.

| quantity | first pass | after fix round 1 |
|---|---|---|
| requirements | 29 | 29 (`nfr:latency` dropped: null target; `INT-LLM` added) |
| `REUSE_AS_IS` / `REUSE_WITH_CONFIGURATION` / `NEW_REQUIRED` / `UNRESOLVED` | 10 / 14 / 0 / 5 | 2 / 11 / 9 / 7 |
| G-1 reuse ratio | 8275 bp | 4482 bp (= floor(10000 x 13 / 29)) |
| bindings in the blueprint | 24 | 22 |
| `WRITE` tools allowlisted | 2 (on unresolved stores) | 0 |
| test cases | 18 | 19 (`TC-N09` model failure), all marked `CONDITIONAL on IQ-nn` |
| `pending_question_ids` | 6 (`Q-`) | 13 (7 `Q-`, 6 `IQ-`); `resume_cursor` `IQ-01` |

Where the drop comes from: WT-04 alone moves 9 rows (BR-04..BR-11, DP-01) to `NEW_REQUIRED`: 15 of 29 = 5172 bp; WT-01 with the r1 inventory moves DP-02 and OUT-02 to `UNRESOLVED` (the exception-queue rule BR-10 is already counted in the WT-04 nine): 13 of 29 = 4482 bp; the `INT-LLM` row adds one `UNRESOLVED`, and the dropped `nfr:latency` removes one.

Does each MAJOR fix leave the example valid (by reading and the node check)?

| finding | result |
|---|---|
| WT-01 | Valid. DP-02, OUT-01, OUT-02 are `UNRESOLVED` with their row id and owner path, rationale prefix `qualification_state=...; target_class=REUSE_WITH_CONFIGURATION`. |
| WT-02 | Valid, but only after the schema change: the verbatim `sfx-capability-inventory.v1+r1` fails the old `$defs/id` pattern. The digest is 64 zeros. |
| WT-03 | Valid. Every `evidence_ref` was produced by the stated extraction rule (the node script applied the same rule to the inventory) and matches the id pattern; `BR-03` now cites `intergrax/runtime/execution/runtime.py:241`, the first word of CAP-01, with the NexusLoop path in the rationale. |
| WT-04 | Valid. The nine hosted rules are `NEW_REQUIRED` with `capability_id` SFX-CAP-03; the blueprint binds them with that resolution; the ratio no longer counts them. |
| WT-22 | Valid. No schema field was added. Every `UNRESOLVED` rationale names an `IQ` id that is a row of `open_questions.md` and an entry of `pending_question_ids`; no binding or tool exists for an unresolved requirement; all 19 cases carry the `CONDITIONAL` marker. |
| WT-23 | Valid. `INT-LLM` is in the specification, the report (`UNRESOLVED`, IQ-05), the plan (`STUB_HTTP`, `adapter_ref` `UNRESOLVED:IQ-05`, egress `DENY_ALL`, `model_access` `secret_ref`) and the blueprint notes. |

### Open after fix round 1

- WT-12 NOT FIXED (above). The v0 UCA gap-class table (WT-13) and the `INT-LLM` naming convention are toolkit conventions, not platform definitions; they need the design owner's confirmation.
- Other design documents that restate the report or problem statement contracts (outside this edit scope) may need the same three schema changes.
- The `.claude/` and `.cursor/` adapters were not regenerated (outside scope; the orchestrator does it).
- sha256 of the regenerated files (LF): problem statement `74ed967f26b2f17b3b66075dc9baa8a91f878e50883275909737b7fee4cdb2c9`, specification `20c8611cd61efd8667c06204f3254fef9ff926f80c87de97dda85af81007bcc3`, report `c1082aac7d516926c081b8fb691f516be4a3320d1c9cd4b41fb729b9c5da8399`, blueprint `b4c38425ab5fbf9969f8be8990bdf2edbb639199392f3f3dc031026918e98511`, integration plan `d38a789125818790653ebe2fbb961b328c8c90584b32a24ed78027dbcda3e436`, test plan `071090c2be2f0beb6ce1a29a691788a2f93924f341bf28006a1f944fdf384de9`, session state `17deed0e5bd63119c26efb6c8ad236696329bfaaa00eb450d64676a2fcd9908a`. The table in section 2 keeps the first-pass hashes.

**The operator walkthrough (acceptance of unit SFX-P4-A01) is still pending.** Fix round 1 changes no unit, gate or lifecycle status.
