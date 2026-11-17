# ECAP revision — W2 DRAFT (T-AF2-ECAP-REVISION, ticket T-IGX-050-I)

> **DRAFT proposal. It is not applied to ECAP, not accepted, not independently audited, and it promotes, closes and certifies nothing.** Every count below is a measurement taken at the stated SHA, not a status verdict. FRZ statuses stay as the checklist records them; the certifier role promotes rows, never this document. The final version is the W5 task T-IGX-052-D, and the applying edit of the protocol is a later step (section 1).

| Item | Value |
| --- | --- |
| Plan node | `T-AF2-ECAP-REVISION`, W2 draft (merged plan MEP-AF2 v2.1, Appendix Z). Hard edge: `T-B7-LOCK-FREEZE-SHA` waits for the final version |
| Facts read at | `origin/development` `a7f02361540752426f0015b628f8a40b6cbdbb2a` (`a7f023615`) |
| Protocol revised | `docs/project/maintainers/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md` (document ID `ECAP-1`, history: one row, version 1.0 of 2026-10-18), called "ECAP" below |
| Not this node | `T-B8-ECAP-16` (W8): the post-closure §16 handoff record `ARCH_FREEZE_ECAP16_HANDOFF.md`, written after `T-B8-ROADMAP`. This draft is about the text of the protocol, not about that record |

## 1. Which case applies (task step 1)

**Canonical text.** ECAP lives in its own file (path above). At `a7f023615`: §2.C Full Platform Current-HEAD Audit is at lines 160-170, §13 findings classification at 1017-1033, §15 Architecture Freeze Admission at 1088-1103 (12 bullets, lines 1092-1103), §16 Maintenance and Continuous Enforcement at 1107-1123, §17.1 mechanical ID coverage at 1158-1166, §18 validation record at 1169-1190.

**It is not inside an Operator-only file.** The roadmap (`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`) and the FRZ checklist are separate files; ECAP only lists them as companion authorities. So the task's rule "if ECAP is inside an Operator-only file, write a proposal" is not triggered by the letter of the file layout. This draft is a **proposal document anyway**, for three reasons:

1. **The content does not exist yet.** The contract names behind the PT-WA ports, the typed provider error contract, the public/internal split and the counts at M3 come from W3 and W5 work. The binding scope of the RW train is still being planned (T-AF2-RW-PLAN). A W2 edit of ECAP would freeze guesses into an authority document.
2. **ECAP is an input of the EBH-7 F10 harness.** `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:24,1006` reads it and parses its `#### FRZ-<FAMILY> — ` headers and the `Primary closing stage(s)` rows (lines 39-40, 704-716). ECAP is also listed in `docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`. An edit of those parts before the single EBH-7 pin (DP-61) would change an input of the generated F10 matrix (see gap G-08).
3. **The wording must match two Operator-only edits.** PT-D09 (roadmap) and PT-D10 (checklist, including the A/B split) change the statements ECAP defers to. The applying ECAP edit should follow them.

**Applying path (proposed).** The W5 final (T-IGX-052-D) is reviewed with the roadmap and checklist hunks (PT-AUD-DOC), and the ECAP edit is applied after PT-D09 and PT-D10, before the lock. Who may apply it, and whether it counts as a statement-changing hunk for PT-AUD-DOC, is for the Operator (gap G-09).

## 2. What the §16 trigger list says about this program

ECAP §16 (lines 1109-1117) lists seven conditions for updating the protocol. Applied to the program as the plan describes it:

| §16 trigger | Applies? | Basis | ECAP places affected |
| --- | --- | --- | --- |
| New platform domain or responsibility | No new domain section | PDC-X decided: no new ECAP domain sections (ECAP header, PDC-P3-D04). The program adds no domain | none |
| **New public contract or extension mechanism** | **Yes** | Four manifests, planned Tier-0 ports (PT-WA), typed provider error contract, trust classes, projection field maps (section 3) | §2.C, §4.6, §5, §15 items 3, 11 |
| New infrastructure/provider class | None found | The plan keeps C10/C15 (no new vendor integrations) and records E2B/Modal/Daytona as `NOT_CLAIMED`; no new provider class appears in the sources read | §15 item 11 wording only |
| New enterprise invariant or FRZ criterion | Not as new IDs | The inventory is 225 / 26 at `a7f023615` (section 5); the total after the A/B split depends on G-01; how the A/B split is written into the checklist is undecided (G-01) | §3.0, §17.1, header, §18 |
| New architecture-reopen trigger | Depends on W5 | T-B7-PFP-ACTIVATE defines requalification triggers; ECAP §2.D must then agree with it (G-07) | §2.D, §16 |
| **New regression class** | **Yes** | Known-red classes FIX / ENV / OUT / UNKNOWN and the expected-red set (section 4) | §2.A A8, §2.C C4, §2.D D4, §12, §13, §14 |
| **Canonical evidence requirement changes** | **Yes** | The PASS rule is exact SHA = M3 with product tree equal to FCS, evidence executed at M3, an independent audit row; ECAP §12 and §2.C do not state the M3 identity condition | §2.C, §12, §15 |

§16 also requires, on each revision, a re-check of the companion documents at current HEAD and a re-run of the FRZ inventory count (line 1119). Both were done for this draft at `a7f023615`: all eight companion paths of §18 exist, and the recount is in section 5.

## 3. New and changed contract surfaces, and what §16 / ECAP must say (task item 1)

Sources: the draft manifests (`manifests/`, generated from `332c5217c390202a911ad5592e88a8f366845587`), plan sections G and H, and the Operator decisions quoted in the plan. "Planned" means the plan states it and the tree does not have it yet.

| # | Surface | State | What ECAP must say (proposal) |
| --- | --- | --- | --- |
| C1 | **Contract manifest** (FRZ-FRZ-05). Roots `intergrax/contracts`, `intergrax/applications/contracts`; 853 modules (839 `DRAFT_PUBLIC_CANDIDATE`, 2 `DRAFT_INTERNAL`, 12 `PLANNED_MOVE (PT-WA)`); public/internal is a path rule only | Draft; final generated from G2 in W5 (T-IGX-052-B) | §2.C C1/C2 and §5 name the frozen manifests as the closed-world input for contract-surface checks. A "public contract" in §16 means a module the final manifest lists as public, not a path guess. A drift between manifest and tree is a finding, not a note |
| C2 | **Claim exclusions** in the contract manifest: 10 entries (`NOT_CLAIMED`: OD-AF-18 for E2B/Modal/Daytona, EX-D4-02, EX-D5-01; `OUTSIDE_FROZEN_SCOPE`: EX-G-02, EX-G-07, EX-D2-04, EX-D4-03, EX-D5-04; `TRANSFER_TO_LIVE_PROD_Q`: EX-G-04; `BOUNDED_CLAIM`: EX-G-01) | Draft; marks quoted from the residual document and OD-AF-18 | §15 items 3 and 11 ("frozen scope", "claimed scope") must read the scope from the manifest. Add a label rule beside §14's: a `NOT_CLAIMED` or `OUTSIDE_FROZEN_SCOPE` mark is never a PASS and never hides a finding inside the claimed scope |
| C3 | **PT-WA inversion ports** in `intergrax/contracts/` (DP-13): neutral Tier-0 contracts and ports, Tier-3 builds the implementations, Tier-1 applies them through injected ports; no allowlist for the 12 runtime modules; est. about 20 modules | **Planned** (W3, first train ticket). Port names are not fixed in the sources read; the manifest lists 6 candidate symbols (`ApplicationEnvironmentProfile`, `ExecutionMode`, `ApplicationProfile`, `ApplicationGraphSpec`, `GraphEdgeKind`, `ScenarioRuntimeMode`), one of them (`ExecutionMode`) already defined at Tier-0 | §5 gets one sentence: a Tier-0/1 dependency on application code is closed by injection through a Tier-0 port, not by an allowlist; function-level lazy imports count. The port list is generated from the G2 manifest, not typed into ECAP (G-02) |
| C4 | **Layer rules and gates** (FRZ-FRZ-06): R1 and R2 are the AGENTS.md boundaries, R4 the registry-bypass rule; DP-14 rewrites the registry-bypass gate (agents may import `integrations.contracts` leaves); DP-22: AST gates are the tier-boundary mechanism. Draft scan: 92 forbidden edges in 37 files (R1 79, R2 0, R4 13) | Draft counts at the manifest baseline; gate rewrite is W3 | §5 and the FRZ-BND family text refer to the AST gate set and the layer manifest; the "Regression gates (representative)" cell of FRZ-BND must still be checked for file existence at the audited SHA, as ECAP already requires |
| C5 | **Typed provider error contract v1**, served-model attribution, refusal-normalization matrix (T-AF2-RW-RUNTIME, `intergrax/llm_adapters/`) | **Planned** (W3) | FRZ-CTR/FRZ-REL procedures cite it once it exists; §8 (resilience) names stream-open failure and timeout as audited failure paths only after the code exists |
| C6 | **Trust classes** for model messages (T-AF2-RW-TRUST: untrusted content never `role=system`) | **Planned** (W3) | §9 (security) names the trust-class boundary as an adversarial-check target; no wording before the code exists |
| C7 | **Canonical gate/scenario contracts**: `get_steps(context)`, canonical event payload signature and schema (EX-L-13), promotion gate bound to stage plus ticket (T-AF2-RW-GATES) | **Planned** (W3) | §10 and the FRZ-OBS / FRZ-TRC procedures cite the reconciled schema after W3 |
| C8 | **Frozen-surface typing**: ratchet gate against new `Any` in a frozen contract (T-AF2-RW-TYP; DP-36, OD-AF-11) | **Planned** (W3) | FRZ-TYP "Regression gates" cell names the ratchet gate once it exists |
| C9 | **Projection field maps** (DP-17): mirrored profile models are deliberate projections with an explicit field map that fails on unmapped fields | Decision recorded; code planned | FRZ-CTR procedure: a mirrored model without a field map is a finding |
| C10 | **Ownership and composition manifests** (FRZ-FRZ-07, -08): 54 CONFIG-X rows, 16 VBN platform records, 5 fixed-substrate decisions; 7 strict production roots; 11 `NOT_COMPOSED` entries, each non-composition gate `PENDING` | Draft; gates not built | §5 uses the manifests for "exactly one owner" and "sanctioned composition owner" checks. A `NOT_COMPOSED` entry without its gate is an open item, not evidence |

Every "Planned" row is a claim about the future. Nothing in this table says a surface exists or passes.

## 4. Regression classes (task item 2)

**What the register defines.** `KNOWN_RED_REGISTER.md` (T-IGX-050-J, itself a draft) lists 27 rows: FIX 15, ENV 5, OUT 0, UNKNOWN 7. ENV rows are provisional until a proving run: 3 rows carry `env_proof` PENDING and 2 QUOTED. Per the register only 6 rows carry full pytest node ids; 13 carry a count or a class only. Its proposed M3 rule: the red set at M3 equals the proven ENV and OUT rows exactly; FIX and UNKNOWN rows must be gone. No skip or xfail may hide a red.

**What ECAP says today.** ECAP has no "known-red" or "expected-red" wording. It speaks of regression gates (§2.A A8: "only existing tests confirmed at SHA"), "enterprise regression / drift vs last accepted baseline SHA" (§2.C C4), regression selection (§2.D D4), the L10 level (§12), and the finding classes of §13. The text that decides whether a red test blocks a verdict is therefore missing.

| Register class | Meaning (register) | Nearest ECAP §13 class | Gap |
| --- | --- | --- | --- |
| FIX | defect or stale artefact needing a change | `PROVEN_ARCHITECTURAL_DEFECT` / `IN_SCOPE_BLOCKER` when an invariant is violated on a production path | Many FIX rows are test-side staleness or import errors (for example KR-03, KR-16), which is not an invariant violation. §13 has no class for a stale test artefact (G-04) |
| ENV | red only because of the environment | `ENVIRONMENT_TEST_ISSUE` ("qualification infra not representative") | Fits only after the proving run. §13 does not say that an unproven ENV label is a red |
| OUT | outside the frozen scope by a cited Operator decision | `N/A_WITH_EVIDENCE`, which §13 defines for a *criterion* | No row is OUT today. An excluded *test* needs the same cited decision and a manifest entry (C2) |
| UNKNOWN | the evidence read does not decide it | `MISSING_EXECUTION_EVIDENCE` | Fits. The 7 UNKNOWN rows must be resolved before M3 |
| hidden reds | collection errors abort a suite, so the tests behind them are unmeasured | `MISSING_EXECUTION_EVIDENCE` | ECAP has no sentence that a collection error is not a pass (register open point 1) |

**Proposed wording (to be refined in the W5 final).**

1. **§12 / §14.** Add the term *known-red register* and *expected-red set*: the expected-red set is the set of reds registered as proven ENV or OUT with a cited basis; any red outside it is a finding. §14 section J gets a line: unregistered reds, UNKNOWN rows, unproven ENV rows.
2. **§2.C C4.** The drift check against the baseline includes the red set: at M3 the observed red set must equal the registered expected-red set exactly (register proposed rule; plan section I, known-red policy: "the expected-red set equals the registered ENV/OUT set exactly").
3. **§2.A A8 and §2.D D4.** A regression selection that includes a registered red states it as such. A skip or xfail that hides a red is a finding (register rule; the plan: "No skip or xfail may hide a red").
4. **§13.** Decide the class for a FIX row that is a test-side defect and for a collection error (G-04).
5. **Two sets must not be mixed.** The train-only expected-red set (from the W0 probe PR, may only shrink) and the M3 red set are different objects; ECAP names only the M3 one, because the train set is a program mechanism.

## 5. FRZ recount (task item 3)

Counts below were computed with a stdlib script over the checklist and ECAP text at `a7f023615`; the commands are in section 8. They are counts of the checklist's own status cells, not verdicts.

**Inventory.** 225 rows `FRZ-<FAMILY>-<NN>`, all unique, in 26 families. ECAP §3 lists exactly the same 225 IDs in 26 family subsections; the declared per-family counts equal the listed ones; no ID is missing on either side. §3.0, §17.1 and the header (225 criteria in 26 families) therefore agree with the checklist today. ECAP contains no per-row status column (checked: no `| OPEN` or `| PASS ` table cell in ECAP).

**Status cells today.**

| Status cell | Rows |
| --- | ---: |
| `OPEN` | 109 |
| `PASS` | 108 |
| `PASS — SCOPED` (EVOL-01, -06, -08, -09, -10; FRZ-11) | 6 |
| `PASS (EVOL-X SCOPED)` (EVOL-07) | 1 |
| `PASS — COMPENSATING (SCOPED)` (AUD-10) | 1 |
| **Total** | **225** |

By family (OPEN / PASS / off-vocabulary PASS label): OPEN only: BND 6, CTR 6, DEBT 6, DOC 6, OBS 7, OWN 5, PLG 8, PRD 12, REG 10, REL 11, RPL 4, SEC 10, TYP 6 (13 families, 97 rows). PASS only: CFG 8, CMP 8, CTL 12, EXE 7, GOV 10, HRN 8, REC 10, STA 8, TEN 12, TRC 12 (10 families, 95 rows). Mixed: AUD 9 PASS + 1 label; EVOL 4 PASS + 6 labels; FRZ 12 OPEN + 1 label (33 rows; 97 + 95 + 33 = 225, and OPEN 97 + 12 = 109). The checklist's allowed statuses are OPEN, PASS, BLOCKED and `N/A — WITH EVIDENCE`; the 8 labels are outside that vocabulary (OD-AF-08 normalizes them only after the independent W7 row verdict; AUD-10 only after PT-AUD-ENFA is live).

**Target at M3 (planned, not claimed).** Plan section N row N4: architecture-scoped OPEN = 0 and BLOCKED = 0, all TEN rows PASS, only Path B rows OPEN, counters 225 / 26. The PASS rule (plan section D): exact SHA = M3, evidence executed at M3, an independent audit row. What that leaves open:

| Group | Rows | Expected at M3 per the plan | Note |
| --- | --- | --- | --- |
| Off-vocabulary labels | 8 | become PASS with a linked limitation after the W7 verdict (AUD-10 after PT-AUD-ENFA) | OD-AF-08 |
| Path B rows (checklist line 66) | FRZ-PRD-01, FRZ-PRD-02 whole; live parts of FRZ-PRD-03, FRZ-PRD-08, FRZ-SEC-04 | PRD-01, PRD-02 stay OPEN under PROD-Q; PRD-03, -08, SEC-04: architecture part PASS (Dimension A), live part OPEN (Dimension B) | DP-34, DP-35; split matrix in plan section D |
| SEC-05, SEC-06 (OD-AF-18) | 2 | architecture gates PASS; the E2B/Modal/Daytona part is `NOT_CLAIMED`, `N/A — WITH EVIDENCE` for that part. **Not Path B** | needs an evidence file and its own row verdict |
| A/B chain (EQOA-A7-019) | 14 rows: PRD-04, -11, -12, REG-08, REL-03, -05, -08, -09, -10, -11, SEC-01, -05, -06, -09 | architecture sub-part PASS; live sub-part may stay OPEN under PROD-Q | split is a checklist amendment (PT-D10); SEC-05, -06 are in this chain and handled by the previous row; source: `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md:641` |
| PLG-02 | 1 | planned `N/A — WITH EVIDENCE` under SCOPE-03 (EX-G-03) | the manifests list EX-G-03 without a derivable mark |
| All other OPEN rows | rest of the 109 | PASS (or `N/A — WITH EVIDENCE`) after the closing nodes of plan section D | no count is predicted here |

**Expected row-level counts cannot be a single number yet.** They depend on how the split is written into the checklist (G-01):

- **Split inside the row** (the row cell stays one status): rows with an open live part keep the cell `OPEN`. Lower bound 2 (PRD-01, PRD-02), named upper bound 17 (PRD-01, -02, -03, -08, SEC-04 plus the 12 chain rows that have a live sub-part and are not SEC-05/06). Total stays 225 / 26. The checklist has no status for "architecture part PASS, live part OPEN" (allowed statuses: OPEN, PASS, BLOCKED, N/A), so a representation rule is needed.
- **Split into new sub-row IDs:** the total exceeds 225, and ECAP §3.0, §17.1, the header and the §18 count command must change together with the checklist.

ECAP must keep storing **ID counts only** (225 / 26, or the new totals) and never PASS/OPEN counts: §16 forbids a second status ledger (line 1123) and ECAP-§1.4 puts the checklist above it. This draft's status counts are an audit aid for the reviewer and are not text to copy into ECAP.

## 6. §2.C, §15 and the plan's closure matrix

- **§15 has 12 bullets (lines 1092-1103).** The plan's §15 → N map (plan section N) also has 12 items, in the same order, and ties each to closure rows (N1-N7 and N11) or to closure text. No item is added or dropped by the plan; no §15 wording change is proposed except where C2 asks that "frozen scope" and "claimed scope" be read from the manifest.
- **§2.C** has steps C1-C5 and the trigger `ARCH-FREEZE admission`. Plan N6 requires §2.C PASS at M3 with a §14 report. The proposed additions are in section 3 (C1, C2, C10), section 4 (items 2-3) and the M3 identity condition: the audit SHA is the lock SHA M3, and the product tree equals G2/FCS (plan sections D and N, row N2).
- **Independence.** ECAP-P4 and the plan both forbid self-acceptance. The §2.C auditor must differ from the PT-Q02 auditor (plan section L). This draft makes no acceptance claim.

## 7. Gaps for the W5 final (task item 4, T-IGX-052-D)

| ID | Gap | Needed from |
| --- | --- | --- |
| G-01 | The A/B split representation (inside the row or new sub-row IDs) is not specified; it fixes the totals in ECAP §3.0, §17.1, the header and §18, and the expected counts at M3 | PT-D10 (Operator-only checklist); decision on form |
| G-02 | Final names of the PT-WA ports and of the typed error, trust-class and gate contracts: the binding scope is T-AF2-RW-PLAN; the manifests list candidates only | T-AF2-RW-PLAN, then generation from G2 |
| G-03 | The reviewed public/internal contract split (manifest rule says W5) decides what "public contract" means in §16 | T-IGX-052-B |
| G-04 | No §13 class for a FIX-type test-side red or for a collection error; whether to extend §13 or reuse an existing class | W5 ECAP author, Operator if §13 is extended |
| G-05 | §2.C C4 says "last accepted baseline SHA": which baseline applies at M3 (the pre-freeze ENT baseline, AUDIT_BASELINE_2, or both) is not fixed in the sources read | T-AF2-ENT-BASELINE-PREFREEZE |
| G-06 | The known-red register is a draft: 7 UNKNOWN rows, 5 unproven ENV rows, 13 rows without node ids. The M3 rule needs enumerated, proven rows | T-AF2-RED-CLASSES successor, PT-Q13 |
| G-07 | New architecture-reopen triggers depend on T-B7-PFP-ACTIVATE (W5); ECAP §2.D and §16 must agree with its rebaselined policy documents | T-B7-PFP-ACTIVATE |
| G-08 | **Ordering.** The F10 harness parses ECAP's family headers and `Primary closing stage(s)` rows; PT-D11 re-renders the F10 matrix after PT-D09 and PT-D10, and its dependencies in `WP12_DAG.md` do not list the ECAP revision. An ECAP edit of those parts after PT-D11 would stale the matrix | Orchestrator (DAG), PT-D11 owner |
| G-09 | Who applies the ECAP edit, and whether it passes through PT-AUD-DOC; the history table needs a new row and a version number the plan does not name | Operator |
| G-10 | The ECAP header block still carries inline stage notes (MCP-SERVER-X, PDC-X); whether the revision adds an ARCH-FREEZE note there or keeps the header to inventory and companions only | W5 ECAP author |

## 8. How the counts were produced

Run from a checkout of `a7f023615`; each prints numbers only and writes nothing:

```bash
python -I -c "import re,pathlib,collections as C;t=pathlib.Path('docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md').read_text(encoding='utf-8').splitlines();r=[[c.strip() for c in l.strip().strip('|').split('|')] for l in t if re.match(r'^\|\s*FRZ-[A-Z]+-\d+\s*\|',l)];print(len(r),len({x[0] for x in r}),len({x[0].split('-')[1] for x in r}));print(C.Counter(x[4] for x in r))"
python -I -c "import re,pathlib;e=pathlib.Path('docs/project/maintainers/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md').read_text(encoding='utf-8');print(len({i for l in e.splitlines() if l.startswith('| **FRZ IDs (') for i in re.findall(r'FRZ-[A-Z]+-\d+',l)}))"
```

The known-red counts come from `known_red_register.v1.json` (`rows[*].disposition`). Per-sentence checks (`path@sha:line`) are kept outside the repository under `.tmp/session/arch-freeze/ecap-rev/FACT_CHECK.md`.

## 9. Not claimed

This draft claims no FRZ promotion, no certification, no independent verification and no ECAP §2.C or §15 result. It proposes no change to the roadmap or the checklist. Section 3 describes planned surfaces as planned. Proposed wording is a starting point for the W5 final, not accepted text.
