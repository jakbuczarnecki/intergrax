# Campaign A — Enterprise Baseline Audit (EQO-A) — execution plan for Sonnet Agent Mode

Planning HEAD: `f87df3b2a93bd09aa8e1577b80122391249772c8` (`development`, 2026-11-15 08:53 +01:00). This is the discovery pin only; the audit pin is set in W0 (section 6).
Campaign ID: **EQO-A** (Enterprise Quality Oversight, Campaign A). EQO-A is **not** a roadmap stage. It never appears in the roadmap, FRZ checklist, ledger or IIAC activation record.

## 0. Context

The platform has about 50 parent stages and about 253 stage IDs with their own rows in [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) (2375 lines). There are 225 FRZ criteria in 26 families, more than 820 qualification records, and several earlier assurance campaigns (ENT-AUDIT-X, ECAP, IIAC, ROADMAP-REPLAY-X, the historical certification campaign, PRE-ENT, PDC-X). PDC-X is currently in P7 repair loop 1 (tickets T-IGX-041-*). ARCH-FREEZE is OPEN.

No single, independent, evidence-graded picture of the whole program exists yet. Discovery already shows contradictions: dual statuses, stale matrices, off-vocabulary statuses, open child rows under closed parents, and claims of independent review that other rows call unverified. It also shows unaudited worker judgements (85 HIST FRZ dispositions) and a HIGH tier-boundary residual (HIST-F-BND-03B).

EQO-A builds that baseline. It is read-only, pinned to an exact SHA, uses existing mechanisms where they help, and is checked by an independent session.

**Operator decisions already taken (2026-11-15, this planning chat). They are binding and need no re-asking:**
- **OD-EQOA-01 Baseline.** `BASELINE_SHA` = `origin/development` at the moment W0 starts, pinned in a detached worktree. Later commits go into a "post-baseline delta" section and are never mixed into the evidence.
- **OD-EQOA-02 Publication.** Work under `.tmp/` during execution. After the independent auditor's verdict, publish to `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_a/` in one docs-only PR. The Operator merges it. Creating that one publication branch is authorized by this decision.
- **OD-EQOA-03 Tests.** At most **10** bounded, read-only pytest invocations in total. Each names explicit node IDs or files, runs one process at a time, and writes its log to `.tmp/`. They are used only where a certification claim cannot be decided from documents or code. Broad suites and regression tiers (`igx_gov ticket`, `run_regression_gate_ci.py --profile *`, ENT-AUDIT `--mode FULL/FREEZE` execution) are **forbidden**.

## 1. Executive objective

At one exact `BASELINE_SHA`, produce an evidence-graded and mechanically checked baseline of the Enterprise program covering:
- every stage and its certification evidence;
- every FRZ criterion;
- the architecture map of 20 Enterprise concerns;
- post-certification drift;
- coverage gaps in the automated audit system;
- residual debt, exceptions and the integrity of operator decisions;
- cross-domain ownership completeness.

The result is a prioritised findings register, a closure matrix, and a handoff to Campaigns B and C. It is accepted by a separate independent audit session.

EQO-A issues no certificate, promotes nothing and changes no canonical status.

## 2. Scope, limits, non-goals

**In scope:**
- All roadmap stage IDs: rows in the §3 table, rows in the §5 ledger, and IDs mentioned only in prose.
- All FRZ-* IDs in the checklist at `BASELINE_SHA`.
- All certification and qualification records those IDs cite.
- Code and tests in `intergrax/`, `agents/` and `applications/` as far as needed to map concerns and check claims.
- Governance tooling: `scripts/governance`, `scripts/enterprise_audit`, `scripts/qualification` and `intergrax/platform/enterprise_audit`.
- All registers listed in A7.
- The PDC-X, ARCH-FREEZE, SFX, SEF-X, SCENARIO-READINESS-X and SCENARIO-GATE records, **for status and dependencies only**.

**Out of scope (each item gets a row in `SCOPE_EXCLUSIONS.md`):**
- Deep security, adversarial, fault-injection and chaos testing → Campaign B.
- Full re-audit of individual mechanisms → Campaigns B and C.
- Live vendor or production qualification → PROD-Q Path B.
- Re-running PDC-X P7 domain audits. EQO-A consumes `PDC_P7_D1..D8_AUDIT_REPORT.md` as in-flight inputs.
- Scenario implementation.

**Forbidden (no authorization exists in this campaign; on need → STOP and escalate):**
- Any edit to a tracked file. This covers product code, tests, roadmap, checklist, ledger, certificates, policies, tickets, `IIAC_POLICY_ACTIVATION_RECORD.json`, baselines and allowlists.
- FRZ promotion, stage or ticket closure, and accepting exceptions.
- Migrations or anything touching production environments.
- Docker, and new vendor integrations.
- Converting red tests to skipped tests, or widening exceptions.
- `git checkout` or `switch` in the main checkout.
- Committing anything other than the single publication PR in W6-P.

## 3. Verified canonical sources (as of f87df3b2a; re-verify at BASELINE in W0-T03)

Paths are relative to `docs/project/maintainers/` unless a path is given.

| # | Source | Authority role (§3.4 hierarchy) | Size / read method |
|---|---|---|---|
| S01 | `plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` | Order, status, formal closure. §2.2 status lifecycle L292–302, §2.3 who may close L304, §2.4 evidence rule L316, §3 table L338–503, §5 ledger L1921–2172, §1.2 and §2.0.3 Path B L108–123 and L246–280, ARCH-FREEZE row L499 | 2375 lines. **Parse the tables with a script.** Read prose sections by offset only |
| S02 | `qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` | Acceptance criteria. Table header L78, rows L80–337 (L151–183 is closure prose that breaks the table), vocabulary L25–34, evidence log L341+, family→stage owners L495–571, Path B L57–68 | 601 lines, about 125k tokens. **Never read it whole.** Parse with a script and read individual rows by offset |
| S03 | `qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md`, `IIAC_POLICY_ACTIVATION_RECORD.json`, `schemas/iiac_audit_session_manifest.v1.schema.json`, `IIAC_SESSION_HANDOFF_RECORD.template.md` | Certification semantics, closure vocabulary, manifest schema | Small; read fully |
| S04 | `qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md` (ECAP-1; modes §2.A–D, C1–C5), `EBH_7_ECAP_EXECUTABLE_CHECKLIST.md`, `EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` | Audit-mode semantics | Read by section |
| S05 | `qualification/ent_audit_x/` (Annex B FRZ-AUD closure, C control matrix, D VBN 1–31, F gap register, G/G2 gate registry and mapping, SHA ladder, closure status JSON); `intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json` (85 checks), `changed_since_impact_map.v1.json`, `evidence_artifact_catalog.v1.json`; `tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json` | ENT-AUDIT-X control system | JSON parsed by script |
| S06 | `qualification/ARCH_FREEZE_*.md` (21 files) + `arch_freeze_b1_mandatory_dispositions.json`. In particular: FRZ acceptance matrix (stale), closed-world gap register, SHA ladder, B2 debt reconciliation, operator decision register, WBS/DAG, architecture and production surface inventory (9024 modules @ `18a0f68f`) | ARCH-FREEZE in-flight state | Headers + targeted rows |
| S07 | `qualification/platform_domain_completeness/` (`PDC_INDEX.md`, `PDC_OWNERSHIP_MATRIX.md`, `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, `PDC_FRZ_COVERAGE_MATRIX.md`, `PDC_GAP_REGISTER.md`, `PDC_SHA_LADDER.md`, `PDC_P7_D1..D8_AUDIT_REPORT.md`, `operator_decisions/`) | PDC-X in flight, 8 domains D1–D8 | Index first |
| S08 | `qualification/historical_certification_campaign/` (stage SHA and evidence registry, independent audit register, FRZ-85 final disposition JSON, residual registers, Path B deferred obligations, terminal closure report) | Prior HIST assurance. Its own §3 says the 85 dispositions and 62 findings were **never independently audited** | Small MD + JSON |
| S09 | `qualification/pre_ent_enterprise_certification/`, `qualification/iiac_evidence/{ENT-AUDIT-X,EVOL-X,EVOL-X-W4-01-REAUDIT,HIST-P5-2026-11-09,IIAC-GOV-1,MCP-SERVER-X,PDC-X}/`, `qualification/evol_x/`, `qualification/mcp_server_x/MCP_SERVER_X_INDEX.md`, `qualification/ebh_6_p03_independent_replay/`, `qualification/ebh_7_p03_independent_replay/` | Stage evidence bundles | Manifests and indexes first |
| S10 | Stage records: `qualification/{TENANT_X,COMPAT_X,CONFIG_X,CTRL_X,STATE_X,TRACE_X,DUP_X,EBH_3..7,DIST_SCALE_X,TOOL_LIFE,SKILL_X,QUAL_X,PROD_Q,ROADMAP_REPLAY,GOV_X*,HARNESS_*,EE_*,NPSC_*}*.md` | Per-stage certification records | Only those cited by S01 or S02 rows |
| S11 | `qualification/VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md` (schema authority, not a PASS record), `PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md` (READY FOR AUDIT), `EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md` | VBN / PROD-Q | Targeted |
| S12 | `docs/project/architecture/*.md` (about 70; in particular `INTERGRAX_ARCHITECTURE_PRINCIPLES.md`, `DEPENDENCY_OWNERSHIP.md`, `ARCHITECTURE_OVERVIEW.md`), `docs/project/capabilities/architecture/*.md` (10, including `EVOL_X_CHANGE_SAFETY.md`, `COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md`, `MCP_SERVER.md`, `ARTIFACT_LIFECYCLE.md`, `SCENARIO_FACTORY.md`), `docs/project/maintainers/plans/*.md` (domain plans), `HARNESS_ARCHITECTURE_EVOLUTION_P0A_*` | Domain semantic authorities | Headers / owner sections |
| S13 | Registers: `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md`, `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`, `EBH_7_GAP_REGISTER.md`, `EBH_7_B6_BLOCKER_REGISTER.md`, `ent_audit_x/ANNEX_F_GAP_REGISTER.md`, `DIAGNOSTIC_GAP_LEDGER.md`, `DUP_X_FINDINGS_REGISTER.md`, `EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md`, `ROADMAP_REPLAY_X_CORE_HARDCODING_FINDINGS_REGISTER.md`, `EBH_6_CONDITIONAL_REMEDIATION_REGISTER.md`, the `*_OPERATOR_DECISION_REGISTER.md` files (ARCH_FREEZE, DUP_X, EBH_5/6/7, ROADMAP_REPLAY_X), `EBH_7_CROSS_DOMAIN_RISK_MATRIX.md`, `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md`, `ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`, the HIST residual registers, `PDC_GAP_REGISTER.md`, `mcp_server_x/MCP_SERVER_X_GAP_REGISTER.md` | Debt, exceptions, decisions | Rows parsed |
| S14 | `governance/` (`tickets/` 102 tickets + `plans/`, `agent-control-plane/role_registry.yaml`, `policy-registry.yaml`, `TRUSTED_POLICY_PIN.yaml`, `GOVERNANCE-MANIFEST.yaml`) | Delivery governance, approvals | Grep + targeted |
| S15 | `docs/project/technical/DOCUMENTATION_MAP.md` | Navigation hub | Read first in W0 |

Known absences (to be recorded as findings in W0, not searched for again): there is no known-red test register, no dedicated N/A register, no certification-debt register, no README or index for `qualification/`, no import-linter config (boundaries are enforced by AST gates in `tests/unit/architecture`, `tests/unit/runtime/architecture` and `scripts/maintenance/check_*.py`), and no standalone ECAP script.

## 4. Reused mechanisms (command whitelist / blacklist)

Run them from the audit worktree (section 6). Every run is logged in `TOOL_RUN_LOG.md` with the command, cwd, SHA, exit code, duration and output path. Check `git -C $WT status --porcelain` before and after each run. Any change means the tool is not read-only: record a finding, restore the worktree with `git -C $WT checkout -- . && git -C $WT clean -fd` (own worktree only), and never run that tool again.

**Allowed (static or read-only; does not count against the test budget):**
- `git` read commands: `log`, `show`, `diff`, `rev-parse`, `merge-base --is-ancestor`, `cat-file -e`, `ls-tree`, `blame`, `grep`.
- `uv run python -m scripts.governance.igx_gov doctor`. The `inventory` and `lint` subcommands only after checking `--help` shows no write flag.
- `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force`, and `verify_iiac_audit_manifest <manifest>`.
- `verify_iiac_role_separation <manifest>` on the existing published manifests.
- `uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --non-certifying --dry-run --output-dir <.tmp>`. This lists the plan only.
- The same command with `--mode CHANGED-SINCE --since-sha <X> --dry-run`.
- `scripts/enterprise_audit/build_impact_map.py --since-sha <X> --output <.tmp file>`.
- `scripts/enterprise_audit/mechanical_readiness_check.py` and `vbn_domain_report.py`, with output to `.tmp`.
- `scripts/enterprise_audit/check_manifest_annex_c_alignment.py`.
- `scripts/qualification/ent_audit_x_formal_closure_preflight.py --gate A|B`.
- `scripts/qualification/evol_x_change_workflow.py --base-sha A --head-sha B --out-dir <.tmp>`. Check `--help` first and never pass `--write-baseline`.
- `scripts/qualification/render_ebh7_f10_normative_matrix.py --check`.
- `scripts/governance/verify_arch_freeze_plan_traceability.py`.
- `scripts/maintenance/check_*.py` (tier, vendor and boundary scanners), `scripts/check_integration_vendor_imports.py`, `scripts/check_no_upward_application_imports.py`, `scripts/check_agents_no_pseudo_shared_packages.py`.
- `scripts/governance/verify_ent_audit_branch_protection.py`, only if `gh auth status` succeeds. It makes read-only GET calls. If `gh` is not available, record an evidence gap.

**Counted against the 10-invocation test budget:**
- `uv run pytest <explicit node ids/files> -p no:cacheprovider --basetemp <WT_BASE>/_scratch/eqo-a/<run-id>`.
- `run_enterprise_audit.py run --check-id <ID> --non-certifying` (one check per invocation).

**Forbidden:**
- `promote-baseline`, `--bootstrap-candidate`, `--write-baseline`.
- `igx_gov bootstrap`, `igx_gov sync-adapters`, `igx_gov ticket`, `igx_gov before-push`.
- `run_regression_gate_ci.py`.
- `generate_roadmap_replay_x_matrices.py`, `render_*` without `--check`, `sync_agent_env.py`.
- Any ENT-AUDIT run without `--dry-run`, except a single `--check-id`.
- Any script whose `--help` shows it writes into tracked paths.

## 5. Operating model, responsibilities, autonomy, escalation

| Actor | Does | Must never |
|---|---|---|
| **Sonnet (EQO-A executor, role `orchestrator`)** | Runs W0–W5 end to end without per-wave approval. Spawns and consolidates subagents, runs the allowed tools and the test budget, writes the dossier under `.tmp/`, prepares the W6 handoff, applies the auditor's corrections (corrective loop), opens the publication PR | Declare PASS, certify, or label the dossier "independently accepted". Edit tracked files outside W6-P. Run more than 3 subagents at once. Let subagents run pytest or commit |
| **Subagents** (Explore for search; general-purpose for shard writing) | One batch each. Each writes **only** its own shard files (`shards/<WS>/<BATCH>.*`) and returns a summary of at most 300 words with counts | Write shared files, run pytest, commit, issue verdicts beyond proposed per-row values |
| **Independent auditor** (new session, role `independent-quality-auditor`, ≠ executor session) | Replays the mechanical checks, samples rows, re-checks every BLOCKER and HIGH finding, writes `EQOA_INDEPENDENT_AUDIT_REPORT.md` and an IIAC-style session manifest (`stage_id: EQO-A`, verdict `PASS`, `PASS_SCOPED`, `CONDITIONAL` or `FAIL` on the **dossier**) | Edit the dossier (findings only), audit its own work, certify the platform |
| **Operator** | Starts or approves the auditor session (may delegate per memory: delegation is allowed to a new dedicated non-implementing session), decides escalations, merges the publication PR, rules on authority conflicts that block A9 | — |

**Autonomous (already authorized):**
- All W0–W5 analysis tasks.
- Creating the worktree under `_wt`.
- Writing under `.tmp/session/eqo-a/` and `_wt/intergrax/_scratch/eqo-a/`.
- Allowed tools, and the test budget within OD-EQOA-03.
- Authoring helper scripts under `.tmp/session/eqo-a/tools/` (stdlib only, run with `python -I` or `uv run python`).
- Raising findings at any severity.
- Proposing (`start_session`, `own_initiative`) the auditor session card at W6.
- After the auditor's verdict: one publication branch and PR (W6-P).

**Escalate (STOP the affected task only; continue the others; report in chat in one short message):**
- **E1:** a candidate BLOCKER is confirmed at evidence level ≥E3. Notify immediately and keep going.
- **E2:** a test is needed beyond the budget of 10, or needs services, network or secrets.
- **E3:** any tracked-file change is needed.
- **E4:** `BASELINE_SHA` cannot be resolved, or the worktree path violates the CLAUDE.md worktree rule.
- **E5:** an authority conflict whose resolution changes a BLOCKER or HIGH verdict. Record it in the conflict register with the deciding authority, and continue with both readings labelled.
- **E6:** secrets or credentials appear in evidence. Do not copy them; record path and line only.
- **E7:** a tool turns out not to be read-only (section 4).
- **E8:** the independent auditor's verdict is `FAIL`, or a second `CONDITIONAL`.

## 6. Exact-SHA policy and workspace

**W0-T01 pin:**
```bash
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
git -C "$MAIN" fetch origin development
BASELINE_SHA="$(git -C "$MAIN" rev-parse origin/development)"
WT_BASE="$(dirname "$MAIN")/_wt/$(basename "$MAIN")"; mkdir -p "$WT_BASE/_scratch/eqo-a"
git -C "$MAIN" worktree add --detach "$WT_BASE/eqo-a-${BASELINE_SHA:0:8}" "$BASELINE_SHA"
```
- The main checkout stays on `development` and is never checked out, reset or stashed. Its pre-existing untracked file stays untouched.
- `WS=$MAIN/.tmp/session/eqo-a` is the dossier workspace (gitignored, survives worktree removal).
- `WT=$WT_BASE/eqo-a-<sha8>` holds the code under audit.
- Write `BASELINE.json` with: SHA, commit date and subject, the `origin/development` fetch timestamp, `git -C $WT status --porcelain` (must be empty), `uv --version`, the sha256 of `uv.lock`, the Python version, the OS, and the planning-pin delta (`git log --oneline f87df3b2a..BASELINE_SHA`).

**Rules:**
1. **R-SHA-1.** Every statement about the *current* state refers to `BASELINE_SHA`. Read code through `$WT` or `git show BASELINE_SHA:<path>`, never from the moving main checkout.
2. **R-SHA-2.** Every cited historical SHA is stored in full (40 hex), with a role label: `implementation`, `audit`, `test_execution`, `certification`, `closure_publication`, `rejected`, `superseded` or `evidence`. Each is checked with `git cat-file -e` and `git merge-base --is-ancestor <sha> BASELINE_SHA`. A SHA that does not resolve, or is not an ancestor, is a finding (class `EVIDENCE_DEFICIENCY`) and is never silently dropped.
3. **R-SHA-3.** Abbreviated SHAs from documents are expanded with `git rev-parse <abbr>^{commit}`. An ambiguous or unknown abbreviation is a finding.
4. **R-SHA-4.** No mixing. A row's verdict uses evidence from its own cited SHAs plus `BASELINE_SHA`. Findings derived from post-baseline commits are forbidden. W5-T05 records `git log BASELINE_SHA..origin/development` as a delta annex, informative only, and flags "possibly affected" findings without changing their verdicts.
5. **R-SHA-5.** Every final deliverable starts with the same header block: Campaign, `BASELINE_SHA`, generation UTC, and source-registry digest. Check C06 enforces this.

## 7. Evidence model and schemas

**Evidence strength levels (one per row and per finding):**

| Level | Meaning |
|---|---|
| E0 | No evidence found |
| E1 | Document claim only |
| E2 | E1 plus a resolvable SHA and an existing cited artifact |
| E3 | E2 plus artifact content checked against the claim (digest match, log shows exit code and node IDs, or code at SHA confirms the behavior) |
| E4 | Re-executed by EQO-A at `BASELINE_SHA` (test budget) |

Inventory completeness (C01–C05) and verification strength (the E-level histogram) are always reported **separately**.

**A2 certification verdicts:**

| Code | Meaning |
|---|---|
| `CV-VERIFIED` | Certificate valid, evidence verified |
| `CV-SCOPED` | Valid within declared scope; limitations listed |
| `EV-INCOMPLETE` | Evidence incomplete |
| `EV-INCONSISTENT` | Evidence inconsistent |
| `HIST-REQUAL` | Historical; needs requalification |
| `CLAIM-UNSUPPORTED` | Certification claim unsupported |

`NOT-ASSESSED` is allowed in shards only and must be 0 by W5.

**A5 drift classes:**

| Code | Meaning |
|---|---|
| `DRIFT-NONE` | — |
| `DRIFT-BENIGN` | Changes do not touch the certified invariant; rationale required |
| `DRIFT-REQUALIFIED` | Later requalification evidence cited |
| `DRIFT-UNREQUALIFIED` | Invariant-relevant change with no requalification |
| `DRIFT-UNKNOWN` | Surface not determinable; this is a finding |

**Finding schema** (JSON line in `findings/FINDINGS.jsonl`; the Markdown register is generated from it):
`finding_id` (`EQOA-<WS>-<NNN>`), `severity` (`BLOCKER | HIGH | MEDIUM | LOW | INFO`), `class` (`CONFIRMED_DEFECT | EVIDENCE_DEFICIENCY | GOVERNANCE_INCONSISTENCY | CERTIFICATION_DRIFT | ARCHITECTURE_RISK | UNVERIFIED_HYPOTHESIS | ACCEPTED_SCOPED_LIMITATION`), `scope` (stage, FRZ, concern or register IDs), `affected_invariant` (roadmap §1 invariant number, FRZ ID or contract name), `exact_sha` (full), `evidence` (list of `path@sha:line` or a log path plus digest), `evidence_level`, `impact` (including an explicit `arch_freeze_impact: BLOCKS | CONDITIONS | NONE | UNKNOWN`), `recommendation`, `owner` (canonical owner per section 3.4, e.g. "ARCH-FREEZE", "PDC-X D6", "PROD-Q", "Operator"), `next_action` (`OWNER_DECISION | CAMPAIGN_B | CAMPAIGN_C | REQUALIFY | DOC_FIX_BY_OWNER | NONE`), `decision_authority`, `related_findings`, `seed_id` (if from section 15).

**Severity rules:**
- **BLOCKER** means it would invalidate an ARCH-FREEZE entry condition (roadmap L499 / invariants 22–23) or breaks a hard tier boundary in production code.
- **HIGH** means a closed stage's central claim is unsupported, or a security or tenant invariant is unproven.
- **Classification rule:** BLOCKER or HIGH requires evidence level ≥E3. Otherwise the finding is `UNVERIFIED_HYPOTHESIS` with `next_action: CAMPAIGN_B`, and its severity is labelled "potential".

**Row schemas:**
- **Stage row** (A1): the fields in the Campaign A spec A1, plus `row_source` (`S3TABLE | LEDGER | PROSE`), `line`, `status_raw`, `status_normalized` (§2.2 vocabulary or `OFF_VOCAB`), `parent_id`, `disposition` (`PARENT_ASSESSED | CHILD_INHERITS:<parent> | LINEAGE_ONLY | OPEN_NOT_CERTIFIED | PROSE_ORPHAN | OUT_OF_SCOPE:<reason>`) and `assessment_tier` (`T1` = full 12 questions, `T2` = mechanical).
- **FRZ row** (A3): the fields in the A3 spec, plus `checklist_line`, `status_raw`, `status_normalized`, and the statuses in the ARCH-FREEZE matrix, PDC coverage, HIST-85 and Annex B with flags for any disagreement.

## 8. Test and verification decision table

| Situation | Action |
|---|---|
| The claim is about a record, a status, a SHA or an approval | Check the documents and evidence (E2/E3) |
| The claim is about behavior or a boundary, and code at `BASELINE_SHA` decides it | Read the code (E3) |
| The claim is BLOCKER or HIGH relevant, documents and code cannot decide it, and an existing test targets it without services | Run the test (E4), one budget slot |
| The claim needs live services, network, secrets, chaos or adversarial work, or more than about 5 files of suites | Do not run. Raise `UNVERIFIED_HYPOTHESIS` → Campaign B |
| The cross-domain claim depends on several stages | Static cross-check plus `ent_audit --dry-run` plan analysis. Execution goes to Campaign B |
| The historical evidence SHA is not an ancestor, or there is `DRIFT-UNREQUALIFIED` | `HIST-REQUAL` with `next_action: REQUALIFY`, owner = stage owner |

**Budget ledger `tests/TEST_EXECUTION_RECORD.md`:**
- Columns: run-id, purpose (finding or seed ID), exact command, cwd, `BASELINE_SHA` from `git -C $WT rev-parse HEAD`, Python, `uv` and `uv.lock` sha256, OS, start and end UTC, exit code, log path, and log sha256.
- Only bounded node IDs or files that need no external services.
- No environment variables that enable live vendors.
- Default slot plan: up to 4 slots for SEED and BLOCKER/HIGH candidates in W2–W3, up to 4 for A5 drift spot-checks, and 2 held in reserve.
- An unused slot is fine. A failing test is recorded as is and is never rerun until it passes. Not being able to run a test is reported explicitly.

## 9. Workstreams and WBS

Every task below has these properties unless the row says otherwise:
- Evidence is recorded per the section 7 schemas.
- Output goes to `$WS/<WS-id>/`.
- The task finishes when its mechanical check passes and its shard or deliverable is consolidated.

The added workstream **A0** covers authority, sources and tooling, which the A1–A9 brief assumes but does not specify. A1–A9 also gain the sub-tasks marked ★, found in the self-review (section 20).

### A0 — Baseline, source authority, tooling (W0, sequential, parent only)
| Task | Objective / method | Output | Mechanical acceptance |
|---|---|---|---|
| W0-T01 | Pin `BASELINE_SHA` and create the worktree and workspace (section 6) | `BASELINE.json`, `$WS` tree (`A1..A9/`, `shards/`, `findings/`, `tools/`, `tests/`, `logs/`) | `WT` HEAD == `BASELINE_SHA`; porcelain is empty |
| W0-T02 | Read the S15 map, the S03 protocol and the roadmap §2.2–2.4 by offset. Write `CAMPAIGN_A_CHARTER.md`: scope, OD-EQOA-01..03, roles, enums | Charter | Contains all enums from section 7 |
| W0-T03 | `SOURCE_REGISTRY.md` / `.json`: S01–S15 resolved at `BASELINE_SHA` (path exists via `git cat-file -e BASELINE:path`, line count, sha256 of the blob), each with an authority role. Discover more authorities: grep S01 and S02 for every linked `.md`/`.json` path → `LINKED_PATHS.json`, with existence checked at the baseline | Registry, linked-path list | 100% of linked paths classified as exists / missing. Missing → finding |
| W0-T04 | Write the parsers in `tools/`, stdlib only: `parse_roadmap.py` (§3 table + §5 ledger → rows, with prose ID mentions by regex `\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b` filtered against a stage-ID grammar), `parse_frz.py` (FRZ rows from L80–337, including the headerless block after L183, plus the evidence log), `resolve_shas.py` (R-SHA-2/3), `parse_registers.py` (generic Markdown table → JSON). Run them and keep the raw extracts | `extracts/*.json` | Roadmap row counts reconcile with the line ranges. The FRZ unique-ID count is printed (expected 225 at planning; any change is a finding-free fact) |
| W0-T05 | Write `tools/verify_campaign_a.py`, which implements checks C01–C15 (section 14) and reads only `$WS` and git | Checker | Runs; reports "inventory pending" |
| W0-T06 | Tool baseline runs (allowed list): `igx_gov doctor`, `verify_iiac_policy_activation --require-in-force`, ENT-AUDIT `--mode FULL --non-certifying --dry-run`, `check_manifest_annex_c_alignment`, `render_ebh7_f10_normative_matrix --check`, `verify_arch_freeze_plan_traceability` | `TOOL_RUN_LOG.md` + logs | Every exit code recorded. Porcelain unchanged |
| W0-T07 | `AUTHORITY_CONFLICT_REGISTER.md` (empty schema: conflict ID, sources A/B with line, values, hierarchy rule from Campaign A spec section 3.4, deciding authority, resolution procedure, affected rows), `SCOPE_EXCLUSIONS.md` (seeded from section 2), and `FINDINGS.jsonl`. Seed the known absences (section 3) and the section 15 seeds as `UNVERIFIED_HYPOTHESIS` rows with `seed_id` | Registers | Every seed is present |

### A1 — Canonical Program Inventory (W1)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A1-T01 | Mechanical stage inventory from `extracts/roadmap_rows.json`: all §3 rows, ledger rows and prose-only IDs; `status_raw`, normalized status, the SHAs with roles, linked docs, stated dependencies, residuals | `A1/stage_rows.jsonl` | C01: every parsed ID present |
| A1-T02 | Subagent batches (3, in parallel): fill in owner, FRZ links (from checklist L495–571 family→stage), ADRs and tickets (grep `governance/tickets` for the stage ID), open obligations, closure conditions, disposition and tier. Batch α: §3 L340–417 (hardening, GOV, Harness, TRACE-X). Batch β: §3 L418–484 (CONFIG, COMPAT, TENANT, PROD-Q) plus their ledger children. Batch γ: §3 L485–503, the remaining ledger-only rows and prose-only IDs | `shards/A1/{α,β,γ}.jsonl` | Every row has a disposition and tier. `PROSE_ORPHAN` → finding |
| A1-T03 ★ | Cross-source reconciliation: IDs in ledger, dossiers (S08, S09 indexes, ARCH-FREEZE WBS, PDC index) or tickets that are not correctly reflected in the roadmap; dual statuses; checkbox misuse (`[x]` on REJECTED/SUPERSEDED); `READY FOR AUDIT` rows under CLOSED parents; stale "NEXT/CURRENT" text inside closed rows | Findings (`GOVERNANCE_INCONSISTENCY`) | Each seed S-01..S-06 resolved: confirmed, refuted or merged |
| A1-T04 | Consolidate and render `ENTERPRISE_STAGE_INVENTORY.md`: a parent-level table of about 50 rows, plus a child appendix generated from JSONL | Deliverable | C01, C02, C03 (inventory part) pass |

### A3 — FRZ Acceptance Reconciliation (W1, parallel with A1)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A3-T01 | Mechanical FRZ matrix from `extracts/frz_rows.json`, joined with the ARCH_FREEZE FRZ matrix, `PDC_FRZ_COVERAGE_MATRIX`, `HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json`, the ENT-AUDIT Annex B/G2 mapping, ROADMAP_REPLAY_X FRZ-225 reconciliation, and the checklist evidence log (L341+, about 114 rows) | `A3/frz_rows.jsonl` with per-source status columns and disagreement flags | C04 |
| A3-T02 | Subagent batches by owning-stage clusters (6 batches, at most 3 at a time): {AUD, EVOL, FRZ, DOC, DEBT}, {BND, OWN, CTR, TYP, PLG}, {CFG, CMP, TEN, STA, TRC}, {CTL, GOV, HRN, EXE, REC}, {OBS, SEC, REL, REG, RPL}, {PRD}. Per row: owner, responsible stage, implementation, test and audit evidence (E-level), scope limits, history, open obligations; whether PASS evidence applies to the current implementation (does the code path still exist at the baseline: `git cat-file -e`, grep); N/A validity | `shards/A3/*.jsonl` | No `NOT-ASSESSED` rows |
| A3-T03 ★ | Independent sample of the HIST-85 worker dispositions: all PASS rows among the 85, plus 20% of the rest (deterministic: every 5th by ID order), re-judged from evidence | `A3/hist85_sample.md` | Sample size printed; disagreements → findings |
| A3-T04 | Render `ENTERPRISE_FRZ_RECONCILIATION_MATRIX.md` with family roll-ups computed by script, never typed by hand | Deliverable | C04, C08 |

### A2 — Certification Evidence Integrity (W2)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A2-T01 | Tier-2 mechanical pass over every CLOSED-family row (all about 253 own-row IDs with a closed-type status): SHA resolution and ancestry, cited doc exists at the baseline, an audit or acceptance record exists, independence wording present (e.g. "independently accepted", "operator accepted", "scoped") | `A2/t2_mechanical.jsonl` | Questions 1, 2, 4, 5 and 11 answered mechanically for 100% of rows |
| A2-T02 | Tier-1 subagent batches over parent stages: all 12 spec questions + verdict code + limitations. Batches: B1 {EBH-2E..2I, ADR3-IMP-*, HARNESS-QINF-01, HARNESS-W4/W5/W6}; B2 {GOV-X1, INT-EXTCOMP-X, INT-CONFIG-REAL-X, AW-7C, EBH-3, EBH-4}; B3 {HARNESS-W7, W8, RESIDUAL SCHED-01/CE-01, HARNESS-FINAL, GOV-X2, CTRL-X, STATE-X}; B4 {TRACE-X tree, TRACE-X-CERT, P6}; B5 {CONFIG-X, COMPAT-X, TENANT-X}; B6 {PROD-Q, P0/P2/P3, R7-Q tree, D03-A decisions, Path B amendment}; B7 {QUAL-X(-P0), SKILL-X, TOOL-LIFE-X, DIST-SCALE-X}; B8 {EBH-5, EBH-6, DUP-X, ROADMAP-REPLAY-X, EBH-7}; B9 {EVOL-X, ENT-AUDIT-X, MCP-SERVER-X, IIAC-GOV-1 activation chain incl. ACT-N11 re-pins, HIST campaign, PRE-ENT}; B10 {PDC-X, ARCH-FREEZE wave 1, SFX, SEF-X, READINESS-X, SCENARIO-GATE: status and dependency validity only, verdict n/a = `OPEN_NOT_CERTIFIED`} | `shards/A2/B1..B10.jsonl` | Every T1 row: 12 answers, each with evidence or an explicit "no evidence" |
| A2-T03 ★ | Independence chain check: for IIAC-era stages, validate the published manifests with `verify_iiac_audit_manifest` and `verify_iiac_role_separation` (allowed); for pre-IIAC stages, classify the authority type (human Protocol 03, operator-only, process-enforced delegate) | Column `independence_basis` | 100% of T1 rows classified |
| A2-T04 ★ | Operator decision trace: for every T1 stage, list the required ODs (from stage record and roadmap row) → found / approved line present / delegation reference resolves at its cited SHA | Column `od_trace` | Missing OD → finding |
| A2-T05 | Render `ENTERPRISE_CERTIFICATION_EVIDENCE_MATRIX.md`: T1 table + T2 appendix + verdict histogram + E-level histogram | Deliverable | C03, C07 |

### A4 — Architecture and Implementation Baseline (W2, parallel with A2)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A4-T01 | Concern skeleton: the 20 concerns from the spec, plus ★ "Decision System", "Human-in-the-loop / interrupts", "Hosting / application tier (Tier-3)" and "Proof receipts / evidence plane" (they exist as packages and arch docs and none of the 20 obviously owns them) | `A4/concerns.json` | 24 concerns |
| A4-T02 | Package census at the baseline: top-level packages of `intergrax/`, `intergrax/runtime/` (about 80), `agents/`, `applications/`, each mapped to a concern by an owner-doc grep (`DEPENDENCY_OWNERSHIP.md`, `PDC_OWNERSHIP_MATRIX.md`, arch docs) | `A4/package_map.json` | Every package mapped, or flagged `UNOWNED` → feeds A8 |
| A4-T03 | Subagent batches of about 5 concerns (5 batches, at most 3 at a time). Per concern: semantic owner doc, composition owner (composition root / wiring module), public contracts (Protocols/ABCs in `intergrax/contracts` etc.), implementations and adapters, tests, FRZ links, certifications, dependencies; potential duplicates, bypasses, unauthorized fallbacks, undocumented execution paths → `ARCHITECTURE_RISK` / `UNVERIFIED_HYPOTHESIS` | `shards/A4/*.jsonl` | Each concern has an owner, or an "owner missing" finding |
| A4-T04 | Static boundary sweep (allowed `check_*` scanners), plus a targeted grep for Tier-0 → Tier-2/3 imports, including dynamic `importlib` (seed S-12). NPSC/boundary allowlists live inside the gate tests (`tests/unit/runtime/architecture/test_npsc4_1_*`, `test_npsc4_2_*`, `test_npsc5f_r3_*`); there is no standalone allowlist file. Every code scan in A4/A5/A8 excludes the build and venv artefacts `build/` and `applications/*/docker/runtime-context/` | `A4/boundary_scan.md` | Exit codes logged; exclusions listed in `SCOPE_EXCLUSIONS.md` |
| A4-T05 | Render `ENTERPRISE_ARCHITECTURE_BASELINE.md` | Deliverable | C07 |

### A5 — Historical Certification Drift (W3)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A5-T01 | Stage surface definition per certified T1 stage: paths from its certification record, ENT-AUDIT impact map entries, and the A4 concern map | `A5/surfaces.json` | Every T1 CLOSED stage has a surface, or `DRIFT-UNKNOWN` |
| A5-T02 | Mechanical delta for each stage: `git diff --name-status <cert_sha> BASELINE_SHA -- <surface>`, `git log` counts, public-contract file changes, new provider bindings (grep registries and entry points), tier-dependency changes (re-use the A4-T04 scan), test-file changes in the surface. Plus `build_impact_map.py --since-sha <cert_sha>` and `evol_x_change_workflow.py` for the 12 highest-risk stages (TENANT-X, CONFIG-X, COMPAT-X, STATE-X, TRACE-X, CTRL-X, GOV-X2, EBH-5, DUP-X, EVOL-X, ENT-AUDIT-X, MCP-SERVER-X) | `A5/deltas/*.json` | Every run logged |
| A5-T03 | Subagent batches (4): invariant-level reasoning. Does each relevant change touch the certified invariant? Is requalification evidence present (later ledger rows, PDC re-pins, ENT-AUDIT runs, EBH-7 re-pin)? File count is never the criterion | `shards/A5/*.jsonl` | Each stage has a drift class and rationale |
| A5-T04 | Up to 4 test-budget slots on `DRIFT-UNREQUALIFIED` candidates of HIGH relevance | Test record | Budget respected |
| A5-T05 | Render `ENTERPRISE_CERTIFICATION_DRIFT_MATRIX.md` | Deliverable | C07 |

### A6 — Enterprise Audit System Coverage (W3)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A6-T01 | Manifest analysis: 85 checks × kind × mode × severity × `pg_id` → FRZ (Annex B/G2) → the 225 FRZ IDs. FRZ IDs with zero automated checks; PDC domains D1–D8 and tiers with zero checks | `A6/coverage.json` | 100% of FRZ IDs labelled `AUTOMATED`, `PARTIAL`, `MANUAL_ONLY` or `NONE` |
| A6-T02 | PASS-despite-gap analysis (code reading, E3): env-gated markers that skip silently (`ENT_AUDIT_X_SEMANTIC`, `ENT_AUDIT_X_GATE_MATRIX`), existence-only gates (e.g. `check_architecture_debt_register.py`), allowlist gates (NPSC-4.2 allowlist recorded as FAIL), `changed_since` unmapped-path fail-closed behavior (code plus `--mode CHANGED-SINCE --dry-run` from an old SHA), and how PARTIAL/INCOMPLETE map to exit codes | `A6/pass_despite_gap.md` | Each mechanism has a verdict and evidence |
| A6-T03 | Drift detection and new-path handling: does a brand-new package path get covered or fail closed? Determine by impact map plus code | Section in A6 | — |
| A6-T04 | IIAC / validator trust: `process_enforced` assurance limits; validators that test only for an `Approved:` prefix (ticket README states this); ACT-N11 product-path sensitivity | Section in A6 | — |
| A6-T05 | Render `ENTERPRISE_AUDIT_COVERAGE_GAP_MATRIX.md` (no new gates are proposed as implemented; only recommendations) | Deliverable | C07 |

### A7 — Residual Debt and Exception Integrity (W3)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A7-T01 | Parse every S13 register, plus residuals and limitations from A1/A2 rows, ARCH-FREEZE ODs, PDC ODs and delegations, and IIAC manifest `limitations` | `A7/entries.jsonl` | Every entry has owner, severity, invariant, currency, evidence, reason, reopen condition, ARCH-FREEZE impact and production impact, or a finding |
| A7-T02 ★ | Carry-forward chain detection: build a graph of obligation IDs (e.g. `R1-SQLITE-ENV-01`, `HIST-F-*`, `F-*`, `R-P7-*`, `OPEN-BP-01`, `OD-*`) and "deferred to / tracked as / → STAGE" edges across all sources. Flag chains of length ≥2, or chains ending at a stage that is CLOSED, does not own the concern, or is post-freeze → "bouncing obligation" finding | `A7/carry_forward_graph.json` | Every chain classified |
| A7-T03 ★ | Known-red / skip inventory (no register exists): static count of `skip`, `skipif`, `xfail` and `no_ci` markers in `tests/` with reasons, cross-checked against `EBH_7_SKIP_DISPOSITION_REGISTER.md`; N/A decisions collected from inline `NOT_APPLICABLE` / `N/A` across S10 | `A7/skips.json`, `A7/na_decisions.json` | Unregistered skips over HIGH-surface tests → finding |
| A7-T04 ★ | Delegation and approval integrity: every `Approved: operator-delegate` line in tickets → its `Delegation:` reference resolves at the cited SHA; the delegate is not the implementer (as far as records show) | `A7/delegations.json` | 100% of delegate lines checked |
| A7-T05 | Render `ENTERPRISE_RESIDUAL_RISK_REGISTER.md` | Deliverable | C05 |

### A8 — Cross-Domain Completeness and Consistency (W4)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A8-T01 | Domain universe: historical domains (arch doc ↔ plan pairs per AGENTS.md documentation pairs, capability docs), PDC D1–D8, FRZ families, A4 concerns, A4-T02 packages. Build a many-to-many map | `A8/domain_map.json` | — |
| A8-T02 | Per domain: one semantic owner, contracts, lifecycle owner, failure/recovery owner, tenant/security rules, configuration rules, observability, test evidence, certification. Each is marked present (with path), absent or ambiguous (several owners) | `shards/A8/*.jsonl` (3 batches) | Every cell filled |
| A8-T03 | Orphan detection: packages or functionality with no domain, no FRZ family and no PDC domain; doc pairs missing a side; PDC-X in flight is treated as `IN_FLIGHT` (no closure assumed, P7 reports are inputs only) | Findings | Orphans count printed |
| A8-T04 | Render `ENTERPRISE_DOMAIN_COMPLETENESS_MATRIX.md` | Deliverable | C07 |

### A9 — Baseline Verdict and Risk Prioritization (W5)
| Task | Objective / method | Output | Acceptance |
|---|---|---|---|
| A9-T01 | Finding deduplication and merge (same invariant + scope); renumber nothing (keep IDs, add `merged_into`) | `FINDINGS.jsonl` final | C07 |
| A9-T02 | Cross-workstream consistency: A1 status vs A2 verdict vs A3 FRZ status vs A5 drift vs A7 debt. Contradictions → conflict register | Conflict register entries | C08 |
| A9-T03 | Prioritization: severity × `arch_freeze_impact` × evidence level; a list of potential ARCH-FREEZE blockers; recommended requalifications (stage, reason, minimal evidence needed, owner) | Sections | — |
| A9-T04 | Render `ENTERPRISE_BASELINE_AUDIT_REPORT.md`. Contents: executive summary; stage matrix; certification reliability matrix; coverage map; inconsistency, missing-evidence and risk registers; potential architecture violations; requalifications; ARCH-FREEZE-blocking list; the 7-class breakdown; limitations; post-baseline delta annex | Deliverable | C06 |
| A9-T05 | `ENTERPRISE_BASELINE_CLOSURE_MATRIX.md`: each spec section 10/13 condition → the check ID and its result, plus each deliverable → the acceptance criteria it meets | Deliverable | C13 |
| A9-T06 | `ENTERPRISE_CAMPAIGN_B_C_HANDOFF.md`. Every finding with `next_action` in {CAMPAIGN_B, CAMPAIGN_C} goes here, grouped by theme. Campaign B gets security/adversarial/fault-injection and test execution. Campaign C gets the deep mechanism re-audit and requalification design. Each entry has its scope, the hypothesis, the evidence so far and a suggested method | Deliverable | C10 |
| A9-T07 | `MANIFEST.json`: sha256 of every dossier file, plus `BASELINE_SHA`, tool versions and the source-registry digest | Manifest | C12 |

## 10. Dependency DAG

```text
W0: T01 → T02 → T03 → T04 → T05 → T06 → T07
            T03 ──────────────────────────────┐
W1:  A1-T01 → A1-T02(α,β,γ ∥) → A1-T03 → A1-T04
     A3-T01 → A3-T02(6 batches, ≤3 ∥) → A3-T03 → A3-T04        (A1 ∥ A3; share only read-only extracts)
     GATE-W1: C01, C02, C04 + A1↔A3 join (every FRZ closing stage ∈ inventory; every T1 stage's FRZ links ∈ matrix)
W2:  A2-T01 → A2-T02(B1..B10, ≤3 ∥) → A2-T03 → A2-T04 → A2-T05   needs A1-T04 (rows, tiers), A3-T04 (FRZ links)
     A4-T01 → A4-T02 → A4-T03(5 ∥≤3) → A4-T04 → A4-T05            needs W0 only (∥ with A2)
     GATE-W2: C03, C07 partial
W3:  A5 (needs A2 verdicts + A4 concern map) ∥ A6 (needs A3 + A4) ∥ A7 (needs A1 + A2)
     GATE-W3: C05, C09 partial
W4:  A8 (needs A4, A3, A7) → consolidation of all shards → GATE-W4: C01–C11 all pass
W5:  A9-T01 → T02 → T03 → T04/T05/T06 → T07 → GATE-W5: C01–C15 all pass
W6:  H-T01 handoff → [Operator/auditor] H-T02 audit → H-T03 corrective loop (≤2) → W6-P publication PR
```
Rule: subagents are only ever spawned for disjoint shard sets. Each consolidation step is done by the parent alone. No two agents write the same file.

## 11. Waves and exit gates

| Wave | Content | Subagent runs (≤3 at a time) | Exit gate (all must pass in `verify_campaign_a.py`) |
|---|---|---|---|
| W0 | A0 | 0 | `BASELINE.json` valid, sources resolved, parsers' counts printed, tool runs logged, seeds registered |
| W1 | A1 + A3 | 3 + 6 | C01, C02, C04 + A1↔A3 join |
| W2 | A2 + A4 | 10 + 5 | C03 + C07 on A2/A4 findings |
| W3 | A5 + A6 + A7 | 4 + 2 (A6-T01/T02) + 3 (A7-T01 split by register group) | C05, C09 (no BLOCKER/HIGH below E3) |
| W4 | A8 + consolidation | 3 | C01–C11 |
| W5 | A9 | 0 | C01–C15; zero `NOT-ASSESSED`; header SHA identical everywhere |
| W6 | Handoff, audit, corrections, publication | Auditor = separate session | Auditor verdict `PASS` or `PASS_SCOPED` (or `CONDITIONAL` resolved within ≤2 loops), then the PR |

There are about 39 subagent runs in total. No Operator approval is needed between W0 and W5.

## 12. Delegation strategy

- At session start, Sonnet loads `/intergrax-agent-roles`. The session role is `orchestrator`, which activates the DCP subagent exception in the token-budget rule. It follows `igx-devos-parallel-orchestration`: at most 3 parallel subagents, no subagent commits, no subagent PASS.
- **Subagent brief template** (every brief is self-contained):
  - goal and batch ID;
  - an exact row list (IDs and line numbers) taken from the extracts;
  - `BASELINE_SHA` and the `WT` path;
  - the allowed source paths (from `SOURCE_REGISTRY`), with "read by offset/limit, Grep first";
  - the row schema (paste the JSON keys);
  - the enum lists;
  - the output file path (own shard only);
  - "do not run pytest, do not edit outside your shard, do not commit";
  - "return at most 300 words: counts per verdict, candidate BLOCKER/HIGH IDs with one-line evidence, unresolved questions".
- Subagent types: `Explore` for pure discovery (it returns text and the parent writes it); `general-purpose` when the subagent must write a shard.
- The parent never re-reads full shards. It consolidates with `tools/consolidate.py` and spot-reads only the rows flagged BLOCKER, HIGH or conflict.
- On a subagent failure or nonsense output: re-run once with a narrower batch. If it fails again, the parent does the batch itself.

## 13. Context and token management

- Canonical first: parsers turn S01, S02, S05 and S13 into JSON once. After that everything works from the extracts. Never read S01 or S02 whole. Never bulk-read `docs/`, `governance/tickets/`, `iiac_evidence/` or logs.
- Historical dossiers are opened only when a row's evidence path points at them (dependency-driven). Within them, read the manifest or index first, then the cited section.
- Every file read is recorded once in `READ_LOG.jsonl` (path, sha, line range) so files are not analysed twice, and the auditor can see what was actually examined.
- Reuse existing evidence by citing it with its SHA and E-level. Never present reused evidence as re-executed: only budget runs get E4.
- Subagent reports are capped at 300 words, and their findings live in shards, not in the chat context.
- One central truth: `FINDINGS.jsonl` plus the `*/rows.jsonl` files. Every Markdown deliverable is rendered from them by `tools/render.py`. Hand edits to deliverables are forbidden; fix the data and re-render instead.
- Do not narrow the audit to save tokens. If a batch is too big, split it.

## 14. Mechanical completeness checks (`tools/verify_campaign_a.py`)

| ID | Check | How proven (not by record presence alone) |
|---|---|---|
| C01 | 100% of roadmap stage IDs inventoried or explicitly `OUT_OF_SCOPE` | Set equality between the IDs from an independent second parse of S01 at `BASELINE_SHA` (regex over raw text, not the table parser) and the IDs in `A1/stage_rows.jsonl`. Differences are listed |
| C02 | Every stage row has a normalized status, an owner and a disposition | Field non-empty plus enum membership |
| C03 | Every CLOSED-type row has ≥1 certification source that resolves at its SHA, or a linked `EVIDENCE_DEFICIENCY` finding | `git cat-file -e` per path@sha; finding-link check |
| C04 | 100% of FRZ IDs in the matrix, each with an owner and a status | Set equality with an independent regex `FRZ-[A-Z]+-\d+` over raw S02; count printed |
| C05 | Every active exception or debt entry has an owner or a finding | Field / link check |
| C06 | SHA consistency | Every deliverable header has the identical `BASELINE_SHA`; every 40-hex string in deliverables resolves; every abbreviated SHA was expanded |
| C07 | Finding completeness | All 10 spec fields plus `evidence_level`, enums valid, every evidence path exists at its SHA or in `$WS` |
| C08 | No contradiction hidden in a roll-up | Every roll-up count is recomputed from rows and must equal the rendered number; every row with any disagreement flag links to a conflict-register entry or a finding |
| C09 | Strength gate | BLOCKER/HIGH ⇒ `evidence_level ≥ E3`, otherwise class must be `UNVERIFIED_HYPOTHESIS` |
| C10 | Handoff completeness | Every finding with `next_action` in {CAMPAIGN_B, CAMPAIGN_C} appears in the handoff by ID |
| C11 | Read-only integrity | `git -C $WT status --porcelain` is empty. In `$MAIN`, `git status --porcelain` equals the W0 snapshot. No commit in `$MAIN` by this session before W6-P |
| C12 | Manifest | Each recomputed sha256 equals `MANIFEST.json` |
| C13 | Exclusions | Every `OUT_OF_SCOPE` / skip / `MANUAL_ONLY` item has a reason and owner in `SCOPE_EXCLUSIONS.md` |
| C14 | Test budget | ≤10 rows in the test record; each row has all fields; each log digest matches |
| C15 | Seeds closed | Every section 15 seed is resolved: confirmed (finding), refuted (with evidence) or merged |

The checker prints two separate blocks. **Inventory completeness** reports C01, C02, C04 and C05 as percentages. **Verification strength** reports the E0–E4 histogram per workstream and the A2 verdict histogram. Both blocks are copied verbatim into the report.

## 15. Seed hypotheses from planning discovery (verify at BASELINE; none is a finding yet)

| Seed | Hypothesis (planning source) | Workstream |
|---|---|---|
| S-01 | DIST-SCALE-X has dual status: roadmap §3.0.6 L963 "IN PROGRESS … Parent OPEN" vs L489/L564/L747/L2155 "CLOSED — OPERATOR SCOPED ACCEPTANCE" | A1/A2 |
| S-02 | PROD-Q has dual status: L484 `[x] CLOSED BY OPERATOR — SCOPED` vs L259/L564 "OPEN/BLOCKED", "NEXT / CURRENT" | A1/A2 |
| S-03 | TOOL-LIFE-X and SKILL-X: L751/L635 claim "external independent review", while L488/L487 say no review artefact (F-03), with AUD-TOOLLIFE CONDITIONAL/STOP. The next stage is given inconsistently (EBH-5 vs DIST-SCALE-X). P0 SHA prefix `76a98cb3` collides with the SKILL-X publication SHA | A2 |
| S-04 | The ARCH-FREEZE L499 requirement of "independent exact-SHA evidence" for SKILL-X, TOOL-LIFE-X and DIST-SCALE-X is not met by the recorded evidence (conditional / conditional-STOP / operator-only) | A2/A9 (potential BLOCKER) |
| S-05 | `[ ] READY FOR AUDIT` rows under CLOSED parents (L401 `5f348257`, L410) and ledger CLOSE/RFA rows (L2027, L2030, L2034, L2117, L2124, L2145) conflict with "mandatory OPEN = 0" and the §5 rule at L1923 | A1 |
| S-06 | Checkbox misuse (`[x]` on REJECTED/SUPERSEDED, L436/437/446/450/477); off-vocabulary statuses; malformed ledger rows L2075–2078; placeholder at L2172; self-dependency text at L570–574; stale NEXT/CURRENT text (L397, L486, L631, L1647–1654, L1952, L2009, L2019, L2053, L2161) | A1 |
| S-07 | FRZ checklist: 8 off-vocabulary statuses ("PASS — SCOPED" ×6, "PASS (EVOL-X SCOPED)", "PASS — COMPENSATING (SCOPED)") against vocabulary L25–34; stale L47 sentence "all FRZ-AUD OPEN" vs rows PASS; table broken by prose L151–183 (rows 184–337 have no header) | A3 |
| S-08 | `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` is stale vs the checklist on 21 rows (AUD-01..10, EVOL-01..10, FRZ-11). `ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md` says "130 OPEN" | A3 |
| S-09 | FRZ cardinality "22 families / 168" (roadmap L1939) vs "225 / 26" (L1880, L2363) is never reconciled; PDC requirements 172 rows vs `PDC_INDEX` L32 "168" | A3/A8 |
| S-10 | The HIST-85 FRZ dispositions and 62 worker findings were never independently audited (HIST independent audit register §3), yet may be relied on downstream | A3-T03/A2 |
| S-11 | PROD-Q-P0-R1×8 "OPERATOR ACCEPTED" with "no independent exact-SHA audit"; DIST-SCALE-X "no Protocol 03 audit"; EBH-5 "51 rows NOT YET QUALIFIED" | A2 |
| S-12 | HIST-F-BND-03B (HIGH): `intergrax/applications/_shared/product_manifest_registry.py` dynamically imports Tier-3 manifest modules from Tier-0, a potential breach of the AGENTS.md hard boundary. `FRZ-BND-03` is OPEN | A4/A9 (potential BLOCKER) |
| S-13 | HIST-F-DUR-01: hard-coded `is_durable=True` on KV/document pin stores (qualification honesty, `FRZ-PRD-07`) | A4/A7 |
| S-14 | Unpublished SHAs cited as evidence (HIST registry "on origin/development: False": `0adb7338`, `10089121`, `29c38787`, `98b4bf51`), plus the HELD fix 030-C | A2 (R-SHA-2) |
| S-15 | ENT-AUDIT semantic/matrix gates are env-gated (`ENT_AUDIT_X_SEMANTIC=1`, `ENT_AUDIT_X_GATE_MATRIX=1`) and may skip in default runs; the debt-register gate checks existence only; the NPSC-4.2 allowlist is recorded as FAIL in DUP-X / ROADMAP-REPLAY-X | A6 |
| S-16 | Ticket approval validators accept any line starting `Approved:` (ticket README), so delegate-form assurance is process-only | A6/A7-T04 |
| S-17 | ARCH-FREEZE operator decisions OD-EVOL-11, OD-EVOL-FORMAL, OD-ENT-00/01/08 and OD-ARCH-FREEZE-01 are OPEN; OD-ENT-02 is BLOCKED_EXTERNAL; OPEN-BP-01 is under the compensating path | A7 |
| S-18 | MCP-SERVER-X is CLOSED (SCOPED) with F-P8-01 HIGH and R-P7-01..06 residuals open; the ENT-AUDIT-X ledger L2161 says "MCP-SERVER-X (not CLOSED)" | A2/A7 |
| S-19 | PDC P7 audits at `c08efe46`: D5 FAIL (R015), others PARTIAL. ARCH-FREEZE WBS notes make PDC-X CLOSED a precondition. EQO-A treats this as in-flight and records dependency health only | A8/A9 |
| S-20 | IDs referenced without row or section (PROD-Q-P1, PROD-Q-R1, D03-A/B/C, IMPL-01, QUAL-X-P1..P4/FINAL, EVOL-X-P0, STATE-X-R3..R6, GOV-X2 children, GATE-SFX-ENTRY, GATE-SEF-*); "ECAP" used in S01 without definition | A1 |

## 16. Risk and blocker handling

| Risk | Mitigation |
|---|---|
| Parallel sessions move `development` (PDC P7) | Detached worktree at `BASELINE_SHA`; R-SHA-4; delta annex at W5 |
| Parser misses rows (malformed tables, the headerless FRZ block) | C01/C04 use a second, independent raw-regex parse; differences are reviewed by hand |
| Subagent hallucinated evidence | Every evidence entry is `path@sha:line` and checked by C07 (`git cat-file`, line exists). The parent spot-reads all BLOCKER/HIGH rows |
| Anchoring on the planning seeds | Seeds are `UNVERIFIED_HYPOTHESIS` until checked at the baseline; C15 requires an explicit refutation path |
| A tool writes to tracked files | Porcelain before and after; E7 |
| Windows resource exhaustion | ≤3 subagents; one pytest at a time; no `uv run` fan-out (test-process budget rule) |
| Confirmed BLOCKER found mid-campaign | Record it immediately (E1 chat note to the Operator). Do not invalidate any certificate. Set owner and `next_action: OWNER_DECISION`. Continue the campaign |
| Secrets in evidence | E6; record path and line only |
| Overlap with PDC-X P7/P8 | PDC artifacts are inputs; no PDC verdicts; findings about PDC go to owner "PDC-X" |

## 17. Deliverable registry

Working location: `$WS/deliverables/`. Publication location: `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_a/`.

| # | File | Produced by | Source of truth |
|---|---|---|---|
| D01 | `ENTERPRISE_STAGE_INVENTORY.md` | A1-T04 | `A1/stage_rows.jsonl` |
| D02 | `ENTERPRISE_CERTIFICATION_EVIDENCE_MATRIX.md` | A2-T05 | `A2/*.jsonl` |
| D03 | `ENTERPRISE_FRZ_RECONCILIATION_MATRIX.md` | A3-T04 | `A3/frz_rows.jsonl` |
| D04 | `ENTERPRISE_ARCHITECTURE_BASELINE.md` | A4-T05 | `A4/*.json(l)` |
| D05 | `ENTERPRISE_CERTIFICATION_DRIFT_MATRIX.md` | A5-T05 | `A5/*` |
| D06 | `ENTERPRISE_AUDIT_COVERAGE_GAP_MATRIX.md` | A6-T05 | `A6/*` |
| D07 | `ENTERPRISE_RESIDUAL_RISK_REGISTER.md` | A7-T05 | `A7/*` |
| D08 | `ENTERPRISE_DOMAIN_COMPLETENESS_MATRIX.md` | A8-T04 | `A8/*` |
| D09 | `ENTERPRISE_BASELINE_AUDIT_REPORT.md` | A9-T04 | All |
| D10 | `ENTERPRISE_BASELINE_CLOSURE_MATRIX.md` | A9-T05 | Checker output |
| D11 | `ENTERPRISE_CAMPAIGN_B_C_HANDOFF.md` | A9-T06 | `FINDINGS.jsonl` |
| D12 | `CAMPAIGN_A_CHARTER.md`, `SOURCE_REGISTRY.md`, `AUTHORITY_CONFLICT_REGISTER.md`, `SCOPE_EXCLUSIONS.md`, `FINDINGS_REGISTER.md` (rendered from JSONL), `TOOL_RUN_LOG.md`, `TEST_EXECUTION_RECORD.md`, `READ_LOG.jsonl`, `BASELINE.json`, `MANIFEST.json` | W0/A9 | — |
| D13 | `data/` (all JSONL/JSON sources), `tools/` (parsers, checker, renderer; stdlib only; published as non-product evidence tooling so the auditor can replay) | All | — |
| D14 | `EQOA_INDEPENDENT_AUDIT_HANDOFF.md` (W6) | H-T01 | — |
| D15 | `EQOA_INDEPENDENT_AUDIT_REPORT.md` + `eqoa_audit_session_manifest.json` (by the auditor) | Auditor | — |

## 18. Acceptance criteria and campaign termination

EQO-A is **ready for independent acceptance** (end of W5) when all of the following hold:
- `verify_campaign_a.py` exits 0 with C01–C15 all PASS.
- D01–D13 exist and carry identical headers.
- `NOT-ASSESSED` = 0.
- Every stage has an evidence status.
- Every FRZ ID is present.
- Every gap and limitation is classified and has an owner.
- Every finding is traceable.
- The B/C handoff is complete.
- The dossier is reproducible from `data/` + `tools/` + git at `BASELINE_SHA`.

EQO-A is **complete** when:
1. the independent auditor verdict is `PASS` or `PASS_SCOPED`, or `CONDITIONAL` resolved within at most 2 corrective loops;
2. the publication PR is open with the auditor report included; and
3. the Operator has merged it.

Completion states only that the baseline was inventoried, evidence-graded and independently checked within the declared scope. It does not state that the platform is defect-free, and nobody may describe it that way.

## 19. Independent audit handoff (W6)

- **H-T01.** Sonnet writes `EQOA_INDEPENDENT_AUDIT_HANDOFF.md` with:
  - scope, `BASELINE_SHA`, and the `MANIFEST.json` digest;
  - the replay instructions:
    1. create your own detached worktree at `BASELINE_SHA` under `_wt/intergrax/eqo-a-aud`;
    2. copy `$WS` read-only into `_wt/intergrax/_scratch/eqo-a-aud/`;
    3. run `tools/verify_campaign_a.py` and check that the digests match;
  - the mandatory sample: 100% of BLOCKER/HIGH findings; 10% (minimum 10) of A2 T1 rows chosen by the auditor; 5% of FRZ rows; 3 drift rows; every conflict-register entry; every seed refutation;
  - the allowed tools (section 4) and a separate auditor test budget of at most 5 runs (the auditor's own; also covered by OD-EQOA-03);
  - the expected outputs: D15, an IIAC-style manifest (`stage_id: EQO-A`, `certification_profile: iiac-v1`, verdict on the dossier, `limitations`), and a handoff record per `IIAC_SESSION_HANDOFF_RECORD.template.md`;
  - the auditor role and "write only findings; do not edit the dossier".
- **H-T02.** Sonnet proposes the auditor session (`start_session`, `own_initiative`, context `fresh`, brief = the handoff file path). The Operator starts it (or delegates). Sonnet never audits itself and never continues as the auditor.
- **H-T03.** Corrective loop. Sonnet fixes **data** (not verdicts the auditor did not contest), re-renders, re-runs the checker, and appends `CORRECTIONS_LOG.md`. The auditor re-reads only the delta. At most 2 loops, then E8.
- **W6-P.** Publication (OD-EQOA-02). In the main checkout's repository, create a branch from `origin/development` in a **new worktree** (not the main checkout): `git -C $MAIN worktree add -b eqo-a/publication $WT_BASE/eqo-a-pub origin/development`. Then:
  1. Copy `$WS/deliverables`, `data/`, `tools/` and the auditor outputs into `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_a/`. Docs only; no other path.
  2. Commit with no AI attribution. No `Ref:` trailer is needed (no product path).
  3. Push with upstream set per memory: `git branch --set-upstream-to=origin/development` and then `git push origin eqo-a/publication:eqo-a/publication`.
  4. Open a PR with `gh pr create` against `development`.
  5. Verify the PR head equals the pushed SHA before asking for the merge.

  The Operator merges. Afterwards: `git worktree remove` for the `eqo-a-*` worktrees; the main checkout is still on `development`.

## 20. Self-review (STEP 6) — gaps found and how the plan closes them

1. **Not specified in A1–A9:** source authority, tooling and SHA mechanics → added A0 (W0-T01..T07).
2. **About 253 IDs are too many for a full 12-question review** → T1 (about 50 parents, full) and T2 (all closed rows, mechanical). Child rows inherit explicitly; nothing is silently skipped (C01–C03).
3. **Prior assurance was never itself audited** (HIST-85, worker findings) → A3-T03 sample + S-10.
4. **IIAC/delegation provenance can be faked by text** → A2-T03, A2-T04, A7-T04, S-16.
5. **No known-red, N/A or certification-debt registers** → A7-T03 builds static inventories; absences become findings.
6. **Obligations bouncing between stages** → A7-T02 carry-forward graph.
7. **Concerns missing from the 20** (Decision System, HITL, Tier-3 hosting, evidence plane) → A4-T01 adds 4; A4-T02 package census feeds A8 orphan detection.
8. **Roll-up PASS hiding contradictions** → C08 recomputes; renders from data only.
9. **Inventory vs strength confusion** → separate checker blocks; C09 strength gate.
10. **Parser fragility** (headerless FRZ block, malformed ledger rows) → independent second parse (C01/C04).
11. **Mixing SHAs from a moving branch** → detached worktree, R-SHA-1..5, C06, delta annex.
12. **Duplicating PDC-X** → PDC is an input only (B10, A8-T03, S-19).
13. **Resource limits on Windows** → ≤3 subagents, one pytest, budget ledger (C14).
14. **Remaining accepted limitations:**
    - Dynamic behavior beyond 10 runs is not re-executed; it is handed off as `UNVERIFIED_HYPOTHESIS`.
    - Session-identity independence is `process_enforced` only.
    - `gh`-based branch-protection proof depends on local auth.

## 21. Ready-to-run instruction for Sonnet Agent Mode

```text
ROLE: EQO-A executor (Delivery DevOS role: orchestrator). Load /intergrax-agent-roles first.
PLAN (binding, read fully once): agent-env/plans/pasted-content-id-c041-intergrax-jolly-newell.md
Reply to the Operator in Polish; every file you write is in English.

AUTHORITY YOU HAVE (OD-EQOA-01..03, 2026-11-15): pin BASELINE_SHA = origin/development at W0 start in a
detached worktree under <parent>/_wt/intergrax/; write only under <main>/.tmp/session/eqo-a/ and
<parent>/_wt/intergrax/_scratch/eqo-a/; run the section-4 allowed tools; up to 10 bounded pytest runs
(section 8); spawn ≤3 subagents at a time with disjoint shard outputs; after the independent auditor's
verdict, one docs-only publication branch + PR (section 19 W6-P).

YOU MUST NOT: edit any tracked file before W6-P; checkout/reset/stash in the main checkout; promote FRZ,
close stages/tickets, change roadmap/checklist/ledger/IIAC records; run forbidden tools (section 4);
run broad suites or regression tiers; declare PASS/certified/independently accepted; audit your own dossier.

EXECUTE: W0 → W1 → W2 → W3 → W4 → W5 without asking for approval between waves. At each wave end run
tools/verify_campaign_a.py and do not start the next wave until its gate passes (fix data, never relax a check).
Use mcp__ccd_turn__update_plan with one step per wave. Keep chat updates to one line per wave plus any
escalation (section 5 E1–E8). On E1 (confirmed BLOCKER ≥E3) notify immediately and continue.

W6: write EQOA_INDEPENDENT_AUDIT_HANDOFF.md, propose the auditor session with start_session
(own_initiative, fresh), then STOP and wait. Apply ≤2 corrective loops on auditor findings. Then W6-P
publication PR; verify PR head == pushed SHA; ask the Operator to merge. Remove your worktrees afterwards and
confirm `git branch --show-current` = development in the main checkout.

FINAL CHAT REPORT (terse, ≤12 lines): BASELINE_SHA, checker result, finding counts by class × severity,
potential ARCH-FREEZE blockers (IDs), test budget used, auditor verdict, PR link.
```
