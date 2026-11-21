# EQO-B — Campaign B: Critical Enterprise Invariants Audit — execution plan

Plan author: planning session (Opus, PLAN MODE, 2026-11-17). Executor: a NEW Claude Sonnet session in Agent Mode ("EQO-B-EXEC"). Independent auditor: a separate NEW Opus session ("EQO-B-INDEPENDENT-AUDITOR" = WP-12 node PT-BA). This file is a plan, not evidence, not a certificate, and closes nothing.

Operator decisions taken during planning (2026-11-17, chat, Polish; English rendering by the planner):
- **OD-EQOB-P1:** EQO-B is the single, canonical, independent Campaign B. It is bound into the existing ARCH-FREEZE DAG (WP-12 nodes PT-B01/PT-B02/PT-T01/PT-B1W/PT-B2W/PT-BA/PT-D13). The B-1/B-2 "alpha" runs of 2026-11-16 are treated as **non-independent input evidence** that must be reproduced independently. Any change to the content of those DAG nodes still needs the ARCH-FREEZE owner/Operator (EQOB-AR-11).
- **OD-EQOB-P2:** `CAMPAIGN_B_BASELINE_SHA` = tip of `origin/development` frozen at W0; `258bbf6ab…` remains the historical baseline for reproducing Campaign A findings; the RW-train head (G2) and FCS are the delta-revalidation targets.

---

## 1. Executive summary

Campaign A (EQO-A, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, dossier PR #589, auditor verdict PASS_SCOPED on the dossier) handed 51 findings to Campaign B and 2 to Campaign C. Since then:
- A-REM (EQO-A remediation) is planned and partly delivered as tickets T-IGX-042..047 (docs, registers, validators); its product fixes (FC-1..FC-4, W-A..W-F) run as ONE product train `arch-freeze/rw-train` under the ARCH-FREEZE program MEP-AF2 v2.1 (`docs/project/maintainers/qualification/arch_freeze_program/`). SCOPE passed on 2026-11-17; lane 0 (PT-WA, T-IGX-043-G, draft PR #642, branch `af2/pt-wa`) is in progress; rw-train has 0 product files different from development yet.
- WP-11 inside MEP-AF2 already pre-registered and ran a **B-1 alpha** (PT-B01, 39 rows, 25 CONFIRMED) and **B-2 alpha** (PT-B02) at `258bbf6ab`, DIAGNOSTIC and NON-CERTIFYING, executed by the ARCH-FREEZE program, results only in gitignored `.tmp/session/arch-freeze/b1-alpha|b2-alpha/`. Alpha CONFIRMED BLOCKS-class defects: A4-027 (LKW run tenant authority), A8-020 (foreign-tenant documents), X-A4-016 (tier routing records shared across tenants), A4-007 (dynamic loader seams), X-A4-004 (registry bypass), X-A2-037 (no live required checks).

EQO-B therefore: (1) independently reproduces and **extends** the alpha evidence at the new baseline `B`; (2) adds closed-world discovery and new adversarial hypotheses (planning recon already found ~24 candidate hypotheses, section 6); (3) hands product-change defects to the Operator **before G2** where possible (WP12-Q03: after SCOPE a product change means RB-3 or a second wave by Operator decision); (4) revalidates everything at FCS (omega = PT-B1W/PT-B2W) with red-first proof of the tracked negative tests (PT-T01); (5) is audited by PT-BA; (6) publishes a digest-only dossier (PT-D13) under `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_b/`.

Campaign B is optimized to find real violations, not to produce PASS. A finished, independently accepted Campaign B may (and probably will) report that critical invariants are violated.

## 2. Actual repository state (planning snapshot — executor re-derives in W0)

| Item | Value at planning |
|---|---|
| `PLANNING_HEAD_SHA` (origin/development) | `2727a2c2364a97492202b931375772a400279a9b` (PR #645) |
| Main checkout | `development` @ `76516c4dd` (behind; **read-only for EQO-B: no pull/checkout/stash there**) |
| Product delta `258bbf6ab..2727a2c23` (`intergrax agents applications`) | 2 commits, 3 files: `intergrax/rag/graph/{generation_visibility.py, indexer/heuristic_graph_indexer.py, providers/inmemory_graph_store.py}` |
| IIAC activation | `IIAC_POLICY_ACTIVATION_RECORD.json` IN_FORCE; audit_execution_sha `a73d15ab3`; 0 product/material changes `a73d15ab3..origin/development` |
| `origin/arch-freeze/rw-train` | `c482c4345`, 6 ahead / 29 behind development, 0 product files differ |
| PT-WA (lane 0) | `origin/af2/pt-wa`, PR #642 draft, ~45 product files (12 runtime modules inversion, agent seams, LKW facades) |
| ARCH-FREEZE | not entered; OD-ARCH-FREEZE-01 OPEN; SCOPE passed 2026-11-17 |
| PDC-X | roadmap: CLOSED (IIAC SCOPED); `PDC_INDEX.md` L44–49 still "awaiting Operator final confirmation" (conflict → record, do not resolve) |
| FRZ checklist | 225 rows / 26 families: OPEN 109, PASS 108, off-vocabulary 8 (PASS — SCOPED ×6, PASS (EVOL-X SCOPED) ×1, PASS — COMPENSATING (SCOPED) ×1); regex `^\| *FRZ-[A-Z0-9]+-[0-9]+[a-z]? *\|`, status = 5th cell |
| EDER | DRAFT, 1069 rows all `INGESTED` (239 FINDING, 636 DEBT_ENTRY, 194 EXCEPTION); impact BLOCKS 112 / CONDITIONS 325 / NONE 147 / UNKNOWN 485 |
| Campaign A findings | 240 rows (239 OPEN + 1 MERGED); BLOCKER 3, HIGH 5; next_action CAMPAIGN_B 51, CAMPAIGN_C 2 |
| Pytest lock | `D:/Projekty/_wt/_locks/pytest.lock` (mkdir lock, ≤15 min hold, one pytest process machine-wide) |

## 3. Canonical source registry (read at `B`; record sha256 per file in `SOURCE_REGISTRY.json`)

Read with `git show <B>:<path>` or from the B worktree; read hubs by grep/offset, never whole.

| Group | Path(s) | Use |
|---|---|---|
| Roadmap | `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` (L499 entry conditions, L572, L2155–2164 ledger) | stage status, entry conditions |
| FRZ | `docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` (L66 = only Path B authority, DP-34) | criteria, owners |
| Campaign A | `…/enterprise_quality_oversight/campaign_a/` — README, CAMPAIGN_A_CHARTER, MANIFEST.json, FINDINGS_REGISTER, `findings/FINDINGS.jsonl`, `briefs/COMMON.md` (L17 E-levels, L21 schema, L22 severity), `deliverables/*` (all 11 listed in the brief + STAGE_INVENTORY), `audit_replay/EQOA_INDEPENDENT_AUDIT_REPORT.md`, `handoff/CORRECTIONS_LOG.md`, `tools/*.py.txt`, `logs/`, `A4/boundary_scan.md` | input findings, method, tools |
| A-REM | `…/qualification/eqo_a_remediation/{OPERATOR_DECISIONS_EQOA_REMEDIATION.md, OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md (addenda 1–4), README.md}`; tickets `governance/tickets/T-IGX-04[0-7]-*.md`, `T-IGX-050-*.md`, `governance/tickets/plans/PLAN_T-IGX-*.md` | OD-REM-01..08, C1–C10, DP-01..68, ticket write sets |
| ARCH-FREEZE | `…/qualification/arch_freeze_program/{MEP-AF2_v2.1.md, WP12_DAG.md, WP12_TICKETS_DRAFT.md, WP12_AUTHORIZATION_REQUESTS.md, WP12_OPEN_QUESTIONS.md, RM499_ENTRY_MAP.md, KNOWN_RED_REGISTER.md, known_red_register.v1.json, ECAP_REVISION_DRAFT.md, manifests/}`; `…/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` | DAG nodes, SHA ladder (AUDIT_BASELINE_2 → G2 → R → FCS → M3), known-red set, authorizations |
| EDER | `…/qualification/enterprise_debt_register/{README.md, register_schema.json, register.jsonl, reconciliation.json, exclusions.jsonl, owner_vocabulary.proposed.json}` | owner/impact vocabularies; new rows only as `EDER-N-*` proposals |
| IIAC | `…/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `IIAC_AMENDMENT_01_*`, `IIAC_POLICY_ACTIVATION_RECORD.json`, `scripts/governance/iiac_*.py`, `verify_iiac_*.py` | role separation, ACT-N11 |
| ECAP / ENT-AUDIT-X | `…/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md` (§2.C L160, §13 classes L1017, §15 L1088); engine `intergrax/platform/enterprise_audit/` (runner, modes, changed_since, impact_map, `data/`); CLI `scripts/enterprise_audit/run_enterprise_audit.py`, `build_impact_map.py` | assurance under test (WB11) |
| Architecture canon | `docs/project/architecture/{INTERGRAX_ARCHITECTURE_PRINCIPLES, PLATFORM_FOUNDATION, GOVERNED_EXECUTION, AGENT_CONTRACTS_AND_ASSEMBLY, ENTERPRISE_RELIABILITY_LAYER}.md` (+ `satellites/*_production_gates.md`), plans pairs under `docs/project/maintainers/plans/` | canonical contracts/owners |
| Stage certifications | grep `docs/project/maintainers/qualification/` for TENANT-X, GOV-X, CONFIG-X, COMPAT-X, EVOL-X, TRACE-X, STATE-X, CTRL-X, DUP-X, ROADMAP-REPLAY-X, TOOL-LIFE-X, SKILL-X, DIST-SCALE-X, EBH-5/6/7, MCP-SERVER-X, PDC-X (`platform_domain_completeness/`), PROD-Q (`PROD_Q_R1_R7_*`), `VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md` | claims to challenge |
| Alpha evidence (non-tracked) | `<MAIN>/.tmp/session/arch-freeze/b1-alpha/` (PREREG.md sha256 `f9f6532003a9…`, RESULTS_B1_ALPHA.md, probes/, runs/*/SHA256SUMS, runner.py) and `b2-alpha/` (PREREG sha256 `2b7d3f07b453…`); erratum: B-1 table titles for A4-026/027/031/037 are shifted — PREREG and probe docstrings are authoritative | reusable probes + E3 input |

Discovery of further sources: `git grep -n -l` over `docs/project/maintainers/qualification` and `docs/project/capabilities` for each stage ID; record every source actually used. No alternative document hierarchy; on conflict apply COMMON.md L25 authority order and record in `AUTHORITY_CONFLICT_REGISTER.md`.

## 4. Campaign A evidence reconciliation (rules)

1. Campaign A results are **historical at `258bbf6ab`**; nothing in them is a current-state claim.
2. For each handed-off finding: re-read the cited `path@sha:line` evidence, check whether the cited files changed `258bbf6ab..B` (`git diff --name-only`) → `code_state ∈ {UNCHANGED, CHANGED, REMOVED}` (UNCHANGED = byte-identical path, not "still true").
3. Alpha results: verify `SHA256SUMS` of each run dir and PREREG sha256 before use; classify `evidence_origin = ALPHA_NON_INDEPENDENT`, max E3; an EQO-B verdict needs its own rerun (E4) or own code analysis (E3).
4. Dossier digest check: recompute sha256 of Campaign A `MANIFEST.json` entries actually used (detect post-publication edits).
5. Record Campaign A severity and impact unchanged in `orig_*` fields; EQO-B ratings are separate fields (no silent re-rating).

## 5. Complete mapping of the 51 Campaign B findings

Legend: **P** = B-1/B-2; **WB** = primary (secondary); **α** = alpha class at `258bbf6ab` (C=CONFIRMED, NR=NOT_REPRODUCED, I=INCONCLUSIVE, –=not run); **Vehicle** = A-REM/ARCH-FREEZE work that changes the surface (to be verified in W0 by grep of tickets/plans/DAG/PR branches); **CI status** = initial change-impact class; **Method** = EQO-B primary method (details in WB sections). Original severity/class/E-level/impact/owner/hypothesis/evidence come verbatim from `FINDINGS.jsonl` into `CAMPAIGN_B_SCOPE_MATRIX` (not repeated here).

| # | Finding | Sev | P | WB | α | Vehicle | CI status | Method (expected SAFE behaviour) |
|---|---|---|---|---|---|---|---|---|
| 1 | A4-006 tenant event gate cannot run | M | B-1 | 02 (10) | C | PT-T01/AB-19 | ACTIVE_CHANGE | run gate at B; independent AST census of every RuntimeEvent emitter for tenant_id propagation (safe: every emitter on a tenant-scoped path sets tenant_id) |
| 2 | A4-007 config-driven dynamic loaders | L | B-1 | 05 | C (3/5) | PT-WA part D, DP-21, PT-G01 S11 | ACTIVE_CHANGE | trace provenance of every import string (22 sites); adversarial config loading a Tier-2/3 module (safe: rejected outside sanctioned registries) |
| 3 | A6-007 CHANGED-SINCE unmapped path | M | B-1 | 11 | C | PT-WD (DP-66: unmapped ⇒ FULL) | ACTIVE_CHANGE | synthetic new product package in scratch copy (safe: selects FULL) |
| 4 | A7-037 164 skip/xfail markers unregistered | M | B-2 | 11 | – | DP-53 EDER SKIP_CATEGORY | STABLE_FOR_AUDIT | AST census of markers + reasons; classify env-gated / optional-dep / no_ci; prove which hide critical invariants |
| 5 | A1-008 roadmap FRZ links not in checklist | L | B-2 | 11 | – | PT-D09/D10 | WAIT_FOR_DEPENDENCY | key-alias search; doc-level |
| 6 | A1-012 COMPAT-X-R5 single gate | L | B-2 | 06 | C (rule) | none | STABLE_FOR_AUDIT | COMPAT gates once at B and FCS (E4) |
| 7 | A1-015 PROD-Q-P0 untracked log | L | B-2 | 11 | I | PT-D03 | STABLE_FOR_AUDIT | reproduce totals from per-file counts; else EVIDENCE_INSUFFICIENT |
| 8 | A1-021 FREEZE-CONTROL-01 audit artefact absent | M | B-2 | 11 | – | PT-D04 relabel | WAIT_FOR_DEPENDENCY | repo/history search (`git log -S`), record result |
| 9 | A2-043 W4 permit/cancellation races drift | M | B-1 | 01 (07) | NR (gates) | none | STABLE_FOR_AUDIT | concurrency probes: permit revoked/cancelled between admission and side effect (safe: no effect) — beyond existing gates |
| 10 | A2-009 EBH-3 `integrations._shared` leakage | M | B-1 | 05 | – | PT-WA S8b, PT-G01 (DP-20) | ACTIVE_CHANGE | AST scan agents/ + applications/ (static + dynamic) vs exception list |
| 11 | A2-047 15 security qualification tests red | M | B-1 | 04 | C (3 of 15) | KNOWN_RED_REGISTER | STABLE_FOR_AUDIT | rerun 15 nodes at B/FCS; root-cause each red (fixture vs real defect) |
| 12 | A2-049 TRACE-X P6 classifier defaults dirs | M | B-1 | 10 (07) | C | PT-Q06 | REVALIDATION_REQUIRED | synthetic unclassified restart file (safe: classifier fails) |
| 13 | A2-019 TRACE-X-P6 restart/resume drift | M | B-1 | 07 (10) | C | PT-Q06 | REVALIDATION_REQUIRED | q02/q10 + 7 red nodes root-cause; behavioural restart probe |
| 14 | A2-051 COMPAT-X R2 47 failed accepted | M | B-2 | 06 | C (2 red) | none | STABLE_FOR_AUDIT | map each red to owner/fix commit |
| 15 | A2-023 PROD-Q P0/P2/P3 no independent audit | M | B-2 | 03 | NR | DP-41 relabel | STABLE_FOR_AUDIT | record-level; P0 replay at FCS |
| 16 | A2-024 PROD-Q P0 inventory not replayed | M | B-2 | 03 | NR | none | REVALIDATION_REQUIRED | P0 gates at FCS |
| 17 | A2-027 SKILL-X drift + escaped PROD-Q regression | M | B-1 | 09 | – | PT-FC4, DP-44 | ACTIVE_CHANGE | diff certified surface; targeted SKILL-X replay; L499 link |
| 18 | A2-030 DUP-X/RRX acceptance on absent artefacts | M | B-2 | 11 | I | PT-D03 | WAIT_FOR_DEPENDENCY | evidence packaging check |
| 19 | A2-031 NPSC-4.2 allowlist semantics | M | B-2 | 11 (05) | – | none | STABLE_FOR_AUDIT | run NPSC-4.2 gate; review relaxation (promote to B-1 if a boundary is unguarded) |
| 20 | A3-011 hasattr probing in vectorstore | L | B-2 | 06 (08) | – | OD-AF-11 TYP ratchet | STABLE_FOR_AUDIT | classify probes semantic/non-semantic |
| 21 | A3-019 EXE/GOV/CTL/REC/HRN PASS at cert-time SHAs | M | B-1 | 01 (07) | NR (gates) | PT-Q12 | REVALIDATION_REQUIRED | behavioural negative probes per row family (gates passing ≠ invariant) |
| 22 | A3-027 FRZ-PRD-07 strict path | M | B-1 | 03 | NR | PT-FC4 | REVALIDATION_REQUIRED | strict fixture replay D4/P2-N02 |
| 23 | A4-044 host-task router ignores production_mode | M | B-1 | 01 (03) | C router | PT-FC1 (DP-26/27) | ACTIVE_CHANGE | E2E via canonical host: non-production agent on host-task path in PRODUCT (safe: rejected) |
| 24 | A4-015 relaxed production_mode sites | M | B-1 | 03 (01) | C | PT-FC1 | ACTIVE_CHANGE | enumerate all `production_mode=False` reaching RuntimeContext on product paths (10 sites + 33 defaults) |
| 25 | A4-020 provider imports + enum switches | M | B-1 | 06 (05) | – | PT-WB (DP-18) | ACTIVE_CHANGE | classify 65+16 files: sanctioned root / constant-only / bypass |
| 26 | A4-025 CONFIG-X ambient env reads | M | B-1 | 06 (03) | – | PT-FC3 + T-IGX-043-F | ACTIVE_CHANGE | full census (27 runtime, 179 applications lines); security-relevant ones under PRODUCT |
| 27 | A4-026 inbound verifier no-op | M | B-1 | 03 (04) | C | PT-FC3 (DP-25) | ACTIVE_CHANGE | unsigned/forged Slack/Teams request on PRODUCT intake (safe: 401/403, no execution) — see H-01 |
| 28 | A4-027 Tier-3 client tenant_id + 'default' | M | B-1 | 02 | C (BLOCKS) | PT-WF (G-09, DP-23) | ACTIVE_CHANGE | tenant-from-payload probes on all 9 routers + detector kw-arg/field-default gap |
| 29 | A4-031 UAEP skips context assembly | L | B-1 | 08 (03) | C (probe v6) | PT-FC1 conditional | ACTIVE_CHANGE | PRODUCT composition without engine/adapter (safe: fail closed) |
| 30 | A4-034 dormant retry budgets | L | B-2 | 07 | – | DP-32 docs | STABLE_FOR_AUDIT | confirm dormancy (no product caller) |
| 31 | A4-037 Nexus HITL persistence no-op | M | B-1 | 01 (10) | C | PT-FC1 | ACTIVE_CHANGE | reject/escalate run E2E; durable record assertion |
| 32 | A4-039 two ProofReceipt models | L | B-2 | 10 | – | PDC-D8 | STABLE_FOR_AUDIT | + H-09 (mutable, untenanted) — promote if evidence can be forged |
| 33 | A4-040 direct uvicorn.run | M | B-1 | 03 (04) | – | PT-WB (DP-19) | ACTIVE_CHANGE | prove PRODUCT profile cannot be served via direct entrypoints; default bind |
| 34 | A4-042 globals Poland defaults | L | B-2 | 06 | – | none | STABLE_FOR_AUDIT | census of import-time defaults |
| 35 | A4-045 artifact ref vocabularies | L | B-2 | 10 | – | PDC-D8 | STABLE_FOR_AUDIT | mapping + digest requirement per ref |
| 36 | A4-047 scenario contracts owner | M | B-2 | 12 | – | T-IGX-042-A | STABLE_FOR_AUDIT | ownership map only |
| 37 | A5-002 stream paths bypassed provider guard (repaired) | M | B-1 | 08 | – | PDC D3 | STABLE_FOR_AUDIT | census of every provider call path (stream/non-stream/structured); see H-06 |
| 38 | A5-003 GOV-X1 gates rewritten | L | B-1 | 11 (01) | – | none | STABLE_FOR_AUDIT | diff GR-10/GR-12 gates `27af801..B` for removed assertions |
| 39 | A5-012 STATE-X/TRACE-X gates not re-run | L | B-1 | 07 (10) | C (7 red) | PT-Q06 | REVALIDATION_REQUIRED | rerun at B and FCS; root-cause 7 red |
| 40 | A5-010 DUP-X not re-examined for 70 modules | M | B-2 | 05 | I | PT-Q05 | REVALIDATION_REQUIRED | 15-family pass over new modules (promote to B-1 if a duplicate authority/execution mechanism) |
| 41 | A5-015 ENT-AUDIT FULL predates 14 security paths | L | B-1 | 11 | – (omega) | PT-Q13 | INTEGRATED_VERIFICATION_REQUIRED | one FULL non-certifying run at FCS (coordinate, never duplicate PT-Q13 certifying run) |
| 42 | A7-003 DEBT-04/05 not zero inventory | M | B-2 | 11 | – | PT-R03 | WAIT_FOR_DEPENDENCY | inventory recount |
| 43 | A7-007 hardcoding/DUP counts pinned pre-baseline | M | B-2 | 06 | – | PT-Q05 | REVALIDATION_REQUIRED | rerun mechanical hardcoding/DUP gates at B |
| 44 | A7-009 GAP-TEN-UNK-05 structural N/A | M | B-1 | 02 | I (duckdb) | PT-Q07 | STABLE_FOR_AUDIT | install duckdb extra in EQO-B venv; E2E |
| 45 | A7-021 HIST-F-BND-03B hard boundary | M | B-1 | 05 | NR | DP-21 | STABLE_FOR_AUDIT | independent AST incl. dynamic/lazy/TYPE_CHECKING |
| 46 | A7-013 MCP exposure security R017 | M | B-1 | 09 (04) | NR (92 passed) | DP-45 | STABLE_FOR_AUDIT | adversarial MCP beyond existing tests (H-12) |
| 47 | A7-028 EBH-7 SCOPED with tenant-guard open | M | B-1 | 02 | – | PT-Q07/Q11 | WAIT_FOR_DEPENDENCY | trace W1-PROD-DELTA/ADR outcome; behavioural tenant guard probe |
| 48 | A7-030 audit obligations without owner/SHA | M | B-1 | 09 (11) | – | PT-Q08 | WAIT_FOR_DEPENDENCY | check TOOL-LIFE-X/SKILL-X audits scheduled/performed (record) |
| 49 | A7-031 EBH-7 eight mandatory skips N/A | L | B-1 | 02 (12) | NR + observation | PT-Q07 | REVALIDATION_REQUIRED | re-execute equivalent nodes; H-18 untenanted credential |
| 50 | A8-006 hubs without tenant rules | L | B-2 | 02 | – | DB-F2 | STABLE_FOR_AUDIT | sample FRZ-TEN-06 on context path |
| 51 | A8-020 LCI trusted tenant scope | M | B-1 | 08 (02) | C (BLOCKS) | PT-WF / RAG owner | ACTIVE_CHANGE | foreign-metadata documents on compat/legacy hit paths (safe: rejected) |

Totals: **B-1 = 31, B-2 = 20, 51/51.** Campaign C (A3-004, A6-004): EQO-B only supplies inputs (A6-004 → PT-WD coverage map evidence; A3-004 → PT-C01).

Promotion rule B-2 → B-1 (mechanical): a B-2 item is promoted when its evidence shows reachability in PRODUCT/STRICT of a security, tenant, governance, fail-closed, recovery/replay or hard-boundary violation, or when a linked FRZ row is BLOCKS under DP-01 (WP12-Q05 ordering BLOCKS > UNKNOWN > CONDITIONS > NONE).

## 6. Additional scope beyond the 51 (all in B-1 unless stated)

**6a. Related Campaign A findings (verification of A-REM fixes; EQO-B never repairs):** A4-003 (BLOCKER, SYS-INV-01, PT-WA), A4-004 (HIGH, registry bypass, PT-WA/PT-G01), A0-007 (HIGH, IIAC activation vs ACT-N11, RB-2), A3-012 (HIGH, TENANT-X TASK memory namespace), A5-007 (HIGH, RRX counters, PT-Q05), A2-001/A1-017/A2-002/A2-025 (L499 TOOL-LIFE-X/SKILL-X/DIST-SCALE-X independence — EQO-B checks evidence currency only; PT-Q08/Q09 own the re-audits), OD-REM-03 fail-closed set A4-009/016/017/019/024/030/032/035/036/002, validator set A6-002/A2-037/A2-038/A6-006 (WB11), A8-003 profile projection, A3-016 seams, A3-021 (41 OPEN OBS/RPL/SEC/REL/REG rows: C8 architecture-vs-live split check, WB12). Admin-only (impact tracking, no tests): A2-017, A7-005, A7-029.

**6b. Alpha-raised IDs (carry as `AX-*` rows):** X-A4-004, X-A4-016, X-A4-024, X-A5-011 (G-09 strict wiring tenant None), X-A5-007, X-A8-003, X-ESC-E1 (child authority narrowing), X-ESC-E2 (durable binder concurrent finalization), X-ESC-E4/E5 (debug surfaces cross-tenant), X-A2-037 (required checks absent), X-A6-003, X-A4-046 (VBN gate blind spot), X-A0-008 (F10 STALE), "R03 v6 extra", G-09b (INCONCLUSIVE), X-S2b, X-A8-022, X-TLX.

**6c. New hypotheses from planning recon (`H-*`, UNVERIFIED — executor must verify before any finding):**

| H | Hypothesis | WB |
|---|---|---|
| H-01 | `intergrax/runtime/interactions/router.py` POST /intake: `tenant` query default `"default"`, `execute` default True (runs NexusLoop); verifier Null unless env set; Slack/Teams verify is no-op unless flag AND secret set; mounted by 8 app factories | 03/04/02 |
| H-02 | 5 Tier-3 routers (attestation_demo, dispute_sim, intergrax_assistant, lab, poc_template) take `body.tenant_id` without auth cross-check; governed_contractor/legal fall back to body when no auth context; research L53 branch; LKW `workspace_routes.py:410-424` body wins; `functional_evidence_qualification_routes.py:26` tenant via query | 02 |
| H-03 | `strict_production_serving_execution_tenant_authority.py`: authoritative tenant = `application_environment_id`; control-plane tenant stored but never compared; binding optional in `harness_host_runtime.py:339/701`, passed only by governed_contractor | 02 |
| H-04 | `AgentRouter`: unknown `task.agent_id` silently falls through to capability match / `fallback_first_routable`, also in production_mode | 01/03 |
| H-05 | `llm_routing_wiring.py:108`: `env.llm_profile or LLMProfile.lab()` — PRODUCT host silently uses LAB LLM profile | 03/06 |
| H-06 | `ollama_adapter.py` `generate_messages` (L349) / `generate_structured` (L614) bypass `_execute` → no provider guard, no retry budget (regression class of A5-002) | 08 |
| H-07 | `fastapi_core` `NoOpJwtVerifier` default of `JwtAuthProvider(verifier=None)` → authenticated with tenant None; NoOp budget/rate-limit defaults (`app_factory.py:61,68`) | 04/03 |
| H-08 | `applications/_shared/integrations/persistence.py:454,709` Kv* stores `is_durable` hard True regardless of backend | 07 |
| H-09 | `intergrax/proofs/receipts/contracts.py::ProofReceipt` mutable, no tenant_id, free-form evidence | 10 |
| H-10 | `RuntimeEvent` payload `redact()` identity default; no secret-redacting logging filter; 3 divergent `_SECRET_PATTERNS` | 10/04 |
| H-11 | prompt-injection defence = substring match on user prompt only, opt-in; no untrusted markers on RAG/tool output | 08/04 |
| H-12 | MCP `LOCAL_DEV_EXPLICIT`: loopback check treats `"testclient"` as loopback; `run_agent` uses `asserted_tenant_id or default_tenant_id` ("mcp") | 09 |
| H-13 | `__import__` inside a route handler (`local_workspace_application/serving/workspace_routes.py:770`) | 04/05 |
| H-14 | `identity_wiring.py:57` tenant from env `INTERGRAX_HARNESS_TENANT_ID` | 02/06 |
| H-15 | MSE authorization port `None` when execution_mode ≠ strict; `RuntimePolicyEngine()` fallback with no rules ALLOWs decision/pre_llm/pre_output/interrupt | 01/03 |
| H-16 | `delegation_authority._intersect_parent_authorities` drops "unrestricted" authorities before intersecting — can a child obtain wider authority than a restricted parent? (with X-ESC-E1) | 01 |
| H-17 | delegation budget no-op unless `enforcement_enabled` | 01 |
| H-18 | untenanted credential ref resolves for any tenant (alpha A7-031 observation) | 02/04 |
| H-19 | `RuntimeEvent.tenant_id` Optional None default — events without tenant on tenant paths | 10/02 |
| H-20 | tenant fallbacks: retrieval metrics `"_platform"`, profile_resolution `"_global"`, collaborative_work schema `"public"`, dashboard falls back to `env.profile_id` | 02 |
| H-21 | debug routers (`intergrax/debug/*`, `_shared/*_routes.py` admin/replay/trace explorer/checkpoint introspection) mountable in PRODUCT hosts | 04/02 |
| H-22 | 33 `production_mode=False` defaults + hard-coded False in `context_manager.py:239`, `uaep_assemble.py:66`, `nexus_llm_plan_builder.py:31`, `nexus_plan_bridge.py:101`, `organization_worker_agent.py:114` | 03 |
| H-23 | `DurableArtifactRef` parse `retention_class or "default"`; artifact digest verification on every read path? | 10/04 |
| H-24 | idempotency: Redis/SQLite/in-memory idempotency stores — claim atomicity under concurrent workers | 07 |

## 7. B-1 / B-2 scope and prioritization

B-1 (31 + section 6 sets): every BLOCKS / BLOCKER / HIGH, every security, tenant, governance/authority, fail-closed, hard-boundary, recovery/idempotency/replay item, and production reachability of every bypass. B-2 (20 + promotions out): evidence, drift, vocabulary, ownership, VBN/compat breadth, family sweeps FS-1..FS-6.

Priority score (recorded per row, not a severity): `R = reach(PRODUCT_REACHABLE 3 | STRICT 3 | LEGACY_BUT_REACHABLE 2 | UNPROVEN 2 | LAB 1 | TEST/UNREACHABLE 0) × impact(auth bypass / cross-tenant / unauthorized external effect 3; integrity/evidence loss 2; other 1) + frz(BLOCKS row 2, CONDITIONS 1) + alpha_confirmed 1`. Ordering inside B-1: by R, then the **pre-G2 window** rule: items whose fix is a product change are executed first so the Operator can decide on adding them to the train before G2 (otherwise RB-3/second wave, WP12-Q03). B-2 never blocks B-1 (B-2 jobs only take the pytest lock when no B-1 job is queued). Every row, B-1 or B-2, gets an individual ARCH-FREEZE impact (C10).

## 8. Critical invariants catalogue (`CAMPAIGN_B_CRITICAL_INVARIANTS_REGISTRY.md`)

| ID | Invariant | Main FRZ | Owner |
|---|---|---|---|
| INV-GOV-01 | Proposal ≠ permission ≠ execution: no side effect without a fresh authorization decision bound to execution identity | GOV, EXE | GOVERNED_EXECUTION / GOV-X |
| INV-GOV-02 | Child authority ⊆ parent; unknown authority ⇒ deny | GOV, EXE | GOV-X |
| INV-GOV-03 | Revocation between admission and side effect prevents the effect; retry after revocation denied | GOV, REC | GOV-X / HARNESS-W4 |
| INV-GOV-04 | HITL mandatory where required; missing/late/replayed decision never approves; every decision durably recorded | GOV-06, AUD-04 | GOV-X |
| INV-GOV-05 | Single execution authority: MCP, scheduler, tools, apps, direct adapters cannot bypass governance | GOV, OWN, BND | Execution Engine |
| INV-TEN-01 | Tenant identity only from authenticated principal/binding; payload/query/header/env cannot set or override it in PRODUCT/STRICT | TEN-01/02/10 | TENANT-X |
| INV-TEN-02 | No default-tenant substitution in PRODUCT/STRICT | TEN-10 | TENANT-X |
| INV-TEN-03 | Every data plane (memory, RAG/vector/graph, state, cache, artifact, trace, evidence, credentials, provider config) partitions by tenant | TEN | TENANT-X + domain owners |
| INV-TEN-04 | Tenant context preserved across async, background, retry, serialization, recovery | TEN, REC | TENANT-X |
| INV-FC-01 | PRODUCT/STRICT: missing/invalid/unknown config ⇒ startup or call fails; no silent Null/No-op for security-relevant ports | CFG-02/06/08, SEC | CONFIG-X / PROD-Q |
| INV-FC-02 | production_mode/strict propagates end to end; no relaxed default on product paths | EXE-02, PLG-05 | Execution Engine |
| INV-SEC-01 | Inbound requests authenticated/verified in PRODUCT; invalid ⇒ reject with no side effect | SEC | Runtime interactions / PROD-Q |
| INV-SEC-02 | Untrusted content (RAG, tool output, plugin, user) cannot raise authority or bypass policy | SEC-05 | Context/Tools owners |
| INV-SEC-03 | No secrets in logs, errors, events, evidence, argv | SEC, OBS | Observability |
| INV-SEC-04 | No path traversal; artifact identity/digest verified on read | SEC, PRD | ARTIFACT_LIFECYCLE (PDC-D8) |
| INV-SEC-05 | No unauthorized dynamic component loading | BND-03, PLG | ARCH-FREEZE |
| INV-BND-01 | Tier rules hold for static, dynamic, reflective and DI paths | BND | ARCH-FREEZE |
| INV-BND-02 | One canonical owner per mechanism; no shadow registry / duplicate authority | OWN, DUP | DUP-X |
| INV-CFG-01 | Provider selection only via typed config + canonical registry inside named composition roots (DP-18) | CTR-02, PLG-04 | Integrations (EBH-2F) |
| INV-REL-01 | Crash before/after side effect ⇒ no duplicate effect, consistent recovery | REL, REC | ERL / HARNESS-W4 |
| INV-REL-02 | Leases/claims exclusive under concurrency and duplicate delivery | REL | ERL |
| INV-REL-03 | Durability declarations truthful | STA, REL | STATE-X |
| INV-TRC-01 | Every side effect has a trace with tenant + execution identity + correlation; no trace of an effect that did not happen | TRC | TRACE-X |
| INV-TRC-02 | Evidence immutable, digest-bound, tenant-scoped | TRC, AUD | PDC-D8 / PROOF_RECEIPTS |
| INV-MOD-01 | Every provider call path (stream, non-stream, structured) passes provider guard + usage enforcement | CTL-12 | Model Runtime (PDC-D3) |
| INV-MOD-02 | Canonical context assembly mandatory in PRODUCT (CE-01) | CTL-12 | Context Engineering |
| INV-ASR-01 | Assurance cannot silently skip: unmapped path ⇒ FULL, missing env ⇒ fail not skip, validators reject forged manifests/SHAs/approvals | AUD-03/10 | ENT-AUDIT-X / IIAC |

W1 maps each INV to FRZ rows (exact IDs) and to every scope row; an INV with no scope row gets one.

## 9. Threat model (`CAMPAIGN_B_THREAT_MODEL.md`)

Adversaries: **T1** unauthenticated network client; **T2** authenticated user of tenant A (malicious tenant); **T3** author of untrusted content (RAG document, tool output, plugin/skill package, webhook payload); **T4** mis-configuring operator (no malice: typo, missing var, wrong profile); **T5** crash/fault environment (process kill, storage error, duplicate delivery, lease loss); **T6** assurance gaming (an agent session that wants a green verdict: forged manifest, wrong SHA, skipped test, weakened gate).

Surfaces (from planning recon, executor completes in W1): Tier-3 HTTP routers (37 files, ~134 routes), `/intake` interactions, MCP (`_shared/mcp_*`, 8 `applications/*/mcp/server.py`), debug routers, background workers (`intergrax/queueing/*`, `runtime/long_running/*`, LKW `background_worker_main.py`, kafka ingest), tool invocation (`runtime/nexus/tools/*`), skills (`intergrax/skills/*`, dynamic acquisition), plugins (`core/plugins/discovery.py` entry points), integrations registries (`integrations/registry/*`, `runtime/integrations/registry_v2.py`), provider adapters (`llm_adapters/*`), RAG ingestion/retrieval, artifact import/export, workspace operations, credentials resolution, hosting (`intergrax/hosting/runner.py`), CLI (`intergrax/cli/run.py`, `intergrax/harness/app.py`).

Per surface × STRIDE: Spoofing (forged identity/signature), Tampering (artifact identity, checkpoint, manifest), Repudiation (missing trace/HITL record), Information disclosure (cross-tenant, secrets in errors), Denial of service (config-level only — **no load testing**), Elevation (authority widening, confused deputy, dynamic load). Each cell → invariant(s) → probe ID or explicit "not tested + reason".

## 10. Production reachability strategy

Classes: `PRODUCT_REACHABLE`, `STRICT_REACHABLE`, `LAB_ONLY`, `TEST_ONLY`, `LEGACY_BUT_REACHABLE`, `UNREACHABLE_WITH_EVIDENCE`, `REACHABILITY_UNPROVEN` (default; absence of proof is never unreachability).

Method per surface (DP-33: reachability first for E3/E4 findings):
1. **Static:** callers via `git grep -n` + AST call graph from every composition root (353 `*wiring*/*composition*/*bootstrap*` files; key: `intergrax/applications/_shared/harness_host_runtime.py`, `production_host_composition.py`, governed_contractor strict roots, `intergrax/hosting/runner.py`).
2. **Profile:** which `ApplicationProfile` (LAB/PRODUCT, `applications/contracts/application_host.py:12`), `ExecutionMode` (STRICT/BALANCED/EXPLORATORY), `ScenarioRuntimeMode` reach it; PRODUCT hosts list from `applications/shipped_product_manifest_catalog.py` + PROD-Q scope + DP-19 (direct `uvicorn.run` entrypoints declared non-production).
3. **Runtime (E4):** build the host through its canonical composition root in-process under PRODUCT/STRICT with synthetic config and assert the concrete instance (e.g. verifier class, authorization port not None, router production_mode) — `UNREACHABLE_WITH_EVIDENCE` requires a failing construction or a proof that no root wires it.
4. Exclusion classes need a written justification row in `EXCLUSIONS.jsonl`.

## 11. Workstreams WB01–WB12

Common for every WB: inputs = scope rows with that WB primary + section 6 rows + INV rows; outputs = `B_<NAME>_AUDIT.md` + `data/wbXX_results.jsonl` + probes under `probes/wbXX/`; every claim cites `path@sha:line` or `ws:` evidence; every probe pre-registered.

**WB01 Governance & Execution Authority → `B_GOVERNANCE_EXECUTION_AUDIT.md`.** Code: `runtime/governance/*` (ExecutionGuard, DefaultCanonicalInnerExecutionGuard, RootExecutionAuthorityAdmissionService, active_execution_authority), `runtime/policy/runtime_policy_engine.py`, `_shared/harness_meaningful_side_effect_authorization_wiring.py`, `_shared/agent_runtime_governance_wiring.py`, `runtime/agent_governance/*`, `contracts/delegation_authority.py`, `_shared/delegation_budget_wiring.py`, revocation (`runtime/external_operations/worker_execution_ownership_revocation*.py`), HITL (`runtime/nexus/orchestration/human_response.py`, `runtime/human/store.py`), `runtime/nexus/agent_router.py`. Negative cases (each with no-side-effect oracle): no permit; invalid permit; expired authorization; permission change mid-execution; revocation between admission and side effect; direct adapter call bypassing gateway; retry after revocation; child scope widening (H-16, X-ESC-E1); HITL timeout; replay of an old HITL decision; unknown agent id (H-04); no-rules policy engine (H-15). Rows: A4-044, A4-015, A4-037, A2-043, A3-019, A5-003(sec.), X-ESC-E1/E2, H-04, H-15..H-17, INV-GOV-01..05.

**WB02 End-to-End Tenant Isolation → `B_TENANT_ISOLATION_AUDIT.md`.** Code: strict serving binding files, `harness_host_runtime.py`, `contracts/request_identity_spine.py`, `memory/contracts/memory_control.py`, `rag/vectorstore/providers/native_provider_boundary.py`, provider `rag_store.py`, `rag/graph/tenant/graph_isolation_contract.py`, `tenant_storage_wiring.py`, TENANT-X detector `tests/qualification/tenant_x/_r1_silent_substitution.py`. Negative cases: A reads B; A modifies B artefact; tenant switch via payload/query/header/env (H-02, H-14); missing tenant; default fallback (H-20); namespace collision (length-prefix encoding check); retry with wrong tenant context; unauthorized provider binding; foreign trace/evidence access; tool/plugin hijacking tenant context; untenanted credential (H-18). Detector coverage: synthetic corpus in WS (kw-args `tenant_id=x or "default"`, field defaults, dynamic delegation, composition roots, optional binding, (de)serialization, async) run against the detector functions imported from the B worktree (detector code unchanged). Rows: A4-006, A4-027, A7-009, A7-028, A7-031, A8-006, A3-012, X-A4-016, X-A5-011, X-ESC-E4/E5, H-02, H-03, H-14, H-18..H-20.

**WB03 Fail-Closed & Production Strictness → `B_FAIL_CLOSED_STRICTNESS_AUDIT.md`.** Census (mechanical, AST): every product-path parameter `X | None = None` on a security/authority/durability port; every `Null*/NoOp*` class (25 found) and where it is the default; every `production_mode`/`strict` default (33 False, 7 True) and hard-coded value; every `except …: pass/return default` around config parsing; every `or <literal>` fallback for profile/provider/tenant. For each: reachable in PRODUCT/STRICT? formally allowed (cite contract)? weakens which INV? signals error? stops protected operation? Runtime check under canonical PRODUCT/STRICT composition with missing config (empty profile, missing durable store, missing HITL store, unknown verifier mode). Rows: A4-015, A4-026, A4-031, A4-037, A4-044, A4-017, A4-019, A4-024, A4-032, A4-035, A4-036, A4-040, A3-027, A2-023/024, H-01, H-05, H-07, H-15, H-22; reconcile with FC-1..FC-3 write sets (verify at FCS, never pre-empt).

**WB04 Security & Adversarial → `B_SECURITY_ADVERSARIAL_AUDIT.md`.** Threat model execution for T1–T3/T6 surfaces: authorization bypass, privilege escalation, confused deputy (tool acting with platform authority for tenant), indirect prompt injection (RAG doc / tool output instructing a tool call), untrusted tool output, forged execution identity, invalid request signatures (Slack/Teams/business webhook HMAC), cross-tenant requests, invalid credentials, unsafe error serialization (stack/secret in HTTP error body), path traversal (artifact/workspace paths), artifact identity tampering, stale/replayed authorization, unauthorized dynamic loading (H-13). Only in-process TestClient/ASGI, synthetic data, `testing_support` fakes for external services; outbound network blocked by a socket guard in every probe. Observability check: rejected attack produces an audit/trace record without secret material. Rows: A2-047, A7-013, H-01, H-07, H-10..H-13, H-21, H-23.

**WB05 Architecture Boundaries & Canonical Ownership → `B_ARCHITECTURE_BOUNDARIES_AUDIT.md`.** Independent AST analyzer (WS script) covering static imports, `TYPE_CHECKING`, lazy imports inside functions, `importlib`/`__import__` with constant and computed strings (trace provenance), `lazy_export`, entry points, DI/provider factories, registry lookups by string. Gate-bypass constructions: synthetic files in a scratch copy of the tree (never the worktree) to test whether `check_intergrax_no_applications_imports.py`, `check_agent_registry_bypass.py`, `check_agents_no_tier3_imports.py` detect aliasing (`import intergrax as ig; ig.applications…`), `importlib.import_module("intergrax." + "applications")`, `__import__`, relative imports, re-export through a neutral module. Classify each crossing: authorized composition seam (named in canon/DP-13/DP-18 list) vs violation; allowlist entries re-justified individually (C2). Rows: A4-003, A4-004, A4-007, A4-016, A4-020(sec.), A4-034, A2-009, A5-010, A7-021, A3-016, A2-031, H-13.

**WB06 Configuration & Vendor/Backend Neutrality → `B_CONFIGURATION_REPLACEABILITY_AUDIT.md`.** Typed config contracts, `integrations/registry/factory.py` (`build_profile_from_env`, `resolve_slug`, `resolve_from_profile`), second registry `runtime/integrations/registry_v2.py`, string/enum factories (speech, vision, llm_guardrail, `_FACTORIES` maps), ambient env census, tenant-scoped provider config. Tests: missing / invalid config, non-existent provider slug, unsupported capability, version mismatch, backend swap (two fakes behind the same contract), unauthorized activation, primary failure, invalid fallback config. Keep provider-neutral architecture apart from live-provider qualification (PROD-Q Path B, checklist L66 only). Rows: A4-020, A4-025, A4-042, A3-011, A5-002(sec.), A7-007, A1-012, A2-051, A8-003, X-A4-046, H-05, H-14.

**WB07 Reliability, Recovery & Durable Execution → `B_RELIABILITY_RECOVERY_AUDIT.md`.** Code: `runtime/long_running/*`, `runtime/execution/continuation/*`, `contracts/lease_claim.py`, `contracts/idempotency_store.py` + implementations, retry models (`contracts/retry_budget.py`, `runtime/resilience/local_provider_retry_budget.py`, dormant `fastapi_core/.../RetryBudget`, `StepRetryBudget`), reconciliation (`runtime/enterprise_reliability/*`), existing crash seams (`NoOpExecutionSuspendedWorkReentryCrashInjection`, `NoOpToolRuntimeEffectCrashInjection` — use their test doubles to inject crashes). Fault tests: interrupt before side effect; after side effect before result persisted; process restart (new store instance on same SQLite file); lease loss; two workers racing (threads, deterministic barriers); duplicate event delivery; storage error during recovery; corrupt checkpoint; revocation during retry; failure while writing execution evidence. Isolated SQLite under basetemp, fake providers; no Postgres/Redis/Docker. Rows: A2-019, A2-043, A2-049, A4-034, A4-040(sec.), A5-012, X-ESC-E2, H-08, H-24.

**WB08 Model, Context, Memory & RAG → `B_MODEL_CONTEXT_MEMORY_AUDIT.md`.** Provider-call census: every adapter method that reaches a vendor SDK vs `base_llm_adapter._execute/_execute_streaming` and `deadline_provider_guard.assert_protected_provider_call_allowed` (H-06); usage enforcement (`llm_adapters/governance/quota.py`, `usage_log.begin_call`, `budget_ticks`); capability matching; mandatory instructions; context budget; CE-01 (`context/assembly_runtime.py`); UAEP (`runtime/nexus/uaep/uaep_executor.py`); memory namespaces/retention/deletion (`memory/*`, `retention_sweep.py`); RAG tenant scope and foreign metadata (A8-020); provenance; untrusted context (H-11). Fake LLM adapter is the external system; the guard, context assembly and memory are real. Rows: A4-027(sec.), A4-031, A5-002, A8-020, A3-011, A3-012, H-06, H-11.

**WB09 MCP, Tools, Skills & Plugins → `B_MCP_TOOL_SKILL_PLUGIN_AUDIT.md`.** Integration with governance/tenant/execution: MCP exposure + discovery (`list_agents`, `run_agent`, catalog tools, custom host tools principal gating `mcp_server_engine.py:83`), identity modes, tool invocation (`ExecutionBoundCatalogToolInvoker`, `tool_gateway.py`, `uaep_tool_gateway.py`), capability identity and revocation, plugin entry-point discovery + admission, skill lifecycle (`skills/registry/*`, `dynamic_acquisition.py`, `execution_binding.py`), deployment trust, durability claims (FC-4). Do NOT repeat the TOOL-LIFE-X re-audit (PT-Q08 RA-1): check its cross-domain invariants and evidence currency only. Rows: A2-027, A7-013, A7-030, A2-001/A1-017 (currency only), H-12, X-TLX, X-A8-022.

**WB10 Trace, Evidence & Observability Integrity → `B_TRACE_EVIDENCE_INTEGRITY_AUDIT.md`.** `contracts/runtime_event.py` (correlation, `parent_event_id` as causation, traceparent), `runtime/events/*`, both ProofReceipt models, `contracts/artifact_ref.py::DurableArtifactRef`, redaction (`privacy_redaction.py`, `runtime_inspection/redaction.py`, `_SECRET_PATTERNS` ×3). Find: effects without trace; traces implying effects that did not happen (emit-before-commit); tenant/execution identity missing; tamperable evidence; observability failure (exporter error) swallowing audit events; secret leakage. Rows: A4-006(sec.), A4-037(sec.), A4-039, A4-045, A2-019(sec.), A5-012(sec.), H-09, H-10, H-19, H-23.

**WB11 Enterprise Assurance System Adversarial → `B_ASSURANCE_SYSTEM_ADVERSARIAL_AUDIT.md`.** Targets: ENT-AUDIT-X (`run_enterprise_audit.py`, `changed_since.py`, `impact_map.py`, check manifest 85 checks), ECAP, IIAC verifiers (`verify_iiac_{audit_manifest,delegation_refs,g0_inventory_completeness,manifest_artifacts,manifest_corpus,policy_activation,role_separation}.py`, `iiac_activation_gate.py` ACT-N11), ticket validators (`scripts/governance/ticket_approval.py`, `verify_ticket_approvals.py`, `verify_plan_approval.py`, `verify_ticket_lifecycle.py`, `verify_commit_ticket_refs.py --ci`), `legacy_approval_exceptions.json`, skip/xfail classification, architecture gates, required-check verifier (`scripts.governance.verify_ent_audit_branch_protection`, read-only). Synthetic corpus (copies under WS, never the repo): wrong manifest accepted? wrong SHA (short SHA, non-ancestor, tag/branch name, uppercase) accepted? incomplete evidence passes? missing env ⇒ silent skip (e.g. `ENT_AUDIT_X_SEMANTIC` unset)? new package not selected (A6-007)? classifier change hides violation? PASS after skipped path? forged `Approved:` / delegate line accepted (A6-002, A2-038 after T-IGX-044-C/E)? option-like git operand injection? No GitHub setting change, no ruleset touch (AR-45 trial PRs #647/#649/#652 belong to another session — read only). Rows: A6-007, A5-003, A5-015, A7-037, A7-031, A2-031, A2-030, A1-015, A1-021, A1-008, A7-003, A6-002/A2-037/A2-038/A6-006, X-A2-037, X-A6-003, X-A0-008.

**WB12 Cross-Domain Enterprise Invariants → `B_CROSS_DOMAIN_INVARIANTS_AUDIT.md`.** Flows through real canonical composition roots (no mock of the mechanism under test; only external systems faked):

| Flow | Path | Root to use | Composed INV |
|---|---|---|---|
| F01 | Agent → Governance → Execution → Tool → Artifact → Trace | governed_contractor strict roots (`strict_external_work_host_wiring.py`, `strict_gca_production_process_composition.py`) | GOV-01/05, TRC-01, SEC-04 |
| F02 | MCP → Authorization → Runtime → Provider → Evidence | `compose_application_mcp_server` HARNESS_BOUND + TestClient | GOV-05, TEN-01, TRC-02 |
| F03 | Scheduler → Worker → Execution → Retry → Recovery | `runtime/long_running/wiring.py` + SQLite checkpoint store + crash seams | REL-01/02, GOV-03 |
| F04 | Agent → Context → Memory → RAG → LLM | host context wiring + fake LLM | MOD-01/02, TEN-03, SEC-02 |
| F05 | Tenant A → request → runtime → storage → tenant A | LKW + governed_contractor | TEN-01/03 |
| F06 | Tenant A → resource of tenant B → safe rejection | same, every data plane | TEN-03 |
| F07 | PRODUCT host, required config missing → fail-closed | `production_host_composition.py`, `run_hosted_application` | FC-01 |
| F08 | Execution → HITL → decision → durable evidence | NexusLoop via `environment_orchestration_materialization.py` | GOV-04, TRC-02 |
| F09 | Config update → compatibility → requalification | profile revision adoption + COMPAT-X | CFG-01 |
| F10 | Restart → recovery → idempotency → trace continuity | continuation + idempotency stores | REL-01, TRC-01 |
| F11 | Inbound webhook → intake → NexusLoop execute | `interaction_wiring.py` (H-01) | SEC-01, TEN-01 |
| F12 | Background ingest worker → tenant → storage | LKW `background_worker_main.py` with fake consumer | TEN-04 |
| F13 | Skill/plugin acquisition → trust → execution | `skills/dynamic_acquisition.py`, `core/plugins/discovery.py` | SEC-05, GOV-05 |

Each flow row: canonical contract, owner, reachability, invariants, expectation, positive control, negative control, fault injection, evidence, acceptance criteria. Also: C8 per-row architecture/live/mixed check of the 41 OPEN OBS/RPL/SEC/REL/REG rows (A3-021) — no architectural obligation may sit in Path B.

## 12. Adversarial test strategy

Per critical invariant, 12 cases: normal path, positive control, negative control, invalid input, missing dependency, invalid configuration, unauthorized request, cross-tenant (where applicable), partial failure, recovery path, concurrent execution (where applicable), evidence verification. A case not applicable gets an explicit reason.

Probe contract (reused from alpha): each probe asserts the SAFE behaviour; probe PASS = REFUTED, FAIL = CONFIRMED, skip/setup error = INCONCLUSIVE (never PASS). Every negative test asserts the absence of the forbidden effect, not only an exception: tool not executed (spy call count 0), no write (store row count / file sentinel unchanged), no outbound network (socket guard raises if touched), correct authorization outcome object, audit/trace record present and secret-free. Each probe declares `real_under_test` (modules exercised) and `faked` (only external systems); a probe whose fake replaces the invariant under test is invalid.

Pre-registration: `PREREG_B1.md` / `PREREG_B2.md` frozen with sha256 before the first run of the wave (rows: probe id, invariant, scope row, SHA, command, predicted outcome, CONFIRMED/REFUTED rule, side-effect oracles). Deviations only via `PREREG_ADDENDUM_n.md`.

Tracked negative tests (PT-T01, NT-01..16 + new NT-17..): EQO-B writes specs and scratch reference implementations; an implementer ticket (not EQO-B) puts them in the repo. EQO-B verifies red-first at `B` (fail on unfixed code; an already red gate does not count) and green at FCS.

## 13. Resource and execution budget

- One pytest process machine-wide via `D:/Projekty/_wt/_locks/pytest.lock` (mkdir; hold ≤ 15 min; stale-lock rule: only if older than 20 min AND no `python.exe` with pytest in the command line — otherwise wait). Chunk long suites; per-chunk kill limit 840 s; always `-p no:xdist`.
- Before heavy runs: free RAM ≥ 6 GB (PowerShell `Get-CimInstance Win32_OperatingSystem`); otherwise wait. After every run: confirm no orphan `python.exe` from EQO-B (by recorded PID tree) — kill only own PIDs.
- No Docker/WSL (C10/C15 freeze), no bound ports (ASGI TestClient only; if unavoidable, port 0 on 127.0.0.1), SQLite under `--basetemp=D:/Projekty/_wt/intergrax/_scratch/eqob/<run>`; disk check ≥ 10 GB free.
- **B-1 budget:** 36 run slots + 8 reserve at B; 30 + 6 at FCS; heavy suites (`tests/qualification/trace_x` ~830 s, `state_x` ~340 s, `tenant_x` ~75 s, ebh_7 tenant E2E ~190 s) at most once per SHA each. **B-2 budget:** 20 + 4 at B, 16 + 3 at FCS (family sweeps FS-1..FS-6 at FCS only). ENT-AUDIT FULL non-certifying: 1 at FCS. Escalation: +25% slots only with a written evidence-gap reason in `BUDGET_LOG.md` and Operator notice.
- Subagents: ≤ 4 concurrent in-session subagents, each briefed with ≤ 2 000 words + file list; reports ≤ 1 500 words; bulk data goes to `data/*.jsonl`, never into the chat. Do not exceed MEP-AF2's ≤ 6 live sub-sessions machine-wide (EQO-B uses 1 session).

## 14. Exact-SHA policy (`CAMPAIGN_B_BASELINE.json`)

| Key | Definition |
|---|---|
| `CAMPAIGN_A_BASELINE_SHA` | `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` |
| `PLANNING_HEAD_SHA` | `2727a2c2364a97492202b931375772a400279a9b` |
| `CAMPAIGN_B_BASELINE_SHA` (`B`) | `git rev-parse origin/development` after `git fetch` at W0; record product-tree relation to `a73d15ab3` and to `258bbf6ab` |
| `TEST_EXECUTION_SHA` | per run = `git -C <WT> rev-parse HEAD`, must equal the run's declared target; porcelain empty before/after |
| `AUDITED_SHA` | `B` for pre-remediation verdicts; FCS for omega verdicts |
| `G2_SHA` | RW train head audited by PT-Q02 (delta preview only) |
| `INTEGRATED_REVALIDATION_SHA` | FCS (merge of the train; product tree identical to G2). If RB-3/second wave occurs, the new FCS |
| `M3_SHA` | EBH-7 pin; EQO-B checks `git diff --quiet FCS M3 -- intergrax agents applications` and lists test/script deltas affecting its evidence |
| `EVIDENCE_PUBLICATION_SHA` | merge commit of the PT-D13 dossier PR (docs-only; product tree == FCS) |

Evidence transfer rule: a run at G2 counts for FCS only if `git diff G2 FCS` is empty for product paths AND for every test/script path the run used; otherwise rerun.

## 15. Safe parallelism policy

- Workspace: `MAIN=D:/Projekty/intergrax`, `WT_BASE=D:/Projekty/_wt/intergrax` (per CLAUDE.md formula). Worktrees (detached, created by EQO-B only): `$WT_BASE/eqob-b` (at B), `$WT_BASE/eqob-a` (at `258bbf6ab`, historical reproduction), later `$WT_BASE/eqob-g2`, `$WT_BASE/eqob-fcs`. Scratch: `$WT_BASE/_scratch/eqob/`. Dossier workspace: `$MAIN/.tmp/session/eqo-b/` (gitignored). Probes and scratch copies live in WS/scratch, never in a worktree; worktrees stay porcelain-clean.
- venv: dedicated `uv` environment for the EQO-B worktree (`uv sync --frozen --extra dev-ci --extra mcp` + `integrations-duckdb` if needed) — needs EQOB-AR-03; never `uv sync` in the main checkout `.venv`.
- Never touch others' worktrees (`af2-*`, `041*`, `_scratch/b1-alpha|b2-alpha|closed-world|s1-diag|pt-wa-base`), the main checkout state, shared DBs, GitHub settings or rulesets. No push, no PR until publication authorization.
- Change-impact registry (`CAMPAIGN_B_CHANGE_IMPACT_REGISTER.md`): per scope row → linked findings, tickets, PRs/branches (`af2/*`, `arch-freeze/rw-train`, `ci/eqo-a-*`), changed code areas, expected test conflicts, contract changes, IIAC dependency (ACT-N11/RB-2), mechanism owner, final revalidation condition; status ∈ `STABLE_FOR_AUDIT | ACTIVE_CHANGE | AUDITABLE_HISTORICAL_BASELINE | WAIT_FOR_DEPENDENCY | REVALIDATION_REQUIRED | INTEGRATED_VERIFICATION_REQUIRED`. Refresh weekly and at every G2/FCS/M3 event (`git fetch`; `git log`/`git diff --stat` of train branches only — read-only). File non-overlap is never proof of no semantic impact: check contracts, DI, composition roots, config and the whole flow.
- Testing an ACTIVE_CHANGE mechanism at B is allowed; its verdict is labelled `PRE_REMEDIATION@B` and never counts as post-fix confirmation.

## 16. A-REM / PDC / ARCH-FREEZE dependency matrix (initial; W0 completes it)

| Vehicle | Scope rows | EQO-B dependency |
|---|---|---|
| PT-WA (T-IGX-043-G, PR #642) | A4-003, A4-004, A4-007, A2-009, A3-016 | verify at G2/FCS: gates + bypass constructions (WB05) |
| PT-WB | A4-020, A4-040, A8-003 | WB05/WB06 at FCS |
| PT-FC1 (T-IGX-043-A) | A4-044, A4-009, A4-037, A4-015, A4-031, A4-024 | WB01/WB03 at FCS |
| PT-FC2 (T-IGX-043-B, 043-E) | A4-030, A4-032a/b, A4-035, A4-036 | WB03 at FCS |
| PT-FC3 (T-IGX-043-C, 043-F) | A4-017, A4-019, A4-026, A4-032c, A4-025 | WB03/WB06 at FCS |
| PT-FC4 (T-IGX-043-D) | A4-002, A2-027, A3-027, H-08 | WB07/WB09 at FCS |
| PT-WD (ENT-AUDIT engine) | A6-003/005/007/008, A6-004 | WB11 at FCS |
| PT-WF (tenant G-09, owner repairs) | A4-027, X-A5-011, A8-020?, X-A4-016? | WB02/WB08 at FCS (confirm in W0 which alpha defects it covers) |
| PT-E02/E04/E05 (T-IGX-044-C/E) | A6-002, A2-038 | WB11 synthetic corpus at B and FCS |
| PT-T01 | NT-01..16, A4-006 | EQO-B specs; red-first at B, green at FCS |
| PT-Q05 / Q06 / Q07 / Q12 | A5-007, A5-010, A7-007, A2-030/031 / A5-012, A2-019, A2-049 / A3-012, A7-009, A7-028, A7-031, A2-023/024, A3-027, A7-013 / A3-019 | consume their results as E2/E3 input; EQO-B samples independently |
| PT-Q08 / PT-Q09 | A2-001, A1-017, A2-002, A2-025, A7-030 | currency of evidence only |
| PT-Q13 (FULL at M3) | A5-015 | EQO-B result is a prerequisite (PT-BA → PT-Q13) |
| PDC-X | A5-002, A4-039, A4-045, A7-013, PDC_INDEX conflict | record only |

Every row without a vehicle and with a confirmed defect → "UNOWNED_FIX" → Operator handoff (section 24).

## 17. Complete WBS (`CAMPAIGN_B_WBS.json` is generated from this table)

Default task fields (apply unless overridden): preconditions = hard deps done + authorizations granted; input = scope rows of the WB + section 3 sources; expected actions = static analysis → reachability → PREREG rows → lock-runner execution → classification; evidence = `path@sha:line` / run dir with SHA256SUMS; acceptance = every assigned row has verdict + scope + SHA + limits; revalidation = any change-impact event on its rows; stop = auth missing, porcelain dirty, HEAD ≠ target, lock unavailable > 60 min, budget exhausted, secret encountered; owner = EQO-B-EXEC; escalation = BLOCKER/HIGH (section 24).

| Task | Wave | Objective | Hard deps | Soft / other | Parallel | Budget |
|---|---|---|---|---|---|---|
| T00-01 | W0 | Fetch; freeze B; SHA ladder file; product-tree checks | – | – | no | 0 runs |
| T00-02 | W0 | Worktrees eqob-b/eqob-a, WS skeleton, lock helper, runner (adapt alpha `runner.py` after reading it), venv (AR-03) | T00-01 | – | no | 1 smoke run |
| T00-03 | W0 | Source registry + READ_LOG (sha256 per source at B) | T00-01 | – | yes | – |
| T00-04 | W0 | Ingest scope: 51 + 2 C + section 6a/6b/6c → `data/scope.jsonl` | T00-03 | – | yes | – |
| T00-05 | W0 | Change-impact registry + dependency matrix (grep tickets, plans, WP12, PR branches) | T00-04 | – | yes | – |
| T00-06 | W0 | Alpha evidence reuse check (digests, PREREG sha, erratum) | T00-04 | – | yes | – |
| T00-07 | W0 | Tools: copy Campaign A tools (`*.py.txt` → `.py`, sha256 vs MANIFEST), parameterize `common.py` BASELINE/CAMPAIGN; checker skeleton | T00-02 | – | yes | – |
| T00-08 | W0 | Authorization + budget check; taxonomy mapping table (A classes ↔ B subclasses ↔ EDER ↔ ECAP §13) | T00-04 | – | yes | – |
| T01-01 | W1 | Closed-world surface discovery (10 categories, repeatable commands) → `data/surfaces.jsonl` | T00-* | – | yes (2 subagents) | – |
| T01-02 | W1 | Static + profile reachability per surface; PRODUCT host list | T01-01 | – | yes | – |
| T01-03 | W1 | Invariants registry INV-* ↔ FRZ ↔ scope rows | T00-04 | – | yes | – |
| T01-04 | W1 | Threat model (surfaces × STRIDE × adversary) | T01-01, T01-03 | – | yes | – |
| T01-05 | W1 | Final B-1/B-2 assignment, R scores, initial ARCH-FREEZE impact | T01-02, T01-03 | – | no | – |
| T01-06 | W1 | Adversarial test matrix + PREREG_B1 freeze | T01-04, T01-05 | – | no | – |
| T01-07 | W1 | PREREG_B2 freeze | T01-05 | T01-06 | yes | – |
| T02-01..04 | W2 | WB01, WB02, WB03, WB04 at B (B-1 rows) | T01-06 | lock (R) | analysis parallel, runs serial | 22 slots |
| T03-05/06/09 | W3 | WB05, WB06, WB09 at B (B-1 rows) | T01-06 | lock (R) | yes | 8 slots |
| T04-07/08/10 | W4 | WB07, WB08, WB10 at B (B-1 rows) | T01-06 | lock (R) | yes | 10 slots |
| T04-11/12 | W4 | WB11 synthetic assurance corpus; WB12 flows F01..F13 at B | T02-*, T03-*, T04-07/08/10 (S) | lock (R) | yes | 8 slots |
| T05-01 | W5 | New-findings consolidation, dedup, severity calibration (COMMON L22 + AUD-06) | T02..T04 | – | no | – |
| T05-02 | W5 | Critical handoff packet (each confirmed product-change defect) → Operator, **target before G2** | T05-01 (rolling: may fire per finding from W2) | – | no | – |
| T05-03 | W5 | NT spec package for PT-T01 + red-first evidence at B | T05-01 | – | yes | 4 slots |
| T05-04 | W5 | B-1 interim closure matrix at B (`B1-AT-B`) + interim independent review by fresh subagent | T05-01 | – | no | – |
| T05-05 | W5 | G2 delta preview (optional; when G2 exists) | G2 exists | C | yes | ≤ 6 slots |
| T06-01..05 | W6 | B-2 at B per WB (grouped) | T01-07 | lock (R, low priority) | yes | 20 slots |
| T07-01 | W7 | Delta impact analysis B→FCS (semantic) → `REVALIDATION_MATRIX` | FCS exists | C | no | – |
| T07-02 | W7 | Worktree eqob-fcs; B-1 reruns + PT-T01 green check | T07-01, PT-T01 merged | F | runs serial | 30 slots |
| T07-03 | W7 | B-2 reruns + family sweeps FS-1..FS-6 | T07-01 | F, after T07-02 in lock | runs serial | 16 slots |
| T07-04 | W7 | ENT-AUDIT FULL non-certifying at FCS (coordinate with PT-Q13 owner) | T07-01 | F | no | 1 run |
| T07-05 | W7 | M3 identity check (when M3 exists) | M3 exists | F | yes | – |
| T08-01 | W8 | Final verdicts, ARCH-FREEZE impact matrix, four status dimensions | T07-* | – | no | – |
| T08-02 | W8 | Render dossier, MANIFEST, checker pass | T08-01 | – | no | – |
| T08-03 | W8 | Executor self-review by a fresh adversarial subagent (dossier red team) | T08-02 | – | no | – |
| T08-04 | W8 | Independent audit handoff → PT-BA (EQOB-AR-08) | T08-03 | – | no | – |
| T08-05 | W8 | Corrective loops (≤ 2, consolidated) | T08-04 verdict | – | no | ≤ 4 slots |
| T08-06 | W8 | Campaign C/D handoff | T08-05 | – | yes | – |
| T08-07 | W8 | Publication PR (PT-D13, EQOB-AR-09) | PT-BA PASS/PASS_SCOPED | – | no | – |

## 18. Dependency DAG (`CAMPAIGN_B_DEPENDENCY_DAG.json`)

Schema: `{"nodes":[{id, wave, kind}], "edges":[{from, to, type}]}`, `type ∈ {hard, soft, shared_resource, change_impact, final_revalidation}`. Edges = the "Hard deps" (hard), "Soft" (soft), "lock (R)" (shared_resource on node `RES-PYTEST-LOCK`), "C" (change_impact from external nodes `EXT-G2`, `EXT-FCS`, `EXT-M3`, `EXT-PT-T01`), "F" (final_revalidation) columns. External nodes from WP-12: `EXT-SCOPE` (passed), `EXT-G2`, `EXT-FCS`, `EXT-M3`, `EXT-PT-T01`, `EXT-PT-Q13`. Binding edges into WP-12 (needs EQOB-AR-11): T05-04 ⇒ PT-B01 content; T06-* ⇒ PT-B02; T05-03 ⇒ PT-T01; T07-02 ⇒ PT-B1W; T07-03 ⇒ PT-B2W; T08-04 ⇒ PT-BA; T08-07 ⇒ PT-D13. Validate: acyclic hard edges, every task reachable from T00-01, critical path printed.

Critical path: T00-01 → T00-04 → T01-03 → T01-05 → T01-06 → T02-01..04 → T05-01 → T05-02 (pre-G2) … EXT-FCS → T07-01 → T07-02 → T08-01 → T08-02 → T08-04 → PT-BA → T08-07 → (PT-Q13). B-2 (T06, T07-03) is off the critical path for B-1 status but required for "Campaign B complete".

## 19. Execution waves

W0 init (day 1) → W1 discovery/threat model/PREREG (days 1–3) → W2 security invariants (from day 3; rolling critical handoff) → W3 architecture/config/MCP → W4 runtime + assurance + cross-domain (W2–W4 analysis parallel, runs serialized by lock, B-1 first) → W5 B-1 at B consolidation + handoffs + NT specs (target: before G2) → W6 B-2 at B (starts after W1, low lock priority) → W7 integrated revalidation at FCS (external trigger) → W8 consolidation, self-review, PT-BA, corrective loops, publication, C/D handoff. If FCS is not reached, Campaign B stops at `B1-AT-B-COMPLETE / B2-AT-B-COMPLETE` and says so; it never claims final status.

## 20. Subagent delegation strategy

EQO-B-EXEC (Sonnet main session) is the Campaign Execution Orchestrator: owns WS, PREREG, the lock-runner, all pytest execution, consolidation, dedup, budget, handoffs. Subagents (Agent tool, in-session):
- `Explore` — discovery sweeps (T01-01, census tasks), read-only.
- `general-purpose` — one per WB analysis/probe authoring; writes only under `WS/probes/wbXX/` and `WS/data/wbXX_*.jsonl`; does not run pytest (submits run requests to the orchestrator queue); one WB per subagent, never two subagents on the same WB at once.
- Fresh `general-purpose` "devil's advocate" reviewer per WB report (did not author it) and for the dossier (T08-03).
Brief template (each ≤ 2 000 words): workstream, assigned scope rows + INV IDs, SHA + worktree path, sources (paths + line ranges), forbidden actions (section 22), probe contract (section 12), report schema (section 21 fields), acceptance criteria, budget, stop conditions. The orchestrator verifies every subagent claim it promotes to a verdict (spot-check the cited lines).

## 21. Evidence policy

E-levels exactly as Campaign A `briefs/COMMON.md` L17: E0 none; E1 claim; E2 + resolvable SHA and existing artefact; E3 + content checked against claim; E4 re-executed **by EQO-B** (alpha = E3 max, `ALPHA_NON_INDEPENDENT`). Formats L18: `path@<40-hex>:<line>`, `ws:<rel>`, `E1:<text>`. Critical invariants require E4 where a safe execution exists; otherwise an explicit `EVIDENCE_GAP` row.

Finding verdicts: `REPRODUCED | NOT_REPRODUCED_WITHIN_SCOPE | FIXED_VERIFIED | PARTIALLY_FIXED | REGRESSION_DETECTED | NOT_REACHABLE_WITH_EVIDENCE | EVIDENCE_INSUFFICIENT | WAITING_FOR_DEPENDENCY | REVALIDATION_REQUIRED | ACCEPTED_SCOPED_LIMITATION` (the last only when an accepting authority record exists — EQO-B never accepts). Each verdict: scope, SHA, test conditions, limits. `FIXED_VERIFIED` only at FCS with a red-first negative test.

New findings `EQOB-WBxx-NNN` (deterministic numbering with Campaign A `fl.py` logic). Fields: the 18 of the brief section 13 + `class` (Campaign A vocabulary: CONFIRMED_DEFECT, ARCHITECTURE_RISK, EVIDENCE_DEFICIENCY, CERTIFICATION_DRIFT, GOVERNANCE_INCONSISTENCY, UNVERIFIED_HYPOTHESIS, ACCEPTED_SCOPED_LIMITATION) + `b_subclass` (SECURITY_VULNERABILITY, TENANT_ISOLATION_VIOLATION, GOVERNANCE_BYPASS, ARCHITECTURE_VIOLATION, FAIL_CLOSED_VIOLATION, RELIABILITY_DEFECT, TEST_COVERAGE_GAP, CERTIFICATION_DRIFT, EVIDENCE_DEFICIENCY, UNVERIFIED_HYPOTHESIS) + `eder_register_class` + `ecap_class` (mapping table from T00-08). Severity rule = COMMON L22 + AUD-06 gate rule; BLOCKER/HIGH need ≥ E3 (B-1 critical: E4 where safe). EDER: proposals only, as `EDER-N-*` rows in `data/eder_proposals.jsonl` (schema-valid against `register_schema.json`, lifecycle `INGESTED`); no write to `register.jsonl`.

ARCH-FREEZE impact: `BLOCKS_ARCH_FREEZE | CONDITIONS_ARCH_FREEZE | NON_BLOCKING_WITH_EVIDENCE | LIVE_QUALIFICATION_ONLY | IMPACT_UNKNOWN`, mapped to EDER `BLOCKS | CONDITIONS | NONE | NONE+live_flag | UNKNOWN`; DP-01 definition of BLOCKS; `LIVE_QUALIFICATION_ONLY` only with a checklist L66 basis (DP-34); mixed findings split into an architecture part and a live part; architecture/governance/tenant/security/fail-closed violations never go to Path B.

Mocks: allowed only for external systems; the record names what stays real.

## 22. Deliverable registry and READ-ONLY boundaries

Working location: `$MAIN/.tmp/session/eqo-b/`; publication target (after EQOB-AR-09): `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_b/` (README.md + the 22 deliverables of the brief: CHARTER, BASELINE.json, SCOPE_MATRIX, FINDING_DISPOSITION_MATRIX, CRITICAL_INVARIANTS_REGISTRY, PRODUCTION_REACHABILITY_MATRIX, THREAT_MODEL, ADVERSARIAL_TEST_MATRIX, CHANGE_IMPACT_REGISTER, TEST_EXECUTION_RECORD, WBS.json, DEPENDENCY_DAG.json, `B_*_AUDIT.md` ×12, NEW_FINDINGS_REGISTER, ARCH_FREEZE_IMPACT_MATRIX, REVALIDATION_MATRIX, B1_CLOSURE_MATRIX, B2_CLOSURE_MATRIX, FINAL_AUDIT_REPORT, INDEPENDENT_AUDIT_HANDOFF, C_D_HANDOFF, MANIFEST.json) plus `data/*.jsonl` (scope, surfaces, invariants, results, findings, eder_proposals, sha_registry), `PREREG_B1.md`/`PREREG_B2.md` (+ sha256), `probes/` and `tools/` as `*.py.txt`, `SOURCE_REGISTRY.json`, `READ_LOG.jsonl`, `EXCLUSIONS.jsonl`, `LIMITATIONS.md`, `AUTHORITY_CONFLICT_REGISTER.md`, `BUDGET_LOG.md`, run manifests with digests only (no gitignored-log dependency: logs summarized + sha256; A2-030/A1-015 lesson).

Allowed: read anything; write only WS/scratch/own worktrees' untracked venv; run probes/tests; `git fetch`; read-only `gh` GET. Forbidden without separate authorization: product code, existing tests/gates/allowlists, FRZ statuses, roadmap, IIAC records, EDER register rows, GitHub settings/rulesets, real external effects, closing findings, certifying own work, PRs/pushes, `uv sync` in the main `.venv`, Docker. Never change the system to get a positive result.

## 23. Mechanical acceptance criteria (`tools/verify_campaign_b.py`, adapted from `verify_campaign_a.py`)

| Check | Rule |
|---|---|
| CB01 | 51/51 handed-off IDs present in SCOPE and DISPOSITION matrices, each with a verdict ≠ empty |
| CB02 | every scope row has `priority ∈ {B-1,B-2}` or a documented transfer (owner + record) |
| CB03 | every BLOCKER/HIGH (old and new) has ≥ E3 evidence and an ARCH-FREEZE impact ≠ empty |
| CB04 | every INV-* has ≥ 1 scope row and (positive control AND negative control) or an EVIDENCE_GAP row |
| CB05 | every surface in `surfaces.jsonl` has a reachability class; every exclusion class has a justification |
| CB06 | every run row: SHA is 40-hex, equals declared target, porcelain clean before/after, exit code, junit/stdout sha256 match files |
| CB07 | every verdict row has scope, SHA, conditions, limits |
| CB08 | every change-impact row has a status; every ACTIVE_CHANGE/REVALIDATION_REQUIRED row has a FCS revalidation entry or WAITING_FOR_DEPENDENCY |
| CB09 | no `FIXED_VERIFIED` without red-first at a pre-fix SHA + green at FCS |
| CB10 | no PASS/CLOSED/CERTIFIED wording in EQO-B outputs about platform state (regex lint, Campaign A `lint_iiac_closure_regex` style) |
| CB11 | no tracked-file change: `git -C <each WT> status --porcelain` empty; `git diff --quiet B -- .` in eqob-b |
| CB12 | no aggregate verdict hides an unresolved row: roll-ups recomputed independently (`rollup_calc.py` logic) equal the rendered numbers |
| CB13 | PREREG sha256 unchanged since freeze; every run maps to a PREREG row or an addendum |
| CB14 | new findings: all mandatory fields, ID format `^EQOB-WB(0[1-9]|1[0-2])-[0-9]{3}$`, EDER proposal schema-valid |
| CB15 | MANIFEST sha256 of every dossier file matches |
| CB16 | alpha-origin evidence never at E4; every alpha row cited has a digest check record |

The checker certifies inventory completeness and internal consistency only — never technical correctness (stated in its output header).

## 24. Risk and blocker handling

On BLOCKER/HIGH or any security/tenant/unauthorized-effect defect: (1) preserve evidence (run dir + digests); (2) minimal deterministic reproduction; (3) exact SHA; (4) confirm the invariant impact; (5) reachability (section 10); (6) canonical owner; (7) check A-REM/WP-12 coverage (change-impact registry); (8) ARCH-FREEZE impact; (9) consolidated fix recommendation + required regression test (NT spec); (10) continue unaffected WBs. Notify the Operator **immediately** in the EQO-B chat with an `ALERTS.md` entry (ID, SHA, invariant, reachability, covered-by vehicle or UNOWNED_FIX, pre-G2 decision needed: add to train / RB-3 / second wave — WP12-Q03). EQO-B does not message other sessions; the Operator forwards. Pause only the tasks the blocker invalidates. Other risks: lock starvation by other sessions (wait, then report); FCS slipping (stop at AT-B statuses); alpha/EQO-B disagreement (record both, never overwrite); evidence contamination by a dirty tree (discard run).

## 25. Delta revalidation strategy

At each of G2 (preview), FCS (mandatory), M3 (identity), and any RB-3/second wave: (1) record SHA; (2) `git diff --name-status B..<SHA>` for product, tests, scripts, `.github`; (3) semantic impact: changed contracts (`intergrax/contracts/**`), DI/composition roots, config/profile modules, registries, gate scripts, test fixtures used by EQO-B probes → map to scope rows and INV; (4) mark affected evidence `REVALIDATION_REQUIRED`; (5) rerun; (6) check new execution paths created by fixes (new surfaces → reachability); (7) keep `@B` and `@FCS` evidence in separate columns, never merged.

## 26. Independent audit strategy (PT-BA, EQOB-AR-08)

New Opus session started by the Operator; not the planner, executor, ARCH-FREEZE/A-REM/PDC implementer or signer (IIAC role separation; host-issued session id + Operator-countersigned handoff per DP-68). Brief = SHAs + scope file + evidence index + dossier MANIFEST only (no executor context). Auditor: verifies baseline/SHA ladder; executor independence; every BLOCKER/HIGH from source; evidentiary integrity (digests, PREREG); independent replay of a sample (all cheap B-1 runs + seeded sample of heavy runs, seed recorded); 51/51 completeness; unjustified exclusions; red-first of NT tests at B and FCS; delta revalidation; ARCH-FREEZE impact; credibility of the final verdict. Verdicts PASS / PASS_SCOPED / CONDITIONAL / FAIL on the dossier only (IIAC semantics; a dossier PASS is not a platform PASS). Corrective loops: ≤ 2, consolidated, delta re-read only.

## 27. Final closure conditions and terminal statuses

Four independent dimensions, reported separately for B-1, B-2 and full Campaign B:
1. **Execution completeness:** `NOT_STARTED | IN_PROGRESS | AT_B_COMPLETE | FCS_REVALIDATED | COMPLETE`.
2. **Evidence integrity:** checker CB01–CB16 result + auditor verdict.
3. **Enterprise invariant compliance:** per INV: `HOLDS_WITHIN_SCOPE | VIOLATED | PARTIALLY_HOLDS | UNVERIFIED`.
4. **ARCH-FREEZE readiness impact:** counts of BLOCKS/CONDITIONS/UNKNOWN rows + exact SHA.
Plus: required corrections (open UNOWNED_FIX and vehicle-covered defects) and final integrated revalidation status at FCS. Full Campaign B = COMPLETE only when B-1 and B-2 are both FCS_REVALIDATED and audited. No defect is ever closed by EQO-B.

Campaign C handoff: certifications contradicted by new evidence, areas needing full requalification, insufficient gate coverage, changes affecting historical guarantees, findings requiring FRZ re-assessment. Campaign D handoff: every unresolved BLOCKER, conditional requirements, remaining critical invariants, integrated revalidation results, security evidence, unresolved exceptions, residual risk, exact-SHA freeze impact. Nothing moves implicitly: each handed-off item names the receiving campaign row.

## 28. Authorization requests (Operator; map to existing AR/AB where they overlap — check the decision register, never assume a grant)

| AR | What | Overlaps |
|---|---|---|
| EQOB-AR-01 | Start the EQO-B-EXEC Sonnet session (independent of ARCH-FREEZE/A-REM/PDC sessions) | AB-03, AR-61 |
| EQOB-AR-02 | Create detached worktrees eqob-b, eqob-a (later eqob-g2, eqob-fcs) | AB-01 |
| EQOB-AR-03 | Dedicated uv environment with `--extra dev-ci --extra mcp` (+ duckdb) — package-index network once | AB-12 |
| EQOB-AR-04 | B-1 run budget at B (44 slots) and FCS (36) | AB-05..09, AB-14, AR-08 |
| EQOB-AR-05 | B-2 run budget at B (24) and FCS (19), incl. `ENT_AUDIT_X_SEMANTIC=1` measured run (A6-006) | AB-10, AR-13 |
| EQOB-AR-06 | Read-only `gh` GET calls (CI runs, branch-protection read) | AB-11 |
| EQOB-AR-07 | One ENT-AUDIT FULL non-certifying run at FCS | AR-10 |
| EQOB-AR-08 | Start EQO-B-INDEPENDENT-AUDITOR (PT-BA) | AB-15/16, AR-57 |
| EQOB-AR-09 | Docs-only publication PR of the dossier (PT-D13) | AB-17 |
| EQOB-AR-10 | Hand EDER-N proposals to the EDER steward (no register mutation) | – |
| EQOB-AR-11 | ARCH-FREEZE owner/Operator: bind EQO-B tasks as the content of PT-B01/B02/T01/B1W/B2W/BA/D13 | WP-12 |

## 29. Executable instruction for Claude Sonnet (EQO-B-EXEC)

Paste as the first message of a NEW Sonnet session in this repository (Agent Mode):

```
You are EQO-B-EXEC, the Campaign Execution Orchestrator of EQO-B (Campaign B - Critical Enterprise
Invariants Audit) for the Intergrax repository. Plan (authoritative, read it fully first):
agent-env/plans/pasted-content-id-9441-intergrax-elegant-lollipop.md. Reply to the Operator in Polish;
every file you write is English.

Identity and independence: you are not the ARCH-FREEZE orchestrator, an A-REM/PDC implementer, a reviewer
or signer of any ticket in the change-impact registry. Record your session id in CAMPAIGN_B_CHARTER.md.
You never certify your own work; the final verdict belongs to PT-BA.

Hard rules: READ-ONLY on tracked files (plan sections 15 and 22). No product/test/gate/allowlist edits,
no FRZ/roadmap/IIAC/EDER mutation, no GitHub settings, no push/PR before EQOB-AR-09, no Docker, no
uv sync in the main .venv, no external production systems, synthetic data and test credentials only,
outbound network blocked in every probe. Main checkout D:/Projekty/intergrax: no pull/checkout/stash.
One pytest process machine-wide via D:/Projekty/_wt/_locks/pytest.lock (section 13). Respect the
always-on repo rules (session artifacts under .tmp/, token budget, test-process budget, git branch safety,
no AI attribution).

Procedure:
1. W0 (tasks T00-01..T00-08): check which EQOB-AR-01..11 are granted in chat or in
   ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md / WP12_AUTHORIZATION_REQUESTS.md; list missing ones in ONE
   batch question and continue only with granted scope. Fetch, freeze B, build the SHA ladder, worktrees,
   WS at .tmp/session/eqo-b/, venv, source registry, scope ingest (51 + sections 6a/6b/6c), change-impact
   registry, alpha reuse check, tools + checker skeleton, taxonomy mapping.
2. W1 (T01-01..T01-07): closed-world discovery, reachability, invariants, threat model, final B-1/B-2,
   adversarial test matrix, freeze PREREG_B1 and PREREG_B2 (sha256) before any run.
3. W2-W4: run B-1 workstreams in R-score order, product-change defects first (pre-G2 window). Delegate
   one workstream per subagent (section 20), max 4 concurrent; only you run pytest, batched, under the
   lock. Each confirmed BLOCKER/HIGH or security/tenant/unauthorized-effect defect: section 24 at once
   (ALERTS.md + chat message to the Operator), then continue.
4. W5: consolidate, dedup, calibrate; NT spec package for PT-T01 with red-first evidence at B; B-1
   interim closure matrix (status B1-AT-B-COMPLETE); devil's-advocate review by a fresh subagent.
5. W6: B-2 at B with low lock priority; apply the promotion rule (section 5).
6. W7 when FCS exists: semantic delta analysis, revalidation matrix, reruns at FCS, PT-T01 green check,
   ENT-AUDIT FULL non-certifying (coordinate with PT-Q13), M3 identity check when M3 exists.
7. W8: final verdicts and four status dimensions, render + MANIFEST + checker CB01..CB16, self-review,
   handoff to PT-BA, at most 2 consolidated corrective loops, C/D handoff, publication after EQOB-AR-09.
Stop and report (do not improvise) when: an authorization is missing, a tracked tree is dirty, HEAD
differs from the target SHA, the budget is exhausted, a secret appears in output, or the plan
contradicts a canonical source (record in AUTHORITY_CONFLICT_REGISTER.md and ask).
Report to the Operator at the end of each wave in at most 12 lines: status per dimension, new
BLOCKER/HIGH, decisions needed.
```

## 30. Self-review (planner) — gaps found and how the plan covers them

- **Independence gap of WP-11 alpha** (run by the ARCH-FREEZE program) → alpha capped at E3, independent rerun required (sections 4, 21, CB16).
- **SCOPE already passed** → pre-G2 handoff window and WP12-Q03 route (sections 7, 24).
- **Domains not in the brief's list but found in recon:** interactions intake, debug routers, CLI/harness serving, fastapi_core auth defaults, queueing brokers, codecraft/ephemeral registry, capability acquisition, agent distribution revocation, speech/vision adapters, marketplace → covered via section 6c and the T01-01 closed-world discovery (each gets a reachability class).
- **Gate-bypass blind spots** (regex/AST gates) → WB05 synthetic bypass constructions; detector coverage corpus (WB02); assurance corpus (WB11).
- **Mock-replaces-invariant risk** → probe `real_under_test/faked` declaration (section 12).
- **DoS / load** → explicitly out of scope (config-level only), recorded in LIMITATIONS.md.
- **Live vendor qualification** → out of scope except architecture split (PROD-Q Path B, L66 only).
- **PDC_INDEX vs roadmap conflict** → recorded, not resolved by EQO-B.
- **Residual planning uncertainty:** WP-11 original files (`WP-11/AUTHORIZATION_REQUESTS.md`, omega W00..W35 list) are session artefacts not in the repo — the executor locates them under `.tmp/session/arch-freeze/` and reconciles; if absent, this plan's WBS is authoritative.

## Verification of this plan (by the executor, before W1)

1. `git rev-parse origin/development` vs section 2; re-derive the counts (FINDINGS.jsonl next_action, FRZ regex, EDER rows) and record differences.
2. `python -c` over `findings/FINDINGS.jsonl`: the set of `next_action == CAMPAIGN_B` IDs equals the 51 IDs of section 5 (fail W0 otherwise).
3. Validate WBS/DAG JSON: acyclic hard edges, all tasks reachable, critical path printed.
4. Dry run of the checker on the empty dossier skeleton: all inventory checks report MISSING, none crash.

## Addendum W0 (EQO-B-EXEC, 2026-11-17) - corrections from W0/W1 subagent results

Facts verified at B (`2727a2c2`); the plan text above is not rewritten, this addendum prevails where they differ.
- Section 3 paths: `FINDINGS_REGISTER.md` (with extension); plans-pair of the reliability architecture is `docs/project/maintainers/plans/ENTERPRISE_RELIABILITY_LAYER_IMPLEMENTATION_PLAN.md`; campaign_a `deliverables/` has 11 files in total (STAGE_INVENTORY included, not a 12th); README.md is tracked but not in the Campaign A MANIFEST. Roadmap differs from its A-baseline by 6 lines (L498, 570, 572, 1847, 1884, 2163); anchors L499/L572/L2155-2164 and FRZ L66 still hold.
- Section 10 / 11: 375 (not 353) `*wiring*|*composition*|*bootstrap*` product files at B (53 groups); 23 import_module/__import__ sites confirmed; plus `entry_points` (core/plugins/discovery.py:98), `exec` (mcp_server_engine.py:159), `pickle.loads` of stored idempotency blobs (runtime/tools/sqlite_idempotency_store.py, distributed/providers/redis_idempotency_store.py) and config/manifest-derived module strings (runtime/execution/context_engine_resolver.py:29, _shared/wiring.py:64, agent_resolution.py:42, venv bundle resolvers) - new WB05/WB09 targets.
- Section 6c: H-13 NOT_CONFIRMED as stated (literal import, no external input); H-02 body-wins detail NOT_CONFIRMED for LKW workspace routes (order: context > X-Tenant-Id header > body > "default"). New leads H-25..H-31: fail-open `require_harness_api_key` without configured key; LKW has no auth dependency under serving/; dispute_sim `identity_source` dead control; research factory hard-codes ApiConfig(DEV); `/runs` GET/cancel and `/v1/tasks/{id}/status` without tenant filter; reference hosts bind 0.0.0.0 by default; fastapi_core falls back to NoAuth + NoOp budget/rate-limit also under PROD (FC-0005); invalid `INTERGRAX_LLM_TENANT_MAX_TOKENS` becomes 0 (quota disabled).
- Section 5 / 16: 10 initial change-impact statuses differ (e.g. A7-021 and A4-047 -> ACTIVE_CHANGE; A2-043, A2-023, A8-006 -> REVALIDATION_REQUIRED); ~29 vehicles unconfirmed; PT-Q12/Q06/Q07 row lists, PT-D03 (#618) and PT-FC4 do not cover what section 16 assumes; see `reports/change_impact_disagreements.md` (authoritative for W0). Vehicles already merged into B: PT-E01, E02, E04, D01-D07, D12, R01, D03. Only unmerged product change: `arch-freeze/rw-train` (23 commits, 68 files, includes PR #642). Tickets T-IGX-043-A..F have no implementation branch; PT-WB, PT-WD, PT-WF, PT-T01, PT-E05 have no ticket or branch; T-IGX-051/052 and 045 do not exist.
- Alpha reuse: all 53 SHA256SUMS verify; PREREG digests match; b2 `R25c/out/` is not hashed; b1 runner.py changed after PREREG (declared in ADDENDUM_1); PREREG_ADDENDUM_2 was written after R00-R04 results were seen; every alpha run executed at A (258bbf6ab) with the main-checkout `.venv` python and no exclusivity check. Alpha class over the 53 rows: CONFIRMED 29, NOT_REPRODUCED 12, INCONCLUSIVE 8, not run 4 (A3-004, A6-004, A5-015, A7-028). All alpha-derived evidence stays E3 `ALPHA_NON_INDEPENDENT`.
- Section 21 / CB14: ECAP section 13 mandatory fields (file/symbol, production path, closure criterion, independent verification) must be added to the EQO-B finding schema and checked by CB14; EDER ids are `EDER-N-NNNN` with `origin.kind = NEW`; taxonomy ambiguities A1-A8 in `reports/taxonomy_mapping.md`.
- Tools: Campaign A `verify_campaign_a.py` and `render.py` need a structural rewrite; `run_pytest.py` (10-run flat budget, no lock, no junit/digest) is replaced by a new lock-runner.
