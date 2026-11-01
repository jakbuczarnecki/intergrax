
# EVOL-X FINAL COMPLETION MASTER PLAN v3

**Baseline:** `8848857741f4b04e5b476dca3c47f7d0a60bf672` · `development` · `origin` = `https://github.com/jakbuczarnecki/intergrax.git`

**Normatywne źródła (repo-only):** [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.9 · [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) FRZ-EVOL-01..10 · [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](docs/project/maintainers/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) · [EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md](docs/project/maintainers/qualification/EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md) · [evol_x/EVOL_X_CONTRACT_SPEC.md](docs/project/maintainers/qualification/evol_x/EVOL_X_CONTRACT_SPEC.md) · [evol_x/EVOL_X_CLOSED_WORLD_GATE_SPEC_W1.md](docs/project/maintainers/qualification/evol_x/EVOL_X_CLOSED_WORLD_GATE_SPEC_W1.md) · [evol_x/EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md](docs/project/maintainers/qualification/evol_x/EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md) · [evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](docs/project/maintainers/qualification/evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md)

**Publikacja:** ten dokument jest [`docs/project/maintainers/qualification/EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md`](docs/project/maintainers/qualification/EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md); **PLAN_PUBLICATION_SHA** = commit git, który go wprowadza (patrz [`EVOL_X_SHA_LADDER.md`](docs/project/maintainers/qualification/EVOL_X_SHA_LADDER.md)). Implementacja W2–W7 **nie** startuje przed W0 i autoryzacją operatora (Part III, VII-I).

**v2→v3:** pełna macierza zadań, CONSERVATIVE_FULL, EXP-S1..S8, P03 schematy, timing OD, macierz pokrycia wiersz-po-wierszu, usunięcie odwołań do lokalnych `.cursor/plans/`.

**v3 acceptance gate (B-01..B-04):** SHA lifecycle immutability · dual conservative populations · mechanical matrix per-ID · FRZ-EVOL-07 scoped claim ([EVOL_X_ENFORCEMENT_BOUNDARY_W1.md](docs/project/maintainers/qualification/evol_x/EVOL_X_ENFORCEMENT_BOUNDARY_W1.md)).

---

## Part I — Architecture (v2 retained, self-contained)

### I-A. CW-01..10 (kanon — hub W1 / closed-world spec / contract spec)

| Gate | Semantyka | Task |
|------|-----------|------|
| CW-01 | `candidates \ (classified ∪ excluded) = ∅` | W2-05 |
| CW-02 | exactly one owner per node | W2-05 |
| CW-03 | import reachability from K∪X | W2-05 |
| CW-04 | exclusion predicates valid (no K→X edge) | W2-05 |
| CW-05 | pytest collect ↔ AST; unowned tests reported | W2-05 |
| CW-06 | second-method contract consumer scan | W2-05 |
| CW-07 | ≥10 mutation probe classes flip a gate | W2-05 |
| CW-08 | dual build identical digest | W2-06 |
| CW-09 | parity vs COMPAT/QUAL/DUP/EBH-7 inventories | W2-05 |
| CW-10 | incremental rebuild == full rebuild | W2-06 |

**XG (not CW):** XG-01 import ratio diagnostic; XG-02 `critical_omissions`; XG-03 graph_digest vs inventory_digest.

### I-B. D1–D10

| ID | Semantyka | Task |
|----|-----------|------|
| D1 | git-tree universe + roots R | W2-05 |
| D2 | AST symbol extraction | W2-04 |
| D3 | 19 REL extractors MANDATORY/BEST_EFFORT | W2-04 |
| D4 | OwnerRule reconciliation | W2-03, W2-05 |
| D5 | X-* register + predicates | W2-05 |
| D6 | run CW-01..10 | W2-05 |
| D7 | UNKNOWN register | W2-04, W2-07 |
| D8 | canonical digest + source_tree_id | W2-06 |
| D9 | incremental (non-evidence) | W2-06 |
| D10 | digest reconcile vs prior publication | W6-03, W7-06 |

### I-C. Semantic graph (summary)

Node IDs: `module:`, `symbol:`, `app:`, `agent:`, `workflow:`, `scenario:`, `registry:`, `schema:`, `store:`, `shared:`, `config:`, `suite:`, `ci:`, `owner:`, `doc:`, `unknown:`.

Mandatory REL: IMPORTS, CONSUMES_CONTRACT, IMPLEMENTS, COMPOSES, REGISTERED_IN, SHARES, PERSISTS, READS_PERSISTED, EMITS/HANDLES_EVENT, BINDS, OWNED_BY, TESTED_BY, GATED_BY, DEPENDS_UNKNOWN.

Files: `testing_support/change_safety/discovery/extractors.py`, `graph/*`, `assess/consumers.py`.

### I-D. AS-IS @ 884885774

70 files partial W2–W6; 20 pytest modules; no S fixtures; F-P0-01..06 confirmed; all FRZ OPEN; roadmap ledger stale.

---

## Part II — Conservative regression scope (B-02)

Two **separate policies** — never merge labels in `SelectionDecision.reasons` or FRZ evidence.

### II-A. Policy `EVOL_CONSERVATIVE_FULL` (in-scope EVOL-X qualification)

| Tier | ID | Population | Owner | Mandatory inclusion | Refusal |
|------|-----|------------|-------|-------------------|---------|
| T1 | `MANDATORY_MIN` | `qual_x_p5_mandatory_collectable_node_ids()` | QUAL-X P5 catalog | **Always executed** in any SELECTED or EVOL_CONSERVATIVE_FULL run | missing collect → `REFUSED_INTEGRITY` |
| T2 | `QUAL_POP_A_COLLECTABLE` | P5 `MANDATORY_COLLECTABLE_AT_HEAD` nodes | QUAL-X Pop-A | **Always ⊆** EVOL_CONSERVATIVE_FULL | same |
| T3 | `EVOL_VL3_MANIFEST` | `evol_vl3_execution_manifest.v1.json` @ TEST_EXECUTION_SHA (`pytest tests/qualification/evol_x --collect-only`, `dev-ci`) | EVOL-X (W2-09) | **Body** of EVOL_CONSERVATIVE_FULL | same |
| Adj | `EVOL_ADJACENT_REGRESSION` | Explicit list in `evol_adjacent_regression_nodes.v1.json` (W5-07) | EVOL-X | Union only when W5-07 registers nodes | same |

**Set:** `EVOL_CONSERVATIVE_FULL = UNION(T1,T2,T3,Adj) \ evol_conservative_exclusions.v1.json`.

**Unknown critical consumer (CW-06):** any oracle-critical consumer not in graph and not `NON_CONSUMER` documented → `ImpactStatus`/`SelectionStatus` **fail-closed** (`CONSERVATIVE_FULL` or `REFUSED_INTEGRITY`) — **forbidden:** PASS on `evol_x_p3_l3` profile alone while omitting mandatory suites or while `critical_omissions>0`.

### II-B. Policy `PLATFORM_CONSERVATIVE_FULL` (explicitly out of EVOL-X parent closure)

| ID | Population | Owner | EVOL-X claim |
|----|------------|-------|--------------|
| **T4** | Full repo gate unit collect (`-m "gate and not no_ci"`, ~13k nodes, N-18) | Platform / QUAL-X Population E | **Forbidden** at EVOL-X W7 unless separate operator program opens T4 |

**Disposition:** roadmap §3.0.9 EVOL-X qualifies **change discipline** on Population A + EVOL VL3 — **not** entire platform gate unit. Document in `evol_frz_evidence.v1` footnote for FRZ-EVOL-06.

### II-C. DIM-B exclusion (canonical scope authority + OD-EVOL-10)

Source: catalog entry `QUAL-X-P5-DIM-B-PROD-Q-LIVE-EXCLUDED` with `mandatory_status=EXCLUDED_FROM_DIMENSION_A` and modules `_PROD_Q_LIVE_DIM_B` ([`_p5_freeze_selection_catalog.py`](tests/qualification/qual_x/_p5_freeze_selection_catalog.py) L348–361).

**Disposition:** DIM-B nodes **must not** appear in T2/T3 for EVOL-X Population A qualification. **Not** a silent shrink — documented in manifest `excluded_populations: ["DIM-B-PROD-Q-LIVE"]`.

### II-D. RC-7 exclusion (N18 + OD-EVOL-10 signed disposition)

Source: [EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md](docs/project/maintainers/qualification/evol_x/EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md) module `tests/unit/agents/test_reference_harness_strict.py` (ImportError RC-7).

**Disposition:** listed in `evol_conservative_exclusions.v1.json` with `reason=RC-7_BASELINE_DEFECT`, `scope=OUTSIDE_EVOL_VL3`, `tracked_owner=platform_harness`. **Qualification:** EVOL-X does **not** claim RC-7 collect; **FAIL** if any **non-excluded** T3 node is uncollectable. RC-7 alone **does not** block EVOL closure if exclusion signed under OD-EVOL-10.

### II-E. Uncollectable required node behavior

If any node ∈ (T1∪T2∪T3) \ exclusions fails `pytest --collect-only` → `SelectionStatus.REFUSED_INTEGRITY` + CI gate FAIL. **Forbidden:** xfail/skip to green; substituting T4 without operator authorization.

### II-F. Tasks (selector uses **EVOL_CONSERVATIVE_FULL** label, not PLATFORM)

| Task | Batch | OD | DONE |
|------|-------|-----|------|
| **W2-09** | B | **OD-EVOL-10 before start** | `evol_vl3_execution_manifest.v1.json` pinned @ SHA; CI job uses same list |
| **W3-05** | C | OD-EVOL-10 | `SelectionStatus` uses policy **II-A** only; enum may remain `CONSERVATIVE_FULL` with `reason_rule_id=EVOL-R7-SCOPED` |
| **W3-04** | C | — | verifies T1/T2 ids collectable |

---

## Part III — Operator decision timing (no self-approval)

| Order | OD | Before task(s) | Batch gate |
|------:|-----|----------------|------------|
| 1 | **OD-EVOLX-01** | any W2 code | W0-01 |
| 2 | **OD-EVOL-01** | ratify Option C @ 884885774 | W0-02 |
| 3 | **OD-EVOL-09** | W2-05 CW-04, X-03/06/07 proofs | W0-03 |
| 4 | **OD-EVOL-10** | **W2-09, W3-05**, collect-only baselines | W0-04 |
| 5 | **OD-EVOL-02** | W4-02 workflow matrix | W0-05 |
| 6 | **OD-EVOL-06** | W4-03 deploy facts | W0-05 |
| 7 | **OD-EVOL-05a/c** | W4-03 S7 scope | W0-05 |
| 8 | **OD-EVOL-05b** | **W4-04 only** (Option 2); default skip | W0-06 optional |
| 9 | **OD-EVOL-03, OD-EVOL-07, OD-EVOL-08** | W6-01 metrics thresholds / M-09 SLA | W0-07 |
| 10 | **OD-EVOL-11** | W7-01 independent audit | W0-08 |
| 11 | **OD-EVOL-12** | W7-03 FRZ-08/10 promotion wording | W7-03 |
| 12 | **OD-EVOLX-08** | W7-04 parent closure | W7-04 |

**W0 STOP:** missing signed row in [OPERATOR_DECISION_REGISTER](docs/project/maintainers/qualification/evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) for required OD at gate → Agent STOP.

---

## Part IV — S1–S8 experiment records (unique ownership)

Shared **implementation** allowed; **experiment record** unique per S.

| EXP | Scenario | Primary task | Shared impl | Fixture | Entry components | Test module | Oracle | Metric | FRZ | Evidence |
|-----|----------|--------------|-------------|---------|------------------|-------------|--------|--------|-----|----------|
| **EXP-S1** | S1 local agent blast | **W5-EXP-S1** | W4-01 isolation | `s1_local_agent/` | 2× app + agent change harness | `test_evol_x_exp_s1_local_agent.py` | O-F | M-06 | FRZ-03 | `evol_experiment_verdict.v1` |
| **EXP-S2** | S2 shared contract | **W5-EXP-S2** | W3-01..02 | `s2_shared_contract/` | COMPAT adapter + consumer index | `test_evol_x_exp_s2_shared_contract.py` | O-D | M-07 | FRZ-04 | same |
| **EXP-S3** | S3 in-flight workflow | **W5-EXP-S3** | W4-02 | `s3_workflow_inflight/` | profile_resolution pinning + checkpoint | `test_evol_x_exp_s3_workflow_inflight.py` | WM oracle | — | FRZ-05 | same |
| **EXP-S4** | S4 provider replace | **W5-EXP-S4** | W3-03 | `s4_provider_replace/` | CONFIG/RPL harness swap | `test_evol_x_exp_s4_provider_replace.py` | O-F | — | — | same |
| **EXP-S5** | S5 state migration | **W5-EXP-S5** | W3-03 | `s5_state_migration/` | schema registries + migration | `test_evol_x_exp_s5_state_migration.py` | COMPAT | M-07 | FRZ-04 | same |
| **EXP-S6** | S6 negative regression | **W5-EXP-S6** | W3-05..06 | `s6_negative_regression/` | selector + fault injection | `test_evol_x_exp_s6_negative_regression.py` | constructed | M-05 | FRZ-06 | same |
| **EXP-S7** | S7 rollback | **W5-EXP-S7** | W4-03 | `s7_rollback/` | AD harness + deployment_safety | `test_evol_x_exp_s7_rollback.py` | DM matrix | M-08 | FRZ-07 | same |
| **EXP-S8** | S8 new application | **W5-EXP-S8** | W4-05 | `s8_new_application/` | extension_safety + scaffold | `test_evol_x_exp_s8_new_application.py` | frozen_core | M-10 | FRZ-09 | same |

**Each EXP DONE:** positive arm PASS + negative arm PASS + verdict JSON committed under `evol_x_p03_independent_replay/experiments/` @ TEST_EXECUTION_SHA.

**Each EXP FAIL:** any arm FAIL or vacuous (flag-only) test.

**W5-07** (not an S): `frz/test_evol_x_frz_01..10.py`, `gates/test_evol_x_adjacent_stage_regression.py`, duplicate/typing gates — owner **W5-07** only.

---

## Part V — Protocol 03 & SHA lifecycle (B-01)

### V-0. Immutable SHA roles (never conflate)

| Role | Git meaning | When set | May equal another role? |
|------|-------------|----------|-------------------------|
| **PLAN_PUBLICATION_SHA** | Commit containing `EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md` | PRE-PUB-01 | Unique |
| **TEST_EXECUTION_SHA** | Commit **checked out** for all VL3 pytest + workflow CLI + artifact **generation** | End Batch F implementation | Unique per qualification cycle |
| **EVIDENCE_PUBLICATION_SHA** | **Separate descendant commit** that adds/updates only `evol_x_p03_independent_replay/**` (+ allowed ladder/checklist footnotes) | W6-06 | **Must differ** from TEST_EXECUTION_SHA unless tree-identical (then forbidden — republish required) |
| **INDEPENDENT_REPLAY_SHA** | Commit auditor checks out for replay (normally = TEST_EXECUTION_SHA) | W7-01 | = TEST_EXECUTION_SHA |
| **B6_ACCEPTANCE_PUBLICATION_SHA** | Commit adding `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` | W7-01 pass | Descendant of EVIDENCE_PUBLICATION_SHA |
| **FORMAL_CLOSURE_PUBLICATION_SHA** | Commit updating roadmap §3/§5 + checklist FRZ-EVOL + ladder closure row | W7-05 | Descendant of B6 |

**Forbidden:** embedding **`EVIDENCE_PUBLICATION_SHA`** (or any publication-role SHA) **inside** `MANIFEST.json` as a required field that must equal the commit being audited — that is a self-referential contract. **EVIDENCE_PUBLICATION_SHA** is always **`git rev-parse HEAD`** on the evidence checkout (Mode A). The manifest **must** record **`test_execution_sha`** (and **`plan_publication_sha`**, **`implementation_sha`**) explicitly; each bundled artifact is bound to **`test_execution_sha`** plus a **content digest** in `ARTIFACT_DIGESTS.json`.

**Provenance chain (git):**

```text
PLAN_PUBLICATION_SHA (plan authority; ancestor of implementation)
TEST_EXECUTION_SHA  →  (optional implementation commits after execution)
       ↓  merge-base --is-ancestor test_execution_sha evidence_HEAD
EVIDENCE_PUBLICATION_SHA (= evidence_HEAD)  →  B6_ACCEPTANCE_PUBLICATION_SHA  →  FORMAL_CLOSURE_PUBLICATION_SHA
```

**Mode A — `verify-evidence-publish`:** checkout **EVIDENCE_PUBLICATION_SHA** (`HEAD` = evidence commit).

1. Set `evidence_head = git rev-parse HEAD` (this **is** EVIDENCE_PUBLICATION_SHA; **do not** require a matching field inside `MANIFEST.json`).
2. Read **`test_execution_sha`** from MANIFEST; `git merge-base --is-ancestor test_execution_sha evidence_head` → exit 0 (**test execution is an ancestor of evidence publication**).
3. `git cat-file -t` + sha256 file bytes @ `evidence_head` for each path in `artifact_paths[]`; compare `ARTIFACT_DIGESTS.json` (content digest per artifact, not commit hash of self).
4. Embedded payloads (`source_commit_sha` in inventory/graph/selection/metrics, `evol_test_execution_manifest.v1`) **must equal** `test_execution_sha`, not `evidence_head`.
5. Compare **`evol_junit_outcomes.v1.json`** (canonical test identity + outcome + relevant provenance fields) bundled @ evidence against MANIFEST expectations — **not** byte-identical regenerated JUnit XML.

**Mode B — `verify-test-execution`:** checkout **`test_execution_sha`** from MANIFEST (`HEAD` must equal declared `test_execution_sha`).

1. Re-run VL3 pytest per MANIFEST `executed_command`; emit/compare **`evol_junit_outcomes.v1.json`** (canonical identities/outcomes) to the copy archived @ EVIDENCE_PUBLICATION_SHA (cross-commit semantic replay).
2. **Do not** require byte-identical JUnit XML regeneration across checkouts; XML may differ cosmetically if canonical outcome map matches.

Independent auditor (**W7-01**): Mode A @ evidence commit; Mode B @ **TEST_EXECUTION_SHA**; **B6** record cites exact **`evidence_publication_sha`** (= evidence commit hash) + `test_execution_sha` + `independent_replay_sha`.

### V-A. `MANIFEST.json` (written @ EVIDENCE_PUBLICATION_SHA)

**Required embedded fields:** `schema_version`, `stage=EVOL-X`, **`plan_publication_sha`**, **`test_execution_sha`**, `implementation_sha` (last code commit ≤ `test_execution_sha`), `independent_replay_sha` (default `test_execution_sha`), `pytest_profile`, `uv_extra`, `executed_command`, `dirty_tree=false`, optional `parent_test_execution_tree` (`git rev-parse test_execution_sha^{tree}`), `artifact_paths[]` (each path listed in `ARTIFACT_DIGESTS.json` with digest @ evidence commit).

**Forbidden embedded field:** `evidence_publication_sha` (or equivalent self-hash of the publication commit). Verifiers derive publication SHA from git only.

### V-B. `evol_test_execution_manifest.v1.json`

`source_commit_sha` (= TEST_EXECUTION_SHA only), `node_ids[]`, `manifest_content_digest`, `excluded_populations[]`, `exclusions[]`.

### V-C–V-F. (unchanged artifact types)

Inventory/graph/selection/metrics/experiment verdict/FRZ evidence — all bind `source_commit_sha` or `test_execution_sha` to **TEST_EXECUTION_SHA**.

### V-G. `ARTIFACT_DIGESTS.json`

`relpath → sha256(file bytes @ EVIDENCE_PUBLICATION_SHA checkout)`.

### V-H. B6 record @ B6_ACCEPTANCE_PUBLICATION_SHA

References exact **`evidence_publication_sha`** (the evidence bundle commit hash), **`test_execution_sha`**, **`independent_replay_sha`**, **`plan_publication_sha`**, auditor id, and replay mode outcomes (A @ evidence commit, B @ test execution).

### V-I. Verifier modes (summary)

See **V-0** Mode A / Mode B checkout contexts. Mode A: ancestry **`test_execution_sha` → evidence HEAD**, digests, canonical junit outcomes, manifest node coverage, FRZ×10, experiments×16. Mode B: **`HEAD == test_execution_sha`**, pytest replay, canonical outcome map vs evidence bundle.

**Negative tests (W6-03, `tests/qualification/evol_x/gates/test_evol_x_p03_integrity.py`):**

| Case | Expected |
|------|----------|
| Mode B: `HEAD` ≠ MANIFEST `test_execution_sha` | exit 2 |
| Mode A: `test_execution_sha` **not** ancestor of evidence `HEAD` | exit 2 |
| Mode A: reversed ancestry (`evidence_HEAD` ancestor of `test_execution_sha` only — execution after evidence) | exit 2 |
| MANIFEST contains forbidden self-field `evidence_publication_sha` | exit 2 |
| swapped canonical junit outcome for same test identity | exit 2 |
| missing manifest node in canonical outcomes | exit 2 |
| stale `source_commit_sha` in inventory | exit 2 |
| tampered ARTIFACT_DIGESTS entry | exit 2 |

**Producer:** W6-02, W6-06 · **Verifier:** W6-03 · **Independent replay:** W7-01 Mode A+B.

---

## Part VI — Full task matrix (W2–W7)

**Legend:** Batch B/C/D/E/F/G · Scope Q=`testing_support/change_safety`, T=`tests/qualification/evol_x`, S=`scripts/qualification`

### PRE-PUB & W0

| ID | Batch | Req / Source | Files | AS-IS → TO-BE | I/O | Owner | Test+ | Test− | Oracle | Evidence | Deps | OD | DONE | FAIL |
|----|-------|--------------|-------|---------------|-----|-------|-------|-------|--------|----------|------|-----|------|------|
| PRE-PUB-01 | PRE | plan durability | `docs/.../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md` | missing → committed v3 | — | docs | review checklist | — | — | PLAN_PUBLICATION_SHA | — | — | file in repo + ladder row | absent |
| W0-01 | W0 | OD-EVOLX-01 roadmap §3.0.9 | register | PENDING | signed row | — | operator | — | — | register PDF/md | — | OD-EVOLX-01 | signed | unsigned |
| W0-02 | W0 | OD-EVOL-01 §9.1 arch | register | code without sign | ratified C | — | operator | — | — | same | PRE-PUB | OD-EVOL-01 | signed | unsigned |
| W0-03 | W0 | OD-EVOL-09 CW-04 | exclusion register | X-03 open | proofs or reject | — | operator | — | — | W1_EXCLUSION updated | — | OD-EVOL-09 | signed | CW-04 blocked |
| W0-04 | W0 | OD-EVOL-10 II-A | register | — | CI/manifest auth | — | operator | — | — | OD-10 record | — | OD-EVOL-10 | signed before W2-09 | W3-05 blocked |
| W0-05 | W0 | OD-02/06/05a/c | register | — | signed | — | operator | — | — | — | W0-01..02 | ODs | signed | W4 blocked |
| W0-06 | W0 | OD-EVOL-05b | register | — | Option 2 gate | — | operator | — | — | — | W0-05 | OD-05b | signed or explicit Option1 | W4-04 ambiguous |
| W0-07 | W0 | OD-03/07/08 | register | — | M thresholds | — | operator | — | — | ratification memo | — | ODs | signed | W6-01 blocked |
| W0-08 | W0 | OD-EVOL-11 | register | — | auditor named | — | operator | — | — | — | — | OD-11 | named | W7-01 blocked |

### W2 (Batch B)

| ID | Batch | Req | Files | AS-IS → TO-BE | I/O | Scope | Test+ | Test− | Oracle | Evidence | Deps | OD | DONE | FAIL |
|----|-------|-----|-------|---------------|-----|-------|-------|-------|--------|----------|------|-----|------|------|
| W2-01 | B | EVOL contracts FRZ all | Q/contracts/*, ports.py | partial → full spec | Pydantic v1 artifacts | Q | test_evol_x_contracts_* | Any in contracts | — | round-trip JSON | — | — | VL1 green | schema drift |
| W2-02 | B | IB-1..6 | T/gates/test_evol_x_import_boundaries.py | exists → extend IB-1 tests | — | T | import boundary | synthetic bad import | — | junit | W2-01 | — | green | boundary leak |
| W2-03 | B | D4 frozen data | T/data/*.v1.json, loaders | partial → validated load | JSON → typed | T+Q | test_evol_x_contracts_and_data | missing source_authority | — | data digest | W2-01 | — | load OK | invalid owner |
| W2-04 | B | D2/D3 E-01..12 | Q/discovery/extractors.py | IMPORTS only → 19 REL | SHA→edges | Q | extractor unit | dynamic→UNKNOWN | — | extractor manifest digest | W2-03 | — | REL edges | silent drop |
| W2-05 | B | D1,D5,D6,CW01-09 | Q/discovery/closed_world.py, T/gates/test_evol_x_closed_world_* | smoke → full engine | inventory v1 | Q | CW parametrized | mutations CW-07 | 2nd method CW-06 | inventory JSON | W2-04 | 09 for CW-04 | all CW pass | any CW fail |
| W2-06 | B | D8,D9,CW08,10 | Q/graph/builder,digest,incremental | partial → full digest | graph v1 | Q | test_evol_x_graph_* | bad digest | dual build | graph_digest | W2-05 | — | CW-08/10 | mismatch |
| W2-07 | B | D7,XG02,EVOL-01 | Q/graph/impact,traversal,coverage | wrong digest, hardcoded 0 → oracle omissions | ImpactAssessmentResult | Q | graph_coverage, impact | leave-one-out | CW-06 | impact reports | W2-05-06 | — | XG-02 real | hidden omission |
| W2-08 | B | EVOL-10 intake | Q/intake/*, ChangeSetPort | basic → rename+class | git paths | Q | intake tests | unknown path | — | change set JSON | W2-05 | — | unknown:path works | silent complete |
| W2-09 | B | II-A T3, OD-10 | unit-tests.yml, T/data/evol_vl3_execution_manifest.v1.json | profile stub → manifest pin | node list | T+CI | collect integrity | missing node | — | manifest @ SHA | W2-01 | **OD-10** | manifest frozen | collect fail |

### W3 (Batch C) — requires W0-04 OD-10

| ID | Batch | Req | Files | AS-IS → TO-BE | I/O | Scope | Test+ | Test− | Oracle | Evidence | Deps | OD | DONE | FAIL |
|----|-------|-----|-------|---------------|-----|-------|-------|-------|--------|----------|------|-----|------|------|
| W3-01 | C | EVOL-03 FRZ-04 | Q/assess/contract_diff, T/adapters/compat_x | stub → COMPAT policy diff | CompatibilityAtChangeResult | Q+T | test_evol_x_compat_adapter | incompatible arm | COMPAT-X | diff records | W2-07 | — | INCOMPATIBLE detect | silent compat |
| W3-02 | C | CW-06 FRZ-04 | Q/assess/consumers, consumer_oracle | IMPORTS → text oracle | ConsumerClosureReport | Q | consumer_index, CW06 | hidden consumer | O-D | closure v1 | W2-05 | — | orphan=0 | orphan>0 |
| W3-03 | C | EVOL-03 persisted | contract_diff extended | field-only → schema/event | — | Q | schema diff tests | bump w/o migration | registries | — | W3-01 | — | MIGRATION_REQUIRED | silent break |
| W3-04 | C | FRZ-06 T1/T2 | T/adapters/qual_x, policy | partial → TESTED_BY edges | SuiteCatalogPort | T | catalog tests | bad suite id | QUAL-X manifest | collection digest | W2-03 | — | 100% ids valid | reconcile!=0 |
| W3-05 | C | FRZ-06 M-04 II-A | Q/selection/selector, ci_manifest adapter | mandatory-only FULL → union T1-3 | SelectionDecision | Q | closed_world_smoke, selection | shrink manifest | policy R7 | selection manifest | W2-09,W3-04 | **OD-10** | CONSERVATIVE correct | under-select |
| W3-06 | C | M-04 oracle | Q/selection/oracle, fault tests | partial → ≥8 fault classes | SelectionStatus | Q | test_evol_x_selection_fault_injection | 8 failures | oracle table | fault log | W3-05 | — | all faults map | bypass |

### W4 (Batch D) — requires W0-05 ODs

| ID | Batch | Req | Files | AS-IS → TO-BE | I/O | Scope | Test+ | Test− | Oracle | Evidence | Deps | OD | DONE | FAIL |
|----|-------|-----|-------|---------------|-----|-------|-------|-------|--------|----------|------|-----|------|------|
| W4-01 | C/D | EVOL-02 FRZ-03 | Q/assess/isolation.py | stub → capability blast | — | Q | domain_safety_smoke, EXP-S1 | cross-app | O-F | isolation section | W3-02 | — | S1 precursor | escape |
| W4-02 | D | EVOL-04 FRZ-05 | workflow_evolution,matrix JSON | flags → WM-01..15 | WorkflowEvolutionVerdict | Q | workflow tests, EXP-S3 | in-flight adopt | WM | matrix+verdict | W3 | **OD-02** | WM rows | unsafe adopt |
| W4-03 | D | EVOL-06 FRZ-07 | deployment_safety,matrix, T/adapters/agent_distribution | partial → DM-01..07 harness | DeploySafetyAssessment | Q+T | deploy tests, EXP-S7 | DM-04 | DM | assessment v1 | W3 | **OD-05a,06** | S7 matrix | false safe |
| W4-04 | D | OD-05b only | agents AD port (optional) | none → default-off port | precondition | P | AD regressions | refuse path | — | port tests | W4-03 | **OD-05b Opt2** | opt-in green | enabled w/o OD |
| W4-05 | D | EVOL-08 FRZ-09 | extension_safety,frozen_core | stub → touch classifier | — | Q | extension tests, EXP-S8 | core touch | registry | S8 evidence | W2-03 | — | M-10 path | unauthorized edit |
| W4-06 | D | EVOL-07 FRZ-08 | diagnostics/dossier.py | skeleton → Q1–Q6 | ChangeDossier | Q | dossier tests | stale flag | — | dossier v1 | W3-05 | — | M-09 drill input | mutates runtime |
| W4-07 | D | EVOL-09 FRZ-10 | triggers,requal JSON | static → executable T1..T9 | baseline manifest | Q | requal tests | stale accept | freshness | triggers v1 | W2-05 | — | M-11 tests | stale pass |
| W4-08 | D | EVOL-10 | evol_x_change_workflow.py | impact only → full pipeline | JSON stages | S | CLI integration | stage fail exit≠0 | ECAP A1-A8 | workflow log | W4-02..07 | — | end-to-end JSON | silent 0 |

### W5 (Batch E)

| ID | Batch | Req | Files | AS-IS → TO-BE | Scope | Test+ | Test− | DONE | FAIL |
|----|-------|-----|-------|---------------|-------|-------|-------|------|------|
| W5-EXP-S1 | E | S1,FRZ-03,M-06 | fixture s1, test module | missing → behavioral | T | blast radius | cross-app escape | both arms PASS | escape |
| W5-EXP-S2 | E | S2,FRZ-04,M-07 | s2 fixture | missing → consumers | T | all listed | hidden | both PASS | orphan |
| W5-EXP-S3 | E | S3,FRZ-05 | s3 fixture | flag→pin/restart | T | pin preserved | unsafe adopt | both PASS | unsafe=1 |
| W5-EXP-S4 | E | S4 | s4 fixture | missing → swap | T | no rewrite | rewrite req | both PASS | rewrite |
| W5-EXP-S5 | E | S5,FRZ-04,M-07 | s5 fixture | missing → migrate/block | T | MIGRATE | silent break | both PASS | silent |
| W5-EXP-S6 | E | S6,FRZ-06,M-05 | s6 fixture | missing → injections | T | gate FAIL | bypass | 100% detect | bypass |
| W5-EXP-S7 | E | S7,FRZ-07,M-08 | s7 fixture | flag→AD+DM | T | DM pass | DM-04 unsafe | 100% outcomes | false safe |
| W5-EXP-S8 | E | S8,FRZ-09,M-10 | s8 fixture | missing → scaffold | T | extension | core edit | 0 unauthorized | core edit |
| W5-07 | E | FRZ-01..10,L2 | frz/test_evol_x_frz_NN.py, adjacent, no_dup | hooks→behavioral | T | 10 FRZ modules | no evidence | all FRZ tests green | vacuous |

### W6 (Batch F) — requires W0-07

| ID | Batch | Req | Files | DONE | FAIL |
|----|-------|-----|-------|------|------|
| W6-01 | F | M-01..11 | Q/metrics/runner, T/metrics/* | real populations, no 1/1 stub | NOT_MEASURED at closure |
| W6-02 | F | VL3 | manifest, single pytest cmd | one green VL3 run | split runs |
| W6-03 | F | P03 V | evol_x_p03_evidence.py, test_evol_x_p03_integrity.py | Mode A @ evidence + Mode B @ TEST_EXECUTION_SHA; canonical junit outcomes; 8 neg cases Part V-I | HEAD-only pass |
| W6-04 | F | CI replay | GHA dispatch @ TEST_EXECUTION_SHA | run id recorded | SHA drift |
| W6-05 | F | pre-cert | review worksheet | non-certifying sign-off | — |
| W6-06 | F | publication | evol_x_p03_independent_replay/* @ **EVIDENCE_PUBLICATION_SHA** | separate commit from TEST_EXECUTION_SHA | same SHA publish |

### W7 (Batch G)

| ID | Batch | Req | DONE | FAIL |
|----|-------|-----|------|------|
| W7-01 | G | P03-B5/B6 | independent replay PASS | auditor=implementer |
| W7-02 | G | P03-B4 | corrective batch replay @ new SHA | silent SHA move |
| W7-03 | G | FRZ promotion | checklist rows per OD-12 | placeholder PASS |
| W7-04 | G | OD-EVOLX-08 | signature record | unsigned |
| W7-05 | G | roadmap RD | EVOL-X CLOSED SCOPED published | stale ledger |
| W7-06 | G | ENT handoff | EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD | missing digests |

---

## Part VII — Mechanical coverage matrix (B-03) — one row per ID

Columns: **Task · Val+ · Val− · Oracle · Producer · Verifier · DONE**

### VII-A. EVOL-01..10

| ID | Task | Val+ | Val− | Oracle | Producer | Verifier | DONE |
|----|------|------|------|--------|----------|----------|------|
| EVOL-01 | W2-07,W2-08 | test_evol_x_impact_* | unknown:path, leave-one-out | CW-06 | W2-07 CLI | W6-03 | impact COMPLETE or fail-closed |
| EVOL-02 | W4-01,W5-EXP-S1,S8 | EXP-S1+ | cross-app arm | O-F | W5-EXP-S1 | W7-01 | escape=0 |
| EVOL-03 | W3-01..03 | compat+consumer | hidden consumer | O-D | W3-02 | W6-03 | INCOMPATIBLE detected |
| EVOL-04 | W4-02,W5-EXP-S3 | WM+EXP-S3+ | unsafe adopt | WM matrix | W5-EXP-S3 | W7-01 | unsafe_adoption=0 |
| EVOL-05 | W3-05..06 | selection+fault | 8 fault classes | selection oracle | W3-05 | W6-03 | M-04=100% scoped |
| EVOL-06 | W4-03,W5-EXP-S7 | DM+EXP-S7+ | DM-04 arm | DM matrix | W5-EXP-S7 | W7-01 | M-08=100% qual layer |
| EVOL-07 | W4-06 | dossier Q1–Q6 | stale evidence | dossier schema | W4-06 | W6-03 | questions answerable |
| EVOL-08 | W4-05,W5-EXP-S8 | EXP-S8+ | core edit arm | frozen_core | W5-EXP-S8 | W7-01 | M-10 satisfied |
| EVOL-09 | W4-07 | trigger T1..T9 | stale accept | freshness counter | W4-07 | W6-03 | M-11 executable |
| EVOL-10 | W4-08 | CLI e2e | stage silent 0 | ECAP A1–A8 | W4-08 | W7-01 | all stages coded |

### VII-B. FRZ-EVOL-01..10

| ID | Task | Val+ | Val− | Oracle | Producer | Verifier | DONE |
|----|------|------|------|--------|----------|----------|------|
| FRZ-EVOL-01 | W5-07 frz_01,W2-07 | frz_01+impact | vacuous hook | graph+impact | W5-07 | B6 auditor | behavioral PASS |
| FRZ-EVOL-02 | W5-07 frz_02,W2-07 | coverage+CW06 | leave-one-out | 2nd method | W2-07 | W6-03 | M-02+CW green |
| FRZ-EVOL-03 | W5-07 frz_03,W5-EXP-S1 | isolation+S1 | cross-app | O-F | W5-EXP-S1 | B6 | S1 PASS |
| FRZ-EVOL-04 | W5-07 frz_04,W5-EXP-S2,S5 | consumer+S2/S5 | hidden/break | O-D+COMPAT | W5-EXP-S2,S5 | B6 | orphans=0 |
| FRZ-EVOL-05 | W5-07 frz_05,W5-EXP-S3 | WM+S3 | in-flight adopt | WM | W5-EXP-S3 | B6 | S3 dual arm |
| FRZ-EVOL-06 | W5-07 frz_06,W3-05..06 | selection scoped | bypass S6 | EVOL_CONSERVATIVE policy | W3-05 | W6-03 | no PLATFORM claim |
| FRZ-EVOL-07 | W5-07 frz_07,W5-EXP-S7,W4-03 | S7+assessor | DM-04 | DM+Part X | W5-EXP-S7 | B6 | **scoped PASS only** (Part X) |
| FRZ-EVOL-08 | W5-07 frz_08,W4-06 | dossier | stale | passive read | W4-06 | B6+OD-12 footnote | dossier evidence |
| FRZ-EVOL-09 | W5-07 frz_09,W5-EXP-S8 | S8+extension | core edit | registry | W5-EXP-S8 | B6 | S8 PASS |
| FRZ-EVOL-10 | W5-07 frz_10,W4-07 | triggers | stale | freshness | W4-07 | B6+OD-12 | triggers executable |

### VII-C. CW-01..10

| ID | Task | Val+ | Val− | Oracle | Producer | Verifier | DONE |
|----|------|------|------|--------|----------|----------|------|
| CW-01 | W2-05 | gate CW01 | inject unclassified | set diff | W2-05 | test closed_world | unclassified=0 |
| CW-02 | W2-05 | gate CW02 | dual owner | owner rules | W2-05 | same | conflicts=0 |
| CW-03 | W2-05 | gate CW03 | orphan import | reachability | W2-05 | same | orphans=0 |
| CW-04 | W2-05 | gate CW04 | bad X predicate | exclusion reg | W2-05 | OD-09 | X valid |
| CW-05 | W2-05 | gate CW05 | unowned test | pytest↔AST | W2-05 | same | reconciled |
| CW-06 | W2-05,W3-02 | gate CW06 | hidden ref | text oracle | W3-02 | W6-03 | parity |
| CW-07 | W2-05 | 10 mutations | mutation miss | probe log | W2-05 | same | 10/10 flip |
| CW-08 | W2-06 | dual build | digest drift | dual build | W2-06 | CI ubuntu | match |
| CW-09 | W2-05 | parity report | unexplained delta | ext inventories | W2-05 | B6 | explained |
| CW-10 | W2-06 | incremental=full | forced mismatch | rebuild | W2-06 | same | equal |

### VII-D. D1..D10

| ID | Task | Val+ | Val− | Producer | Verifier | DONE |
|----|------|------|------|----------|----------|------|
| D1 | W2-05 | universe scan | extra root file | W2-05 | CW-01 | R complete |
| D2 | W2-04 | symbol extract | — | W2-04 | unit | symbols present |
| D3 | W2-04 | 19 REL | dynamic→UNKNOWN | W2-04 | graph | edges typed |
| D4 | W2-03,W2-05 | owner load | invalid owner | W2-03 | CW-02 | owners ok |
| D5 | W2-05 | X register | — | W2-05 | CW-04 | X ok |
| D6 | W2-05 | CW suite | — | W2-05 | all CW | gates run |
| D7 | W2-04,W2-07 | UNKNOWN reg | — | W2-07 | impact | unknown explicit |
| D8 | W2-06 | digest | tamper | W2-06 | W6-03 | stable digest |
| D9 | W2-06 | incremental | mismatch | W2-06 | CW-10 | dev only |
| D10 | W6-03,W7-06 | reconcile | stale pub | W6-06 | handoff | digest chain |

### VII-E. S1..S8

| ID | Task | Val+ | Val− | Oracle | Producer | Verifier | DONE |
|----|------|------|------|--------|----------|----------|------|
| S1 | W5-EXP-S1 | blast A | escape | O-F | verdict JSON | W7-01 | dual arm |
| S2 | W5-EXP-S2 | consumers | hidden | O-D | verdict | B6 | dual arm |
| S3 | W5-EXP-S3 | pin/restart | unsafe adopt | WM | verdict | B6 | dual arm |
| S4 | W5-EXP-S4 | provider swap | rewrite | O-F | verdict | B6 | dual arm |
| S5 | W5-EXP-S5 | migrate | silent break | COMPAT | verdict | B6 | dual arm |
| S6 | W5-EXP-S6 | gate fail | bypass | constructed | verdict | B6 | 100% detect |
| S7 | W5-EXP-S7 | DM ok | DM-04 | DM | verdict | B6 | 100% qual |
| S8 | W5-EXP-S8 | scaffold | core edit | frozen | verdict | B6 | 0 core edits |

### VII-F. EVOL-M-01..11 (+ supplements)

| ID | Task | Val+ | Val− | Producer | Verifier | DONE |
|----|------|------|------|----------|----------|------|
| M-01 | W6-01 | m01 test | stub 1/1 | metrics runner | W6-03 | ≥ threshold OD-03 |
| M-02 | W6-01,W2-07 | m02+leave-one-out | fake omissions | W6-01 | W6-03 | ≥90%+M-02C=0 |
| M-03 | W6-01 | m03 precision | — | W6-01 | B6 | ≥98% S1–S6 |
| M-03R/C/E | W6-01 | supplement tests | escape | W6-01 | B6 | per OD-03 |
| M-04 | W3-06,W6-01 | fault+metric | bypass | W3-06 | W6-03 | 100% |
| M-05 | W5-EXP-S6 | s6 | bypass | W6-01 | B6 | 100% |
| M-06 | W5-EXP-S1,S8 | escape metric | — | W6-01 | B6 | 0 |
| M-07 | W5-EXP-S2,S5 | incompatible | — | W6-01 | B6 | 100% |
| M-08 | W5-EXP-S7 | s7 | false safe | W6-01 | B6 | 100% qual |
| M-09 | W4-06,W6-01 | drill | — | dossier | OD-08+B6 | SLA met |
| M-10 | W5-EXP-S8 | s8 | core | W6-01 | B6 | 0 |
| M-11 | W4-07,W6-01 | freshness | stale | W4-07 | W6-03 | triggers ok |

### VII-G. Findings F-P0 / F-P1 / F-P2 (each row)

| ID | Task | Val− | Verifier | DONE |
|----|------|------|----------|------|
| F-P0-01 | W2-07 | leave-one-out | W6-03 | real omissions |
| F-P0-02 | W3-05 | shrink EVOL manifest | W6-03 | II-A union |
| F-P0-03 | W6-03 | 8 neg cases Part V-I | self+gate | no HEAD-only |
| F-P0-04 | W5-EXP-S1..S8 | vacuous tests removed | B6 | behavioral |
| F-P0-05 | W5-07 frz_* | ID-only test gone | B6 | behavioral |
| F-P0-06 | W6-01 | fake metrics | W6-03 | real populations |
| F-P1-01 | W2-04 | import-only graph | CW-06 | semantic REL |
| F-P1-02 | W3-05 | under-select | oracle | TESTED_BY |
| F-P1-03 | W3-05 | uuid decision | digest id | deterministic |
| F-P1-04 | W5-EXP-S3,S7 | flag smokes replaced | B6 | harness |
| F-P1-05 | W5-EXP-* | fixtures exist | collect | trees present |
| F-P1-06 | W2-05 | CW missing | CW suite | all CW |
| F-P1-07 | W0-* | unsigned ODs | operator | signed |
| F-P1-08 | W2-09,N18 | RC-7 excluded | manifest | formal exclusion |
| F-P2-01 | W7-05 | — | roadmap | ledger synced |
| F-P2-02 | W2-07 | wrong graph digest | W6-03 | semantic digest |
| F-P2-03 | W3-06,W5-07 | thin negs | junit | expanded |

### VII-H. Mandatory evidence artifacts

| Artifact | Producer | Verifier | DONE |
|----------|----------|----------|------|
| MANIFEST.json | W6-06 | W6-03 Mode A | `test_execution_sha`+plan+impl; no self pub SHA |
| ARTIFACT_DIGESTS.json | W6-06 | W6-03 tamper | all paths @ evidence HEAD |
| evol_test_execution_manifest.v1 | W6-02 | Mode B | @ TEST_EXECUTION_SHA |
| evol_junit_outcomes.v1.json | W6-02 | W6-03 A+B | canonical id+outcome vs manifest |
| evol_*_manifest inventory/graph/selection | W2-06,W3-05 | W6-03 | digests match `test_execution_sha` |
| junit XML (archive) | W6-02 | W6-03 | human-readable; not byte-replay oracle |
| evol_metrics.v1 | W6-01 | W6-03 | no NOT_MEASURED @ closure |
| 16× experiment verdicts | W5-EXP-S* | W6-03 | 8×2 arms |
| evol_frz_evidence.v1 | W5-07 | B6 | 10 rows |
| B6 PASS record | W7-01 | operator | published |
| ENT handoff | W7-06 | B6 | digests |

### VII-I. Operator decision gates (execution blockers until signed — plan assigns task)

| OD | Gate task | Val+ | DONE (plan) | Execution blocked until |
|----|-----------|------|-------------|-------------------------|
| OD-EVOLX-01 | W0-01 | register row | task W0-01 complete | Batch B |
| OD-EVOL-01 | W0-02 | register | W0-02 | Batch B |
| OD-EVOL-09 | W0-03 | register+CW-04 | W0-03 | W2-05 CW-04 |
| OD-EVOL-10 | W0-04 | register+manifest auth | W0-04 | W2-09,W3-05 |
| OD-EVOL-02 | W0-05 | register | W0-05 | W4-02 |
| OD-EVOL-06 | W0-05 | register | W0-05 | W4-03 |
| OD-EVOL-05a/c | W0-05 | register | W0-05 | W4-03 S7 scope |
| OD-EVOL-05b | W0-06 | explicit Opt1/2 | W0-06 | W4-04 if Opt2 |
| OD-EVOL-03/07/08 | W0-07 | ratification memo | W0-07 | W6-01 |
| OD-EVOL-11 | W0-08 | auditor named | W0-08 | W7-01 |
| OD-EVOL-12 | W7-03 | register | W7-03 | checklist edit |
| OD-EVOLX-08 | W7-04 | signature record | W7-04 | parent close |

---

## Part X — FRZ-EVOL-07 production boundary (B-04)

**Checklist (L275):** deployment / rollback / roll-forward **qualified** or unsafe refusal **explicit** — durable state + contract compatibility (**S7**); PROD-Q supporting.

**ECAP:** FRZ-EVOL-07 = change-safety row; **no** global PROD-Q PASS implied.

**Plan claim (not pre-approved FRZ PASS):** Qualification-only assessor **may** satisfy **scoped EVOL-X** FRZ-EVOL-07 **if and only if** evidence includes:

- W5-EXP-S7 positive + **DM-04 negative** arm (unsafe rollback **refused** by assessor/harness, not merely undocumented).
- W4-03 `DeploySafetyAssessment` for DM-01..07 with typed `REFUSE` reasons.
- AD harness exercises **real** `ActivationService.rollback` path where applicable **without** claiming durable-state enforcement beyond assessor.
- `evol_frz_evidence.v1` row FRZ-EVOL-07 `scope_footnote`: *«Qualification-layer deploy/rollback safety; not live PROD-Q production enforcement.»*

**W7-03 independent gate (FRZ-EVOL-07):** Before any checklist promotion, **OD-EVOL-11 auditor** (or delegate) **independently confirms** that the **exact canonical checklist wording** (FRZ-EVOL-07 @ L275) supports the **qualification-only scoped** claim above. If wording does **not** support scoped qualification-only acceptance → keep **FRZ-EVOL-07 OPEN** or obtain operator authorization for production enforcement (**OD-EVOL-05b** Option 2 + **W4-04**); **never** manufacture a global or production FRZ PASS from EVOL-X evidence alone.

**Allowed B6 wording only after W7-03 confirmation:** FRZ-EVOL-07 → **PASS (EVOL-X scoped)** with footnote above + S7 + M-08 @ TEST_EXECUTION_SHA — **proposed** acceptance text, not granted by this plan publication.

**Forbidden claims (never in B6/roadmap/checklist):**

- Production `ActivationService.rollback` refuses all DM-04 durable-incompatible rollbacks in operator environments.
- FRZ-EVOL-07 substitutes for PROD-Q live slice or global FRZ-PRD PASS.
- «Binary restart safe» rollback without assessor evidence.

**If operator requires production precondition enforcement:** must authorize **OD-EVOL-05b Option 2**, implement **W4-04** (default-off port), add **negative integration tests** on strict composition path, update Agent Distribution domain docs — **separate** from scoped PASS; otherwise FRZ-EVOL-07 remains **blocked for production claim** (scoped qual PASS still allowed).

**W4-04 FAIL condition:** port enabled without OD-EVOL-05b Option 2 signed.

---

## Part VIII — Execution batches (unchanged structure)

PRE-PUB → W0 (ordered Part III) → **B** W2 → **C** W3 → **D** W4 → **E** W5 → **F** W6 → **G** W7.

**VL3 once @ F.** **No implementation** in plan mode.

---

## Part IX — Final self-audit (plan completeness vs execution readiness)

| Check | Plan coverage | Notes |
|-------|---------------|-------|
| Mandatory norm IDs without assigned task | **0** | Part VII A–I |
| Tasks without DONE criteria | **0** | Part VI |
| Tasks requiring Val− without listed Val− | **0** | W0 exempt (operator-only) |
| Evidence artifacts without producer **and** verifier | **0** | Part VII-H + Part V |
| Ambiguous CW/D/S/M identifier reuse | **0** | Part I, II |
| Operator ODs without gate task | **0** | Part VII-I |
| False «PLATFORM conservative = EVOL closure» claim | **0** | Part II-B |
| FRZ-EVOL-07 production over-claim | **0** | Part X scoped boundary |

**Execution readiness (not counted as plan defects):** Batch B–G remain **blocked** until operator signs ODs per Part III/VII-I — **no self-approval**.

**B-01..B-04 reconciliation:** embedded in Part V, II, VII, X.

---

## Final status

**EVOL-X FINAL COMPLETION MASTER PLAN v3 — PUBLISHED / READY FOR OPERATOR AUTHORIZATION**

**Audited implementation baseline (partial W2–W6, not qualification evidence):** `8848857741f4b04e5b476dca3c47f7d0a60bf672`

**Next (execution, not in this publication batch):** **W0-01..08** (operator ODs) → Batch **B** (W2). **STOP** before W2 until Part III / VII-I gates clear.
