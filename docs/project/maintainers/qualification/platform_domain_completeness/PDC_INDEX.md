# PDC-X — campaign index (SSOT hub)

**Stage ID:** `PDC-X` (`PLATFORM-DOMAIN-COMPLETENESS-X`)
**Roadmap status:** `PLANNED / MANDATORY / CURRENT` (roadmap §3, §3.2, §5)
**Position:** `MCP-SERVER-X` → **`PDC-X`** → `ARCH-FREEZE`
**Baseline pin:** `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b`
**ROADMAP_AMENDMENT_SHA:** `bc9ef52bd690738e71638a599eec64c76ac356c9` (PR #461 merge; see [`PDC_SHA_LADDER.md`](./PDC_SHA_LADDER.md))
**Tickets:** P0 [`T-IGX-036-A.md`](../../../../governance/tickets/T-IGX-036-A.md) · P1 [`T-IGX-036-B.md`](../../../../governance/tickets/T-IGX-036-B.md) · P2 [`T-IGX-036-C.md`](../../../../governance/tickets/T-IGX-036-C.md) · P3 [`T-IGX-036-D.md`](../../../../governance/tickets/T-IGX-036-D.md) · P3-08 lock [`T-IGX-036-E.md`](../../../../governance/tickets/T-IGX-036-E.md) · ODs [`T-IGX-036-F.md`](../../../../governance/tickets/T-IGX-036-F.md) · P4 [`T-IGX-036-G.md`](../../../../governance/tickets/T-IGX-036-G.md) · P5-DOC [`T-IGX-036-H.md`](../../../../governance/tickets/T-IGX-036-H.md)

PDC-X is **one** campaign with eight certification workstreams (`PDC-D1`..`PDC-D8`). They are **not** separate roadmap stages.

## Authority map (one SSOT per concern; this dossier links and never duplicates)

| Concern | SSOT artifact |
| --- | --- |
| Program sequence, stage status, closure | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3 / §3.2 / §5 |
| Freeze criteria (225 / 26) | [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) |
| Assurance operating protocol | [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) |
| Independent certification | [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) + [`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json); applicability to PDC-X → [`OD-PDC-IIAC-SCOPE`](./operator_decisions/OD-PDC-IIAC-SCOPE.md) |
| ARCH-FREEZE WBS (successor; 54 tasks preserved) | [`ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md`](../ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md) |
| Predecessor campaign | [`mcp_server_x/MCP_SERVER_X_INDEX.md`](../mcp_server_x/MCP_SERVER_X_INDEX.md) |

## Dossier artifacts

| Artifact | Phase | Status |
| --- | --- | --- |
| `PDC_MASTER_PLAN` → [`PDC_MASTER_PLAN.md`](./PDC_MASTER_PLAN.md) | P0 | Published (pointer to the approved plan) |
| `PDC_BASELINE_AND_SCOPE` → [`PDC_BASELINE_AND_SCOPE.md`](./PDC_BASELINE_AND_SCOPE.md) | P0 | Published |
| `PDC_SHA_LADDER` → [`PDC_SHA_LADDER.md`](./PDC_SHA_LADDER.md) | P0 → P9 | Open |
| `PDC_OPERATOR_DECISIONS` → [`PDC_OPERATOR_DECISIONS.md`](./PDC_OPERATOR_DECISIONS.md) | P0 → P9 | Open |
| `PDC_CLOSED_WORLD_INVENTORY` → [`PDC_CLOSED_WORLD_INVENTORY.md`](./PDC_CLOSED_WORLD_INVENTORY.md) + [`.json`](./PDC_CLOSED_WORLD_INVENTORY.json) | P1 | Published (T-IGX-036-B) |
| `PDC_DOMAIN_REQUIREMENTS_MATRIX` → [`.md`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.md) + [`.json`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.json) | P2 | Published (T-IGX-036-C; generated; **172 rows** per `pdc_x_requirements.py --check`, this index said 168 earlier) |
| `PDC_GAP_REGISTER` → [`PDC_GAP_REGISTER.md`](./PDC_GAP_REGISTER.md) | P2 | Published (generated P2 snapshot — byte-locked by `pdc_x_requirements.py --check`; **superseded for current verdict authority** by IA-GLOBAL Addenda 1–3, [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md), and signed [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) — see §P2-Snapshot Supersession below) |
| `PDC_FRZ_COVERAGE_MATRIX` → [`PDC_FRZ_COVERAGE_MATRIX.md`](./PDC_FRZ_COVERAGE_MATRIX.md) | P2 | Published (generated; OD-PDC-FRZ-01 approved) |
| `PDC_OWNERSHIP_MATRIX` → [`PDC_OWNERSHIP_MATRIX.md`](./PDC_OWNERSHIP_MATRIX.md) | P3 | Locked (OD-PDC-ADR-01 approved; T-IGX-036-E) |
| `PDC_ARCHITECTURE_DECISIONS` → [`PDC_ARCHITECTURE_DECISIONS.md`](./PDC_ARCHITECTURE_DECISIONS.md) | P3 | Locked (ADR-PDC-01..08 Accepted) |
| `PDC_IMPLEMENTATION_WBS` → [`PDC_IMPLEMENTATION_WBS.md`](./PDC_IMPLEMENTATION_WBS.md) | P4 | Published (generated; T-IGX-036-G) |
| `PDC_DEPENDENCY_DAG` → [`PDC_DEPENDENCY_DAG.md`](./PDC_DEPENDENCY_DAG.md) | P4 | Published (generated) |
| `PDC_CROSS_DOMAIN_TEST_MATRIX` → [`PDC_CROSS_DOMAIN_TEST_MATRIX.md`](./PDC_CROSS_DOMAIN_TEST_MATRIX.md) | P4 | Published (generated) |
| `PDC_REG_DIAGNOSIS` → [`PDC_REG_DIAGNOSIS.md`](./PDC_REG_DIAGNOSIS.md) | P4 / P5-REG | Published (12 stale gates, 1 product regression) |
| `PDC_DOMAIN_CERTIFICATION_MATRIX` → [`PDC_DOMAIN_CERTIFICATION_MATRIX.md`](./PDC_DOMAIN_CERTIFICATION_MATRIX.md) | P7 / P8-04 | Published (CERT-PDC consolidation of domain dispositions; no row re-grade; not a stage certification) |
| `PDC_INDEPENDENT_AUDIT` → [`PDC_INDEPENDENT_AUDIT.md`](./PDC_INDEPENDENT_AUDIT.md) | P8 | Published (IA-GLOBAL; addenda 1-3; final verdict `PASS_SCOPED` at `969cbd964`, with declared exceptions) |
| `PDC_P8_04_ACTIVATION_CERTIFIER_REPORT` → [`PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md`](./PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md), [`PDC_P8_ACTIVATION_RUNBOOK.md`](./PDC_P8_ACTIVATION_RUNBOOK.md) | P8-04 | Published (binding manifest `PASS` scoped; IIAC activation `IN_FORCE`) |
| `PDC_FINAL_ACCEPTANCE_REPORT` → [`PDC_FINAL_ACCEPTANCE_REPORT.md`](./PDC_FINAL_ACCEPTANCE_REPORT.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (Operator signed 2026-11-16, outcome A; PR #603 `feaa2a87d`; no new FRZ promotion claimed here) |
| `PDC_ARCH_FREEZE_HANDOFF` → [`PDC_ARCH_FREEZE_HANDOFF.md`](./PDC_ARCH_FREEZE_HANDOFF.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, outcome A; ARCH-FREEZE closure remains separate) |
| `PDC_EXCEPTION_REGISTER` → [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, Decisions 2 & 4; authoritative exception disposition register) |
| `PDC_DEBT_REGISTER` → [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, Decisions 2 & 4; authoritative documented-debt register) |
| `PDC_FRZ_PROMOTION_PROPOSAL` → [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, outcome A; checklist FRZ promotion remains untouched: 225 / 26) |
| `PDC_ROADMAP_CLOSURE_TEXT` → [`PDC_ROADMAP_CLOSURE_TEXT.md`](./PDC_ROADMAP_CLOSURE_TEXT.md) | P9 | Signed via [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, outcome A; roadmap closure published in PR #603 `feaa2a87d`; no FRZ promotion claimed) |

## Workstreams

| ID | Domain |
| --- | --- |
| `PDC-D1` | AGENT-LIFECYCLE |
| `PDC-D2` | CONTEXT-KNOWLEDGE |
| `PDC-D3` | MODEL-RUNTIME |
| `PDC-D4` | HOST-RUNTIME |
| `PDC-D5` | EVALUATION |
| `PDC-D6` | RUNTIME-SECURITY |
| `PDC-D7` | BACKGROUND-SCHED |
| `PDC-D8` | ARTIFACT-LIFECYCLE |

## P2-Snapshot Supersession & Per-Row P9 Disposition Authority (EQO-W4, `EDER-F-0191` / `EDER-F-0195` / `EDER-D-0112..0154`)

> **Governance note (EQO-W4 reconciliation, `T-IGX-090-B`, analysis pin `646e63953`):** [`PDC_GAP_REGISTER.md`](./PDC_GAP_REGISTER.md) is a generated P2 snapshot byte-locked by `scripts/qualification/pdc_x_requirements.py --check` and is intentionally preserved untouched for generator determinism. Current P8/P9 verdict authority lives in [`PDC_INDEPENDENT_AUDIT.md`](./PDC_INDEPENDENT_AUDIT.md) (IA-GLOBAL Addenda 1–3, `PASS_SCOPED` at `969cbd964`), [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md), and Operator-signed [`OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (2026-11-16, outcome A). This table reconciles documentation pointers only; it claims no new PASS, no IIAC/G6 certification, and no FRZ promotion (225 / 26 preserved).

### Rollup across the 43 P2/P7/EDER PDC-X Gap Rows (`EDER-D-0112..0154`)

- **36 Declared-Exception Rows** (`PDC_EXCEPTION_REGISTER.md` + signed `OD-PDC-X-CLOSE.md` Decisions 2 & 4): **35** rows covered by per-domain `EX-D1..D8` declared exceptions (several also carrying domain-section LOW debt in `PDC_DEBT_REGISTER.md §§1–8`) plus **1** `PARTIALLY_FIXED + DECLARED_EXCEPTION` row (`PDC-D3-R009` / `F-D3-02`, open `EX-L-07` per `PDC_INDEPENDENT_AUDIT.md §§7, 15.2` and `PDC_DEBT_REGISTER.md §12`). Zero rows in this 43-row set are claimed `REPAIRED / VERIFIED_FIXED`.
- **3 No-Exception `PARTIAL` Documented-Debt Rows** (`PDC-D5-R002`, `PDC-D6-R007`, `PDC-D6-R015` — stay `PARTIAL` with no exception per signed `OD-PDC-X-CLOSE.md:25` Decision 3 and `PDC_DEBT_REGISTER.md §13 L220–221`).
- **4 P5-DOC Reconciliation & Deduplication Rows (`EDER-F-0195`):** `PDC-D1-R009` (`EDER-D-0152` → `plans/AGENT_DISTRIBUTION.md:327`, also Decision 3 `PARTIAL` debt `F-D1-15` in `PDC_DEBT_REGISTER.md §1 L26, §13 L217`), `PDC-D7-R021` (`EDER-D-0153` → `architecture/BACKGROUND_TASKS.md`, `F-D7-07`/`F-D7-12` `PARTIALLY_FIXED` per `PDC_P7_D7_AUDIT_REPORT.md:57` & `EX-L-03` in `PDC_EXCEPTION_REGISTER.md §3 L55`), `PDC-D7-R022` (`EDER-D-0112` → `architecture/BACKGROUND_TASKS.md`, also Decision 3 `PARTIAL` debt `F-D7-16` in `PDC_DEBT_REGISTER.md §7 L120, §13 L222`), and `PDC-D7-R023` (`EDER-D-0154` → `plans/BACKGROUND_TASKS.md:35`, `F-D7-12` per `PDC_P7_D7_AUDIT_REPORT.md:59` & `PDC_INDEPENDENT_AUDIT.md §15`). Duplicate obligation references across P2, P7, and EDER are deduplicated onto the single canonical requirement ID below.

### Complete Per-Row P9 Disposition Mapping (`EDER-D-0112..0154`)

| Obligation ID | Canonical Requirement ID | Domain | P8 / P9 Disposition Authority | Source Register / Reconciled Doc |
| --- | --- | --- | --- | --- |
| `EDER-D-0112` | `PDC-D7-R022` | `PDC-X:D7` | `P5-DOC_RECONCILIATION + DOCUMENTED_DEBT` (`PARTIAL`, no exception; `F-D7-16`; deduped per `EDER-F-0195`) | `docs/project/architecture/BACKGROUND_TASKS.md` + `PDC_DEBT_REGISTER.md §7 L120, §13 L222` + `OD-PDC-X-CLOSE.md:25` Decision 3 |
| `EDER-D-0113` | `PDC-D1-R003` | `PDC-X:D1` | `DECLARED_EXCEPTION` (`EX-D1-01`) | `PDC_EXCEPTION_REGISTER.md §4 L76` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0114` | `PDC-D1-R012` | `PDC-X:D1` | `DECLARED_EXCEPTION` (`EX-D1-02`) | `PDC_EXCEPTION_REGISTER.md §4 L77` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0115` | `PDC-D1-R013` | `PDC-X:D1` | `DECLARED_EXCEPTION` (`EX-D1-02/03/04`) | `PDC_EXCEPTION_REGISTER.md §4 L77-79` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0116` | `PDC-D1-R014` | `PDC-X:D1` | `DECLARED_EXCEPTION` (`EX-D1-05`) | `PDC_EXCEPTION_REGISTER.md §4 L80` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0117` | `PDC-D1-R018` | `PDC-X:D1` | `DECLARED_EXCEPTION` (`EX-D1-06` + `F-D1-09/10` debt) | `PDC_EXCEPTION_REGISTER.md §4 L82` + `PDC_DEBT_REGISTER.md §1 L21-22` |
| `EDER-D-0118` | `PDC-D2-R005` | `PDC-X:D2` | `DECLARED_EXCEPTION` (`EX-D2-03`) | `PDC_EXCEPTION_REGISTER.md §4 L97` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0119` | `PDC-D2-R014` | `PDC-X:D2` | `DECLARED_EXCEPTION` (`EX-D2-02` + `F-D2-15` debt) | `PDC_EXCEPTION_REGISTER.md §4 L96` + `PDC_DEBT_REGISTER.md §2 L43` |
| `EDER-D-0120` | `PDC-D2-R015` | `PDC-X:D2` | `DECLARED_EXCEPTION` (`EX-D2-03`) | `PDC_EXCEPTION_REGISTER.md §4 L97` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0121` | `PDC-D3-R009` | `PDC-X:D3` | `PARTIALLY_FIXED + DECLARED_EXCEPTION` (`F-D3-02`; open `EX-L-07`) | `PDC_INDEPENDENT_AUDIT.md §§7 L144, 15.2 L448` + `PDC_EXCEPTION_REGISTER.md §3 L59` + `PDC_DEBT_REGISTER.md §12 L195` (`A-04`) |
| `EDER-D-0122` | `PDC-D3-R015` | `PDC-X:D3` | `DECLARED_EXCEPTION` (`EX-D3-02`) | `PDC_EXCEPTION_REGISTER.md §4 L113` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0123` | `PDC-D4-R005` | `PDC-X:D4` | `DECLARED_EXCEPTION` (`EX-D4-01`) | `PDC_EXCEPTION_REGISTER.md §4 L129` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0124` | `PDC-D4-R007` | `PDC-X:D4` | `DECLARED_EXCEPTION` (`EX-D4-02`) | `PDC_EXCEPTION_REGISTER.md §4 L130` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0125` | `PDC-D4-R015` | `PDC-X:D4` | `DECLARED_EXCEPTION` (`EX-D4-05`) | `PDC_EXCEPTION_REGISTER.md §4 L133` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0126` | `PDC-D5-R002` | `PDC-X:D5` | `DOCUMENTED_DEBT` (`PARTIAL`, no exception; `F-D5-09`) | `PDC_DEBT_REGISTER.md §5 L84, §13 L220` + `OD-PDC-X-CLOSE.md:25` Decision 3 (`PDC_EXCEPTION_REGISTER.md §4 L148, §5 L200`) |
| `EDER-D-0127` | `PDC-D5-R004` | `PDC-X:D5` | `DECLARED_EXCEPTION` (`EX-D5-02`) | `PDC_EXCEPTION_REGISTER.md §4 L143` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0128` | `PDC-D5-R005` | `PDC-X:D5` | `DECLARED_EXCEPTION` (`EX-D5-01`) | `PDC_EXCEPTION_REGISTER.md §4 L142` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0129` | `PDC-D5-R013` | `PDC-X:D5` | `DECLARED_EXCEPTION` (`EX-D5-02` + `F-D5-11` debt) | `PDC_EXCEPTION_REGISTER.md §4 L143, L148` + `PDC_DEBT_REGISTER.md §5 L86` |
| `EDER-D-0130` | `PDC-D5-R014` | `PDC-X:D5` | `DECLARED_EXCEPTION` (`EX-D5-03/04`) | `PDC_EXCEPTION_REGISTER.md §4 L144-145` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0131` | `PDC-D6-R007` | `PDC-X:D6` | `DOCUMENTED_DEBT` (`PARTIAL`, no exception; `F-D6-03`, `F-D6-14`) | `PDC_DEBT_REGISTER.md §6 L99, L106, §13 L221` + `OD-PDC-X-CLOSE.md:25` Decision 3 (`PDC_EXCEPTION_REGISTER.md §4 L162, §5 L201`) |
| `EDER-D-0132` | `PDC-D6-R009` | `PDC-X:D6` | `DECLARED_EXCEPTION` (`EX-D6-02`) | `PDC_EXCEPTION_REGISTER.md §4 L155` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0133` | `PDC-D6-R015` | `PDC-X:D6` | `DOCUMENTED_DEBT` (`PARTIAL`, no exception; `F-D6-20`) | `PDC_DEBT_REGISTER.md §6 L112, §13 L221` + `OD-PDC-X-CLOSE.md:25` Decision 3 (`PDC_EXCEPTION_REGISTER.md §4 L162, §5 L201`) |
| `EDER-D-0134` | `PDC-D6-R016` | `PDC-X:D6` | `DECLARED_EXCEPTION` (`EX-D6-02/03`) | `PDC_EXCEPTION_REGISTER.md §4 L155-156` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0135` | `PDC-D6-R020` | `PDC-X:D6` | `DECLARED_EXCEPTION` (`EX-D6-01/02`) | `PDC_EXCEPTION_REGISTER.md §4 L154-155` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0136` | `PDC-D6-R024` | `PDC-X:D6` | `DECLARED_EXCEPTION` (`EX-D6-02`) + `DOCUMENTED_DEBT` (`F-D6-10`) | `PDC_EXCEPTION_REGISTER.md §4 L155, L162` + `PDC_DEBT_REGISTER.md §6 L102` |
| `EDER-D-0137` | `PDC-D7-R006` | `PDC-X:D7` | `DECLARED_EXCEPTION` (`EX-D7-02`) | `PDC_EXCEPTION_REGISTER.md §4 L169` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0138` | `PDC-D7-R008` | `PDC-X:D7` | `DECLARED_EXCEPTION` (open `EX-D7-02`, `F-D7-05` unwired; recurrence library fixed in `T-IGX-041-G`/`M`) | `PDC_EXCEPTION_REGISTER.md §4 L169, §5 L202` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0139` | `PDC-D7-R009` | `PDC-X:D7` | `DECLARED_EXCEPTION` (`EX-D7-02`) + `DOCUMENTED_DEBT` (`F-D7-15`) | `PDC_EXCEPTION_REGISTER.md §4 L169, L174` + `PDC_DEBT_REGISTER.md §7 L119` |
| `EDER-D-0140` | `PDC-D7-R013` | `PDC-X:D7` | `DECLARED_EXCEPTION` (`EX-D7-02`) + `DOCUMENTED_DEBT` (`F-D7-17`) | `PDC_EXCEPTION_REGISTER.md §4 L169, L174` + `PDC_DEBT_REGISTER.md §7 L121` |
| `EDER-D-0141` | `PDC-D7-R016` | `PDC-X:D7` | `DECLARED_EXCEPTION` (`EX-D7-03`) | `PDC_EXCEPTION_REGISTER.md §4 L170` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0142` | `PDC-D8-R004` | `PDC-X:D8` | `DECLARED_EXCEPTION` (`EX-D8-02` + D8 LOW debt) | `PDC_EXCEPTION_REGISTER.md §4 L181, L187` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0143` | `PDC-D8-R005` | `PDC-X:D8` | `DECLARED_EXCEPTION` (`EX-D8-03/04`, `EX-L-02` inferred) | `PDC_EXCEPTION_REGISTER.md §3 L54, §4 L182-183` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0144` | `PDC-D8-R006` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-02`, `EX-D8-03`, `EX-D8-04`; `EX-L-02` inferred) | `PDC_EXCEPTION_REGISTER.md §3 L54, §4 L181-183, §5 L203` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0145` | `PDC-D8-R007` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-02`) | `PDC_EXCEPTION_REGISTER.md §4 L181, §5 L203` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0146` | `PDC-D8-R008` | `PDC-X:D8` | `DECLARED_EXCEPTION` (`EX-D8-01` + D8 LOW debt) | `PDC_EXCEPTION_REGISTER.md §4 L180, L187` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0147` | `PDC-D8-R009` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-01`) + `DOCUMENTED_DEBT` (`F-D8-09/14`) | `PDC_EXCEPTION_REGISTER.md §4 L180, L187, §5 L203` + `PDC_DEBT_REGISTER.md §8 L134, L139` |
| `EDER-D-0148` | `PDC-D8-R010` | `PDC-X:D8` | `DECLARED_EXCEPTION` (`EX-D8-02`, untouched by repair) | `PDC_EXCEPTION_REGISTER.md §4 L181, L185` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0149` | `PDC-D8-R012` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-02`; untouched by repair per `PDC_EXCEPTION_REGISTER.md:181`) | `PDC_EXCEPTION_REGISTER.md §4 L181, §5 L203` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0150` | `PDC-D8-R013` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-02`; untouched by repair per `PDC_EXCEPTION_REGISTER.md:181`) | `PDC_EXCEPTION_REGISTER.md §4 L181, §5 L203` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0151` | `PDC-D8-R016` | `PDC-X:D8` | `DECLARED_EXCEPTION` (open `EX-D8-01`) | `PDC_EXCEPTION_REGISTER.md §4 L180, §5 L203` + signed `OD-PDC-X-CLOSE.md` Decisions 2/4 |
| `EDER-D-0152` | `PDC-D1-R009` | `PDC-X:D1` | `P5-DOC_RECONCILIATION + DOCUMENTED_DEBT` (`PARTIAL`, no exception; `F-D1-15`; deduped per `EDER-F-0195`) | `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md:327` + `PDC_DEBT_REGISTER.md §1 L26, §13 L217` + `OD-PDC-X-CLOSE.md:25` Decision 3 |
| `EDER-D-0153` | `PDC-D7-R021` | `PDC-X:D7` | `P5-DOC_RECONCILIATION` (`F-D7-07`/`F-D7-12` `PARTIALLY_FIXED`, open `EX-L-03`; deduped per `EDER-F-0195`) | `docs/project/architecture/BACKGROUND_TASKS.md` + `PDC_P7_D7_AUDIT_REPORT.md:57` + `PDC_EXCEPTION_REGISTER.md §3 L55` (`EX-L-03`) |
| `EDER-D-0154` | `PDC-D7-R023` | `PDC-X:D7` | `P5-DOC_RECONCILIATION` (`F-D7-12`; deduped per `EDER-F-0195`) | `docs/project/maintainers/plans/BACKGROUND_TASKS.md:35` + `PDC_P7_D7_AUDIT_REPORT.md:59` + `PDC_INDEPENDENT_AUDIT.md §15` |

