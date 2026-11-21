# EQO-E-03 — OD-EQO-09 operator packet (draft)

**Task:** `EQO-E-03` · **Decision ID:** `OD-EQO-09` — confirm proposed EDER row owners (batch).
**Recorded:** 2026-11-21 · **Worktree:** `eqo-wv-e3` / branch `eqo/wave-1/e3`.
**Sources (read-only):** `register.jsonl` @ DOC_PIN `8cee3c5a913f234e8fb65053b6ab4f34ed78249b`; `enterprise_debt_register/reconciliation.json` §`owner` (ingest reconciliation meta).

**Out of scope for this draft:** no `register.jsonl` edit, no `accountable_person`, no row-level `owner_status=CONFIRMED`, no FRZ/roadmap/product change, no independence claim.

## Operator ask

1. **Batch A — `SINGLE_PROPOSED`:** confirm that each row’s `owner_ref` (DP-47 vocabulary, already confirmed for the ID list) is the correct program/stage owner for disposition.
2. **Batch B — five owner-gap buckets:** assign or resolve each listed row (`MISSING`, `UNMAPPED`, `SHARED_UNRESOLVED`, `ROLE_NOT_OWNER`, `GENERIC_NOT_AN_OWNER`) per `EQO_EDER_COMPLETE_DISPOSITION_PLAN.md` §E-03.
3. **Note:** `OPERATOR_ONLY` rows (132) are proposed `OPERATOR` ownership; listed in summary only — not one of the five gap lists.

## Reconciliation meta (owner)

- Vocabulary: `owner_vocabulary.proposed.json` — CONFIRMED (DP-47 option a; addendum 4 of eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md, 2026-11-17)
- `rows_owner_status_confirmed`: 0
- `rows_by_owner_status`:
  - `OPERATOR_ONLY`: 132
  - `SINGLE_PROPOSED`: 817
  - `SHARED_UNRESOLVED`: 75
  - `UNMAPPED`: 14
  - `GENERIC_NOT_AN_OWNER`: 4
  - `MISSING`: 15
  - `ROLE_NOT_OWNER`: 12

Gap list row counts from `register.jsonl` @ DOC_PIN match `reconciliation.json` §`owner` for all five buckets (15 / 14 / 75 / 12 / 4).

## Batch A — proposed single owners (`SINGLE_PROPOSED`)

**821** rows · **54** distinct `owner_ref` values (`reconciliation.json` ingest slice: 817; delta = four `EDER-N-0001..0004` `POST_INGEST` rows).

| owner_ref | row_count |
| --- | --- |
| PROD-Q | 161 |
| ARCH-FREEZE | 77 |
| ENT-AUDIT-X | 37 |
| MCP-SERVER-X | 37 |
| HARNESS | 34 |
| PDC-X | 29 |
| DOMAIN:TOOLS | 25 |
| EBH-7 | 25 |
| DUP-X | 22 |
| DOMAIN:INTEGRATIONS | 21 |
| QUAL-X | 18 |
| COMPAT-X | 17 |
| EBH-6 | 17 |
| DOMAIN:GOVERNANCE | 16 |
| PRE-ENT | 16 |
| SCENARIO-FACTORY-X | 16 |
| EVOL-X | 15 |
| CONFIG-X | 13 |
| TRACE-X | 13 |
| DOMAIN:AW-7C-INTEGRATIONS | 12 |
| DOMAIN:OBSERVABILITY | 12 |
| PDC-X:D8 | 12 |
| TENANT-X | 12 |
| DIST-SCALE-X | 11 |
| DOMAIN:SKILLS | 11 |
| IIAC-GOV-1 | 10 |
| PDC-X:D7 | 10 |
| DOMAIN:CONTEXT-ENGINEERING | 9 |
| EBH-3/4 | 9 |
| EBH-5 | 9 |
| PDC-X:D1 | 8 |
| ROADMAP-REPLAY-X | 8 |
| CTRL-X | 7 |
| DOMAIN:MEMORY | 7 |
| PDC-X:D6 | 7 |
| STATE-X | 7 |
| DOMAIN:RAG | 6 |
| PDC-X:D5 | 6 |
| DOMAIN:BACKGROUND-EXECUTION | 5 |
| HIST-CAMPAIGN | 5 |
| DOMAIN:LLM-BOUNDARY | 4 |
| PDC-X:D2 | 4 |
| PDC-X:D3 | 4 |
| PDC-X:D4 | 3 |
| DOMAIN:PROOF-RECEIPTS | 2 |
| EBH | 2 |
| SKILL-X | 2 |
| TOOL-LIFE-X | 2 |
| DOMAIN:AGENT-CONTRACTS | 1 |
| DOMAIN:APPLICATION-HOSTING | 1 |
| DOMAIN:EXECUTION | 1 |
| DOMAIN:PLATFORM-FOUNDATION | 1 |
| DOMAIN:PLATFORM-PLUGINS | 1 |
| EBH-2G | 1 |

**Default if Operator silent:** rows remain `INGESTED`; disposition `WAIT_FOR_OPERATOR` per disposition plan.

## Owner gap — `MISSING` (15 rows)

| reg_id | finding_id | title | owner_raw | owner_ref | co_owner_refs |
| --- | --- | --- | --- | --- | --- |
| EDER-D-0002 | — | Historical execution debt registers R1-R13 treated as out of ARC | OWNER_MISSING | — | — |
| EDER-D-0019 | — | GAP-P-08: Historical PLATFORM_WIDE audit | OWNER_MISSING | — | — |
| EDER-D-0046 | — | FRZ-SEC-01: Multi-instance (Kubernetes) trust anchor/global firs | OWNER_MISSING | — | — |
| EDER-D-0048 | — | FRZ-SEC-05: Real provider sandboxes (see SEC-06). | OWNER_MISSING | — | — |
| EDER-D-0050 | — | FRZ-SEC-09: Operator/infra retention config (not platform). | OWNER_MISSING | — | — |
| EDER-D-0051 | — | FRZ-REL-03: Multi-instance global first-writer PIS (Decision-03) | OWNER_MISSING | — | — |
| EDER-D-0052 | — | FRZ-REL-05: Provider native termination live behaviour. | OWNER_MISSING | — | — |
| EDER-D-0053 | — | FRZ-REL-08: Production capacity (see PRD-11). | OWNER_MISSING | — | — |
| EDER-D-0054 | — | FRZ-REL-09: Production capacity (see PRD-11). | OWNER_MISSING | — | — |
| EDER-D-0055 | — | FRZ-REL-10: Production capacity (see PRD-11). | OWNER_MISSING | — | — |
| EDER-D-0056 | — | FRZ-REL-11: Production capacity (see PRD-11). | OWNER_MISSING | — | — |
| EDER-D-0060 | — | FRZ-PRD-11: Load-level capacity of live deployment. | OWNER_MISSING | — | — |
| EDER-D-0061 | — | FRZ-PRD-12: Operator runbook for live environments. | OWNER_MISSING | — | — |
| EDER-D-0105 | — | Capability sampling/roots extensions: DEFERRED_WITH_JUSTIFICATIO | OWNER_MISSING | — | — |
| EDER-X-0044 | — | N/A decision category: 558 mentions in 147 files in the baseline | no owner named by DP-53 or README 9.3 for the N/ | — | — |

## Owner gap — `UNMAPPED` (14 rows)

| reg_id | finding_id | title | owner_raw | owner_ref | co_owner_refs |
| --- | --- | --- | --- | --- | --- |
| EDER-D-0171 | — | Branch protection (CI enforcement) | github_admin | — | — |
| EDER-D-0217 | — | 51 matrix rows NOT YET QUALIFIED at baseline 33ecf19c (historica | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0218 | — | T-B7-VBN-INDEPENDENT-ACCEPT: independent re-verification at the  | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0219 | — | 1 skipped test (reachability graph module absent); SQLite RPL-F  | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0220 | — | Not global FRZ PASS | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0763 | — | Independence is an external model session (ChatGPT); no auditor  | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0764 | — | 67 passed/1 skipped is historical at 33ecf19cc; suite and matrix | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0765 | — | OD-EBH5-03/05/06 remain OPEN/RECOMMENDED/TRACKED at baseline | Replaceability and vendor/backend program - docs | — | — |
| EDER-D-0766 | — | Scoped closure only; global FRZ PASS not claimed | Replaceability and vendor/backend program - docs | — | — |
| EDER-F-0124 | EQOA-A4-015 | Governance and routing strictness depend on production_mode/non- | Governance + Execution owners | — | — |
| EDER-F-0232 | EQOA-A8-016 | PLATFORM_PLUGINS closure claim vs FRZ-PLG-01..08 OPEN | PLATFORM_PLUGINS owner | — | — |
| EDER-F-0235 | EQOA-A8-019 | Capability architecture files without plan pairs, and a pair gat | Docs governance / feature owners | — | — |
| EDER-F-0236 | EQOA-A8-020 | Accepted Protocol-v2 remediation (LCI trusted tenant scope, Toke | RAG / LCI owner; Token Optimization owner | — | — |
| EDER-F-0239 | EQOA-A8-023 | Feature hubs omit cross-cutting rule cells (tenant/security, con | Feature doc owners | — | — |

## Owner gap — `SHARED_UNRESOLVED` (75 rows)

**52** distinct co-owner sets (primary `owner_ref` + `co_owner_refs`).

| proposed_owner_set | count | example_reg_id |
| --- | --- | --- |
| ARCH-FREEZE, OPERATOR | 6 | EDER-F-0024 … |
| EBH-7, QUAL-X | 5 | EDER-F-0216 … |
| DOMAIN:OBSERVABILITY, HARNESS | 3 | EDER-D-0709 … |
| OPERATOR, PROD-Q | 3 | EDER-F-0010 … |
| OPERATOR, TOOL-LIFE-X | 3 | EDER-F-0025 … |
| ARCH-FREEZE, DOMAIN:INTEGRATIONS | 2 | EDER-F-0129 … |
| ARCH-FREEZE, HARNESS | 2 | EDER-F-0017 … |
| ARCH-FREEZE, PDC-X:D6 | 2 | EDER-F-0039 … |
| ARCH-FREEZE, TENANT-X | 2 | EDER-F-0093 … |
| ARCH-FREEZE, TRACE-X | 2 | EDER-F-0018 … |
| DUP-X, EBH-6 | 2 | EDER-F-0190 … |
| EBH-6, OPERATOR | 2 | EDER-F-0058 … |
| EBH-7, OPERATOR | 2 | EDER-F-0187 … |
| ARCH-FREEZE, CTRL-X | 1 | EDER-F-0096 |
| ARCH-FREEZE, DOMAIN:LLM-BOUNDARY | 1 | EDER-F-0125 |
| ARCH-FREEZE, DOMAIN:OBSERVABILITY | 1 | EDER-F-0181 |
| ARCH-FREEZE, EBH-7 | 1 | EDER-F-0099 |
| ARCH-FREEZE, HIST-CAMPAIGN | 1 | EDER-F-0084 |
| ARCH-FREEZE, PDC-X | 1 | EDER-F-0101 |
| ARCH-FREEZE, PROD-Q | 1 | EDER-F-0105 |
| ARCH-FREEZE, QUAL-X | 1 | EDER-F-0076 |
| COMPAT-X, PDC-X:D8, STATE-X | 1 | EDER-F-0094 |
| CONFIG-X, STATE-X | 1 | EDER-F-0133 |
| CTRL-X, PDC-X:D6, QUAL-X | 1 | EDER-F-0104 |
| DIST-SCALE-X, OPERATOR | 1 | EDER-F-0026 |
| DOMAIN:PROOF-RECEIPTS, PDC-X:D8 | 1 | EDER-F-0148 |
| DUP-X, EBH-7 | 1 | EDER-F-0189 |
| DUP-X, OPERATOR | 1 | EDER-D-0633 |
| ENT-AUDIT-X, EVOL-X, PDC-X | 1 | EDER-F-0147 |
| EVOL-X, IIAC-GOV-1 | 1 | EDER-F-0036 |
| EVOL-X, MCP-SERVER-X, PDC-X | 1 | EDER-F-0234 |
| HARNESS, PROD-Q | 1 | EDER-F-0143 |
| HARNESS, TOOL-LIFE-X | 1 | EDER-F-0160 |
| HIST-CAMPAIGN, OPERATOR | 1 | EDER-F-0203 |
| HIST-CAMPAIGN, TOOL-LIFE-X | 1 | EDER-F-0035 |
| IIAC-GOV-1, PDC-X | 1 | EDER-F-0007 |
| MCP-SERVER-X, PROD-Q | 1 | EDER-F-0238 |
| OPERATOR, PDC-X | 1 | EDER-F-0215 |
| OPERATOR, PDC-X:D6 | 1 | EDER-F-0192 |
| OPERATOR, PRE-ENT | 1 | EDER-F-0170 |
| OPERATOR, TRACE-X | 1 | EDER-F-0199 |
| PDC-X, ROADMAP-REPLAY-X | 1 | EDER-F-0163 |
| PDC-X:D2, TENANT-X | 1 | EDER-F-0092 |
| PDC-X:D4, PROD-Q | 1 | EDER-F-0149 |
| PDC-X:D4, TENANT-X | 1 | EDER-F-0115 |
| PDC-X:D6, TENANT-X | 1 | EDER-F-0169 |
| PDC-X:D7, TRACE-X | 1 | EDER-F-0049 |
| PROD-Q, TENANT-X | 1 | EDER-F-0136 |
| PROD-Q, TRACE-X | 1 | EDER-F-0011 |
| QUAL-X, SKILL-X, TOOL-LIFE-X | 1 | EDER-F-0055 |
| QUAL-X, TOOL-LIFE-X | 1 | EDER-F-0103 |
| STATE-X, TRACE-X | 1 | EDER-F-0168 |

| reg_id | finding_id | title | owner_raw | owner_ref | co_owner_refs |
| --- | --- | --- | --- | --- | --- |
| EDER-D-0084 | — | HIST-F-BND-03B: intergrax/applications/_shared/product_manifest_ | Operator / ARCH-FREEZE B7 (future product ticket | ARCH-FREEZE | OPERATOR |
| EDER-D-0474 | — | Independent audit "pending on reconciliation commit SHA" - no re | PROD-Q program / roadmap governance - docs/proje | PROD-Q | OPERATOR |
| EDER-D-0475 | — | Docs-only reconciliation; preserved OPEN: DIM6, DIM17, PROD-Q-P1 | PROD-Q program / roadmap governance - docs/proje | PROD-Q | OPERATOR |
| EDER-D-0633 | — | DUPX-F-COMPAT-R3-FENCE: COMPAT-R3 bypass ratified as lab-only fe | DUP-X (inferred) / Operator ratification | DUP-X | OPERATOR |
| EDER-D-0644 | — | EBH6-DEBT-BND-COMPAT-R3: lab-only compat mechanism ALLOWED - int | Operator / EBH-6 | EBH-6 | OPERATOR |
| EDER-D-0709 | — | Independent audit asserted in the record header; no audit report | Observability domain / Harness W5 - docs/project | DOMAIN:OBSERVABILITY | HARNESS |
| EDER-D-0710 | — | Replay logs in gitignored .tmp (not in repo) | Observability domain / Harness W5 - docs/project | DOMAIN:OBSERVABILITY | HARNESS |
| EDER-D-0711 | — | FRZ rows remain OPEN | Observability domain / Harness W5 - docs/project | DOMAIN:OBSERVABILITY | HARNESS |
| EDER-F-0007 | EQOA-A0-007 | IIAC policy activation is not IN_FORCE at BASELINE_SHA: 71 ACT-N | PDC-X / IIAC-GOV-1 | PDC-X | IIAC-GOV-1 |
| EDER-F-0010 | EQOA-A1-002 | PROD-Q has three simultaneous status expressions (CLOSED BY OPER | Roadmap owner (Operator) / PROD-Q | PROD-Q | OPERATOR |
| EDER-F-0011 | EQOA-A1-003 | 8 'READY FOR AUDIT' rows sit under CLOSED parents (table L401, L | TRACE-X / PROD-Q owners | TRACE-X | PROD-Q |
| EDER-F-0017 | EQOA-A1-009 | SCHED-01 closure rests on roadmap/downstream claims; the stage q | HARNESS residual owner (Harness AI program) / AR | HARNESS | ARCH-FREEZE |
| EDER-F-0018 | EQOA-A1-010 | Two TRACE-X RFA rows remain READY FOR AUDIT in the roadmap while | ARCH-FREEZE / TRACE-X SSOT owner (SIGN-6_TRACE p | ARCH-FREEZE | TRACE-X |
| EDER-F-0024 | EQOA-A1-016 | PROD-Q is CLOSED BY OPERATOR while its mandatory live children s | Operator / ARCH-FREEZE | ARCH-FREEZE | OPERATOR |
| EDER-F-0025 | EQOA-A1-017 | TOOL-LIFE-X labelled CLOSED - INDEPENDENTLY ACCEPTED while the o | Operator (roadmap status) / TOOL-LIFE-X stage ow | TOOL-LIFE-X | OPERATOR |
| EDER-F-0026 | EQOA-A1-018 | DIST-SCALE-X closed on operator-only scoped acceptance while SCE | Operator / DIST-SCALE-X stage owner | DIST-SCALE-X | OPERATOR |
| EDER-F-0033 | EQOA-A2-001 | TOOL-LIFE-X carries the label 'INDEPENDENTLY ACCEPTED' while its | TOOL-LIFE-X / Roadmap owner (Operator) | TOOL-LIFE-X | OPERATOR |
| EDER-F-0034 | EQOA-A2-002 | ARCH-FREEZE entry condition L499 ('independent exact-SHA evidenc | ARCH-FREEZE / Operator | ARCH-FREEZE | OPERATOR |
| EDER-F-0035 | EQOA-A2-004 | Four SHAs cited by the HIST registry are not ancestors of the ba | HIST campaign / TOOL-LIFE-X owner | HIST-CAMPAIGN | TOOL-LIFE-X |
| EDER-F-0036 | EQOA-A2-006 | 3 of 39 published IIAC manifests fail the repo's own manifest an | IIAC-GOV-1 / EVOL-X owners | IIAC-GOV-1 | EVOL-X |
| EDER-F-0037 | EQOA-A2-007 | Qualification records of closed stages still show pre-acceptance | Stage owners / ARCH-FREEZE | ARCH-FREEZE | OPERATOR |
| EDER-F-0039 | EQOA-A2-009 | EBH-3 R2 claim 'cross-domain integrations._shared.* production l | PDC-X D6 / ARCH-FREEZE | PDC-X:D6 | ARCH-FREEZE |
| EDER-F-0049 | EQOA-A2-019 | TRACE-X-P6 certified restart/resume surface changed after the ac | TRACE-X stage owner / PDC-X D7 | TRACE-X | PDC-X:D7 |
| EDER-F-0055 | EQOA-A2-026 | Independent-audit artefacts (AUD-QUALX, AUD-SKILLX, AUD-TOOLLIFE | QUAL-X / SKILL-X / TOOL-LIFE-X owners | QUAL-X | SKILL-X, TOOL-LIFE-X |
| EDER-F-0058 | EQOA-A2-029 | EBH-6 closure rests on an unnamed independent auditor; independe | EBH-6 stage owner / Operator | EBH-6 | OPERATOR |
| EDER-F-0076 | EQOA-A2-047 | CTRL-X FRZ-CTL-01 (incl. regression protection) is PASS while 15 | QUAL-X / ARCH-FREEZE | QUAL-X | ARCH-FREEZE |
| EDER-F-0078 | EQOA-A2-049 | P6 restart/resume classifier defaults whole directories to canon | TRACE-X stage owner / ARCH-FREEZE | TRACE-X | ARCH-FREEZE |
| EDER-F-0084 | EQOA-A3-004 | HIST-85 worker dispositions were not independently re-judged; an | HIST campaign / ARCH-FREEZE | HIST-CAMPAIGN | ARCH-FREEZE |
| EDER-F-0092 | EQOA-A3-012 | TENANT-X closed-world certification missed unqualified TASK-scop | TENANT-X / PDC-X D2 | TENANT-X | PDC-X:D2 |
| EDER-F-0093 | EQOA-A3-013 | FRZ-TEN-01..12 PASS rests on a pre-PDC TENANT-X acceptance; reva | TENANT-X / ARCH-FREEZE | TENANT-X | ARCH-FREEZE |
| EDER-F-0094 | EQOA-A3-014 | COMPAT-X and STATE-X closed-world counters/classifications re-pi | COMPAT-X / STATE-X / PDC-X D8 | COMPAT-X | STATE-X, PDC-X:D8 |
| EDER-F-0096 | EQOA-A3-016 | Tier-2 agent-registry bypass gate was already violated at the CT | CTRL-X / ARCH-FREEZE | CTRL-X | ARCH-FREEZE |
| EDER-F-0097 | EQOA-A3-017 | HARNESS-FINAL: FRZ-HRN-08 'no PARTIAL/GAP remains' is contradict | HARNESS-FINAL owner / ARCH-FREEZE | HARNESS | ARCH-FREEZE |
| EDER-F-0099 | EQOA-A3-019 | Row-level PASS evidence for EXE/GOV/CTL/REC/HRN is tied to certi | EBH-7 / ARCH-FREEZE | EBH-7 | ARCH-FREEZE |
| EDER-F-0101 | EQOA-A3-021 | 41 of 42 OBS/RPL/SEC/REL/REG criteria are OPEN although every na | ARCH-FREEZE / PDC-X | ARCH-FREEZE | PDC-X |
| EDER-F-0103 | EQOA-A3-023 | Closing-stage lists are stale: TOOL-LIFE-X omitted from timeout/ | QUAL-X / TOOL-LIFE-X stage owners | QUAL-X | TOOL-LIFE-X |
| EDER-F-0104 | EQOA-A3-024 | Ten criteria carry zero checklist evidence; FRZ-REG-10 checklist | CTRL-X / QUAL-X / PDC-X D6 | CTRL-X | QUAL-X, PDC-X:D6 |
| EDER-F-0105 | EQOA-A3-025 | Architecture-scoped FRZ-PRD-04/05/06/07/09/10/11/12 are OPEN wit | PROD-Q / ARCH-FREEZE | PROD-Q | ARCH-FREEZE |
| EDER-F-0115 | EQOA-A4-006 | Runtime event tenant propagation audit gate cannot run: emitter  | TENANT-X / PDC-X D4 | TENANT-X | PDC-X:D4 |
| EDER-F-0121 | EQOA-A4-012 | EBH-2A..2D (agent/application contract boundary hardening) have  | ARCH-FREEZE / roadmap owner | ARCH-FREEZE | OPERATOR |
| EDER-F-0122 | EQOA-A4-013 | Decision System has no roadmap stage, PDC workstream or FRZ fami | ARCH-FREEZE / roadmap owner | ARCH-FREEZE | OPERATOR |
| EDER-F-0125 | EQOA-A4-016 | Two parallel host LLM resolvers (runtime.wiring and applications | LLM domain owner (EBH-2E) / ARCH-FREEZE | DOMAIN:LLM-BOUNDARY | ARCH-FREEZE |
| EDER-F-0129 | EQOA-A4-020 | Concrete integration providers are imported and selected by enum | Integrations owner (EBH-2F) / ARCH-FREEZE | DOMAIN:INTEGRATIONS | ARCH-FREEZE |
| EDER-F-0133 | EQOA-A4-024 | Tier-1 NexusLoop selects concrete TaskMemory and runtime-event S | STATE-X owner / CONFIG-X owner | STATE-X | CONFIG-X |
| EDER-F-0136 | EQOA-A4-027 | Tier-3 non-strict serving surfaces accept client-supplied tenant | TENANT-X owner / PROD-Q | TENANT-X | PROD-Q |
| EDER-F-0143 | EQOA-A4-034 | Dormant and duplicate retry-budget mechanisms next to the active | HARNESS-W4 / PROD-Q | HARNESS | PROD-Q |
| EDER-F-0147 | EQOA-A4-038 | Owner-document status lines are stale against roadmap closure (E | EVOL-X / ENT-AUDIT-X / PDC-X owners (ARCH-FREEZE | EVOL-X | ENT-AUDIT-X, PDC-X |
| EDER-F-0148 | EQOA-A4-039 | Two unrelated ProofReceipt models and two ProofReceiptStore defi | PDC-X D8 / PROOF_RECEIPTS owner | PDC-X:D8 | DOMAIN:PROOF-RECEIPTS |
| EDER-F-0149 | EQOA-A4-040 | Most Tier-3 application entrypoints start serving with direct uv | PDC-X D4 / PROD-Q | PDC-X:D4 | PROD-Q |
| EDER-F-0157 | EQOA-A5-001 | EBH-3 gate scans only intergrax/: Tier-2/3 production modules im | Integrations owner / ARCH-FREEZE | DOMAIN:INTEGRATIONS | ARCH-FREEZE |
| EDER-F-0159 | EQOA-A5-003 | GOV-X1 gates rewritten after being found stale; no gate-weakenin | PDC-X D6 / ARCH-FREEZE | PDC-X:D6 | ARCH-FREEZE |
| EDER-F-0160 | EQOA-A5-004 | HARNESS-W4 gate semantics changed by TOOL-LIFE-X without a W4 le | Harness / TOOL-LIFE-X owner | HARNESS | TOOL-LIFE-X |
| EDER-F-0163 | EQOA-A5-007 | ROADMAP-REPLAY-X own invalidation counter is non-zero at baselin | ROADMAP-REPLAY-X / PDC-X | ROADMAP-REPLAY-X | PDC-X |
| EDER-F-0167 | EQOA-A5-011 | TENANT-X global tenant-isolation PASS is not maintained after ce | TENANT-X stage owner / ARCH-FREEZE | TENANT-X | ARCH-FREEZE |
| EDER-F-0168 | EQOA-A5-012 | STATE-X and TRACE-X closed-world gates were not executed after t | STATE-X / TRACE-X stage owners | STATE-X | TRACE-X |
| EDER-F-0169 | EQOA-A5-013 | Closed-world counters of TENANT-X and COMPAT-X were re-pinned af | PDC-X D6 / TENANT-X owner | PDC-X:D6 | TENANT-X |
| EDER-F-0170 | EQOA-A5-014 | PRE-ENT final matrix blocker OD-HIST04-A still reports EVOL-X fo | PRE-ENT campaign owner (T-IGX-034-A) / Operator | PRE-ENT | OPERATOR |
| EDER-F-0181 | EQOA-A7-002 | Stale currency in closed-world gap register, diagnostic ledger a | ARCH-FREEZE / Observability / platform | ARCH-FREEZE | DOMAIN:OBSERVABILITY |
| EDER-F-0187 | EQOA-A7-008 | Open owned gaps without closure criterion inside frozen scope (E | EBH-7 owner / Operator | EBH-7 | OPERATOR |
| EDER-F-0189 | EQOA-A7-010 | Delegated or deferred EBH-7 obligations lack evidenced acceptanc | EBH-7 / DUP-X owners | EBH-7 | DUP-X |
| EDER-F-0190 | EQOA-A7-011 | DUP-X and EBH-6 hardcoding register rows carry no owner, severit | DUP-X / EBH-6 owners | DUP-X | EBH-6 |
| EDER-F-0192 | EQOA-A7-013 | PDC-D6-R017 (MCP exposure security) waits on a stage that is alr | PDC-X D6 / Operator | PDC-X:D6 | OPERATOR |
| EDER-F-0199 | EQOA-A7-020 | Two lineage TRACE-X roadmap rows remain unchecked READY FOR AUDI | TRACE-X / roadmap owner | TRACE-X | OPERATOR |
| EDER-F-0203 | EQOA-A7-024 | HIST and stage-registry statements overtaken by baseline sources | HIST campaign / roadmap owner | HIST-CAMPAIGN | OPERATOR |
| EDER-F-0207 | EQOA-A7-028 | EBH-7 closed SCOPED while W1-PROD-DELTA (tenant guard), W1-PROD- | EBH-7 / Operator | EBH-7 | OPERATOR |
| EDER-F-0209 | EQOA-A7-030 | Audit obligations attached to scoped delegated decisions have no | Operator / TOOL-LIFE-X and SKILL-X owners | TOOL-LIFE-X | OPERATOR |
| EDER-F-0211 | EQOA-A7-032 | Operator decision registers carry stale status text after later  | EBH-6 / DUP-X | EBH-6 | DUP-X |
| EDER-F-0215 | EQOA-A7-036 | 45 obligation carry-forward chains are longer than one hop or en | Roadmap owner / PDC-X | PDC-X | OPERATOR |
| EDER-F-0216 | EQOA-A7-037 | 164 skip/xfail markers in qualification and architecture test tr | QUAL-X / EBH-7 | QUAL-X | EBH-7 |
| EDER-F-0234 | EQOA-A8-018 | Feature and dossier hubs lag the roadmap: closed stages describe | EVOL-X / MCP-SERVER-X / SFX / PDC-X doc owners | EVOL-X | MCP-SERVER-X, PDC-X |
| EDER-F-0238 | EQOA-A8-022 | MCP-SERVER-X is closed with no FRZ mapping and open scoped limit | MCP-SERVER-X owner / PROD-Q | MCP-SERVER-X | PROD-Q |
| EDER-X-0040 | — | Skip category: no_ci marker category - 284 markers in the baseli | DP-53: per-marker disposition lives in an extend | QUAL-X | EBH-7 |
| EDER-X-0041 | — | Skip category: optional-dependency skip category (importorskip)  | DP-53: per-marker disposition lives in an extend | QUAL-X | EBH-7 |
| EDER-X-0042 | — | Skip category: conditional skip category (pytest.skip calls and  | DP-53: per-marker disposition lives in an extend | QUAL-X | EBH-7 |
| EDER-X-0043 | — | Skip category: unconditional skip and xfail marker category - 8  | DP-53: per-marker disposition lives in an extend | QUAL-X | EBH-7 |

## Owner gap — `ROLE_NOT_OWNER` (12 rows)

| reg_id | finding_id | title | owner_raw | owner_ref | co_owner_refs |
| --- | --- | --- | --- | --- | --- |
| EDER-D-0009 | — | FIND-DEBT-REG: Blocks promotion (FRZ-DEBT-01..06) | ARCH-FREEZE:implementer | — | — |
| EDER-D-0010 | — | FIND-VBN-PARTIAL: ECAP-Â§15 risk (FRZ-RPL-01..04) | ARCH-FREEZE:implementer + independent_auditor | — | — |
| EDER-D-0011 | — | FIND-FORMAL-B6: ARCH-FREEZE blocked (FRZ-EVOL; FRZ-AUD; FRZ-FRZ- | ARCH-FREEZE:independent_auditor | — | — |
| EDER-D-0035 | — | EBH-6 Layer A-J L9 attestation: CANDIDATE-RESOLVED, independent  | EBH-6 / Protocol 03 independent auditor | — | — |
| EDER-D-0036 | — | EBH6-LAYER-01..16: all 16 layer rows CANDIDATE-RESOLVED with L3  | EBH-6 / Protocol 03 independent auditor | — | — |
| EDER-D-0037 | — | EBH6-LAYER-07 (example): class G internal mechanism, L9_EVIDENCE | EBH-6 / Protocol 03 independent auditor | — | — |
| EDER-D-0072 | — | H01-F015: Asymmetric error handling between the two branches of  | ticket + implementer + fresh auditor (no named o | — | — |
| EDER-D-0073 | — | HIST02-F-013: Worker thesis (not independently audited): TOOL-LI | ticket + implementer + fresh auditor (no named o | — | — |
| EDER-D-0074 | — | HIST03-F-015: Worker thesis (not independently audited): Code de | ticket + implementer + fresh auditor (no named o | — | — |
| EDER-D-0075 | — | F-H4-03: Operator-written authorship is only process_enforced; m | ticket + implementer + fresh auditor (no named o | — | — |
| EDER-D-0086 | — | C4 CLI identity fixture: tests/.../test_cli_verification_hardeni | Optional env/fixture ticket (outside HIST mandat | — | — |
| EDER-D-0168 | — | EVOL stage closure publication (after FRZ-EVOL; before OD-ENT-00 | Operator + auditor | — | — |

## Owner gap — `GENERIC_NOT_AN_OWNER` (4 rows)

| reg_id | finding_id | title | owner_raw | owner_ref | co_owner_refs |
| --- | --- | --- | --- | --- | --- |
| EDER-D-0022 | — | Flat ApplicationEnvironmentProfile namespace growth (43+ top-lev | platform | — | — |
| EDER-D-0039 | — | GAP-FRZ-IMPACT: FRZ impact records only, OWNED (W8) | Program | — | — |
| EDER-D-0042 | — | GAP-LAYER-L9: layer L9 attestation OWNED, reuse of EBH-6 evidenc | Architecture | — | — |
| EDER-F-0183 | EQOA-A7-004 | Closed ARCH_DEBT entries carry no evidence pointer, closure date | platform | — | — |

## Reconciliation `unmapped_owner_raw` (distinct free-text labels)

- Docs governance / feature owners
- Feature doc owners
- Governance + Execution owners
- PLATFORM_PLUGINS owner
- RAG / LCI owner; Token Optimization owner
- Replaceability and vendor/backend program - docs/project/maintainers/qualification/VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md; stage record docs/project/maintainers/qualification/EBH_5_IMPLEMENTATION_WAVE_RECONCILIATION.md
- github_admin

## Steward closure (EQO-E-03 acceptance)

- [x] Five owner-gap lists published from reconciliation/meta + register scan.
- [x] `SINGLE_PROPOSED` grouping table for OD-EQO-09 batch A.
- [ ] Operator decision recorded (separate artefact; not part of this draft).

**Ticket Ref (commit):** `governance/tickets/T-IGX-046-F.md`
