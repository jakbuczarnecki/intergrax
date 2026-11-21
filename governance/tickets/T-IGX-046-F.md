# T-IGX-046-F - EQO-A remediation: release of the nine held DB-C hunks as dated gap and custody notes

Implementer session: local_398ae9d6-d95a-48cb-9957-e428fed258e5

## Scope

Documentation-only follow-up to T-IGX-046-E (PR #656, merged as `cb60f683f409baf2b87c27e31902a14d918e1e0b`), which held nine of the sixteen prepared DB-C hunks. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589); re-verified at `origin/development` `44d1e5d105beb9b62d0a12e693066b0b26c125eb`.

- Nine append-only dated notes in nine stage records (98 added lines, no line changed or deleted): five gap notes (H03, H06, H07, H08, H12), one gap note on the SHA ladder (H11), two custody notes for replay logs (H04, H05) and one custody note with a decision cross-reference (H09). Findings: EQOA-A2-007, EQOA-A2-036, EQOA-A2-040, EQOA-A2-044, EQOA-A2-046, EQOA-A3-017.
- Publication of three local logs (W5 batch A and B, PRE-ENT W6 bounded regression) under `iiac_evidence/EQO-A-DBC-HELD/` with a manifest, sizes and sha256. The two W4 replay logs (T11-A, T11-B) are not published because the pre-publication scan found user-profile paths; the notes record their local digests.
- The G-SKILL note quotes only the published decision DP-44.

Out of scope: the roadmap, the freeze checklist, the FRZ matrices, `PDC_*` records, generated registers and register data, the PRE-ENT final matrix, `scripts/`, `tests/`, `.github/`, every product path, redaction of any log, supply of any missing owner fact, person assignments, row-level owner confirmation. No PASS, FRZ promotion, stage closure, independent-verification claim or independence claim.

## Plan digest

Plan digest: a8eef929756b7089ce4fe35dbd416780d4b4269d4ddb007d697c8f67c707fab9

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-F.md`](plans/PLAN_T-IGX-046-F.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W7_HOST_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md
docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/EVIDENCE_MANIFEST.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v11-batch-a.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v12-batch-b.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/pre-ent-w6-cross-audit/parent_regression_5gate.log
```

## Approved

Approved: operator-delegate @ 2026-11-17T15:11:25+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@cd93d7e2b519bc2d894e2da82f4f8156d281af2d
Delegate session: local_7b06d036-e322-483f-8e3f-8d2010ae9e29

## Test contract

One process at a time, under the shared pytest lock; logs under `.tmp/session/dbc-held/`:

- `uv run pytest tests/unit/docs/`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `python -m scripts.governance.verify_ticket_approvals --since origin/development`
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the write set plus this ticket and its plan; `git diff --numstat` shows 0 deleted lines for the nine records; a read-only reviewer confirms that no added sentence adds an independence claim without a preserved artefact, that every cited SHA resolves and that every cited line reads as cited (AR-9-07).

## Out of scope

Edits of the roadmap, the freeze checklist, PDC-owned records, generated registers, `scripts/`, `tests/`, product paths, person assignments, row-level owner confirmation, FRZ promotion, Docker, vendor integration, production configuration.

## Checkpoint — EQO wave-2 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-2/integ` |
| Base | `origin/development` `84a7a76ec` |
| Lanes merged | `eqo/wave-2/w4`, `eqo/wave-2/x`, `eqo/wave-2/dis` |
| W4-02 | Tranches 05–07 (queue complete outside Campaign B) |
| X | X-01 / X-06 / X-08 read memos under `eqo_closure/x/` |
| AUD | Prep checklists only — session E deferred to wave-3 |
| C-04 | DEFERRED (prep under `runs/c/`) |
| #680 | Track-only (`pr680_governance_fast.md`) |
| BLOCKED EXE | WB01,04,06,07,09,10,11 non-zero pytest (manifests retained) |
| PR | [#690](https://github.com/jakbuczarnecki/intergrax/pull/690) merged `2026-11-21T06:45:59Z` |
| `development` head | `8631b1e8459e096b74d1c80247d3e432b44fff30` (merge commit; integrator tip `eb15b8a36`) |
| CI at merge | Red on required contexts — pre-existing on `development`, outside EQO wave-2 write set: `ci-smoke` (DF5 `vector_store` config), `governance-fast` (ACT-N11 / IIAC activation drift), `ent-audit-fast` cascade (`ENT-A-PRD-01/02`, `ENT-A-REG-03`). Triage: subagent session `41aa0042`. |
| CI record | PR #690 conversation comment «CI triage (post-`gh pr checks 690 --watch`)»; completion table in «EQO wave-2 completion signal». |

## Checkpoint — EQO wave-3 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-3/integ` |
| Base | `origin/development` `8631b1e84` (PR #690) |
| Deliverables | `AUD_PACKET.md` WB01–12, `W4-03_OWNER_DECISION_BATCH.md`, `EQO_WAVE3_MANIFEST.md`, matrix AUD § |
| PR | [#692](https://github.com/jakbuczarnecki/intergrax/pull/692) merged `b533ed051` |
| `development` head | `b533ed0510ee3b6eef88e5a19249404540064937` (merge commit) |
| AUD verdicts | **none** — packets `PRE_AUD` / `OPEN` only |
| EQO-C-04 | executed @ TRAIN_PIN — **BLOCKED** (pytest exit 1); `runs/c/C04_RUN_MANIFEST.json` |
| E-02 apply | out of scope |
| Session E — **PRE_AUD** | WB02, WB03, WB05, WB12 |
| Session E — **OPEN** | WB01, WB04, WB06, WB07, WB08, WB09, WB10, WB11 |

**AUD packet summary** (from `EQO_WAVE3_MANIFEST.md`; entry path `eqo_closure/runs/wbNN/AUD_PACKET.md`):

| WB | Packet status | EXE exit |
|----|---------------|---------:|
| WB01 | OPEN | 1 |
| WB02 | PRE_AUD | 0 |
| WB03 | PRE_AUD | 0 |
| WB04 | OPEN | 1 |
| WB05 | PRE_AUD | 0 |
| WB06 | OPEN | 4 |
| WB07 | OPEN | 1 |
| WB08 | OPEN | — |
| WB09 | OPEN | 1 |
| WB10 | OPEN | 1 |
| WB11 | OPEN | 1 |
| WB12 | PRE_AUD | 0 |

Session E: separate chat per WB; implementer must not self-PASS (`EQO_WAVE3_MANIFEST.md`).

## Checkpoint — EQO wave-4 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-4/integ` |
| Base | `origin/development` `bb4738c94` (post #692 waiver; wave-4 lanes rebased before commits) |
| Lanes merged | `eqo/wave-4/w2`, `w4`, `x`, `matrix` |
| W2 | `78744c1b5` — W2-02..11 read pass; `#680` track-only |
| W4 | `42276e586` — `W4-04_REQUALIFY_BATCH.md`, `W4-05_NONE_JUSTIFICATION.md` |
| X | `a1912e895` — X-01..08 read memos / refreshes |
| Matrix | `84d14f559` — partial CLS § (no AUD verdicts) |
| Deliverables | `EQO_WAVE4_MANIFEST.md`, README phase index row 26 |
| AUD verdicts | **none** — DES/read band only |
| EQO-C-04 | out of scope this wave |
| E-02 apply | out of scope |
| #680 | track-only (no new ticket) |

**Operator merge policy correction:** After incident PR #690, an autonomous agent MUST NOT run `gh pr merge` unless required docs gates are green (`ticket-governance` minimum) and operator `merge_on_green` applies to the full required check set, or an explicit operator waiver is documented in the ticket or PR thread. Ref: T-IGX-046-F.

## Checkpoint — EQO AUD-1 (2026-11-21)

Session E verdicts @ TRAIN_PIN `28c27cb` (`runs/wbNN/AUD_SESSION_E.md`): WB02 FAIL, WB03 FAIL, WB05 BLOCKED, WB12 BLOCKED; integrator `eqo/aud/wave-1-integ`; no product change, no G6/IIAC/FRZ claim.

## Checkpoint — EQO AUD-2 (2026-11-21)

Session E verdicts @ TRAIN_PIN `28c27cb` (`runs/wbNN/AUD_SESSION_E.md`): WB01 FAIL, WB04 BLOCKED, WB06 FAIL, WB07 FAIL, WB08 BLOCKED, WB09 FAIL, WB10 FAIL, WB11 FAIL; integrator `eqo/aud/wave-2-integ`; read-only source inspection, pytest not re-run; no product change, no G6/IIAC/FRZ claim. WB04, WB06, WB08 and parts of WB09 need an EXE re-run with the optional extras (`mcp`, `llm-langchain-ollama`).

## Checkpoint — EQO post-AUD integrator (2026-11-21)

AUD waves 1–2 merged on `development` ([#695](https://github.com/jakbuczarnecki/intergrax/pull/695), [#696](https://github.com/jakbuczarnecki/intergrax/pull/696), [#697](https://github.com/jakbuczarnecki/intergrax/pull/697)); matrix AUD column WB01–12 + partial W3 `CLS_PRE.md`; session E aggregate **8 FAIL / 4 BLOCKED / 0 PASS**; integrator `eqo/post-aud/matrix-cls`; docs only — no `EQO_CLOSED`, pytest not re-run, no product change.

## Operator waiver — docs merge platform CI

Operator decision **`waiver_both`** @ **2026-11-21** (coordinator chat): retroactive acceptance of docs/ticket merges **[#690](https://github.com/jakbuczarnecki/intergrax/pull/690)** (EQO wave-2 integration) and **[#691](https://github.com/jakbuczarnecki/intergrax/pull/691)** (wave-2 checkpoint + CI triage on this ticket) despite **red platform CI** on `development` at merge time (`ci-smoke`, `governance-fast`, `ent-audit-fast`). Docs gates were **green**: `ticket-governance`, `governance-policy-pin`.

**Going forward (docs-only integrator PRs under T-IGX-046-F / same merge-policy pattern):** merge when **`ticket-governance`** + **`governance-policy-pin`** pass; **do not** block on full `ci-smoke` / `governance-fast` / `ent-audit-fast` green when failure is acknowledged pre-existing platform drift outside the PR write set. Autonomous `gh pr merge` still requires either that docs gate pair green **or** a fresh operator waiver line on this ticket (see merge policy correction above). PR [#692](https://github.com/jakbuczarnecki/intergrax/pull/692) thread noted `AWAITING_OPERATOR_WAIVER` — resolved by this section.

## Checkpoint — EQO wave-8 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-8/integ` |
| Base | `origin/development` `569254840` (PR #698) |
| Lanes | `eqo/wave-8/w2` `a8fde582c`, `eqo/wave-8/x` `fe3a92adc`, `eqo/wave-8/w9` `b53793858` |
| Deliverables | W2-02 replay read; X memos English; W9-01 matrix prep (`EQO_NOT_CLOSED_EXACT_BLOCKERS`) |
| #680 | Track-only (no merge claim) |
| PR | _(pending push — docs gates `ticket-governance` + `governance-policy-pin`)_ |

## Checkpoint — EQO wave-6 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-6/integ` |
| Base | `origin/development` `2061c6d3a` (post #700) |
| Lane merged | `eqo/wave-6/exe-docs` — WB04 `2ab004c37`, WB06 `949a76b76`, WB09 `a13958b49` |
| Deliverables | EXE logs + `RUN_MANIFEST.json` refresh; `EQO_WAVE6_MANIFEST.md`; README index row 32 |
| WB06 | Non-zero pytest exit recorded; does **not** assert EQO closure |
| PR | [#704](https://github.com/jakbuczarnecki/intergrax/pull/704) merged `2026-11-21` |

## Checkpoint — EQO wave-12 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-12/integ` |
| Base | `origin/development` `292ca3348` (PR #704) |
| Lanes | `eqo/wave-12/w2` `a4fd55ade`, `eqo/wave-12/x` `feabdfc70`, `eqo/wave-12/w9` `23bbead01` |
| Deliverables | #680 track; X disposition reads; W9-01 C01–C25 sweep (`EQO_WAVE12_MANIFEST.md`) |
| #680 | Track-only (no merge claim) |
| Terminal | **No** `EQO_CLOSED` |
| PR | _(pending push — docs gates `ticket-governance` + `governance-policy-pin`)_ |

## Checkpoint — EQO wave-11 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-11/integ` |
| Base | `origin/development` `e1f850bdc` |
| Lanes merged | `eqo/wave-11/c`, `eqo/wave-11/w4` |
| C-01 / C-02 | Closure band finalized — `DISPOSITIONED_NOT_CLOSED` |
| C-04 | Formal `BLOCKED` record — `runs/c/C04_BLOCKED_RECORD.md` |
| C-CLS | `BLOCKED` disposition — `c/C_CLS.md` |
| W4-CLS | Registry 186/186 — `w4/W4-CLS.md` |
| Deliverables | `EQO_WAVE11_MANIFEST.md`, README phases 33–35, matrix § |
| pytest | out of scope |

**Docs gate policy:** merge when `ticket-governance` + `governance-policy-pin` pass (waiver pattern above).


## Checkpoint — EQO wave-9 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-9/integ` |
| Base | `origin/development` `00ec80d3e` (post PR #706 wave-12) |
| Lanes | `eqo/wave-9/cor-wb01` … `cor-wb12` (12 worktrees `eqo-w9-cor-wbNN`; missing origin lanes pushed at integrator close) |
| Deliverables | `runs/wbNN/COR_DRV.md`, `EQO_WAVE9_MANIFEST.md`, matrix §, README index row 37 |
| Scope | Docs disposition paths from `AUD_SESSION_E.md`; no product CL-01 |
| PR | [#707](https://github.com/jakbuczarnecki/intergrax/pull/707) merged |
| Integrator tip | `4e3b9c0a03736db5ba0b3030f23541926b2f4e8a` |
| Merge commit | `6b01b63f6264ec62602ba6349d532686dbf8c54c` |

## Checkpoint — EQO wave-10 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-10/integ` |
| Base | `origin/development` `e61e5136a` (includes wave-9 `6b01b63f`) |
| EXE lane | `eqo/wave-10/exe` @ `7b5f9efc7` |
| Workstreams | WB01, WB06, WB07, WB10, WB11 @ TRAIN_PIN (`uv sync --all-extras` on `eqo-w3`) |
| Deliverables | `EQO_WAVE10_MANIFEST.md`, `runs/wbNN/RUN_MANIFEST.json`, `pytest_wbNN_exe.log` |
| EXE | WB01 `1`; WB06 `0`; WB07/10/11 `1` |
| PR | [#708](https://github.com/jakbuczarnecki/intergrax/pull/708) merged 2026-11-21T09:13:04Z; [#709](https://github.com/jakbuczarnecki/intergrax/pull/709) merged 2026-11-21T09:20:17Z |
| Merge commit | 8a401496dc3cca7080b9d25fd720e4537beab4c7 (#708); 1cf91738f82602a7a946eb880a7148d2983291e5 (#709) |


## Checkpoint — EQO wave-13 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-13/integ` |
| Base | `origin/development` `ad9094036` |
| EXE lanes | `eqo/wave-13/exe-wb02` … `exe-wb12` @ TRAIN_PIN |
| Doc lanes | `eqo/wave-13/e`, `c04`, `w4` |
| Deliverables | `EQO_WAVE13_MANIFEST.md`, README row 39, matrix EXE refresh |
| EXE exits | WB02 `1`; WB03/05/08/12 `0` |
| Session E | **unchanged** |
| Wave 14 | **follow-on #712** |
| PR | [#711](https://github.com/jakbuczarnecki/intergrax/pull/711) merged 2026-11-21T09:51:01Z |
| Merge commit | 9a673b1b23ee283d096218146239db9782e1d7ad (#711) |

**Docs gates:** `ticket-governance` + `governance-policy-pin`.

## Checkpoint — EQO wave-14 (2026-11-21)

| Field | Value |
|-------|-------|
| Integrator branch | `eqo/wave-14/integ` |
| Base | `origin/development` `9a673b1b2` (post wave-13 #711) |
| CLS lanes | `eqo/wave-14/cls-wb01` … `cls-wb12` — terminal `W3_CLS.md` |
| DRV lane | `eqo/wave-14/drv` — doc-only DRV deferrals |
| W9 lane | `eqo/wave-14/w9` — C01–C25 sweep (no `FULLY_MET`) |
| Deliverables | `EQO_WAVE14_MANIFEST.md`, README row 40 |
| Session E | **unchanged** |
| Aggregate | **0 PASS** @ pin |
| Terminal | **No** `EQO_CLOSED` |
| PR | _(pending push)_ |

**Docs gates:** `ticket-governance` + `governance-policy-pin`.

## Lifecycle state

plan_approved
