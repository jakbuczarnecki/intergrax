# RC-0 evidence dossier

Read at `origin/arch-freeze/rw-train` `28c27cb896d02e0185b9fa02047411fa152f6469`. Worktree `D:/Projekty/_wt/intergrax/af-rc0`, branch `af2/rc0-continuation`.

No PASS is claimed for a measurement that was not run, and no run below is a G2, FCS, M3, or ARCH_FREEZE_CLOSED statement. OD-AF-RC-KR08 was not granted.

## 1. Supply-ab probe secret scan

Probe worktree `D:/Projekty/_wt/intergrax/rw-supply-ab-s0` (detached). `git status` showed two untracked tests and nothing else untracked. Those two files were hashed and scanned. They were not copied into this dossier. No probe log was published.

Pattern set: PEM private key, AWS `AKIA` access key, GitHub token prefixes, Slack `xox` token, `sk-` key, `Bearer` token, URL userinfo, and an assignment of `api_key` / `secret` / `password` / `token` / `passwd` to a quoted value of at least 8 characters.

| File | Bytes | SHA-256 | Verdict |
| --- | ---: | --- | --- |
| `tests/unit/agent_distribution/test_rw_supply_ab_probe.py` | 38140 | `4df984e73c67b18086f475f3ad6d4ba12b6dfdf877a3fa9debe7c24361537395` | CLEAN |
| `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` | 39587 | `f5c7f3a4372a0261b26b3d5d394e6ae29068a5eed40ac543a495c26b234af209` | CLEAN |

CLEAN means zero hits for that pattern set.

## 2. PT-E03 — `NOT_MERGED`

Card: ENT-AUDIT acceptance conftest hook and debt-gate rewrite (A6-001, A6-006), under `tests/qualification/ent_audit_x/acceptance` and `scripts/maintenance`.

On `origin/arch-freeze/rw-train`:

- `git log --grep=PT-E03`, `--grep=T-ENF-3`, `--grep=A6-001`, and `--grep=A6-006` are empty.
- `git log --grep=044 --grep=conftest --all-match` returns only `90f6dfdb3` (`fix(ent-audit-x): exclude BND-03 smoke from 85/85 semantic matrix marker`), which is not this hook.
- `git ls-files` has no `conftest.py` under `tests/qualification/ent_audit_x`.
- `git grep A6-001` in `tests/qualification/ent_audit_x`, `scripts/maintenance`, and `governance/tickets` hits only `governance/tickets/T-IGX-043-I.md` and its plan.
- `scripts/maintenance` has no `A6-001`, `A6-006`, or debt-gate rewrite hit.

There is no 044 ticket file for E03 on this HEAD (the series files present are 044-A, 044-B, 044-C, 044-E, 044-F).

## 3. T-IGX-051-E Part D / EX-D1-07 — `NOT_IN_PR_674_DIFF`

PR #674 is merged as `e7dd2f0bc` (`Merge pull request #674 from jakbuczarnecki/af2/rw-recovery`).

The firm Part D change is removal of the neutralization of `revision_state` and `activated_at` in `intergrax/applications/_shared/durable_registry_projection_input_store.py`, plus a full binding compare (plan section 6, NT-RW-REC-05).

The merge diff against the first parent does not change that store. `git log -S _in_persisted_representation` on the train last touches the function in `f38a91f14` (T-IGX-041-A), not in #674. At HEAD the function still copies `revision_state` to `VALIDATED` and `activated_at` to `None` (lines 40–41).

What #674 did add is `6d6160657` (`test(applications): pin the control leg of NT-RW-REC-05; part D held`). `tests/unit/applications/test_rw_recovery_durable_cas_projection.py` pins the identical-bundle accept and the different-slug refusal, and its module docstring says the red legs (`revision_state`, `activated_at`, a binding beyond the slug) are held under the plan section 6 STOP and Q-RC-02. Those red legs are not in the diff.

## 4. T-IGX-043-F — `NOT_IMPLEMENTED_ON_TRAIN`

The ticket's write set is `tests/qualification/config_x/_config_x_semantic_production_scan.py` plus new `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` (NT-FC3-CX1..12). Lifecycle state on this HEAD is `draft`. The approval block is still the pending placeholder.

On the train:

- `git log --grep=043-F` and `--grep=T-IGX-043-F` are empty.
- `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` is not in HEAD.
- `tests/qualification/config_x` has no `NT-FC3-CX`, `vendor_slug`, `fc3_vendor`, or `A4-019` hit.
- History of the scan file on the train stops at the CONFIG-X-FINAL commits (`fac9a43f3`, `de5701f09`, `dc42728a7`), none of which is this ticket.

No train commit implements the ticket.

## 5. Published-evidence search — `BLOCKED_EVIDENCE`

`git ls-files` on this HEAD has no tracked `closed-world` result, `bnd_closure`, `typ_measure`, or RW-PLAN-AUDIT verdict. `governance/tickets/T-IGX-050-E.md` is absent. RM499_ENTRY_MAP.md section 7 says the closed-world, boundary, TYP, S1, and B-1 measurements live under gitignored `.tmp/session/arch-freeze/` and are not repository records. The untracked `RESULT.md` that exists only in a local session folder was measured at `7960ae7f6c82da60ca9acdc0ceebf63810002a42`, not at this HEAD, and its own header says no commit and no PR. That file is not a PASS on the train.

These commands were not run in RC-0.

| ID | Status | Replay command |
| --- | --- | --- |
| T-AF2-CLOSED-WORLD | BLOCKED_EVIDENCE | `uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py --strict-certify --output-dir .tmp/session/arch-freeze/closed-world/b1` |
| T-AF2-BND-CLOSURE | BLOCKED_EVIDENCE | `python -I bnd_closure.py <repo_root> <out_json>` as documented in the untracked session script `.tmp/session/arch-freeze/closed-world/bnd_closure.py` (not tracked on the train) |
| T-AF2-TYP-MEASURE | BLOCKED_EVIDENCE | `python -I typ_measure.py <repo_root> <out_json>` and `python -I measure_typ05.py <root> <out_json>` from the untracked session scripts `.tmp/session/arch-freeze/closed-world/typ_measure.py` and `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py` (neither is tracked; `git ls-files` has no `typ_measure`) |
| PT-V02 | BLOCKED_EVIDENCE | No shell line and no C-3 node id are published on the train. The published binding is HIST01: C-1 module `tests/unit/runtime/tools/test_tools_side_effect_safety.py` and C-2 node `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py::test_f3_streamable_http_parent_cancel_during_catalog_child`. Command limited to those published nodes: `uv run pytest tests/unit/runtime/tools/test_tools_side_effect_safety.py tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py::test_f3_streamable_http_parent_cancel_during_catalog_child` |
| PT-V03 | BLOCKED_EVIDENCE | No published shell line for the WP-5 set. The harness_01 directory named by the card is `uv run pytest tests/qualification/harness_01`. The card's `ebh_5 etc.` has no single tracked command. |
| PT-V04 | BLOCKED_EVIDENCE | No single published command covers ROADMAP-REPLAY-X counters, STATE-X/TRACE-X 7 nodes, tenant_x, and F10. The tracked F10 check is `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py --check`. That one script is not the whole node. |
| PT-V05 | BLOCKED_EVIDENCE | WP12 names ENT-AUDIT/IIAC small bounded checks, the impact-map baseline loop, and a measured semantic-wrapper run (AR-11, AR-12, AR-13). No tracked shell entry point for that set was found. |
| PT-V06 | BLOCKED_EVIDENCE | WP12 names a replay of the ticket-approval validators over the 115 existing tickets (AR-14). No tracked command iterates those 115 tickets. `tests/unit/governance/test_ticket_approval_validator.py` is a unit suite, not that replay. |
| T-AF2-RW-PLAN-AUDIT | BLOCKED_EVIDENCE | DAG type SESSION. No PASS verdict file is tracked. Replay is an independent session that recomputes WP-12 phase check 3 with the strict criterion over the RW plan and the FRZ matrix. No tracked CLI implements that check. |

## 6. EDER packages

Source: `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl` at the HEAD above. SHA-256 `f4949fbeb115b2892b8d04a3cc4b6bd0ebfa7d398dd65463bda055df2234555b`, 3377921 bytes, 1073 rows, every `lifecycle_state` is `INGESTED`.

The register has no field named `impact` or `owner`. Grouping uses `frz_arch_freeze_impact`, `disposition`, `next_track`, `plan_nodes`, and `owner_ref`. Full counters are in `EDER_CLASSIFICATION.json`. The register was not edited. No ticket was created. No owner was assigned.

Impact totals: UNKNOWN 489, CONDITIONS 325, NONE 147, BLOCKS 112.

### EDER-CLASS-1 — 61

Definition that matches the plan's 61 UNCLASSIFIED CONDITIONS: `frz_arch_freeze_impact=CONDITIONS`, `disposition=TRIAGE_PENDING`, `next_track=UNCLASSIFIED`. There is no disposition token `UNCLASSIFIED`. All 61 have empty `plan_nodes`.

`owner_ref`: DOMAIN:TOOLS 10, MCP-SERVER-X 7, TENANT-X 5, DIST-SCALE-X 5, EBH-6 5, EBH-7 5, ROADMAP-REPLAY-X 4, DOMAIN:LLM-BOUNDARY 3, DOMAIN:INTEGRATIONS 3, DOMAIN:RAG 3, DOMAIN:MEMORY 3, HARNESS 3, TRACE-X 3, EBH 2.

### EDER-CLASS-UNKNOWN — 489

`frz_arch_freeze_impact=UNKNOWN`.

Disposition: TRIAGE_PENDING 251, CURRENCY_UNVERIFIED 154, ACCEPTED_AS_STATED_RECORD_REQUIRED 42, DOC_PATCH_PENDING 17, DECISION_PENDING 10, STALE_SNAPSHOT_REVERIFY 6, VERIFICATION_PENDING 3, REQUALIFY_PENDING 2, FIX_PENDING_AFTER_PDC 2, DEFERRED_AS_STATED_IMPACT_CLASS_REQUIRED 2.

`next_track`: UNCLASSIFIED 367, DECISION 46, ACCEPTED_LIMITATION 42, DOC_PATCH 24, VERIFY_TEST 4, CODE_FIX 4, REQUALIFY 2.

`plan_nodes`: empty 464; the other 25 rows are the 14 non-empty keys in the JSON (PT-D07 6, PT-B02|PT-B2W 3, PT-D09 3, PT-D10 2, DS-5 2, and nine keys of 1).

`owner_ref`: 38 distinct values. Largest: PROD-Q 142, OPERATOR 54, ARCH-FREEZE 26, PDC-X 21, QUAL-X 18, HARNESS 18, ENT-AUDIT-X 15, SCENARIO-FACTORY-X 14. The rest are in the JSON. Seven of the 489 also have an empty `owner_ref` and are therefore inside EDER-OWNER-1 as well.

### EDER-OWNER-1 — 45

`owner_ref` missing or blank. Impact: CONDITIONS 21, NONE 10, BLOCKS 7, UNKNOWN 7. Same split as the continuation plan.

Disposition: DEFERRED_AS_STATED 11, DECISION_PENDING 7, DOC_PATCH_PENDING 6, CURRENCY_UNVERIFIED 6, STALE_SNAPSHOT_REVERIFY 6, ACCEPTED_AS_STATED_RECORD_REQUIRED 4, VERIFICATION_PENDING 3, DEFERRED_AS_STATED_IMPACT_CLASS_REQUIRED 1, TRIAGE_PENDING 1.

`next_track`: DECISION 18, DOC_PATCH 14, UNCLASSIFIED 5, VERIFY_TEST 4, ACCEPTED_LIMITATION 4.

`plan_nodes`: empty 35; non-empty keys are in the JSON (PT-D07 3, DS-Q 2, and five keys of 1).

Existing fallback fields, not an assignment: `accountable_person` is empty on all 45. `plan_owner_text` is present on 6. The other 39 have neither. Of those 39, 31 have a non-empty `linked_finding_ids`. RC-0 does not turn that link into an owner and does not take the operator decision the plan leaves for the remainder.

### CODE_FIX

`FIX_PENDING_AFTER_PDC`: 19 with impact BLOCKS and 6 with impact CONDITIONS. They stay on the existing W3 tickets named in the continuation plan. The 2 UNKNOWN rows with that disposition were grouped, not reticketed.

## 7. KR-18

Node: `tests/unit/architecture/test_ebh_2d_d_application_contract_surface_recertification.py::test_whole_surface_forbids_contract_to_implementation_edges`.

Command, one process, lock `D:/Projekty/_wt/_locks/pytest.lock` created before the run and removed after it:

```text
uv run --frozen pytest tests/unit/architecture/test_ebh_2d_d_application_contract_surface_recertification.py::test_whole_surface_forbids_contract_to_implementation_edges -q --tb=line
```

Result on this HEAD: `1 passed in 8.15s`. Process exit 0. The log was left under gitignored `.tmp/session/af-rc0/kr18.log` and is not published. The worktree had no `.venv`; `uv` created one (gitignored) and installed the frozen set. This is a local replay of the one node. It is not PT-Q02 and not an independent acceptance. The test was not edited.

## 8. RC-0 exit

Section J exit: every UNKNOWN from section A has a status other than UNKNOWN, or an explicit BLOCKED_EVIDENCE with a replay command.

| Item | Status |
| --- | --- |
| PT-E03 | NOT_MERGED |
| Part D / EX-D1-07 | NOT_IN_PR_674_DIFF |
| T-IGX-043-F | NOT_IMPLEMENTED_ON_TRAIN |
| T-AF2-CLOSED-WORLD, T-AF2-BND-CLOSURE, T-AF2-TYP-MEASURE, PT-V02, PT-V03, PT-V04, PT-V05, PT-V06, T-AF2-RW-PLAN-AUDIT | BLOCKED_EVIDENCE, commands in section 5 |

That exit record is met. It does not open RC-1 and it does not close the freeze.
