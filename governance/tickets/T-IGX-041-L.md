# T-IGX-041-L — PDC-X batch B-P7-CX: re-pin of the COMPAT-X counters (debt item Z-D1) at the last product change

## Scope

Re-pin of the COMPAT-X counters named in debt item D1 of `PLAN_T-IGX-036-Z.md`, delegated by `DELEGATION-PDC-P7-REPIN.md` part (a): `ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES` 7356 to 7417, `REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES` 7355 to 7416, the remainder fingerprint, `ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES` 2031 to 2060, `ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES` 1933 to 1963, `ACCEPTED_CURRENT_HEAD_EXCLUSIONS` 5325 to 5357 and four paths of the accepted shim delta fixture. Measured on `63005e290` against the replay lineage `05549f9acb`: 64 raw candidates added and 3 removed, every one attributed to a ticketed change (B-D1, B-D4, B-D5, B-D6, B-D7, B-D8, MCP-SERVER-X and their repairs), none unclassified, no `defect.persisted_without_version`; the four new shim paths are `source.contains:adapter` heuristic hits. Re-pin with classification and an id-exact evidence fixture, plus negative tests; no gate, classifier or heuristic is changed. Tests and fixtures only, no product code; merges before the EBH-7 pin `P`. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-L.md`.

## Plan digest

Plan digest: 92d87572e0683afaccda381e7f497a1fb4c95c2e5ab2bf5dd2dca62be5019003

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-L.md`](plans/PLAN_T-IGX-041-L.md).

## Approved

Approved: operator-delegate @ 2026-11-15T14:57:19+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPIN.md@664b054ec20c8964edb8b3b5c786e1769fa67d70

Delegate checks (scope of `DELEGATION-PDC-P7-REPIN.md` read at `origin/development`, last changing commit `664b054ec`; signed at the batch branch, implementer commit `ee1f20bd6`; all passed before this line was written):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`92d87572e0683afaccda381e7f497a1fb4c95c2e5ab2bf5dd2dca62be5019003`).
- The plan write set (4 files: `_compat_x_semantic_identity_reconciliation.py`, `test_compat_x_pdc_x_p7_repin_delta.py`, `accepted_shim_delta_replay_x_05549f9.json`, `repin_pdc_x_p7_delta.json`) equals, as a list, the write set of batch row `B-P7-CX` in `PDC_IMPLEMENTATION_WBS.md` and in `scripts/qualification/pdc_x_wbs.py`; `pdc_x_wbs.py --check` is OK. The implementer commit changes only the ticket, the plan, the WBS, the DAG and the batch register `pdc_x_wbs.py`; nothing under `intergrax/`, `agents/`, `applications/`, `scripts/governance/` or `.github/`, and the plan write set contains none of them. `tests/qualification/compat_x/` and `tests/fixtures/` are not on the executable-identity allowlist of `verify_ebh7_p03_immutable_evidence.py` (lines 19 to 37), so the pull request must merge before `P`.
- The two red tests were run once each at the batch branch (product code identical to `63005e290`): `test_cx_r3_r1_combined_gate_passes_on_current_head` fails with raw 7417 vs 7356, exclusions 5357 vs 5325, semantic surfaces and remainder fingerprint `9dfd146f…`; `test_r5_p0_closed_world_gate_passes_on_current_head` fails with the four unexpected shim paths. The plan's explanation was recomputed by running `build_closed_world_report()` in a throwaway checkout of `05549f9acb` and in this tree: 7356/5325/2031/1933/7355 and fingerprint `9dfd146f…` (the accepted values) against 7417/5357/2060/1963/7416 and `8461d44e…`; on the gate's stable inventory identity 64 added and 3 removed (26 wire.persistence, 16 migration.mechanism, 8 public.export, 7 class.field.version, 4 compat.shim, 3 version.constant; removed: `RunArtifactBundle.schema_version` and two `online_evaluation_registry.py` wire sites); surfaces +30 −1; exclusions +68 −36 (8 exports, the rest wire sites); `unclassified_candidate_ids` empty and 6 `defect.persisted_without_version` at both commits; the four new shim paths carry `source.contains:adapter` with class `TRANSLATION_ONLY` and the cited lines (`admin_service.py:194`, `runtime_revision.py:18`, `mcp_host_mount.py:68`, `security_audit_trail.py:34`); the 25 accepted paths split 19/3/2/1 as stated; per-group counts and the owning commits (all ancestors of `63005e290`, subjects as named) match.
- Re-pin only: the plan moves six constants to measured values, adds four classified paths to the accepted shim delta (the loader ignores extra keys; the exact production-versus-fixture comparison in `test_compat_x_deprecation_r5_p0_gates.py` stays), adds an id-exact evidence fixture and negative tests (hidden removal, extra candidate, balanced substitution, fifth or missing shim path); probes P, Q, R and S and probe M exist and stay; no classifier, heuristic, discovery rule or gate meaning is edited; the guard that no `defect.persisted_without_version` or unclassified candidate hides behind a count is added.
- Cited code read: pin lines `_compat_x_semantic_identity_reconciliation.py:39-44`, `_compat_x_closed_world.py:318`, `_compat_x_ast_signals.py:547` and the signal logic of `build_compatibility_candidate_context` (defined at line 516), `PLAN_T-IGX-036-Z.md` line 53 (both quotes), the helper functions named in the plan. Wording notes, not factual errors: "candidate ids" of the 64/3 delta are stable inventory identities (wire sites without line numbers; raw `candidate_id` strings differ by 98 added and 37 removed, same net +61), so the evidence fixture must use `stable_inventory_candidate_identity`; the D1 group file list omits `agent_distribution/runtime_revision.py`, whose shim hit is the 18th D1 candidate; the comparison with the fixture sits at `test_compat_x_deprecation_r5_p0_gates.py` lines 100 to 101 (95 loads it); "origin/development is now `cba32d247`" is a point-in-time statement (it has moved by docs only since `63005e290`: no non-docs path differs).
- Operator-decision item of the plan (editing the accepted shim delta fixture): judged covered by `DELEGATION-PDC-P7-REPIN.md` part (a), because debt item Z-D1 itself names `test_r5_p0_closed_world_gate_passes_on_current_head`, which can only be made green by that fixture; the change adds four paths, each reproduced by test, and keeps the exact comparison. The signer's reading, open to the Operator.
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict. The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Merge gate (stated as the delegation record requires): pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness, EBH-7 identity gate), `headRefOid` equal to the verified SHA, before the EBH-7 pin `P`. The implementer re-measures at the `development` tip (freshness rule). Write-set amendments are not delegated. Not re-verified: the numbering of the standing authorizations.

## Test contract

```text
uv run pytest tests/qualification/compat_x -q
uv run ruff check tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run pyright tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Any change of the shim heuristic, a classifier, a discovery rule or the meaning of a gate; renaming the pinned constants; the `ACCEPTED_P0_*` historical pins; every file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`.
- The EBH-7 re-pin (T-IGX-041-J), the F10 classifier entries (T-IGX-041-K), the IIAC activation and its rebind, OD signatures, FRZ changes, certification.

## Lifecycle state

plan_approved
