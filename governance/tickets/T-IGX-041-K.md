# T-IGX-041-K — PDC-X batch B-P7-F10: B6-F10 C4 classifier entries for the three UNKNOWN paths

## Scope

Test-infrastructure ticket, preparatory to the final EBH-7 re-pin (T-IGX-041-J), by `DELEGATION-PDC-P7-REPIN.md` part (a) (Operator answer "Yes, I approve (a)-(c)", 2026-11-15). `certify_f10()` classifies 3 of the 3338 paths of `git diff --name-only 5a3a05347 HEAD` (the F10 range starts at `EBH7_PARENT_REPLAY_ACCEPTANCE_SHA`) as UNKNOWN, which blocks the C4 gate (`C4-UNCLASSIFIED`): `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py` and `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`. This ticket adds one exact-path `C4Rule` entry per path to `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` (`RUNTIME_BEHAVIOR` + `SECURITY_GOVERNANCE_TENANT`, `SECURITY_GOVERNANCE_TENANT`, `QUALIFICATION_INFRASTRUCTURE` respectively; reasons and evidence in the plan) and a new test file with positive and negative cases (a sibling or any unlisted path still yields UNKNOWN and a BLOCKER). Entries are added; the default classification, every existing rule and every regression map stay unchanged. Product code: no. It must merge before the pin `P` because `governance/tickets/` and the classifier input file are outside the executable-identity allowlist. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-K.md`.

## Plan digest

Plan digest: 924349b7f694436aae9bc0a8eafa44857d3e6f8ed0e8441fec41864395c1498e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-K.md`](plans/PLAN_T-IGX-041-K.md).

## Approved

Approved: operator-delegate @ 2026-11-15T14:01:39+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPIN.md@664b054ec20c8964edb8b3b5c786e1769fa67d70

Delegate checks (scope of `DELEGATION-PDC-P7-REPIN.md`, all passed before this line was written; read at the batch branch after fast-forward to `ac3d89fac`, base `63005e290`; delegation file read from `origin/development`, last changed by `664b054ec`). A first pass at `52d629199` was refused for one untrue sentence (the plan said the integration test imports `IncidentInvestigatorAgent`; it does not) and two wording points; the planner amended the plan in `ac3d89fac` and every check below was redone from scratch:

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`924349b7f694436aae9bc0a8eafa44857d3e6f8ed0e8441fec41864395c1498e`).
- Write set: the two lines of the plan's Write set block (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`, `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py`) equal, as lists, the `writes` tuple of `B-P7-F10` in `scripts/qualification/pdc_x_wbs.py` and the `Write set` of the `B-P7-F10` section in `PDC_IMPLEMENTATION_WBS.md`. `pdc_x_wbs.py --check`: OK.
- Forbidden paths: nothing under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/` or `.github/` in the write set; both files are tests.
- Classifier measurement reproduced by the delegate with its own probe (`certify_f10` on identity-neutral canonical inputs, live `git diff --name-only <accepted> HEAD`): for `5a3a05347..HEAD` (3340 paths) and `dc18e36d9..HEAD` (3126 paths; the plan's 3338 and 3124 plus the two new ticket and plan files) exactly the three named paths are UNKNOWN and `C4-UNCLASSIFIED` FAILS; with the three exact-path rules appended 0 paths are UNKNOWN and C4 PASSES. Four sibling paths (`intergrax/experiments/zz_unclassified.py` and its three analogues) stay UNKNOWN. The three paths lie outside the current pin range `5a3a05347..146e7acf1`.
- Factual sentences and file:line references re-read against the code (about 40 spot reads, no discrepancy): `classify_change` (`_ebh7_f10_ecap_normative_matrix.py:654-660`, exact-path or `/`-terminated prefix, longest wins, default UNKNOWN), `evaluate_c4`, `_ebh7_f10_canonical_inputs.py:942, 991`, `_constants.py:15`, `MANIFEST.json:8` (full path now given), the three last-change commits, the import evidence of `IncidentInvestigatorAgent` (`scenario.py:44`, `agent_factory.py:22`, `evidence_builder.py:20`, and the unit tests `test_architecture_scaffold_gate.py:17`, `test_autonomous_evidence_gathering.py:40`, `test_baseline_evidence_gathering.py:29`; no file under `intergrax/`, `agents/` or `applications/` imports it; the integration test only names a factory path string), the quotation of `PLAN_T-IGX-036-Y.md` ("one more directory entry in the classifier (a documentation-and-test-only follow-up), never a default change") and the plan's statement that exact-path rules are the narrower form of it.
- No gate is weakened or deleted: classification entries only, appended; the default classification, every existing rule, every regression map and every existing test stay byte-identical; each entry has a negative test (a sibling or any unlisted path still yields UNKNOWN and a BLOCKER).
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict (certifies nothing, does not close B6 or promote any FRZ criterion). The implementer (planner lane) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Ordering and merge gate (as the delegation requires): the ticket, the plan and the classifier change merge before the pin `P`; the pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation until the P8-04 rebind, EBH-7 identity until the final re-pin), `headRefOid` equal to the verified SHA.
- Note for the implementer: any later change of the plan changes the digest and voids this signature; write-set amendments are not delegated.

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py -q
uv run ruff check tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any default change of the classifier, directory entries for the three paths, a change of a class definition or regression map, the COMPAT-X counters (T-IGX-041-L), the re-pin (T-IGX-041-J), the IIAC audit, activation and rebind, FRZ promotion, OD signatures, certification.
- Everything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`, and every existing test (byte-identical).

## Implementation checkpoint

- Tests first (`98040d3c3`): `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py`, 14 cases. On the unchanged classifier 8 failed (the three positives, the three exact-path checks, `each_entry_occurs_exactly_once`, and the rule-removal negative) and 6 passed (the four sibling negatives, the default-is-unknown check and the regression binding): log `.tmp/proof/pdc-x-041k/red.log` (not committed).
- Entries: three exact-path `C4Rule` appended at the end of `C4_RULES`; nothing else in `_ebh7_f10_canonical_inputs.py` changed (19 inserted lines).
- Test contract: the new file plus `test_ebh_7_b6_f10_c1_c4_closure_gates.py` in one pytest command: 36 passed (`.tmp/proof/pdc-x-041k/green.log`); ruff check and format clean, pyright 0 errors; `pdc_x_wbs.py --check` OK.
- Probe on the implemented tree after merging `origin/development` (`.tmp/proof/pdc-x-041k/probe_after.log`): `5a3a05347..HEAD` 3402 paths, 0 UNKNOWN, C4 PASS; `dc18e36d9..HEAD` 3188 paths, 0 UNKNOWN, C4 PASS. `git diff --name-only 5a3a05347 146e7acf1` lists none of the three paths.
- No deviation from the signed plan.

## Lifecycle state

plan_approved
