# T-IGX-041-K — PDC-X batch B-P7-F10: B6-F10 C4 classifier entries for the three UNKNOWN paths

## Scope

Test-infrastructure ticket, preparatory to the final EBH-7 re-pin (T-IGX-041-J), by `DELEGATION-PDC-P7-REPIN.md` part (a) (Operator answer "Yes, I approve (a)-(c)", 2026-11-15). `certify_f10()` classifies 3 of the 3338 paths of `git diff --name-only 5a3a05347 HEAD` (the F10 range starts at `EBH7_PARENT_REPLAY_ACCEPTANCE_SHA`) as UNKNOWN, which blocks the C4 gate (`C4-UNCLASSIFIED`): `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py` and `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`. This ticket adds one exact-path `C4Rule` entry per path to `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` (`RUNTIME_BEHAVIOR` + `SECURITY_GOVERNANCE_TENANT`, `SECURITY_GOVERNANCE_TENANT`, `QUALIFICATION_INFRASTRUCTURE` respectively; reasons and evidence in the plan) and a new test file with positive and negative cases (a sibling or any unlisted path still yields UNKNOWN and a BLOCKER). Entries are added; the default classification, every existing rule and every regression map stay unchanged. Product code: no. It must merge before the pin `P` because `governance/tickets/` and the classifier input file are outside the executable-identity allowlist. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-K.md`.

## Plan digest

Plan digest: 924349b7f694436aae9bc0a8eafa44857d3e6f8ed0e8441fec41864395c1498e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-K.md`](plans/PLAN_T-IGX-041-K.md).

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

## Lifecycle state

plan_pending
