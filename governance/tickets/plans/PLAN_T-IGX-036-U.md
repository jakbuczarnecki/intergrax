# PLAN T-IGX-036-U — PDC-P6 batch B-P6-M: model binding

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-M**) and the XD-01 findings 9 and 10 recorded in the B-P6 report (ticket T-IGX-036-S, PR 514, test file `test_xd_01_agent_model.py`). Companion of T-IGX-036-T, which it follows. Requirements: no new rows; XD-01 scenario-level (Agent and Model).

Created on the Operator decision "A separate ticket" (chat, original wording in Polish: "Osobny ticket"), in answer to the question what to do with XD findings 9 and 10 after they were left out of T-IGX-036-T. The two findings add functional scope and one of them lies under the hotspot `intergrax/contracts/`; the Operator's decision covers a ticket for exactly these findings and exactly the two hotspot files named in the write set (`intergrax/contracts/agent_contract_meta.py`, `intergrax/applications/_shared/llm_resolver.py`).

A merge of this batch changes product paths; the IIAC activation is already invalidated and PDC-P7-00 re-activates it.

## Deliverables

1. **Revision pins the model profile (XD finding 9).** `RuntimeRevision` gains an optional `llm_profile_digest` (default `None`, so existing revisions and callers that build `RuntimeRevision` stay valid). `build_application_revision` in `admin_service.py` sets it from the environment's LLM profile when one is configured, and activation verifies it against the profile that resolves at activation time and fails closed on a mismatch. A config edit that changes the model therefore changes the revision identity or blocks activation. Revisions without a pinned digest keep working (additive).
2. **Required model features are checked at binding time (XD finding 10).** `AgentContract` in `agent_contract_meta.py` gains an optional `required_llm_features` (default empty, additive; existing contracts unchanged). The resolver that binds an adapter to an agent (`resolve_llm_adapter` in `llm_resolver.py`) accepts the agent's required features and calls `LLMAdapterRegistry.require_capabilities` before returning the adapter, so a capability mismatch from a provider swap fails closed at composition, not at the first unsupported call. When no feature is required, behavior is unchanged.
3. **Flip the XD-01 stand-in cases.** The two strict `xfail` cases for these findings in `test_xd_01_agent_model.py` become plain passing tests (only the marker and reason change; no assertion is weakened).
4. **Stop rule.** If a repair needs a path outside the write set (for example a persistence schema or a store that serializes `RuntimeRevision`), the batch stops and the orchestrator decides between an amendment under `DELEGATION-PDC-P6-MODEL.md` and a return to the Operator.

Design constraints: smallest change per item; additive public symbols with safe defaults; no new registry, scheduler, engine, trust, policy or audit authority; `LLMAdapterRegistry` stays the single owner of capability checks; no weakening of any gate.

## Write set

`intergrax/agent_distribution/runtime_revision.py`, `intergrax/agent_distribution/admin_service.py`, `intergrax/contracts/agent_contract_meta.py`, `intergrax/applications/_shared/llm_resolver.py`, `tests/qualification/pdc_x/xd/test_xd_01_agent_model.py`, `tests/qualification/pdc_x/p6_model/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Start from `development` after T-IGX-036-T is merged. Write the tests first, under `tests/qualification/pdc_x/p6_model/`.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Do not pass `-p no:cacheprovider`.
3. Commit with `Ref: governance/tickets/T-IGX-036-U.md` (no AI-tool attribution lines, repo rule), push to a branch and open a PR against `development`. The orchestrator merges it under `DELEGATION-PDC-STANDING.md` authorization 3 (known red classes only: IIAC activation).
4. Record the implementation SHA; the publication SHA is recorded after the merge.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Findings 9 and 10 each have a passing test at the implementation SHA; the two XD-01 stand-in cases are plain passing tests.
- Existing agent contracts and revisions without the new fields behave as before (tests for the defaults).
- Layer boundaries hold; no new authority; no weakened gate.
- The full `tests/qualification/pdc_x` suite still passes.
- Test log under `.tmp/proof/pdc-x-p5/B-P6-M/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_model tests/qualification/pdc_x/xd/test_xd_01_agent_model.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Anything in T-IGX-036-T; persistence schema migrations; the IIAC protocol header and activation (PDC-P7-00); FRZ changes; OD signatures; certification.
