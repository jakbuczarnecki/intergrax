# T-IGX-081-A — EC-W3 qualification execution 08 (EQO-W3 B-1)

## Scope

PRE_G2_BATCHED_QUALIFICATION_EXECUTION EC-W3-QUALIFICATION-EXECUTION-08: corrective host policy test semantics (Package A), GCA differential qualification, BND-03B semantic witness, Q01–Q09 qualification batches, mapped unit regression, mechanical obligation reconciliation, and evidence publication under `eqo_closure/runs/pre_g2_qualification_execution_08_245d4717/`.

## Plan digest

Plan digest: n/a (EC-W3-QUALIFICATION-EXECUTION-08 operator batch; queue source `pre_g2_host_policy_corrective_07_063b6ee9/REMAINING_QUALIFICATION_QUEUE.json`)

## Approved

Approved: operator-delegate @ 2026-11-27T12:45:00+01:00

Delegation: EC-W3-QUALIFICATION-EXECUTION-08 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-08 QUALIFICATION)

## Write set

```text
tests/unit/applications/test_eqo_reference_host_policy_fragments.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_qualification_execution_08_*/
governance/tickets/T-IGX-081-A.md
```

## Test contract

```text
uv run pytest tests/unit/applications/test_eqo_reference_host_policy_fragments.py tests/unit/applications/test_skill_bridge_wiring.py tests/unit/applications/test_arch_freeze_skill_bridge_package_d.py tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py tests/unit/applications/test_mcp_server_p4_lifecycle.py -q
uv run ruff check tests/unit/applications/test_eqo_reference_host_policy_fragments.py
git diff --check
uv run python -m scripts.governance.igx_gov ticket
```

Q01–Q09 and mapped groups per DES_PREREG in evidence run directory.

## Out of scope

Production code, GCA test edits, EDER/FRZ register edits, formal G2 certification, EQO_CLOSED promotion, lockfile changes.

## Lifecycle state

in_progress