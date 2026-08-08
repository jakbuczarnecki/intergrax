---
id: IJ-2026-06-11-037
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-OPS-1
status: completed
commit: pending
adr: none — extends V-CG lineage/impact; no new graph taxonomy
---

# Sprint 12 — STRICT capability graph deploy gate

## Operator request

Continue Tier-3 application architecture sprint queue: APP-OPS-1 — environment capability graph with blast-radius STRICT deploy CI gate.

## Summary

- `capability_graph_deploy_gate.py` — `build_environment_capability_deploy_report`, `validate_strict_capability_graph_deploy`, `check_strict_product_capability_graph`.
- STRICT product hosts: assembly validation + roster agents in graph + impact report + blocked lifecycles (EXPERIMENTAL/DEVELOPMENT/CANDIDATE/DEPRECATED/RETIRED).
- `wire_environment_capability_graph` alias on `capability_graph_wiring.py`.
- `scripts/gates/check_capability_graph_strict_deploy.py` wired into `check_application_production_gates.py`.

## Project impact

STRICT product manifests are CI-gated for capability graph completeness and roster lifecycle posture before deploy. Enables ops blast-radius review per architecture §50.1 without forking ACP §19 graph model.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §50.1 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-OPS-1 · §6.2y step 9 |

## Changed artifacts

- `intergrax/applications/_shared/capability_graph_deploy_gate.py`
- `intergrax/applications/_shared/capability_graph_wiring.py`
- `scripts/gates/check_capability_graph_strict_deploy.py`
- `scripts/gates/check_application_production_gates.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_capability_graph_deploy_gate.py \
  tests/unit/scripts/test_check_capability_graph_strict_deploy.py -q
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (8 tests).

## Risks and follow-ups

- CI gate wires with lab integration profile to avoid optional Neo4j driver — production deploy still uses product integration bindings.
- APP-EVOL-4 will tighten lifecycle to PRODUCTION + certification; APP-OPS-2 next in queue.
