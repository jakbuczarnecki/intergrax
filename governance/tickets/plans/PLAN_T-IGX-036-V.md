# PLAN T-IGX-036-V — PDC-P6 ENT-AUDIT impact-map coverage

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-G**), XD finding 8 of ticket T-IGX-036-S (`test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`, a strict `xfail`), split out of ticket T-IGX-036-T by Operator decision ("Tak, osobny ticket"). Requirements: no new requirement rows.

## Problem

The shipped CHANGED-SINCE impact map (`changed_since_impact_map.v1.json`, built by `build_impact_map_from_manifest`) contains only prefixes derived from manifest entrypoints plus a fixed table of platform modules. About 124 product modules changed by the PDC campaign (hosting, agent_distribution, llm_adapters, runtime/nexus/artifacts, runtime/security, memory, and others) match no prefix, so a change to them selects only the invariant floor. The check manifest must stay at exactly 85 checks (`_validate_manifest`), so no new check can be added.

## Deliverables

1. **Declarative domain rules.** `impact_map.py` gains a typed constant table of product prefix to existing check ids (for example the layer-boundary, execution-engine, security-audit, observability, tenant and type/contract checks that already run against that area), merged by `build_impact_map_from_manifest`. Each prefix names only check ids that exist in the shipped manifest, and each rule is justified in a one-line comment by the existing entrypoint that exercises that area. No new check, no change to the 85 count, no change to `manifest.py` or the manifest JSON.
2. **Regenerated SSOT.** `changed_since_impact_map.v1.json` is regenerated with `uv run python scripts/enterprise_audit/build_impact_map.py --emit-ssot` (the flag that exists; there is no `--write`), not edited by hand.
3. **Coverage test.** Under `tests/qualification/pdc_x/p6_impact/`: every module in the PDC-touched set that the XD-12 test computes maps to at least one check id; every check id used by the new rules exists in the manifest; the SSOT equals the generator output.
4. **XD-12 flips.** The strict `xfail` marker of the finding-8 case in `test_xd_12_evol_ent_audit.py` is removed, so it runs as a plain passing test; only the marker and its reason text change.
5. **Existing impact-map test.** `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py` is edited only if it pins content that the regeneration changes; a gate gains coverage, none is weakened.

Design constraints: a gate may be strengthened, none weakened; fail-closed prefixes and `entrypoint_covered` semantics do not change; no new registry, scheduler, engine, trust, policy or audit authority.

## Write set

`intergrax/platform/enterprise_audit/impact_map.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`, `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py`, `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py`, `tests/qualification/pdc_x/p6_impact/`

Nothing outside this set changes. A need to write elsewhere (for example `manifest.py`, the manifest JSON, `changed_since.py`) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Write the coverage test first, under `tests/qualification/pdc_x/p6_impact/`.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Do not pass `-p no:cacheprovider`.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-036-V.md` trailer and no AI attribution, push to a branch and open a PR against `development`. Start from `development` after T-IGX-036-T and T-IGX-036-U are merged. The orchestrator merges under `DELEGATION-PDC-STANDING.md` authorization 3 (known red classes only: IIAC activation).
4. Record the implementation SHA; the publication SHA is recorded after the merge.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- XD-12 finding 8 runs as a plain passing test; the coverage test passes; the ENT-AUDIT unit tests (`tests/qualification/ent_audit_x`) pass.
- The manifest still has exactly 85 checks; no gate weakened; no new authority.
- The full `tests/qualification/pdc_x` suite still passes apart from known red classes.
- Test log under `.tmp/proof/pdc-x-p5/B-P6-G/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_impact tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py tests/qualification/ent_audit_x -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Changing the manifest or its 85-check rule; new checks; the IIAC protocol header and activation (PDC-P7-00); FRZ changes; OD signatures; certification.
