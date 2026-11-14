# PLAN T-IGX-036-X — PDC-P6 ECAP VBN-REACHABILITY: registration order in the ebh_5 package C v2 harness

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-F**), finding R-04 of `PDC_P6_ACCEPTANCE_RECORD.md` (ticket T-IGX-036-W, Part A), and batch B-REG-P (T-IGX-036-M, PR 505). Requirements: no new requirement rows. Split out by Operator decision ("Tak, test-only ticket").

## Problem

ECAP FULL on the integrated baseline is BLOCKED by one check, `VBN-REACHABILITY`: `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py::test_ebh5_pkgc_v2_dual_slug_integration[integration.embedding_provider]` and `::test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence[integration.embedding_provider]` fail with `UnknownIntegrationError: Integration slug 'ollama' is not registered in the catalog`.

Cause: B-REG-P made `register_ollama_embedding_provider_integration` contribute to the single `ollama` catalog row with `augment_integration_contract_specs`, which requires that the model-serving row (`model_serving_runtime/ollama`) is registered first (this is also the order of `register_default_integrations(preset="full")`). The harness helper `register_catalog_slug` in `tests/qualification/ebh_5/_package_c_v2_primitives.py` clears the catalog and then registers only the embedding provider, so the augment call finds no row. The product behavior is the intended outcome of B-REG-P; the harness precondition is stale.

## Deliverables

1. **Harness precondition.** `register_catalog_slug` registers the owning row first when the provider contributes to an existing row: for `IntegrationCategory.EMBEDDING_PROVIDER` and slug `ollama` it first calls the model-serving `ollama` registration (the function exposed by `intergrax/integrations/providers/model_serving_runtime/ollama/register.py`), then the embedding-provider registration. All other category/slug pairs behave as before. The assertions of the two failing tests and of `test_ebh_7_int2e_bundle_regression.py` are not changed.
2. **Pinned regression test.** Under `tests/qualification/pdc_x/p6_ecapfix/`: on a cleared catalog, registering the embedding provider alone through the harness resolves both `ollama` and `hf`; the catalog holds exactly one `ollama` row carrying the model-serving and the embedding contribution; registering the embedding provider without the owning row still raises `UnknownIntegrationError` (so the product contract is not loosened).
3. **Gate re-run.** One `uv run pytest` over the three ebh_5 package C v2 modules named in the ECAP finding plus the new tests, then the ECAP gate file `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` once in full (86 tests); the 2 failing cases are green and the 84 others stay green. The full ECAP run is not repeated.

Design constraints: test and harness code only; no change under `intergrax/`, `agents/`, `applications/`; no gate weakened (a precondition that mirrors the real registration order, plus a negative test); no new authority.

## Write set

`tests/qualification/ebh_5/_package_c_v2_primitives.py`, `tests/qualification/pdc_x/p6_ecapfix/`

Nothing outside this set changes. A need to write elsewhere (for example a product file of the embedding provider, or another ebh_5 module) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Write the pinned regression test first and show it failing for the right reason.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files; no file-level `# ruff: noqa`. Do not pass `-p no:cacheprovider`.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-036-X.md` trailer and no AI attribution, set the upstream to `origin/development` before the first push, push with an explicit refspec, open a PR against `development` and never merge. The orchestrator merges under `DELEGATION-PDC-STANDING.md` authorization 3 (known red classes only: IIAC activation).
4. This batch merges before the EBH-7 pin of T-IGX-036-W is taken, because `tests/qualification/ebh_5/` is outside the executable-identity allowlist.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- The two failing ECAP cases pass and the whole `test_ebh_5_behavioral_package_c_v2.py` module is green; the new tests pass.
- No product path changed; no gate weakened.
- Test log under `.tmp/proof/pdc-x-p6/ecapfix/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_ecapfix tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product changes to the embedding provider or the catalog; the `pdc_x` pytest process that does not exit after its summary line (recorded as a debt item); the EBH-7 re-pin; IIAC activation (PDC-P7-00); FRZ changes; OD signatures; certification.
