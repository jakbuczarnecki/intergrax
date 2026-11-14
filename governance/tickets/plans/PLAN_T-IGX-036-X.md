# PLAN T-IGX-036-X — PDC-P6 ECAP VBN-REACHABILITY: resolving the `ollama` embedding provider through the catalog

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-F**), finding R-04 of `PDC_P6_ACCEPTANCE_RECORD.md` (ticket T-IGX-036-W, Part A), and batch B-REG-P (T-IGX-036-M, PR 505). Requirements: PDC-D1-R022 (single `ollama` catalog row) is completed, no new requirement rows. Split out by Operator decision ("Tak, test-only ticket"), extended with the product fix by Operator decision of 2026-11-14 ("Tak, rozszerz o produkt", Amendment 1; the implementer's stop report showed the test-only fix is not sufficient).

## Problem

ECAP FULL on the integrated baseline is BLOCKED by one check, `VBN-REACHABILITY`: `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py::test_ebh5_pkgc_v2_dual_slug_integration[integration.embedding_provider]` and `::test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence[integration.embedding_provider]`. There are two layers:

1. **Harness order.** B-REG-P made `register_ollama_embedding_provider_integration` contribute to the single `ollama` row with `augment_integration_contract_specs`, which requires the model-serving row (`model_serving_runtime/ollama`) to exist. `register_catalog_slug` in `tests/qualification/ebh_5/_package_c_v2_primitives.py` clears the catalog and registers only the embedding provider, so the harness gets `UnknownIntegrationError: Integration slug 'ollama' is not registered in the catalog`.
2. **Product defect in the shared row.** With the owning row registered first, `resolve` for the embedding category fails with `TypeError: Integration factory for slug 'ollama' (category 'embedding_provider') returned OllamaModelServingRuntimeIntegration, expected EmbeddingProviderIntegrationContract`. The row's factory is the model-serving factory from `register_from_manifest`; `create_ollama_model_serving_runtime` ignores `integration_category` and always returns the model-serving integration. B-REG-P merged the two contributions into one row but left no per-category dispatch, so resolving `ollama` as an embedding provider through the catalog can never work. The `reg_p` tests check only that the embedding manager builds, which is why this was not caught.

## Deliverables

1. **Pinned regression tests first.** Under `tests/qualification/pdc_x/p6_ecapfix/`: on a cleared catalog, with the model-serving row registered first and then the embedding contribution, (a) the catalog holds exactly one `ollama` row carrying both categories and both contract specs; (b) `resolve_from_profile` for the embedding category returns an `EmbeddingProviderIntegrationContract` instance and for the model-serving category an `OllamaModelServingRuntimeIntegration`; (c) registering the embedding contribution without the owning row still raises `UnknownIntegrationError` (the contract is not loosened). The tests fail on the unchanged code for the two reasons above.
2. **Product fix.** The factory registered for the single `ollama` row dispatches on `integration_category`: for `EMBEDDING_PROVIDER` it returns the integration built by the embedding provider's existing factory (`create_ollama_embedding_provider_integration` in `embedding_provider/ollama/bundle.py`); for every other category it keeps the model-serving behavior unchanged, including the legacy keyword arguments. The slug stays owned by `model_serving_runtime/ollama`; there is still exactly one `ollama` row. The smallest change inside the four files of the write set wins; it is recorded in the checkpoint.
3. **Harness precondition.** `register_catalog_slug` registers the owning model-serving `ollama` row before the embedding contribution for `EMBEDDING_PROVIDER`/`ollama`; all other category/slug pairs behave as before. No assertion of the existing tests changes.
4. **Gate re-run.** One `uv run pytest` over the new tests, `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` (86 tests), `tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py`, `tests/qualification/pdc_x/reg_p` and the unit tests that exercise the ollama catalog row, found by `git grep` for `model_serving_runtime.ollama`; the two former ECAP failures are green and nothing else turns red. The full ECAP run is not repeated; the orchestrator re-runs only the ECAP gate file for the record. Tests that need `duckdb` need a `dev-ci` synced environment; the implementer syncs it once (`uv sync --extra dev-ci --frozen`) or reports them as environment reds.

Design constraints: the catalog keeps one row per slug and remains the single owner; no change to `intergrax/integrations/registry/` or `intergrax/integrations/_shared/`, no new registry, factory registry, scheduler, engine, trust, policy or audit authority; public catalog contracts do not change; no gate weakened; the fix does not change any other provider or category.

## Write set

`intergrax/integrations/providers/model_serving_runtime/ollama/bundle.py`, `intergrax/integrations/providers/model_serving_runtime/ollama/register.py`, `intergrax/integrations/providers/embedding_provider/ollama/bundle.py`, `intergrax/integrations/providers/embedding_provider/ollama/register.py`, `tests/qualification/ebh_5/_package_c_v2_primitives.py`, `tests/qualification/pdc_x/p6_ecapfix/`

Nothing outside this set changes. A need to write elsewhere (the catalog, the shared factories, the plugin registration helper, other ebh_5 modules) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Write the pinned regression tests first and show them failing for the right reasons.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files; no file-level `# ruff: noqa`. Do not pass `-p no:cacheprovider`.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-036-X.md` trailer and no AI attribution, set the upstream to `origin/development` before the first push, push with an explicit refspec, open a PR against `development` and never merge. The orchestrator merges under `DELEGATION-PDC-STANDING.md` authorization 3 (known red classes only: IIAC activation). Product paths change here, so the IIAC activation stays red until PDC-P7-00.
4. This batch merges before the EBH-7 pin of T-IGX-036-W is taken: `intergrax/` and `tests/qualification/ebh_5/` are outside the executable-identity allowlist.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- The two former ECAP failures pass; the whole `test_ebh_5_behavioral_package_c_v2.py` module and the `reg_p` tests are green; the new tests pass.
- `resolve` for `ollama` as an embedding provider returns an embedding provider integration; the model-serving behavior of the row is unchanged.
- No file outside the write set changed; no gate weakened; no new authority.
- Test log under `.tmp/proof/pdc-x-p6/ecapfix/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_ecapfix tests/qualification/pdc_x/reg_p tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Catalog, registry and shared-factory changes; other providers and categories; the `pdc_x` pytest process that does not exit after its summary line (debt item); the EBH-7 re-pin; IIAC activation (PDC-P7-00); FRZ changes; OD signatures; certification.
